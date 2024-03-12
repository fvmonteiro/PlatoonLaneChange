from __future__ import annotations

from enum import Enum
from typing import Mapping

import numpy as np

import configuration
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


class LongitudinalController:

    class States(Enum):
        NOT_INITIALIZED = 0
        CRUISE = 1
        VEHICLE_FOLLOWING = 2

    def __init__(self, vehicle: fsv.FourStateVehicle, max_brake: float):
        self.vehicle = vehicle
        self._kg = 0.2
        self._kv = 0.5
        self._ka = 0.1
        self._threshold_param = 2.
        self._state = self.States.NOT_INITIALIZED
        self._velocity_controller = VelocityController(
            vehicle.brake_comfort_max, vehicle.accel_max)
        self._max_brake = max_brake
        self._time_headway = 0
        self._standstill_distance = self.vehicle.c

    def set_time_headway(self, value: float) -> None:
        self._time_headway = value

    def compute_acceleration(self, vehicles: Mapping[int, base.BaseVehicle],
                             leader_id: int) -> float:
        v_ego = self.vehicle.get_vel()
        v_ff = self.vehicle.get_desired_free_flow_speed(vehicles)

        if leader_id < 0:
            new_state = self.States.CRUISE
        else:
            leader = vehicles[leader_id]
            # if (self.vehicle.is_in_a_platoon()
            #         and self.vehicle.get_platoon().contains_vehicle(leader_id)):
            #     new_state = self.States.VEHICLE_FOLLOWING
            # else:
            is_leader_too_fast = leader.get_vel() > 1.1 * v_ff
            if self._state == self.States.VEHICLE_FOLLOWING:
                if (is_leader_too_fast
                        or (self.vehicle.has_origin_lane_leader_changed()
                            and self._is_vehicle_far_ahead(leader))):
                    new_state = self.States.CRUISE
                else:
                    new_state = self.States.VEHICLE_FOLLOWING
            else:
                if self._is_vehicle_far_ahead(leader) or is_leader_too_fast:
                    new_state = self.States.CRUISE
                else:
                    new_state = self.States.VEHICLE_FOLLOWING

        # Transition back to cruising
        if new_state == self.States.CRUISE and new_state != self._state:
            self._velocity_controller.reset(v_ego)

        # if new_state != self._state:
        #     print(
        #         f"[LongController] "
        #         f"t={self.vehicle.get_current_time():.2f} "
        #         f"veh {self.vehicle.get_name()} long mode from: "
        #         f"{self._state.name} to {new_state.name}")

        self._state = new_state
        if self._state == self.States.CRUISE:
            accel = self._velocity_controller.compute_input(v_ego, v_ff)
        else:
            leader = vehicles[leader_id]
            accel = self._compute_accel_to_a_leader(leader)
            if v_ego <= 0 and accel < 0:
                accel = 0
        accel = self._saturate_accel(accel)
        return accel

    def compute_acceleration_from_interface(
            self, vehicle_interface: base.BaseVehicleInterface,
            ego_states, v_ff, leader_states) -> float:
        v_ego = vehicle_interface.select_state_from_vector(ego_states, 'v')
        if leader_states is None or len(leader_states) == 0:
            accel = self._compute_velocity_control(v_ff, v_ego)
        else:
            gap = (
                vehicle_interface.select_state_from_vector(leader_states, 'x')
                - vehicle_interface.select_state_from_vector(ego_states, 'x')
            )
            v_leader = vehicle_interface.select_state_from_vector(
                leader_states, 'v')
            accel = self._compute_gap_control(gap, v_ego, v_leader)
        # accel = self._saturate_accel(accel, v_ego, v_ff)
        accel = self._saturate_accel(accel)
        return accel

    def _is_vehicle_far_ahead(self, other_vehicle: base.BaseVehicle):
        margin = 0.1
        gap = base.BaseVehicle.compute_a_gap(other_vehicle, self.vehicle)
        return gap > self._compute_vehicle_following_threshold(
            other_vehicle.get_vel()) + margin

    def _compute_vehicle_following_threshold(self, v_leader: float):
        v_ego = self.vehicle.get_vel()
        # v_ff = self.vehicle.get_free_flow_speed()
        g_ref = (self._time_headway * v_ego
                 + self.vehicle.c)
        return g_ref + self._threshold_param * max(v_ego - v_leader, 0.)

    def _saturate_accel_and_velocity(
            self, desired_accel: float, v_ego: float, v_ff: float) -> float:
        """
        Saturates the acceleration based on max and min accel values, and also
        based on the current speed. This prevents the vehicle from traveling
        with negative or above max speed
        """
        if v_ego >= v_ff and desired_accel > 0:
            accel = self._compute_velocity_control(v_ff, v_ego)
        elif v_ego <= 0 and desired_accel < 0:
            accel = 0.
        else:
            accel = desired_accel
        return min(max(self._max_brake, accel), self.vehicle.accel_max)

    def _saturate_accel(self, desired_accel: float) -> float:
        """
        Saturates the acceleration based on max and min accel values
        """
        return min(max(self._max_brake, desired_accel), self.vehicle.accel_max)

    def _compute_accel_to_a_leader(
            self, other_vehicle: base.BaseVehicle
    ) -> float:
        v_ego = self.vehicle.get_vel()
        gap = base.BaseVehicle.compute_a_gap(other_vehicle, self.vehicle)
        if (self.vehicle.is_in_a_platoon() and other_vehicle.is_in_a_platoon()
            and (self.vehicle.get_platoon().get_id()
                 == other_vehicle.get_platoon().get_id())):
            accel_diff = (other_vehicle.get_an_input_by_name("a")
                          - self.vehicle.get_an_input_by_name("a"))
        else:
            accel_diff = 0  # to make the difference zero
        return self._compute_gap_control(gap, v_ego, other_vehicle.get_vel(),
                                         accel_diff)

    def _compute_velocity_control(self, v_ff: float,
                                  v_ego: float) -> float:
        return self._kv * (v_ff - v_ego)

    def _compute_gap_control(self, gap: float, v_ego: float,
                             v_leader: float, accel_diff: float = 0) -> float:
        gap_ref = self.vehicle.compute_reference_gap(self._time_headway,
                                                     v_ego)
        return (self._kg * (gap - gap_ref) + self._kv * (v_leader - v_ego)
                + self._ka * accel_diff)


class VelocityController:

    _v_ref: float
    _v_ff: float

    def __init__(self, max_brake: float,
                 max_accel: float):
        self._k = 0.5
        self._filter_gain = 10.
        dt = configuration.Configuration.time_step
        self._alpha = np.exp(-self._filter_gain * dt)
        self._max_variation = max_accel * dt
        self._min_variation = max_brake * dt

    def get_v_ff(self) -> float:
        return self._v_ff

    def get_current_v_ref(self) -> float:
        return self._v_ref

    def reset(self, v0: float):
        self._v_ref = v0

    def compute_input(self, v_ego: float, v_ff: float) -> float:
        self._apply_filter(v_ff)
        return self._k * (self._v_ref - v_ego)

    def _apply_filter(self, v_ff: float) -> None:
        variation = v_ff - self._v_ref
        if (1 - self._alpha) * variation > self._max_variation:
            filtered_variation = self._max_variation
        elif (1 - self._alpha) * variation < self._min_variation:
            filtered_variation = self._min_variation
        else:
            filtered_variation = (1 - self._alpha) * variation
        self._v_ref += filtered_variation
