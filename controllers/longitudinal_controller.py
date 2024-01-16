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

    def __init__(self, vehicle: fsv.FourStateVehicle):
        self.vehicle = vehicle
        self._kg = 0.2
        self._kv = 0.5
        self._threshold_param = 2.
        self._state = self.States.NOT_INITIALIZED
        self._velocity_controller = VelocityController(vehicle.brake_max,
                                                       vehicle.accel_max)

    def compute_acceleration(self, vehicles: Mapping[int, base.BaseVehicle]
                             ) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.vehicle.get_vel()
        v_ff = self.vehicle.get_desired_free_flow_speed(vehicles)

        if not self.vehicle.has_leader() or self._is_vehicle_far_ahead(
                vehicles[self.vehicle.get_current_leader_id()]):
            if self._state != self.States.CRUISE:
                self._velocity_controller.set(v_ego, v_ff)
            new_state = self.States.CRUISE
        else:
            new_state = self.States.VEHICLE_FOLLOWING

        self._state = new_state
        if self._state == self.States.CRUISE:
            accel = self._velocity_controller.compute_input(v_ego)
        else:
            leader = vehicles[self.vehicle.get_current_leader_id()]
            accel = self._compute_accel_to_a_leader(leader)
        # accel = self._saturate_accel(accel)
        return accel

    def compute_acceleration_from_interface(
            self, vehicle_interface: base.BaseVehicleInterface,
            ego_states, v_ff, leader_states) -> float:
        v_ego = vehicle_interface.select_state_from_vector(ego_states, 'v')
        if leader_states is None or len(leader_states) == 0:
            accel = self._compute_velocity_control(v_ff, v_ego)
        else:
            gap = (vehicle_interface.select_state_from_vector(leader_states,
                                                              'x')
                   - vehicle_interface.select_state_from_vector(ego_states,
                                                                'x'))
            v_leader = vehicle_interface.select_state_from_vector(
                leader_states, 'v')
            accel = self._compute_gap_control(gap, v_ego, v_leader)
        # accel = self._saturate_accel(accel, v_ego, v_ff)
        accel = self._saturate_accel(accel)
        return accel

    def get_more_critical_leader(self, vehicles: Mapping[int, base.BaseVehicle]
                                 ) -> int:
        """
        Compares the acceleration if following the origin or destination lane
        leaders, and chooses as leader the vehicle which causes the lesser
        acceleration
        :param vehicles:
        :return:
        """
        relevant_ids = self.vehicle.get_possible_target_leader_ids()

        candidate_accel = {
            veh_id: self._compute_accel_to_a_leader(vehicles[veh_id])
            for veh_id in relevant_ids if veh_id >= 0
        }

        if len(candidate_accel) > 0:
            return min(candidate_accel, key=candidate_accel.get)
        else:
            return -1

    def _is_vehicle_far_ahead(self, other_vehicle: base.BaseVehicle):
        margin = 0.1
        gap = base.BaseVehicle.compute_a_gap(other_vehicle, self.vehicle)
        return gap > self._compute_vehicle_following_threshold(
            other_vehicle.get_vel()) + margin

    def _compute_vehicle_following_threshold(self, v_leader: float):
        v_ego = self.vehicle.get_vel()
        # v_ff = self.vehicle.get_free_flow_speed()
        g_ref = (self.vehicle.get_reference_time_headway() * v_ego
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
        return min(max(self.vehicle.get_active_brake_max(), accel),
                   self.vehicle.accel_max)

    def _saturate_accel(self, desired_accel: float) -> float:
        """
        Saturates the acceleration based on max and min accel values
        """
        return min(max(self.vehicle.get_active_brake_max(), desired_accel),
                   self.vehicle.accel_max)

    def _compute_accel_to_a_leader(
            self, other_vehicle: base.BaseVehicle
    ) -> float:
        v_ego = self.vehicle.get_vel()
        gap = base.BaseVehicle.compute_a_gap(other_vehicle, self.vehicle)
        return self._compute_gap_control(gap, v_ego, other_vehicle.get_vel())

    def _compute_velocity_control(self, v_ff: float,
                                  v_ego: float) -> float:
        return self._kv * (v_ff - v_ego)

    def _compute_gap_control(self, gap: float, v_ego: float,
                             v_leader: float) -> float:
        h_ref = self.vehicle.get_reference_time_headway()
        return (self._kg * (gap - (h_ref * v_ego + self.vehicle.c))
                + self._kv * (v_leader - v_ego))


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

    def set(self, v0: float, v_ff: float):
        self._v_ref = v0
        self._v_ff = v_ff

    def compute_input(self, v_ego: float) -> float:
        self._apply_filter()
        return self._k * (self._v_ref - v_ego)

    def _apply_filter(self) -> None:
        variation = self._v_ff - self._v_ref
        if (1 - self._alpha) * variation > self._max_variation:
            filtered_variation = self._max_variation
        elif (1 - self._alpha) * variation < self._min_variation:
            filtered_variation = self._min_variation
        else:
            filtered_variation = (1 - self._alpha) * variation
        self._v_ref += filtered_variation
