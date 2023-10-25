from __future__ import annotations

from typing import Dict, Type

import numpy as np

import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


class LongitudinalController:

    def __init__(self, vehicle: fsv.FourStateVehicle):
        self.vehicle = vehicle
        self.kg = 0.5
        self.kv = 0.5

    def compute_acceleration(self, vehicles: Dict[int, base.BaseVehicle]
                             ) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.vehicle.get_vel()
        v_ff = self.vehicle.free_flow_speed
        if not self.vehicle.has_leader():
            accel = self.compute_velocity_control(v_ff, v_ego)
        else:
            leader = vehicles[self.vehicle.get_current_leader_id()]
            gap = self.vehicle.compute_gap_to_a_leader(leader)
            v_leader = leader.get_vel()
            accel = self.compute_gap_control(gap, v_ego, v_leader)
            # Stay under free flow speed
            accel = self.saturate_accel(v_ego, v_ff, accel)
        return accel

    def compute_acceleration_from_interface(
            self, ego_class: Type[base.BaseVehicleInterface],
            ego_states, v_ff, leader_states) -> float:
        v_ego = ego_class.select_state_from_vector(ego_states, 'v')
        if leader_states is None or len(leader_states) == 0:
            accel = self.compute_velocity_control(v_ff, v_ego)
        else:
            gap = (ego_class.select_state_from_vector(leader_states, 'x')
                   - ego_class.select_state_from_vector(ego_states, 'x'))
            v_leader = ego_class.select_state_from_vector(leader_states, 'v')
            accel = self.compute_gap_control(gap, v_ego, v_leader)
            accel = self.saturate_accel(v_ego, v_ff, accel)
        return accel

    def saturate_accel(self, v_ego, v_ff, desired_accel) -> float:
        if v_ego >= v_ff and desired_accel > 0:
            return self.compute_velocity_control(v_ff, v_ego)
        elif v_ego <= 0 and desired_accel < 0:
            return 0.
        else:
            return desired_accel

    def compute_accel_to_a_leader(
            self, other_id: int, vehicles: Dict[int, base.BaseVehicle]
    ) -> float:
        if other_id >= 0:
            other_vehicle = vehicles[other_id]
            gap = self.vehicle.compute_gap_to_a_leader(other_vehicle)
            v_ego = self.vehicle.get_vel()
            return self.compute_gap_control(
                gap, v_ego, other_vehicle.get_vel())
        else:
            return np.inf

    def get_more_critical_leader(self, vehicles: Dict[int, base.BaseVehicle]
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
            veh_id: self.compute_accel_to_a_leader(veh_id, vehicles)
            for veh_id in relevant_ids
        }

        return min(candidate_accel, key=candidate_accel.get)

    def compute_velocity_control(self, v_ff: float,
                                 v_ego: float) -> float:
        return self.kv * (v_ff - v_ego)

    def compute_gap_control(self, gap: float, v_ego: float,
                            v_leader: float) -> float:
        h_ref = self.vehicle.get_reference_time_headway()
        return (self.kg * (gap - h_ref * v_ego - self.vehicle.c)
                + self.kv * (v_leader - v_ego))
