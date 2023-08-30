from __future__ import annotations

from typing import Dict

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
            accel = self._compute_velocity_control(v_ff, v_ego)
        else:
            leader = vehicles[self.vehicle.get_current_leader_id()]
            gap = self.vehicle.compute_gap_to_a_leader(leader)
            v_leader = leader.get_vel()
            accel = self._compute_gap_control(gap, v_ego, v_leader)
            # Stay under free flow speed
            if v_ego >= self.vehicle.free_flow_speed and accel > 0:
                accel = self._compute_velocity_control(v_ff, v_ego)
        # Don't drive backwards
        # if v_ego <= 0 and accel < 0:
        #     accel = 0
        return accel

    def compute_accel_to_a_leader(self, other_id, vehicles) -> float:
        if other_id >= 0:
            other_vehicle = vehicles[other_id]
            gap = self.vehicle.compute_gap_to_a_leader(other_vehicle)
            v_ego = self.vehicle.get_vel()
            return self._compute_gap_control(
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
        relevant_ids = [self.vehicle.get_orig_lane_leader_id(),
                        self.vehicle.get_dest_lane_leader_id(),
                        self.vehicle.get_incoming_vehicle_id()]
        candidate_accel = {
            veh_id: self.compute_accel_to_a_leader(veh_id, vehicles)
            for veh_id in relevant_ids
        }

        return min(candidate_accel, key=candidate_accel.get)

    def _compute_velocity_control(self, desired_vel: float,
                                  v_ego: float) -> float:
        return self.kv * (desired_vel - v_ego)

    def _compute_gap_control(self, gap: float, v_ego: float,
                             v_leader: float) -> float:
        return (self.kg * (gap - self.vehicle.h * v_ego - self.vehicle.c)
                + self.kv * (v_leader - v_ego))
