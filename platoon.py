from __future__ import annotations

from typing import Dict, List

import numpy as np

import controllers.optimal_controller as opt_ctrl
import vehicle_models.base_vehicle as base


class Platoon:
    controller: opt_ctrl.VehicleOptimalController

    _counter = 0
    _ocp_horizon = 10.0
    _solver_wait_time = 10.0

    vehicle_ids: List[int]
    current_inputs: Dict[int, float]

    def __init__(self):
        self.id: int = Platoon._counter
        Platoon._counter += 1
        # self.vehicle_ids.append(platoon_leader_id)
        self._lc_start_time: float = -np.inf
        self._lc_controller: opt_ctrl.VehicleOptimalController = (
            opt_ctrl.VehicleOptimalController(self._ocp_horizon)
        )
        self._solver_attempt_time: float = -np.inf
        self.trajectory_exists: bool = False

    def get_platoon_leader_id(self):
        return self.vehicle_ids[0]

    def get_platoon_last_vehicle_id(self):
        return self.vehicle_ids[-1]

    def set_lc_start_time(self, t):
        self._lc_start_time = t

    def add_vehicle(self, veh_id):
        # TODO: how to ensure vehicle is behind the platoon?
        self.vehicle_ids.append(veh_id)

    def is_lane_changing(self, t):
        delta_t = t - self._lc_start_time
        return delta_t <= self._ocp_horizon

    def compute_lane_change_trajectory(
            self, t: float, vehicles: Dict[int, base.BaseVehicle]):

        # TODO: [Aug 23] No cooperation checks for now
        # If the OPC solver didn't find a solution at first, we do not want to
        # run it again too soon.
        is_cool_down_period_done = (
                t - self._solver_attempt_time >= Platoon._solver_wait_time
        )
        if is_cool_down_period_done:
            self._solver_attempt_time = t
            print("t={:.2f}, veh:{}. Calling ocp solver...".format(t, self.id))
            self._lc_controller.find_lane_change_trajectory(t, vehicles,
                                                            self.vehicle_ids)
        self.trajectory_exists = self._lc_controller.has_solution()

    def retrieve_all_inputs(self, t):
        self.current_inputs = self._lc_controller.get_input(t, self.vehicle_ids)

    def get_input_for_vehicle(self, veh_id):
        return self.current_inputs[veh_id]
