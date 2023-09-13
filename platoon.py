from __future__ import annotations

# import bisect
from typing import Dict, List

import numpy as np

import controllers.optimal_controller as opt_ctrl
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


class Platoon:
    current_inputs: Dict[int, float]

    _counter: int = 0
    _ocp_horizon: float = 10.0
    _solver_wait_time: float = 10.0

    def __init__(self, first_vehicle: fsv.PlatoonVehicle):
        self._id: int = Platoon._counter
        Platoon._counter += 1
        # self.vehicle_ids.append(platoon_leader_id)
        self._lc_start_time: float = -np.inf
        self._lc_controller: opt_ctrl.VehicleOptimalController = (
            opt_ctrl.VehicleOptimalController(self._ocp_horizon)
        )
        self._solver_attempt_time: float = -np.inf
        self.trajectory_exists: bool = False

        self.vehicles = [first_vehicle]

    def get_platoon_leader_id(self):
        return self.vehicles[0].get_id()

    def get_platoon_last_vehicle_id(self):
        return self.vehicles[-1].get_name()

    def set_lc_start_time(self, t: float):
        self._lc_start_time = t

    def add_vehicle(self, new_vehicle: fsv.PlatoonVehicle):
        if new_vehicle.get_x() < self.vehicles[-1].get_x():
            self.vehicles.append(new_vehicle)
        else:
            # bisect.insort(self.vehicles, new_vehicle, key=lambda v: v.get_x())
            idx = np.searchsorted([veh.get_x() for veh in self.vehicles],
                                  new_vehicle.get_x())
            # Possibly slow, but irrelevant for the total run time
            self.vehicles = (self.vehicles[:idx] + [new_vehicle]
                             + self.vehicles[idx:])

    # def add_vehicles(self, vehicles: List[fsv.PlatoonVehicle]):
    #     self.vehicle_ids = [
    #         veh.get_id() for veh
    #         in sorted(vehicles, key=lambda x: x.get_x(), reverse=True)
    #     ]
    #     [veh.set_platoon(self) for veh in vehicles]

    def is_lane_changing(self, t):
        delta_t = t - self._lc_start_time
        return delta_t <= self._ocp_horizon

    def compute_lane_change_trajectory(
            self, t: float, all_vehicles: Dict[int, base.BaseVehicle]):

        # TODO: [Aug 23] No cooperation checks for now
        # If the OPC solver didn't find a solution at first, we do not want to
        # run it again too soon.
        is_cool_down_period_done = (
                t - self._solver_attempt_time >= Platoon._solver_wait_time
        )
        if is_cool_down_period_done:
            self._solver_attempt_time = t
            print("t={:.2f}, veh:{}. Calling ocp solver...".format(t, self._id))
            self._lc_controller.find_lane_change_trajectory(
                t, all_vehicles, [veh.get_id() for veh in self.vehicles]
            )
        self.trajectory_exists = True  # self._lc_controller.has_solution()

    def retrieve_all_inputs(self, t):
        self.current_inputs = self._lc_controller.get_input(
            t, [veh.get_id() for veh in self.vehicles]
        )

    def get_input_for_vehicle(self, veh_id):
        return self.current_inputs[veh_id]
