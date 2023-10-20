from __future__ import annotations

# import bisect
from typing import Dict, List

import numpy as np

import controllers.optimal_controller as opt_ctrl
import vehicle_models.four_state_vehicles as fsv


class Platoon:
    current_inputs: Dict[int, float]

    _counter: int = 0

    def __init__(self, first_vehicle: fsv.PlatoonVehicle):
        self._id: int = Platoon._counter
        Platoon._counter += 1

        # Vehicles and their ids sorted by position (first is front-most)
        self.vehicles: List[fsv.PlatoonVehicle] = []
        self._id_to_position_map: Dict[int, int] = {}
        self.add_vehicle(first_vehicle)

    def get_platoon_leader(self):
        return self.vehicles[0]

    def get_platoon_leader_id(self):
        return self.get_platoon_leader().get_id()

    def get_platoon_last_vehicle_id(self):
        return self.vehicles[-1].get_name()

    def get_vehicle_ids(self):
        return [veh.get_id() for veh in self.vehicles]

    def get_preceding_vehicle_id(self, veh_id: int) -> int:
        preceding_position = self._id_to_position_map[veh_id] - 1
        if preceding_position >= 0:
            return self.vehicles[preceding_position].get_id()
        else:
            return -1

    def get_optimal_controller(self) -> opt_ctrl.VehicleOptimalController:
        return self.get_platoon_leader().opt_controller

    def add_vehicle(self, new_vehicle: fsv.PlatoonVehicle):
        if len(self.vehicles) == 0:
            self._id_to_position_map[new_vehicle.get_id()] = len(self.vehicles)
            self.vehicles.append(new_vehicle)
        elif new_vehicle.get_x() < self.vehicles[-1].get_x():
            self._id_to_position_map[new_vehicle.get_id()] = len(self.vehicles)
            self.vehicles.append(new_vehicle)
            opt_control = self.get_optimal_controller()
            new_vehicle.set_centralized_controller(opt_control)
        else:
            # bisect.insort(self.vehicles, new_vehicle, key=lambda v: v.get_x())
            idx = np.searchsorted([veh.get_x() for veh in self.vehicles],
                                  new_vehicle.get_x())
            # Possibly slow, but irrelevant for the total run time
            self.vehicles = (self.vehicles[:idx] + [new_vehicle]
                             + self.vehicles[idx:])
            for i, veh in enumerate(self.vehicles):
                self._id_to_position_map[veh.get_id()] = i
            opt_control = self.get_optimal_controller()
            new_vehicle.set_centralized_controller(opt_control)
