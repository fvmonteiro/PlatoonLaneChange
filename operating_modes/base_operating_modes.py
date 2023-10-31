from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import vehicle_models.base_vehicle as base


class VehicleMode(ABC):
    vehicle: base.BaseVehicle

    def __init__(self, name):
        self.name = name

    def set_ego_vehicle(self, vehicle: base.BaseVehicle):
        self.vehicle = vehicle

    @abstractmethod
    def handle_lane_keeping_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        pass

    @abstractmethod
    def handle_lane_changing_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        pass

    def __str__(self):
        return self.name

    # def __eq__(self, other):
    #     return self.name == other.name
