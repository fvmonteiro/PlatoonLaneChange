from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import vehicle_models as vm


class VehicleMode(ABC):
    vehicle: vm.BaseVehicle

    def __init__(self, name):
        self.name = name

    def set_ego_vehicle(self, vehicle: vm.BaseVehicle):
        self.vehicle = vehicle

    @abstractmethod
    def handle_lane_keeping_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        pass

    @abstractmethod
    def handle_lane_changing_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        pass

    def __str__(self):
        return self.name

    # def __eq__(self, other):
    #     return self.name == other.name


class CLVehicleMode(VehicleMode, ABC):
    """ Base of vehicle modes for the ClosedLoopVehicle"""

    vehicle: vm.ClosedLoopVehicle


class OCPVehicleMode(VehicleMode, ABC):
    """ Base of vehicle modes for any vehicles using optimal controllers"""

    vehicle: vm.OptimalControlVehicle


class CLLaneKeepingMode(CLVehicleMode):
    def __init__(self):
        super().__init__("CL lane keeping")

    def handle_lane_keeping_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        pass

    def handle_lane_changing_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        self.vehicle.prepare_for_longitudinal_adjustments_start(vehicles)
        self.vehicle.set_mode(CLLongAdjustmentMode())


class CLLongAdjustmentMode(CLVehicleMode):
    def __init__(self):
        super().__init__("CL long adjustment")

    def handle_lane_keeping_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        self.vehicle.set_mode(CLLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        if self.vehicle.is_lane_change_safe(vehicles):
            self.vehicle.prepare_for_lane_change_start()
            self.vehicle.set_mode(CLLaneChangingMode())
        else:
            self.vehicle.request_cooperation()


class CLLaneChangingMode(CLVehicleMode):
    def __init__(self):
        super().__init__("CL lane changing")

    def handle_lane_keeping_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        if self.vehicle.is_lane_change_complete():
            self.vehicle.reset_lane_change_start_time()
            self.vehicle.set_mode(CLLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        pass


class OCPLaneKeepingMode(OCPVehicleMode):
    def __init__(self):
        super().__init__("OCP lane keeping")

    def handle_lane_keeping_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        pass

    def handle_lane_changing_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        self.vehicle.prepare_for_longitudinal_adjustments_start(vehicles)
        self.vehicle.set_mode(OCPLongAdjustmentMode())


class OCPLongAdjustmentMode(OCPVehicleMode):
    def __init__(self):
        super().__init__("OCP long adjustment")

    def handle_lane_keeping_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        self.vehicle.set_mode(OCPLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        if self.vehicle.can_start_lane_change(vehicles):
            self.vehicle.prepare_for_lane_change_start()
            self.vehicle.set_mode(OCPLaneChangingMode())
        else:
            self.vehicle.request_cooperation()


class OCPLaneChangingMode(OCPVehicleMode):
    def __init__(self):
        super().__init__("OCP lane changing")

    def handle_lane_keeping_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        if not self.vehicle.is_lane_changing():
            self.vehicle.reset_lane_change_start_time()
            self.vehicle.set_mode(OCPLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Dict[int, vm.BaseVehicle]) -> None:
        pass
