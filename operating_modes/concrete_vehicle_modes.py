from __future__ import annotations

from abc import ABC
from typing import Mapping

import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv
import operating_modes.base_operating_modes as vom


# ========================= CL Control Vehicle States ======================== #

class CLVehicleMode(vom.VehicleMode, ABC):
    """ Base of vehicle modes for the ClosedLoopVehicle"""

    vehicle: fsv.ClosedLoopVehicle


class CLLaneKeepingMode(CLVehicleMode):
    def __init__(self):
        super().__init__("CL lane keeping")

    def handle_lane_keeping_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        pass

    def handle_lane_changing_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        self.vehicle.prepare_for_longitudinal_adjustments_start()
        self.vehicle.set_mode(CLLongAdjustmentMode())


class CLLongAdjustmentMode(CLVehicleMode):
    def __init__(self):
        super().__init__("CL long adjustment")

    def handle_lane_keeping_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        self.vehicle.set_mode(CLLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        if self.vehicle.can_start_lane_change(vehicles):
            self.vehicle.prepare_for_lane_change_start()
            self.vehicle.set_mode(CLLaneChangingMode())


class CLLaneChangingMode(CLVehicleMode):
    def __init__(self):
        super().__init__("CL lane changing")

    def handle_lane_keeping_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        if self.vehicle.is_lane_change_complete():
            self.vehicle.prepare_for_lane_keeping_start()
            self.vehicle.set_mode(CLLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        pass


# ====================== Optimal Control Vehicle States ====================== #

class OCPVehicleMode(vom.VehicleMode, ABC):
    """ Base of vehicle modes for vehicles equipped with optimal controllers"""

    vehicle: fsv.OptimalControlVehicle


class OCPLaneKeepingMode(OCPVehicleMode):
    def __init__(self):
        super().__init__("OCP lane keeping")

    def handle_lane_keeping_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        pass

    def handle_lane_changing_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        self.vehicle.prepare_for_longitudinal_adjustments_start()
        self.vehicle.set_mode(OCPLongAdjustmentMode())


class OCPLongAdjustmentMode(OCPVehicleMode):
    def __init__(self):
        super().__init__("OCP long adjustment")

    def handle_lane_keeping_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        self.vehicle.set_mode(OCPLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        if self.vehicle.can_start_lane_change(vehicles):
            self.vehicle.prepare_for_lane_change_start()
            self.vehicle.set_mode(OCPLaneChangingMode())
        else:
            self.vehicle.request_cooperation()


class OCPLaneChangingMode(OCPVehicleMode):
    def __init__(self):
        super().__init__("OCP lane changing")

    def handle_lane_keeping_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        if not self.vehicle.is_lane_changing():
            self.vehicle.prepare_for_lane_keeping_start()
            self.vehicle.set_mode(OCPLaneKeepingMode())

    def handle_lane_changing_intention(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        pass


# ========================== Platoon Vehicle States ========================== #

# class PlatoonVehicleMode(vom.VehicleMode, ABC):
#     """ Base of vehicle modes for platoon vehicles"""
#     vehicle: fsv.OptimalControlVehicle

# class PlatoonVehicleLaneKeepingMode(PlatoonVehicleMode):
#     def __init__(self):
#         super().__init__("Platoon vehicle lane keeping")
#
#     def handle_lane_keeping_intention(
#             self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
#         pass
#
#     def handle_lane_changing_intention(
#             self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
#         self.vehicle.prepare_for_longitudinal_adjustments_start()
#         self.vehicle.set_mode(PlatoonVehicleLongAdjustmentMode())
#
#
# class PlatoonVehicleLongAdjustmentMode(PlatoonVehicleMode):
#     def __init__(self):
#         super().__init__("Platoon vehicle long adjustment")
#
#     def handle_lane_keeping_intention(
#             self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
#         self.vehicle.set_mode(PlatoonVehicleLaneKeepingMode())
#
#     def handle_lane_changing_intention(
#             self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
#         if self.vehicle.can_start_lane_change(vehicles):
#             self.vehicle.prepare_for_lane_change_start()
#             self.vehicle.set_mode(PlatoonVehicleLaneChangingMode())
#         else:
#             self.vehicle.request_cooperation()
#
#
# class PlatoonVehicleLaneChangingMode(PlatoonVehicleMode):
#     def __init__(self):
#         super().__init__("Platoon vehicle lane changing")
#
#     def handle_lane_keeping_intention(
#             self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
#         if not self.vehicle.is_lane_changing():
#             self.vehicle.reset_lane_change_start_time()
#             self.vehicle.set_mode(PlatoonVehicleLaneKeepingMode())
#
#     def handle_lane_changing_intention(
#             self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
#         pass
