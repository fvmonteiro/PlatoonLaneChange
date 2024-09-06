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
