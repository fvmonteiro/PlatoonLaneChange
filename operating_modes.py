from abc import ABC, abstractmethod

import vehicle_models as vh


class VehicleMode(ABC):

    def __init__(self, name):
        self.name = name
        self.vehicle = None

    def set_ego_vehicle(self, vehicle):
        self.vehicle = vehicle

    @abstractmethod
    def handle_lane_keeping_intention(self, vehicles) -> None:
        pass

    @abstractmethod
    def handle_lane_changing_intention(self, vehicles) -> None:
        pass

    def __str__(self):
        return self.name

    # def __eq__(self, other):
    #     return self.name == other.name


class LaneKeepingMode(VehicleMode):
    def __init__(self):
        super().__init__("lane keeping")

    def handle_lane_keeping_intention(self, vehicles) -> None:
        pass

    def handle_lane_changing_intention(self, vehicles) -> None:
        self.vehicle.set_mode(LongAdjustmentMode())


class LongAdjustmentMode(VehicleMode):
    def __init__(self):
        super().__init__("long adjustment")

    def handle_lane_keeping_intention(self, vehicles) -> None:
        self.vehicle.set_mode(LaneKeepingMode())

    def handle_lane_changing_intention(self, vehicles) -> None:
        if self.vehicle.is_lane_change_safe(vehicles):
            self.vehicle.set_lane_change_maneuver_parameters()
            self.vehicle.set_mode(LaneChangingMode())


class LaneChangingMode(VehicleMode):
    def __init__(self):
        super().__init__("lane changing")

    def handle_lane_keeping_intention(self, vehicles) -> None:
        if self.vehicle.is_lane_change_complete():
            self.vehicle.reset_lane_change_start_time()
            self.vehicle.set_mode(LaneKeepingMode())

    def handle_lane_changing_intention(self, vehicles) -> None:
        pass
