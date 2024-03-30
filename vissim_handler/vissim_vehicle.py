from __future__ import annotations
from enum import Enum


class VehicleType(Enum):
    HDV = 0
    ACC = 1
    AUTONOMOUS = 2
    CONNECTED = 3
    CONNECTED_NO_LANE_CHANGE = 4
    PLATOON = 5

    def get_print_name(self):
        _vehicle_type_to_print_name: dict[VehicleType, str] = {
            VehicleType.HDV: "HDV",
            VehicleType.ACC: "ACC",
            VehicleType.AUTONOMOUS: "AV",
            VehicleType.CONNECTED: "CAV",
            VehicleType.CONNECTED_NO_LANE_CHANGE: "CAV",
            VehicleType.PLATOON: "Platoon",
        }
        return _vehicle_type_to_print_name[self]

    def get_vissim_id(self):
        _vehicle_type_to_vissim_id: dict[VehicleType, int] = {
            VehicleType.HDV: 100,
            VehicleType.ACC: 105,
            VehicleType.AUTONOMOUS: 110,
            VehicleType.CONNECTED: 120,
            VehicleType.CONNECTED_NO_LANE_CHANGE: 121,
            VehicleType.PLATOON: 140
        }
        return _vehicle_type_to_vissim_id[self]


# TODO: merge with platoon_lane_change_strategies.StrategyMap
class PlatoonLaneChangeStrategy(Enum):
    human_driven = -1  # baseline for comparison
    no_strategy = 0  # baseline for comparison
    single_body_platoon = 1
    leader_first = 2
    last_vehicle_first = 3
    leader_first_and_reverse = 4
    graph_min_time = 5
    graph_min_accel = 6

    def get_print_name(self):
        _strategy_to_print_name: dict[PlatoonLaneChangeStrategy, str] = {
            PlatoonLaneChangeStrategy.human_driven: "HDV",
            PlatoonLaneChangeStrategy.no_strategy: "CAV",
            PlatoonLaneChangeStrategy.single_body_platoon: "SBP",
            PlatoonLaneChangeStrategy.leader_first: "LdF",
            PlatoonLaneChangeStrategy.last_vehicle_first: "LVF",
            PlatoonLaneChangeStrategy.leader_first_and_reverse: "LdFR"
        }
        return _strategy_to_print_name[self]
