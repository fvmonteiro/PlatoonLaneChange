from enum import Enum


# TODO: these VehicleType and Vehicle classes are messy. Consider better
#  organizing the data. Perhaps make VehicleType a regular class.
class VehicleType(Enum):
    HDV = 0
    ACC = 1
    AUTONOMOUS = 2
    CONNECTED = 3
    CONNECTED_NO_LANE_CHANGE = 4
    PLATOON = 5


class PlatoonLaneChangeStrategy(Enum):
    human_driven = -1  # baseline for comparison
    no_strategy = 0  # baseline for comparison
    single_body_platoon = 1
    leader_first = 2
    last_vehicle_first = 3
    leader_first_and_reverse = 4
    graph = 5


vehicle_type_to_print_name_map = {
    VehicleType.HDV: "HDV",
    VehicleType.ACC: "ACC",
    VehicleType.AUTONOMOUS: "AV",
    VehicleType.CONNECTED: "CAV",
    VehicleType.CONNECTED_NO_LANE_CHANGE: "CAV",
    VehicleType.PLATOON: "Platoon",
}


strategy_to_print_name_map = {
    PlatoonLaneChangeStrategy.human_driven: "HDV",
    PlatoonLaneChangeStrategy.no_strategy: "CAV",
    PlatoonLaneChangeStrategy.single_body_platoon: "SBP",
    PlatoonLaneChangeStrategy.leader_first: "LdF",
    PlatoonLaneChangeStrategy.last_vehicle_first: "LVF",
    PlatoonLaneChangeStrategy.leader_first_and_reverse: "LdFR"
}


# Useful when editing vissim simulation parameters
ENUM_TO_VISSIM_ID = {
    VehicleType.HDV: 100,
    VehicleType.ACC: 105,
    VehicleType.AUTONOMOUS: 110,
    VehicleType.CONNECTED: 120,
    VehicleType.CONNECTED_NO_LANE_CHANGE: 121,
    VehicleType.PLATOON: 140
}
