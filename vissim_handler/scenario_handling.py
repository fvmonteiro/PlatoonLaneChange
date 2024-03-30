import itertools
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Union

import vissim_handler.vissim_vehicle as vissim_vehicle


@dataclass
class ScenarioInfo:
    """Defines a VISSIM scenario. The scenario parameters must agree with
    the network being run
    vehicle_percentages: Describes the percentages of controlled
     vehicles in the simulations.
    vehicles_per_lane: Vehicle input per lane on VISSIM. Possible
     values depend on the controlled_vehicles_percentage: 500:500:2500
    platoon_lane_change_strategy: Coordination strategy used in platoon lane
     changing scenarios.
    orig_and_dest_lane_speeds: Mean desired speeds in the platoon lane
     changing scenario
    """
    vehicle_percentages: dict[vissim_vehicle.VehicleType, int]
    vehicles_per_lane: int
    platoon_lane_change_strategy: \
        vissim_vehicle.PlatoonLaneChangeStrategy = None
    orig_and_dest_lane_speeds: tuple[Union[str, int], Union[str, int]] = None
    platoon_size: int = None
    special_case: str = None

    def __str__(self):
        str_list = []
        veh_percent_list = [str(p) + "% " + vt.name.lower()
                            for vt, p in self.vehicle_percentages.items()]
        str_list.append("Vehicles: " + ", ".join(veh_percent_list))
        str_list.append("Input: " + str(self.vehicles_per_lane)
                        + " vehs/lane/hour")
        if self.platoon_lane_change_strategy is not None:
            str_list.append("Platoon LC strat.: "
                            + self.platoon_lane_change_strategy.name.lower())
        if self.orig_and_dest_lane_speeds is not None:
            str_list.append("Orig lane speed "
                            + str(self.orig_and_dest_lane_speeds[0])
                            + ". Dest lane speed: "
                            + str(self.orig_and_dest_lane_speeds[1]))
        if self.platoon_size is not None:
            str_list.append("n_platoon=" + str(self.platoon_size))
        if self.special_case is not None:
            str_list.append("Special case: " + self.special_case)
        return "\n".join(str_list)


def is_all_human(scenario: ScenarioInfo) -> bool:
    return (sum(scenario.vehicle_percentages.values()) == 0
            or scenario.vehicle_percentages.get(
                vissim_vehicle.VehicleType.HDV, 0) == 100)


def create_vehicle_percentages_dictionary(
        vehicle_types: list[vissim_vehicle.VehicleType], percentages: list[int],
        n_vehicle_types: int) -> list[dict[vissim_vehicle.VehicleType, int]]:
    """
    :param vehicle_types:
    :param percentages:
    :param n_vehicle_types: Must be equal to 1 or 2
    :return: List of dictionaries describing the percentage of each vehicle
     type in the simulation
    """
    percentages_list = []
    percentages_ = percentages.copy()
    if n_vehicle_types == 1:
        for vt in vehicle_types:
            for p in percentages_:
                percentages_list.append({vt: p})
            if 0 in percentages_:
                percentages_.remove(0)
    if n_vehicle_types == 2:
        for p1 in percentages_:
            for p2 in percentages_:
                if p1 > 0 and p2 > 0 and p1 + p2 <= 100:
                    percentages_list.append({vehicle_types[0]: p1,
                                             vehicle_types[1]: p2})
    return percentages_list


def create_multiple_scenarios(
        vehicle_percentages: Iterable[dict[vissim_vehicle.VehicleType, int]],
        vehicle_inputs: Iterable[int],
        lane_change_strategies:
        Iterable[vissim_vehicle.PlatoonLaneChangeStrategy] = None,
        orig_and_dest_lane_speeds: Iterable[tuple[Union[str, int],
                                            Union[str, int]]] = None,
        platoon_size: Iterable[int] = None,
        special_cases: Iterable[str] = None) -> list[ScenarioInfo]:
    if lane_change_strategies is None:
        lane_change_strategies = [None]
    if orig_and_dest_lane_speeds is None:
        orig_and_dest_lane_speeds = [None]
    if platoon_size is None:
        platoon_size = [None]
    if special_cases is None:
        special_cases = [None]
    scenarios = []
    for vp, vi, st, speeds, sizes, case in itertools.product(
            vehicle_percentages, vehicle_inputs,
            lane_change_strategies, orig_and_dest_lane_speeds, platoon_size,
            special_cases):
        scenarios.append(ScenarioInfo(vp, vi, st, speeds, sizes, case))
    return scenarios


def vehicle_percentage_dict_to_string(
        vp_dict: dict[vissim_vehicle.VehicleType, int]) -> str:
    if sum(vp_dict.values()) == 0:
        return "100% HDV"
    ret_str = []
    for veh_type, p in vp_dict.items():
        ret_str.append(
            str(p) + "% "
            + veh_type.get_print_name())
    return " ".join(sorted(ret_str))


all_vissim_simulation_configurations: dict[str, Iterable] = {
    "strategies": [
        vissim_vehicle.PlatoonLaneChangeStrategy.single_body_platoon,
        vissim_vehicle.PlatoonLaneChangeStrategy.last_vehicle_first,
        vissim_vehicle.PlatoonLaneChangeStrategy.leader_first_and_reverse,
        vissim_vehicle.PlatoonLaneChangeStrategy.graph_min_accel,
        vissim_vehicle.PlatoonLaneChangeStrategy.graph_min_time
    ],
    "orig_and_dest_lane_speeds": [("70", "50"), ("70", "70"), ("70", "90")],
    "platoon_size": [2, 3, 4, 5],
    "vehicles_per_lane": [500, 1000, 1500]
}


def get_platoon_lane_change_scenarios(
        select: str = None, with_hdv: bool = False,
        include_no_lane_change: bool = False) -> list[ScenarioInfo]:
    """

    :param select: "all" returns all the lane change scenarios;
     "dest_lane_speed" returns the 15 scenarios with varying relative speed;
     "vehicles_per_lane" returns the 60 scenarios with varying vehicle input;
     "platoon_size" returns the 60 scenarios with varying number of vehicles
     in the platoon
    :param with_hdv: If true, non-platoon vehicles are human driven
    :param include_no_lane_change: if True also includes the no lane change
     scenario in the list
    :returns: list with the requested scenarios
    """
    if with_hdv:
        other_vehicles = [{vissim_vehicle.VehicleType.HDV: 100}]
    else:
        other_vehicles = [{vissim_vehicle.VehicleType.CONNECTED: 100}]
    strategies = all_vissim_simulation_configurations["strategies"]
    orig_and_dest_lane_speeds = all_vissim_simulation_configurations[
        "orig_and_dest_lane_speeds"]
    platoon_size = all_vissim_simulation_configurations["platoon_size"]
    vehicles_per_lane = all_vissim_simulation_configurations[
        "vehicles_per_lane"]
    scenarios = []
    if select == "all" or select == "dest_lane_speed":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, [2700],
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            platoon_size=[4]))
    if select == "all" or select == "vehicles_per_lane":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, vehicles_per_lane,
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            platoon_size=[4]))
    if select == "all" or select == "platoon_size":
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, [2700],
            lane_change_strategies=strategies,
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            platoon_size=platoon_size
        ))
    if include_no_lane_change:
        scenarios.extend(create_multiple_scenarios(
            other_vehicles, [2700],
            orig_and_dest_lane_speeds=orig_and_dest_lane_speeds,
            special_cases=["no_lane_change"]))
    if len(scenarios) == 0:
        raise ValueError("No scenarios selected. Probably parameter 'select' "
                         "is incorrect.")
    return scenarios
