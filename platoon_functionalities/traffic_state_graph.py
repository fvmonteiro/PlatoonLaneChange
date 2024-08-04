from __future__ import annotations

import warnings
from collections.abc import Sequence
import copy
import datetime
import functools
import json
import numpy as np
import os
import pickle
import time
from typing import Callable
# import weakref

import configuration
import helper
from platoon_functionalities import graph_explorer, graph_tools, vehicle_platoon
import platoon_functionalities.platoon_lane_change_strategies as lc_strategies
import vehicle_group as vg
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv
from platoon_functionalities.graph_explorer import State
from platoon_functionalities.graph_tools import StateQuantizer

QuantizedState = configuration.QuantizedState
Command = tuple[frozenset[int], int]


def solve_queries_from_simulations(
        simulator_name: str, mode: str, n_platoon: int, cost_type: str,
        scenario_number: int = None, epsilon: float = 0.5,
        verbose_level: int = 0
) -> None:
    """

    """
    start_time = time.time()

    sim_time = 20.0 * n_platoon
    configuration.Configuration.set_scenario_parameters(
        sim_time=sim_time, increase_lc_time_headway=False
    )

    df, file_path = helper.load_queries_from_simulations(
        n_platoon, simulator_name)
    if scenario_number is not None:
        df = df.iloc[scenario_number: scenario_number + 1]

    TrafficStateNode.set_cost_type(cost_type)

    n_queries = len(df.index)
    results = dict()
    print(f"{n_queries} queries to solve.")
    for index, row in df.iterrows():
        print(f"#{index}")

        # TODO: verify if query has already been solved or explored

        initial_state = tuple([int(x) for x in row["qx"].split(",")])
        if isinstance(row["first_movers_set"], str):
            first_movers_set = frozenset(
                int(x) for x in row["first_movers_set"].split(","))
        elif np.isscalar(row["first_movers_set"]):
            first_movers_set = frozenset([row["first_movers_set"]])
        else:
            first_movers_set = frozenset(int(x) for x in row["first_movers_set"])

        # This dict is reset for every query because we want to simulate the
        # effects of solving each query from scratch
        visited_nodes = dict()
        query_results = train(
            initial_state, 1000, first_movers_set, epsilon,
            verbose_level, visited_nodes)
        # Reduce results size: by storing the initial node, we'll have access
        # to the entire graph. No need to store all the paths
        query_results["initial_node"] = query_results["best_path"][-1][0][0]
        del query_results["best_path"]
        if initial_state not in results:
            results[initial_state] = dict()
        results[initial_state][first_movers_set] = query_results

    pickle_results(n_platoon, results, cost_name=cost_type, epsilon=epsilon)
    save_results_to_json(n_platoon, results, cost_name=cost_type,
                         epsilon=epsilon)
    print(f"{n_queries} new queries solved.")
    graph_time = datetime.timedelta(seconds=time.time() - start_time)
    print(f"Graph n={n_platoon} expansion time:",
          str(graph_time).split(".")[0])


def train(
        initial_state: QuantizedState, max_episodes: int,
        first_movers_set: frozenset[int] = None, epsilon: float = 0.5,
        verbose_level: int = 0,
        visited_nodes: dict[State, TrafficStateNode] = None
        ) -> dict[str, list]:

    def is_goal_node(node: TrafficStateNode) -> bool:
        return node.is_terminal()

    initial_node = TrafficStateNode(initial_state)
    if first_movers_set is not None:
        first_command = (first_movers_set, -1)
        initial_node.take_action(first_command, dict())
        initial_node.find_best_edge()
        current_node = initial_node.exploit().destination_node
    else:
        current_node = initial_node

    if is_goal_node(current_node):
        print("Single-step solution")
        results = {"cost": [initial_node.best_edge.cost],
                   "best_path": [[(initial_node, initial_node.best_edge)]],
                   "time": [0.]}
    else:
        results = graph_explorer.train_base(
            current_node, max_episodes, is_goal_node, epsilon, verbose_level,
            visited_nodes)
        for i in range(len(results["cost"])):
            results["cost"][i] += initial_node.best_edge.cost
            if results["best_path"][i][0][0] != initial_node:
                results["best_path"][i].insert(
                    0, (initial_node, initial_node.best_edge))
            else:
                warnings.warn("Repeated path in results dict")
    if verbose_level > 0:
        print("Query solution:\n", graph_explorer.path_to_string(
            graph_explorer.get_best_path(initial_node, is_goal_node)), sep="")
    strategies = get_strategies_from_paths(results["best_path"])
    results["lc_order"] = []
    results["coop_order"] = []
    for strat in strategies:
        # pycharm's warning is wrong here
        results["lc_order"].append(strat[0])
        results["coop_order"].append(strat[1])
    # Quick and dirty solution to avoid Memory Error when working with n > 4.
    # del results["best_path"]
    return results


def get_strategies_from_paths(paths: list[TrafficPath]
                              ) -> list[configuration.Strategy]:
    strategies = []
    for p in paths:
        lc_order, coop_order = [], []
        for element in p:
            maneuver_step = element[1].action
            lc_order.append(maneuver_step[0])
            coop_order.append(maneuver_step[1])
        strategies.append((lc_order, coop_order))
    return strategies


def load_best_results(
        n_platoon: int
) -> dict[QuantizedState, dict[frozenset[int], dict[str, list]]]:
    results = load_pickled_results(n_platoon)
    # Desired: {root: first_mover_set: {cost: , lc_order: , coop_order:}
    best_results = dict()
    for initial_state, outer_dict in results.items():
        if initial_state not in best_results:
            best_results[initial_state] = dict()
        for first_mover_set, query_result in outer_dict.items():
            best_results[initial_state][first_mover_set] = dict()
            for key, value in query_result.items():
                best_results[initial_state][first_mover_set][key] = value[-1]
    return best_results


def load_pickled_results(
        n_platoon: int, cost_name: str = "time"
) -> dict[QuantizedState, dict[frozenset[int], dict[str, list]]]:
    file_name = "_".join(["min", cost_name, "strategies_for",
                          str(n_platoon), "vehicles.pickle"])
    file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                             "strategy_maps", file_name)
    with open(file_path, 'rb') as file:
        results = pickle.load(file)
    return results


def pickle_results(
        n_platoon: int,
        data: dict[QuantizedState, dict[frozenset[int], dict[str, list]]],
        cost_name: str, epsilon: float):
    file_name = "_".join(["min", cost_name, "strategies_for",
                          str(n_platoon), "vehicles.pickle"])
    epsilon_percent = int(epsilon * 100)
    folder_name = os.path.join(configuration.DATA_FOLDER_PATH,
                               "strategy_maps", f"epsilon_{epsilon_percent}")
    os.makedirs(folder_name, exist_ok=True)
    full_path = os.path.join(folder_name, file_name)
    with open(full_path, 'wb') as handle:
        pickle.dump(data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def save_results_to_json(
        n_platoon: int,
        data: dict[QuantizedState, dict[frozenset[int], dict[str, list]]],
        cost_name: str, epsilon: float,
        mode: str = "as") -> None:
    """
    Saves the minimum cost maneuver for every initial condition and
    every first mover set, to a json file.

    :param data: The data to save. The data is a dictionary with structure as:
     results[initial_state][first_mover_set] = {"cost": list[float],
     "lc_order": list[frozenset[int]], "coop_order": list[int],
     "time": list[float]}
    :param cost_name: "accel" or "time". If left empty, computes both
    :param mode: "w": completely overwrites any existing file;
     "ao": appends to an existing strategy map (previously saved to a file)
     and overwrites previous results if there are repeated initial states;
     "as": appends to an existing graph and skips any repeated initial
     states. [Not implemented]
    :return:
    """

    # TODO: multiple costs
    # if cost_name is None:
    #     for c in ["time", "accel"]:
    #         save_minimum_cost_strategies_to_json(c)
    #     return
    epsilon_percent = int(epsilon * 100)
    folder_name = os.path.join(configuration.DATA_FOLDER_PATH,
                               "strategy_maps", f"epsilon_{epsilon_percent}")
    os.makedirs(folder_name, exist_ok=True)

    # TODO: add to existing solution
    # overwrite = mode == "w" or mode[1] == "o"
    # existing_list = LaneChangeStrategyManager.strategy_map_to_dict_list(
    #     self._strategy_maps[cost_name], cost_name)
    strategy_list = _to_json_format(data)
    # strategy_list += existing_list
    json_data = json.dumps(strategy_list, indent=2)
    file_name = "_".join(["min", cost_name, "strategies_for",
                          str(n_platoon), "vehicles.json"])
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "w") as file:
        file.write(json_data)
        print("Saved file ", file_name)


def _to_json_format(
        data: dict[QuantizedState, dict[frozenset[int], dict[str, list]]]):
    strategy_map = []
    for root in data:
        for first_mover_set in data[root]:
            computation_time = data[root][first_mover_set]["time"]
            lc_order: list[configuration.LCOrder] = data[root][
                first_mover_set]["lc_order"]
            coop_order: list[configuration.LCOrder] = data[root][
                first_mover_set]["coop_order"]
            cost = data[root][first_mover_set]["cost"]
            strategy_map.append(
                {"root": [int(i) for i in root],
                 "first_mover_set": list(first_mover_set),
                 "time": computation_time, "cost": cost,
                 "lc_order": [[list(s) for s in order] for order in lc_order],
                 "coop_order": [order for order in coop_order],
                 }
            )
    return strategy_map


def read_from_json(n_platoon: int, cost_name: str, epsilon: float
                   ) -> list[dict[str, list]]:
    epsilon_percent = int(epsilon * 100)
    folder_name = os.path.join(configuration.DATA_FOLDER_PATH,
                               "strategy_maps", f"epsilon_{epsilon_percent}")
    os.makedirs(folder_name, exist_ok=True)
    file_name = "_".join(["min", cost_name, "strategies_for",
                          str(n_platoon), "vehicles.json"])
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "r") as file:
        results = json.load(file)
    return results


class TrafficStateNode(graph_explorer.Node):
    _possible_actions: list[Command]
    _explored_actions: dict[Command, TrafficEdge]
    _best_edge: TrafficEdge
    _simulate: Callable[[QuantizedState, Command], tuple[QuantizedState, float]]
    _cost_type = "time"

    def __init__(self, state: QuantizedState,
                 tracker: graph_tools.PlatoonLCTracker = None):
        n_states = fsv.FourStateVehicle.get_n_states()
        # State contains the platoon vehicles plus ld and lo
        self._n_platoon = int(len(state) / n_states - 2)
        if tracker is None:
            self._tracker = graph_tools.PlatoonLCTracker(self._n_platoon)
        else:
            self._tracker = tracker
        state_quantizer = graph_tools.StateQuantizer()
        free_flow_speeds = state_quantizer.infer_free_flow_speeds_from_state(
            state, self._n_platoon)
        sim_fun = functools.partial(simulate_new, n_platoon=self._n_platoon,
                                    free_flow_speeds=free_flow_speeds,
                                    cost_type=TrafficStateNode._cost_type)
        self._continuous_state = state_quantizer.dequantize_state(state)
        super().__init__(state, sim_fun)

    @property
    def best_edge(self) -> TrafficEdge:
        return self._best_edge

    @staticmethod
    def set_cost_type(cost_type: str):
        print(f"Cost set to: {cost_type}")
        TrafficStateNode._cost_type = cost_type

    def add_edge_from_node(self, successor: TrafficStateNode, action: Command,
                           cost: float) -> None:
        self._explored_actions[action] = TrafficEdge(successor, action, cost)

    def generate_possible_actions(self) -> None:
        initial_state = self._continuous_state
        vehicle_group = self._create_vehicle_group()
        vehicle_group.set_vehicles_initial_states_from_array(initial_state)
        vehicle_group.prepare_to_start_simulation(1)
        [veh.set_lane_change_direction(1) for veh
         in vehicle_group.vehicles.values()
         if (veh.is_in_a_platoon() and veh.get_current_lane() == 0)]
        vehicle_group.update_surrounding_vehicles()
        remaining_vehicles = sorted(self._tracker.remaining_vehicles)

        if not self.is_new():
            raise RuntimeError("Trying to generate possible actions for "
                               "a node that already has an action list")
        for next_pos_to_coop in self._tracker.get_possible_cooperative_vehicles():
            if next_pos_to_coop < 0:
                first_maneuver_steps = (
                    self._determine_next_maneuver_steps_no_coop(
                        vehicle_group, remaining_vehicles)
                )
                self._possible_actions |= first_maneuver_steps
                # Debugging
                # analysis.plot_state_vector(
                #     vehicle_group.get_full_initial_state_vector())
                # print(next_maneuver_steps)
            else:
                for i in range(len(remaining_vehicles)):
                    next_positions_to_move = set()
                    for j in range(i, len(remaining_vehicles)):
                        next_positions_to_move = (next_positions_to_move
                                                  | {remaining_vehicles[j]})
                        self._possible_actions.append(
                            (frozenset(next_positions_to_move),
                             next_pos_to_coop)
                             )

    def add_edge_from_state(self, successor_state: State, action: Command,
                            cost: float) -> None:
        if np.isinf(cost):  # unsuccessful maneuver step
            next_node = self
        else:
            next_pos_to_move, next_pos_to_coop = action
            next_tracker = copy.deepcopy(self._tracker)
            next_tracker.move_vehicles(next_pos_to_move,
                                       next_pos_to_coop)
            next_node = TrafficStateNode(successor_state, next_tracker)
        self.add_edge_from_node(next_node, action, cost)
        # self._explored_actions[action] = Edge(next_node, action, cost)

    def is_terminal(self):
        if self._tracker.is_lane_change_done():
            self.set_as_terminal()
            return True
        return False

    def state_to_str(self) -> str:
        x_idx = fsv.FourStateVehicle.get_idx_of_state("x")
        y_idx = fsv.FourStateVehicle.get_idx_of_state("y")
        states_per_vehicles = helper.split_state_vector(self.state)
        min_x, max_x = np.inf, -np.inf
        min_y, max_y = np.inf, -np.inf
        position_map = {}
        for name, state in states_per_vehicles.items():
            position_map[(state[x_idx], state[y_idx])] = name
            min_x = state[x_idx] if state[x_idx] < min_x else min_x
            max_x = state[x_idx] if state[x_idx] > max_x else max_x
            min_y = state[y_idx] if state[y_idx] < min_y else min_y
            max_y = state[y_idx] if state[y_idx] > max_y else max_y

        ret_str = ""
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (x, y) in position_map:
                    ret_str += f"|{position_map[(x, y)]}|"
                else:
                    ret_str += "|  |"
            ret_str += "\n"
        return ret_str[:-1]

    def _determine_next_maneuver_steps_no_coop(
            self, vehicle_group: vg.VehicleGroup,
            remaining_vehicles: list[int]) -> list[tuple[int, set[int]]]:
        """
        When there is no cooperation, we only allow steps including vehicles
        that are already at safe position or that can perform a "quick"
        adjustment to be safe. It is the caller's duty to check whether there
        is cooperation.
        :return:
        """
        next_maneuver_steps: list[tuple[int, set[int]]] = []
        next_pos_to_coop = -1  # the caller should be sure of this
        for i in range(len(remaining_vehicles)):
            front_most_next_pos_to_move = remaining_vehicles[i]
            front_most_vehicle = (
                vehicle_group.get_platoon_vehicle_by_position(
                    front_most_next_pos_to_move)
            )
            ld = vehicle_group.get_vehicle_by_name("ld")
            if front_most_vehicle.get_x() <= ld.get_x():
                front_most_vehicle.check_surrounding_gaps_safety(
                    vehicle_group.vehicles,
                    ignore_orig_lane_leader=True)
                if front_most_vehicle.get_is_lane_change_safe():
                    next_positions_to_move = set()
                    for j in range(i, len(remaining_vehicles)):
                        next_positions_to_move = (
                                next_positions_to_move
                                | {remaining_vehicles[j]}
                        )
                        next_maneuver_steps.append(
                            (next_pos_to_coop, next_positions_to_move))
                    # We don't need to include cases that do not include
                    # the current front most vehicle if it is at a safe
                    # position
                    break
                else:
                    next_maneuver_steps.append(
                        (next_pos_to_coop,
                         {front_most_next_pos_to_move})
                    )
        return next_maneuver_steps

    def _create_vehicle_group(self, include_ld: bool = True
                              ) -> vg.ShortSimulationVehicleGroup:
        base.BaseVehicle.reset_vehicle_counter()
        lo = fsv.ShortSimulationVehicle(False)
        lo.set_name("lo")
        platoon_vehicles = []
        for i in range(self._n_platoon):
            veh = fsv.ShortSimulationVehicle(True, is_connected=True)
            veh.set_name("p" + str(i + 1))
            platoon_vehicles.append(veh)

        leader_platoon = vehicle_platoon.ClosedLoopPlatoon(
            platoon_vehicles[0], lc_strategies.StrategyMap.template)
        platoon_vehicles[0].set_platoon(leader_platoon)
        for i in range(1, len(platoon_vehicles)):
            leader_platoon.append_vehicle(platoon_vehicles[i])
            platoon_vehicles[i].set_platoon(leader_platoon)

        if include_ld:
            ld = fsv.ShortSimulationVehicle(False)
            ld.set_name("ld")
            leader_platoon._dest_lane_leader_id = ld.get_id()
        else:
            ld = []
        all_vehicles = helper.order_values(lo, platoon_vehicles, ld)
        vehicle_group = vg.ShortSimulationVehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)
        vehicle_group.set_platoon_lane_change_strategy(
            lc_strategies.StrategyMap.template)

        return vehicle_group


class TrafficEdge(graph_explorer.Edge):
    _destination_node: TrafficStateNode
    _action: Command

    def __init__(self, destination_node: TrafficStateNode, action: Command,
                 cost: float):
        super().__init__(destination_node, action, cost)

    @property
    def destination_node(self) -> TrafficStateNode:
        return self._destination_node

    @property
    def action(self) -> Command:
        return self._action

    def action_to_str(self) -> str:
        ret_str = (f"[c={self.action[1]}, lc={{"
                   + ", ".join([str(i) for i in self.action[0]]) + "}]")
        return ret_str


def simulate_new(state: QuantizedState, action: Command, n_platoon: int,
                 free_flow_speeds: Sequence[float], cost_type: str
                 ) -> tuple[QuantizedState, float]:
    """
    Simulates the vehicle group until all listed platoon vehicles have finished
    lane changing. When there is no cooperation, only simulates if the
    front-most lane changing vehicle is behind the destination lane leader.
    :param state:
    :param action:
    :param n_platoon:
    :param free_flow_speeds:
    :return:
    """
    # TODO: we must ensure a single state quantizer all throughout
    state_quantizer = graph_tools.StateQuantizer()
    continuous_state = state_quantizer.dequantize_state(state)
    next_platoon_positions_to_move, next_platoon_position_to_coop = action
    vehicle_group = vg.ShortSimulationVehicleGroup.create_platoon_scenario(
        n_platoon, continuous_state, next_platoon_positions_to_move,
        next_platoon_position_to_coop)
    # Due to quantization, we could end up with initial vel above free
    # flow desired vel. To prevent vehicles from decelerating, we allow
    # these slightly higher free-flow speeds
    adjusted_free_flow_speeds = []
    for i, veh in enumerate(vehicle_group.get_all_vehicles_in_order()):
        adjusted_free_flow_speeds.append(
            max(free_flow_speeds[i], veh.get_vel()))
    vehicle_group.set_free_flow_speeds(adjusted_free_flow_speeds)

    dt = 1.0e-2
    tf = 40.
    sim_time = np.arange(0., tf + dt, dt)
    vehicle_group.prepare_to_start_simulation(len(sim_time))
    [veh.set_lane_change_direction(1) for veh
     in vehicle_group.vehicles.values()
     if (veh.is_in_a_platoon() and veh.get_current_lane() == 0)]
    vehicle_group.update_surrounding_vehicles()

    next_vehs_to_move = (vehicle_group.get_next_vehicles_to_change_lane(
        next_platoon_positions_to_move))

    # We skip the simulation if no cooperation and the front most lc vehicle
    # cannot start a lane change
    if next_platoon_position_to_coop < 0:
        front_most_vehicle = max([veh for veh in next_vehs_to_move],
                                 key=lambda x: x.get_x())
        ld = vehicle_group.get_vehicle_by_name("ld")
        if ld.get_x() < front_most_vehicle.get_x():
            return tuple([]), np.inf  # TODO: 'hidden' return statement

    # This function's caller must ensure safety for the first set of lane
    # changing vehicles. We need the condition below because quantization may
    # transform a safe situation into a non-safe one.
    if vehicle_group.count_completed_lane_changes() == 0:
        vehicle_group.set_ids_must_change_lanes(
            [veh.get_id() for veh in next_vehs_to_move])

    i = 0
    success = False
    while i < len(sim_time) - 1 and not success:
        i += 1
        vehicle_group.simulate_one_time_step(
            sim_time[i], detect_collision=True
        )
        success = np.all([not veh.has_lane_change_intention()
                          for veh in next_vehs_to_move])
    if success:
        new_state = state_quantizer.quantize_state_map(
            vehicle_group.get_state_by_vehicle()
        )
        # TODO: could organize this better
        if cost_type == "time":
            cost = vehicle_group.get_current_time()
        elif cost_type == "accel_cost":
            cost = vehicle_group.compute_acceleration_cost()
        else:
            raise ValueError("Only 'time' and 'accel_cost' are accepted"
                             "costs.")
        return new_state, cost
    else:
        return tuple([]), np.inf


TrafficPath = list[tuple[TrafficStateNode, TrafficEdge]]
