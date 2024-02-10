from __future__ import annotations

import copy
import pickle
import warnings
from collections.abc import Iterable, Sequence
from collections import defaultdict, deque
import json
import os
from typing import Any, Union

import networkx as nx
import numpy as np

import configuration
import platoon
# import scenarios
import vehicle_group as vg
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


def _simulate_till_lane_change(
        vehicle_group: vg.ShortSimulationVehicleGroup,
        initial_state: np.ndarray,
        free_flow_speeds: Sequence[float],
        next_platoon_positions_to_move: set[int],
        next_platoon_position_to_coop: int
) -> bool:
    """
    Simulates the vehicle group until all listed platoon vehicles have finished
    lane changing. When there is no cooperation, only simulates if the
    front-most lane changing vehicle is behind the destination lane leader.
    :param vehicle_group:
    :param initial_state:
    :param free_flow_speeds:
    :param next_platoon_positions_to_move:
    :param next_platoon_position_to_coop:
    :return:
    """

    vehicle_group.set_verbose(False)
    vehicle_group.set_vehicles_initial_states_from_array(
        initial_state)
    vehicle_group.set_platoon_lane_change_order(
        [next_platoon_positions_to_move], [next_platoon_position_to_coop])
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
    time = np.arange(0, tf + dt, dt)
    vehicle_group.prepare_to_start_simulation(len(time))
    [veh.set_lane_change_direction(1) for veh
     in vehicle_group.vehicles.values()
     if (veh.is_in_a_platoon() and veh.get_current_lane() == 0)]
    vehicle_group.update_surrounding_vehicles()

    next_vehs_to_move = (vehicle_group.get_next_vehicles_to_change_lane(
        next_platoon_positions_to_move))
    if next_platoon_position_to_coop < 0:
        front_most_vehicle = max([veh for veh in next_vehs_to_move],
                                 key=lambda x: x.get_x())
        ld = vehicle_group.get_vehicle_by_name("ld")
        if ld.get_x() < front_most_vehicle.get_x():
            return False

    i = 0
    success = False
    while i < len(time) - 1 and not success:
        i += 1
        vehicle_group.simulate_one_time_step(time[i],
                                             detect_collision=True)
        success = np.all([not veh.has_lane_change_intention()
                          for veh in next_vehs_to_move])
    return success


class VehicleStateGraph:
    """
    Data class to contain the DAG with vehicle states along with some important
    information regarding the graph
    """

    def __init__(self):
        self.states_graph = nx.DiGraph()
        self._initial_states = set()
        self._quantization_parameters = {
            "dx": configuration.DELTA_X, "dy": configuration.DELTA_Y,
            "dtheta": 1000,  # irrelevant for current approach because we force
            # q_theta = 0 always
            "dv": configuration.DELTA_V,
            "max_x_lo": np.inf, "min_x_lo": -np.inf,
            "max_x_ld": np.inf, "min_x_ld": -np.inf,
            "max_v_lo": np.inf, "min_v_lo": -np.inf,
            "max_v_ld": np.inf, "min_v_ld": -np.inf,
        }

    @staticmethod
    def load_from_file(n_platoon: int, has_fd: bool
                       ) -> VehicleStateGraph:
        file_name = VehicleStateGraph.get_graph_file_name(n_platoon,
                                                          has_fd)
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "vehicle_state_graphs",
                                 file_name + ".pickle")
        with open(file_path, "rb") as f:
            vsg: VehicleStateGraph = pickle.load(f)
        return vsg

    @staticmethod
    def get_graph_file_name(n_platoon: int, has_fd: bool) -> str:
        return f"graph_{n_platoon}_vehicles_with{'' if has_fd else 'out'}_fd"

    @staticmethod
    def order_values(lo_value: Any, platoon_value: Iterable[Any],
                     ld_value: Any, fd_value: Any = None) -> np.ndarray:
        """
        Used to ensure any stacked vectors always places the vehicles in the
        same order
        :param lo_value:
        :param platoon_value:
        :param ld_value:
        :param fd_value:
        :return:
        """
        if fd_value is None:
            fd_value = []
        platoon_value_array = np.array(platoon_value).flatten()
        return np.hstack((lo_value, platoon_value_array, ld_value, fd_value))

    @staticmethod
    def split_state_vector(values: Sequence[Any]) -> dict[str, Any]:
        # TODO remove hard coded indices?
        return {"lo": values[:4], "platoon": values[4:-4], "ld": values[-4:]}

    @staticmethod
    def split_matrix(matrix: Iterable[Sequence[Any]]) -> dict[str, Any]:
        ret = defaultdict(list)
        for array in matrix:
            for key, split_array in (VehicleStateGraph.
                                     split_state_vector(array).items()):
                ret[key].append(split_array)
        return ret

    def save_to_file(self, n_platoon: int, has_fd: bool):
        self.register_quantization_boundaries()
        file_name = VehicleStateGraph.get_graph_file_name(n_platoon, has_fd)
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "vehicle_state_graphs",
                                 file_name + ".pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print(f"File {file_name} saved")

    def get_initial_states(self) -> set[configuration.QuantizedState]:
        return self._initial_states

    def get_quantization_parameters(self) -> dict[str, float]:
        return self._quantization_parameters

    def get_state_quantizer(self, n_vehicles: int) -> StateQuantizer:
        return StateQuantizer(
            n_vehicles, self._quantization_parameters)

    def get_successor_given_first_movers(
            self, source: configuration.QuantizedState,
            first_mover_set: set[int]):
        dag: nx.DiGraph = self.states_graph
        for node in dag.successors(source):
            if dag[source][node]["lc_vehicles"] == first_mover_set:
                return node
        raise nx.NetworkXNoPath

    def add_initial_node(self, initial_state: configuration.QuantizedState):
        """
        Adds the given state to the list of nodes without any predecessors
        and to the graph's nodes list
        :param initial_state:
        :return:
        """
        self._initial_states.add(initial_state)
        self.states_graph.add_node(initial_state)

    def update_graph(
            self, source_state: configuration.QuantizedState,
            dest_state: configuration.QuantizedState,
            transition_time: float, accel_cost: float,
            lc_vehicles: Iterable[int], coop_vehicle: int):
        """
        Updates all tracking graphs with the same weight
        """
        self.states_graph.add_edge(source_state, dest_state,
                                   time=transition_time, accel=accel_cost,
                                   lc_vehicles=lc_vehicles,
                                   coop_vehicle=coop_vehicle)

        n_nodes = self.states_graph.number_of_nodes()
        if n_nodes % 100 == 0:
            print(f"\t{n_nodes} nodes created")

    def mark_terminal_node(self, node: configuration.QuantizedState) -> None:
        self.states_graph.nodes[node]["is_terminal"] = True

    def delete_all_successors(self, node: configuration.QuantizedState):
        """
        Deletes all children of the given node if they don't have
        any other ancestors
        """
        nodes_to_delete = set()
        self._get_all_successor_nodes(node, nodes_to_delete)
        self.states_graph.remove_nodes_from(nodes_to_delete)

    def find_minimum_cost_strategy_from_node(
            self, starting_node: configuration.QuantizedState, cost_name: str
    ) -> tuple[configuration.Strategy, float]:
        dag: nx.DiGraph = self.states_graph
        all_costs = []  # for debugging
        all_strategies = []  # for debugging
        opt_cost = np.inf
        opt_strategy = None

        if dag.nodes[starting_node].get("is_terminal", False):
            return ([], []), 0

        for node in nx.descendants(dag, starting_node):
            if dag.nodes[node].get("is_terminal", False):
                cost, path = nx.single_source_dijkstra(
                    dag, starting_node, node, weight=cost_name)
                strategy = self.get_maneuver_order_from_path(path)
                all_costs.append(cost)
                all_strategies.append(strategy)
                if cost < opt_cost:
                    opt_cost = cost
                    opt_strategy = strategy
        return opt_strategy, opt_cost

    def get_maneuver_order_from_path(self, path) -> configuration.Strategy:
        lc_order = []
        coop_order = []
        for source_node, target_node in nx.utils.pairwise(path):
            edge = self.states_graph[source_node][target_node]
            lc_order.append(edge["lc_vehicles"])
            coop_order.append(edge["coop_vehicle"])
        return lc_order, coop_order

    def register_quantization_boundaries(self):
        qx0 = self._initial_states
        qx0_split = VehicleStateGraph.split_matrix(qx0)
        x_lo = np.array(qx0_split["lo"])[:,
               fsv.FourStateVehicle.get_idx_of_state("x")]
        x_ld = np.array(qx0_split["ld"])[:,
               fsv.FourStateVehicle.get_idx_of_state("x")]
        v_lo = np.array(qx0_split["lo"])[:,
               fsv.FourStateVehicle.get_idx_of_state("v")]
        v_ld = np.array(qx0_split["ld"])[:,
               fsv.FourStateVehicle.get_idx_of_state("v")]
        bounds = {
            "max_x_lo": int(np.max(x_lo)),
            "min_x_lo": int(np.min(x_lo)),
            "max_x_ld": int(np.max(x_ld)),
            "min_x_ld": int(np.min(x_ld)),
            "max_v_lo": int(np.max(v_lo)),
            "min_v_lo": int(np.min(v_lo)),
            "max_v_ld": int(np.max(v_ld)),
            "min_v_ld": int(np.min(v_ld)),
        }
        for key, value in bounds.items():
            self._quantization_parameters[key] = value

    def _get_all_successor_nodes(self, node: configuration.QuantizedState,
                                 result: set[configuration.QuantizedState]
                                 ) -> None:
        """
        Recursively finds all successor nodes
        :param node: 'root' from where to start looking
        :param result: Set that will contain all successor nodes
        :return:
        """
        if len(list(self.states_graph.predecessors(node))) > 1:
            return
        result.add(node)
        for successor in self.states_graph.successors(node):
            self._get_all_successor_nodes(successor, result)


class PlatoonLCTracker:
    """
    Supporting class to make it easier to keep track of which vehicles have
    already moved to the destination lane and which are still in the origin
    lane
    """

    def __init__(self, n_platoon):
        self._remaining_vehicles: set[int] = set([i for i in range(n_platoon)])
        self._dest_lane_vehicles: list[int] = []
        self._cooperating_order: list[int] = []

    def __repr__(self):
        return (f"[{self.__class__.__name__}] remaining|moved|coop_order = "
                f"{self._remaining_vehicles}|{self._dest_lane_vehicles}|"
                f"{self._cooperating_order}")

    def get_remaining_vehicles(self) -> set[int]:
        return self._remaining_vehicles

    def get_possible_cooperative_vehicles(self) -> list[int]:
        """
        We return all the platoon vehicles that have already changed lanes
        :return:
        """
        return (self._dest_lane_vehicles if len(self._dest_lane_vehicles) > 0
                else [-1])

    def move_vehicles(self, position_in_platoon: Iterable[int],
                      cooperative_vehicle_position: int):
        for p in position_in_platoon:
            self._remaining_vehicles.remove(p)
            self._dest_lane_vehicles.append(p)
        self._cooperating_order.append(cooperative_vehicle_position)

    def bring_back_vehicle(self, position_in_platoon: Iterable[int]):
        self._cooperating_order.pop()
        for p in position_in_platoon:
            self._dest_lane_vehicles.pop()
            self._remaining_vehicles.add(p)


class GraphCreator:

    def __init__(self, n_platoon: int, has_fd: bool):
        self._n_platoon = n_platoon
        self._has_fd = has_fd
        self.vehicle_state_graph = VehicleStateGraph()
        self.state_quantizer: StateQuantizer = (
            self.vehicle_state_graph.get_state_quantizer(
            self.get_n_vehicles_per_state()))

    def get_n_vehicles_per_state(self):
        return self._n_platoon + 2 + self._has_fd

    def load_a_graph(self):
        vsg = VehicleStateGraph.load_from_file(self._n_platoon, self._has_fd)
        self.vehicle_state_graph = vsg
        self.state_quantizer = self.vehicle_state_graph.get_state_quantizer(
            self.get_n_vehicles_per_state())

    def save_vehicle_state_graph_to_file(self) -> None:
        self.vehicle_state_graph.save_to_file(self._n_platoon, self._has_fd)

    def save_quantization_parameters_to_file(self) -> None:
        self.vehicle_state_graph.register_quantization_boundaries()
        json_data = json.dumps(
            self.vehicle_state_graph.get_quantization_parameters())
        file_name = "_".join([str(self._n_platoon), "vehicles.json"])
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "quantization_parameters",
                                 file_name)
        with open(file_path, "w") as file:
            file.write(json_data)
            print("Saved quantization parameters file ", file_name)

    def save_minimum_cost_strategies_to_json(self, cost_name: str = None
                                             ) -> None:
        if cost_name is None:
            for c in ["time", "accel"]:
                self.save_minimum_cost_strategies_to_json(c)
            return

        strategy_map = self.create_strategies_map(cost_name)
        json_data = json.dumps(strategy_map, indent=2)
        file_name = "_".join(["min", cost_name, "strategies_for",
                              str(self._n_platoon), "vehicles.json"])
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "strategy_maps", file_name)
        with open(file_path, "w") as file:
            file.write(json_data)
            print("Saved file ", file_name)

    def create_graph(self, vel_orig_lane: Sequence[float],
                     vel_ff_platoon: float, vel_dest_lane: Sequence[float],
                     max_dist: float, mode: str = "as") -> None:
        """

        :param vel_orig_lane: The traveling speed of non-platoon
         vehicles at the origin lane (only the origin lane leader)
        :param vel_ff_platoon: The desired free-flow speed of the platoon. The
         method creates nodes with initial platoon speed between
         vel_orig_lane and vel_ff_platoon.
        :param vel_dest_lane: The traveling speed of non-platoon
         vehicles at the origin lane (destination lane leader and, possibly,
         the destination lane follower)
        :param max_dist: The maximum distance between the platoon leader and
         its origin and destination lane leaders. The method creates nodes
         with initial leader positions up to the way to max_dist.
        :param mode: "w": ignores any existing file (so saving this object
         will overwrite existing graphs); "ao": appends to an existing graph
         (previously saved to a file) and overwrites previous results if
         there are repeated initial nodes; "as": appends to an existing graph
         and skips any repeated initial nodes.
        :return:
        """
        print(f"Creating graph ")

        if mode not in {"w", "ao", "as"}:
            raise ValueError(
                "[VehicleStatesGraph] Chosen mode is not valid. Must be "
                "'w', 'ao' or 'as'")

        if mode.startswith("a"):
            self.load_a_graph()

        visited_states = set()
        all_velocities = self._create_all_initial_speeds(
            vel_orig_lane, vel_ff_platoon, vel_dest_lane)
        repeated_initial_states_counter = 0
        counter = 0
        for v0_lo, v0_p, v0_dest in all_velocities:
            counter += 1
            print(f"{counter} / {len(all_velocities)} vel. combination")

            # Create and add the initial states nodes (before any lane
            # change)
            initial_states = self._create_initial_states(
                v0_lo, v0_p, v0_dest, v0_dest, max_dist)
            print(f"  {len(initial_states)} roots")

            v0_fd = v0_dest if self._has_fd else None
            free_flow_speeds = VehicleStateGraph.order_values(
                v0_lo, [vel_ff_platoon] * self._n_platoon, v0_dest, v0_fd)
            for x0 in initial_states:
                n_new = self.add_all_from_initial_state_to_graph(
                    x0, visited_states, free_flow_speeds, mode)
                repeated_initial_states_counter += 1 if n_new == 0 else 0

        print(f"Total initial states: "
              f"{len(self.vehicle_state_graph.get_initial_states())}")
        print(f"Total nodes: "
              f"{self.vehicle_state_graph.states_graph.number_of_nodes()}")
        if repeated_initial_states_counter > 0:
            print(("skipped" if mode.endswith("s") else "overwrote")
                  + f" initial states: {repeated_initial_states_counter}")

    def add_all_from_initial_state_to_graph(
            self, initial_state: tuple[int], visited_states: set[tuple],
            free_flow_speeds: Sequence[float], mode: str):
        """
        Adds the node relative to the given initial state and explores all
        succeeding states.
        :return: Number of new nodes
        """
        if initial_state in self.vehicle_state_graph.get_initial_states():
            if mode.endswith("s"):
                return 0
            elif mode.endswith("o"):
                self.vehicle_state_graph.delete_all_successors(initial_state)
            else:
                warnings.warn(
                    "[VehicleStatesGraph] found a repeated initial "
                    "state in writing mode. Overwriting it...")
                self.vehicle_state_graph.delete_all_successors(initial_state)
        n1 = self.vehicle_state_graph.states_graph.number_of_nodes()
        self.vehicle_state_graph.add_initial_node(initial_state)
        # BFS exploration
        self._explore_until_maneuver_completion(
            initial_state, visited_states, free_flow_speeds)
        return self.vehicle_state_graph.states_graph.number_of_nodes() - n1

    def create_strategies_map(self, cost_name: str) -> list[dict]:
        strategy_map: list[dict] = []
        roots_with_issues = []
        dag = self.vehicle_state_graph.states_graph
        for root in self.vehicle_state_graph.get_initial_states():
            for next_node in dag.successors(root):
                first_mover_set = dag[root][next_node]["lc_vehicles"].copy()
                strategy = ([first_mover_set], [-1])
                strategy_from_node, cost = (
                    self.vehicle_state_graph.
                    find_minimum_cost_strategy_from_node(next_node, cost_name)
                )
                if strategy_from_node is None:
                    warnings.warn(f"Could not find a strategy from root "
                                  f"{root} with first mover set "
                                  f"{first_mover_set}")
                    roots_with_issues.append(root)
                    continue
                strategy[0].extend(strategy_from_node[0])
                strategy[1].extend(strategy_from_node[1])
                strategy_map.append(
                    {"root": [int(i) for i in root],
                     "first_mover_set": list(first_mover_set),
                     "lc_order": [list(s) for s in strategy[0]],
                     "coop_order": strategy[1], cost_name: cost}
                )
        # if len(roots_with_issues) > 0:
        file_name = "unsolved_root_nodes.pickle"
        with open(file_name, "wb") as f:
            pickle.dump(roots_with_issues, f, pickle.HIGHEST_PROTOCOL)
            print(f"{len(roots_with_issues)} issues saved in file {file_name}")
        return strategy_map

    @staticmethod
    def _create_all_initial_speeds(
            v_orig: Sequence[float], v_ff_platoon: float,
            v_dest: Sequence[float]) -> list[tuple[float, float, float]]:
        min_step_v = configuration.DELTA_V
        all_combinations = []
        for vo in v_orig:
            # print("")
            v_platoon = [vo]
            while v_platoon[-1] < v_ff_platoon:
                v_platoon.append(v_platoon[-1] + min_step_v)
            for vp in v_platoon:
                for vd in v_dest:
                    all_combinations.append((vo, vp, vd))
        return all_combinations

    def _create_initial_states(
            self, v0_lo: float, v0_platoon: float, v0_ld: float, v0_fd: float,
            max_leader_dist: float
    ) -> set[configuration.QuantizedState]:
        """
        The root node is the quantized states of all platoon vehicles and the
        two destination lane vehicles between which the platoon will move.
        We assume x0_p1 = 0
        :param v0_lo:
        :param v0_platoon:
        :param max_leader_dist: Assumed value of the farthest ahead any leader
         may be
        :return:
        """
        vehicle_group = self._create_vehicle_group()

        # Platoon vehicles (fixed positions given the speed)
        x0_platoon_vehicle = 0
        for i in range(self._n_platoon):
            p_i = vehicle_group.get_vehicle_by_name("p" + str(i+1))
            p_i.set_initial_state(x0_platoon_vehicle, 0., 0., v0_platoon)
            x0_platoon_vehicle -= p_i.compute_initial_reference_gap_to(
                p_i, v0_platoon)  # since all platoon vehicles have the same
            # parameters, the gap reference gap from a vehicle to itself is
            # the same as between any two platoon vehicles
        p1 = vehicle_group.get_vehicle_by_name("p1")
        pN = vehicle_group.get_vehicle_by_name("p" + str(self._n_platoon))

        # Surrounding vehicles (variable positions)
        lo = vehicle_group.get_vehicle_by_name("lo")
        ld = vehicle_group.get_vehicle_by_name("ld")
        x0_lo = p1.compute_initial_reference_gap_to(lo, v0_platoon)
        dx = configuration.DELTA_X
        quantized_initial_states: set[configuration.QuantizedState] = set()
        while x0_lo < max_leader_dist:
            lo.set_initial_state(x0_lo, 0., 0., v0_lo)
            x0_ld = pN.get_x()
            while x0_ld < max_leader_dist:
                ld.set_initial_state(x0_ld, configuration.LANE_WIDTH, 0., v0_ld)
                # it"s not conceptually right to use ld when computing fd"s
                # safe gap, but we know fd and ld have the same characteristics
                x0_fd = pN.get_x() - ld.compute_initial_reference_gap_to(
                    pN, v0_fd)
                while self._has_fd and x0_fd > pN.get_x() - max_leader_dist:
                    fd = vehicle_group.get_vehicle_by_name("fd")
                    fd.set_initial_state(x0_fd, configuration.LANE_WIDTH,
                                         0., v0_fd)
                    quantized_initial_states.add(
                        self.state_quantizer.quantize_without_bounds(
                            vehicle_group.get_initial_state_by_vehicle()))
                    x0_fd -= dx
                # We add again in case self._has_fd is false
                quantized_initial_states.add(
                    self.state_quantizer.quantize_without_bounds(
                        vehicle_group.get_initial_state_by_vehicle()))
                x0_ld += dx
            x0_lo += dx
        return quantized_initial_states

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

        # TODO: improve strategy setting method.
        #  Setting by int is hard to maintain and read
        leader_platoon = platoon.ClosedLoopPlatoon(platoon_vehicles[0], 2)
        platoon_vehicles[0].set_platoon(leader_platoon)
        for i in range(1, len(platoon_vehicles)):
            leader_platoon.append_vehicle(platoon_vehicles[i])
            platoon_vehicles[i].set_platoon(leader_platoon)

        if include_ld:
            ld = fsv.ShortSimulationVehicle(False)
            ld.set_name("ld")
        else:
            ld = []
        if self._has_fd:
            fd = fsv.ShortSimulationVehicle(False)
            fd.set_name("fd")
        else:
            fd = None
        all_vehicles = VehicleStateGraph.order_values(lo, platoon_vehicles,
                                                      ld, fd)
        vehicle_group = vg.ShortSimulationVehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)
        vehicle_group.set_platoon_lane_change_strategy(2)

        return vehicle_group

    def _explore_until_maneuver_completion(
            self, initial_state: configuration.QuantizedState,
            visited_states: set[tuple], free_flow_speeds: Sequence[float]):
        """
        Given an initial state, explore all possibilities until maneuver
         completion
        :param initial_state: state before any lane changes
        :param free_flow_speeds:
        :return:
        """
        root = (PlatoonLCTracker(self._n_platoon), initial_state)
        nodes: deque[tuple[PlatoonLCTracker, tuple]] = deque([root])
        while len(nodes) > 0:
            tracker, quantized_state = nodes.pop()
            if quantized_state in visited_states:
                continue
            visited_states.add(quantized_state)
            remaining_vehicles = tracker.get_remaining_vehicles()
            if len(remaining_vehicles) == 0:
                self.vehicle_state_graph.mark_terminal_node(quantized_state)
                continue

            initial_state = self.state_quantizer.dequantize_state(
                quantized_state)
            for next_pos_to_coop in tracker.get_possible_cooperative_vehicles():
                for starting_next_pos_to_move in remaining_vehicles:
                    next_positions_to_move = set()
                    for p in range(starting_next_pos_to_move, self._n_platoon):
                        if p not in remaining_vehicles:
                            continue
                        next_positions_to_move = next_positions_to_move | {p}
                        vehicle_group = self._create_vehicle_group()
                        success = _simulate_till_lane_change(
                            vehicle_group, initial_state, free_flow_speeds,
                            next_positions_to_move, next_pos_to_coop
                        )
                        # vehicle_group.truncate_simulation_history()
                        # data = vehicle_group.to_dataframe()
                        # analysis.plot_trajectory(data)
                        # analysis.plot_platoon_lane_change(data)
                        if success:
                            next_tracker = copy.deepcopy(tracker)
                            next_tracker.move_vehicles(next_positions_to_move,
                                                       next_pos_to_coop)
                            next_quantized_state = (
                                self.state_quantizer.quantize_without_bounds(
                                    vehicle_group.get_state_by_vehicle())
                            )
                            transition_time = vehicle_group.get_current_time()
                            accel_cost = (
                                vehicle_group.compute_acceleration_cost())
                            self.vehicle_state_graph.update_graph(
                                quantized_state, next_quantized_state,
                                transition_time, accel_cost,
                                next_positions_to_move, next_pos_to_coop)
                            nodes.appendleft((next_tracker,
                                              next_quantized_state))


class LaneChangeStrategyManager:

    _empty_initial_state: configuration.QuantizedState = (0,)

    def __init__(self, n_platoon: int, has_fd: bool):
        n_vehicles = n_platoon + 2 + has_fd
        self.vehicle_state_graph = VehicleStateGraph.load_from_file(
            n_platoon, has_fd)
        self.state_quantizer = self.vehicle_state_graph.get_state_quantizer(
            n_vehicles)
        self._initial_state_per_vehicle: \
            dict[int, configuration.QuantizedState] = dict()

    @staticmethod
    def load_strategy_map(n_platoon: int, cost_name: str
                          ) -> configuration.StrategyMap:
        file_name = "_".join(["min", cost_name, "strategies_for",
                              str(n_platoon), "vehicles.json"])
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "strategy_maps", file_name)

        with open(file_path) as f:
            raw_data: list[dict] = json.load(f)
            strategy_map: configuration.StrategyMap = defaultdict(dict)
            for d in raw_data:
                key1 = tuple(d["root"])
                key2 = frozenset(d["first_mover_set"])
                lc_order = [set(i) for i in d["lc_order"]]
                strategy_map[key1][key2] = ((lc_order, d["coop_order"]),
                                            d[cost_name])
        return strategy_map

    def set_maneuver_initial_state(
            self, ego_position_in_platoon: int, states: dict[str, np.ndarray]
    ) -> None:
        """
        Saves the current system state seen by the ego vehicle as a possible
        starting state for the lane change maneuver
        :param ego_position_in_platoon:
        :param states:
        :return:
        """
        # print("[GraphTools] set_maneuver_initial_state for veh at",
        #       ego_position_in_platoon)
        quantized_states = self.state_quantizer.quantize_without_bounds(
            states)
        if (quantized_states
                not in self.vehicle_state_graph.get_initial_states()):
            raise nx.NodeNotFound(quantized_states)
        self._initial_state_per_vehicle[ego_position_in_platoon] = (
            quantized_states
        )

    def set_empty_maneuver_initial_state(self, ego_position_in_platoon: int):
        self._initial_state_per_vehicle[ego_position_in_platoon] = (
            self._empty_initial_state)

    def find_minimum_cost_maneuver_from_root(
            self, cost_name: str) -> tuple[configuration.Strategy, float]:
        initial_states = set(self._initial_state_per_vehicle.values())
        initial_states.discard(self._empty_initial_state)

        all_costs = []  # for debugging
        all_strategies = []  # for debugging
        opt_cost = np.inf
        opt_strategy = None
        for root in initial_states:
            strategy, cost = (
                self.vehicle_state_graph.find_minimum_cost_strategy_from_node(
                    root, cost_name))
            all_costs.append(cost)
            all_strategies.append(strategy)
            if cost < opt_cost:
                opt_cost = cost
                opt_strategy = strategy
        return opt_strategy, opt_cost

    def find_min_cost_strategy_given_first_mover(
            self, first_mover_platoon_positions: set[int], cost_name: str
    ) -> tuple[configuration.Strategy, float]:
        # Get the initial state as seen by the first mover group
        initial_state_candidates = {self._initial_state_per_vehicle[p]
                                    for p in first_mover_platoon_positions}
        if len(initial_state_candidates) > 1:
            # If multiple first-movers, they should all "see" the same initial
            # state, i.e., the same position for ld
            raise RuntimeError("More than one possible initial state")
        initial_state = initial_state_candidates.pop()

        if initial_state not in self.vehicle_state_graph.get_initial_states():
            print(f"State {initial_state} not in graph. "
                  f"Waiting for next iteration...")
            # print("Adding it now...")
            # v_ff_lo = self.state_quantizer.dequantize_free_flow_velocities(
            #     quantized_states[3], possible_vel_dest)
            # v_ff_ld = self.state_quantizer.dequantize_free_flow_velocities(
            #     quantized_states[-1], possible_vel_dest)
            # free_flow_speeds = ([v_ff_lo] + [v_ff_platoon] * n_platoon
            #                     + [v_ff_ld])
            # self.add_initial_state_to_graph(
            #     quantized_states, set(self.states_graph.nodes),
            #     free_flow_speeds, mode="as")
            raise nx.NetworkXNoPath

        # Get the node after the first-movers have changed lanes
        first_move_state = (
            self.vehicle_state_graph.get_successor_given_first_movers(
                initial_state, first_mover_platoon_positions))

        opt_strategy = ([first_mover_platoon_positions.copy()], [-1])
        opt_strategy_from_first, opt_cost_from_first = (
            self.vehicle_state_graph.find_minimum_cost_strategy_from_node(
                first_move_state, cost_name)
        )
        opt_strategy[0].extend(opt_strategy_from_first[0])
        opt_strategy[1].extend(opt_strategy_from_first[1])
        opt_cost = opt_cost_from_first

        return opt_strategy, opt_cost

    def find_min_cost_strategy_given_first_mover_2(
            self, first_mover_platoon_positions: set[int],
            strategy_map: configuration.StrategyMap
    ) -> tuple[configuration.Strategy, float]:
        # Get the initial state as seen by the first mover group
        initial_state_candidates = {self._initial_state_per_vehicle[p]
                                    for p in first_mover_platoon_positions}
        if len(initial_state_candidates) > 1:
            # If multiple first-movers, they should all "see" the same initial
            # state, i.e., the same position for ld
            raise RuntimeError("More than one possible initial state")
        initial_state = initial_state_candidates.pop()

        opt_strategy = strategy_map[initial_state][
            frozenset(first_mover_platoon_positions)][0]
        opt_cost = strategy_map[initial_state][
            frozenset(first_mover_platoon_positions)][1]

        return opt_strategy, opt_cost

    # def find_minimum_cost_strategy_from_node(
    #         self, starting_node: configuration.QuantizedState, cost_name: str
    # ) -> tuple[configuration.Strategy, float]:
    #     dag: nx.DiGraph = self.vehicle_state_graph
    #     all_costs = []  # for debugging
    #     all_strategies = []  # for debugging
    #     opt_cost = np.inf
    #     opt_strategy = None
    #
    #     if dag.nodes[starting_node].get("is_terminal", False):
    #         return ([], []), 0
    #
    #     for node in nx.descendants(dag, starting_node):
    #         if dag.nodes[node].get("is_terminal", False):
    #             cost, path = nx.single_source_dijkstra(
    #                 dag, starting_node, node, weight=cost_name)
    #             strategy = self.get_maneuver_order_from_path(path)
    #             all_costs.append(cost)
    #             all_strategies.append(strategy)
    #             if cost < opt_cost:
    #                 opt_cost = cost
    #                 opt_strategy = strategy
    #     return opt_strategy, opt_cost
    #
    # def get_maneuver_order_from_path(self, path) -> configuration.Strategy:
    #     lc_order = []
    #     coop_order = []
    #     for source_node, target_node in nx.utils.pairwise(path):
    #         edge = self.vehicle_state_graph[source_node][target_node]
    #         lc_order.append(edge["lc_vehicles"])
    #         coop_order.append(edge["coop_vehicle"])
    #     return lc_order, coop_order


class StateQuantizer:
    """
    Class to manage quantization of states
    """

    def __init__(self, n_vehicles: int, parameters: dict[str, float],
                 veh_type: type[base.BaseVehicle] = None):

        if veh_type is None:
            veh_type = fsv.FourStateVehicle
        self._parameters = parameters
        self._intervals = veh_type.create_state_vector_2(
            parameters["dx"], parameters["dy"], parameters["dtheta"],
            parameters["dv"]
        )
        self._shift = veh_type.create_state_vector_2(0., -2., 0., 0.)
        self._n_states = veh_type.get_n_states()
        self._zero_idx = veh_type.get_idx_of_state("theta")
        # Still not sure if we'll need this [Feb 9]
        max_y = 1
        max_theta = 0
        min_y = -1
        min_theta = 0
        self._max = {
            "lo": veh_type.create_state_vector_2(
                parameters["max_x_lo"], max_y, max_theta,
                parameters["max_v_lo"]),
            "ld": veh_type.create_state_vector_2(
                parameters["max_x_ld"], max_y, max_theta,
                parameters["max_v_ld"]),
            "p": veh_type.create_state_vector_2(
                np.inf, max_y, max_theta, np.inf)
        }
        self._min = {
            "lo": veh_type.create_state_vector_2(
                parameters["min_x_lo"], min_y, min_theta,
                parameters["min_v_lo"]),
            "ld": veh_type.create_state_vector_2(
                parameters["min_x_ld"], min_y, min_theta,
                parameters["min_v_ld"]),
            "p": veh_type.create_state_vector_2(
                -np.inf, min_y, min_theta, -np.inf)
        }

    def quantize_state_array(self, full_system_state: Sequence[float]
                             ) -> configuration.QuantizedState:
        """
        Computes the quantized version of the system state (stack of all
        vehicle states)
        :param full_system_state:
        :return:
        """
        full_system_state = np.array(full_system_state)
        qx = (full_system_state - self._shift) // self._intervals
        qx = np.minimum(np.maximum(qx, self._min), self._max)
        return tuple(qx.astype(int))

    def quantize_with_bounds(self, state_by_vehicle: dict[str, Sequence[float]]
                             ) -> configuration.QuantizedState:
        """
        Computes the quantized version of the system state (stack of all
        vehicle states)
        :param state_by_vehicle:
        :return:
        """
        qx_by_vehicle = dict()
        for veh_name, state in state_by_vehicle.items():
            key = "p" if veh_name.startswith("p") else veh_name
            qx_by_vehicle[veh_name] = ((np.array(state) - self._shift)
                                       // self._intervals)
            qx_by_vehicle[veh_name] = np.minimum(
                np.maximum(qx_by_vehicle[veh_name], self._min[key]),
                self._max[key])
        qx_platoon = [qx for name, qx in qx_by_vehicle.items()
                      if name.startswith("p")]
        return tuple(
            VehicleStateGraph.order_values(
                qx_by_vehicle["lo"], qx_platoon, qx_by_vehicle["ld"], None)
            .astype(int))

    def quantize_without_bounds(
            self, state_by_vehicle: dict[str, Sequence[float]]
    ) -> configuration.QuantizedState:
        """
        Computes the quantized state without applying max or min values. Used
        for tests and debugging.
        :param state_by_vehicle:
        :return:
        """
        qx_by_vehicle = dict()
        for veh_name, state in state_by_vehicle.items():
            # key = "p" if veh_name.startswith("p") else veh_name
            qx_by_vehicle[veh_name] = ((np.array(state) - self._shift)
                                       // self._intervals)
        qx_platoon = [qx for name, qx in qx_by_vehicle.items()
                      if name.startswith("p")]
        return tuple(
            VehicleStateGraph.order_values(
                qx_by_vehicle["lo"], qx_platoon, qx_by_vehicle["ld"], None)
            .astype(int))

    def dequantize_state(self, full_quantized_state: Sequence[float],
                         mode: str = "mean") -> np.ndarray:
        """
        Estimates the continuous system state given the quantized state. The
        estimate is done based on the mode.
        :param full_quantized_state:
        :param mode:
        :return:
        """

        if mode == "min":
            delta = 0
        elif mode == "mean":
            delta = 0.5
        elif mode == "max":
            delta = 1.
        else:
            raise ValueError("Parameter mode must be 'min', 'mean', or 'max'")

        n_vehicles = len(full_quantized_state) // self._n_states
        intervals_long_array = np.tile(self._intervals, n_vehicles)
        shift_long_array = np.tile(self._shift, n_vehicles)
        zero_idx = [self._zero_idx + i * n_vehicles for i in range(n_vehicles)]

        full_quantized_state = np.array(full_quantized_state)
        # full_quantized_state = np.minimum(
        #     np.maximum(full_quantized_state, self._min), self._max)
        x = ((full_quantized_state + delta) * intervals_long_array
             + shift_long_array)
        x[zero_idx] = 0.
        return x

    def dequantize_free_flow_velocities(self, quantized_vel: float,
                                        possible_values: Iterable[float]):
        """
        Useful while debugging. We get a quantized initial velocity, assume it
        to be the free flow value and map it to one of the expected possible
        values
        :param quantized_vel:
        :param possible_values:
        :return:
        """
        vel = quantized_vel * self._parameters["dv"]
        possible_values = np.array(possible_values)
        idx = np.argmin(np.abs(possible_values - vel))
        return possible_values[idx]

    # def quantize_free_flow_velocities(self, velocities):
    #     qv = []
    #     for v in velocities:
    #         qv.append(v - self.min_value)
