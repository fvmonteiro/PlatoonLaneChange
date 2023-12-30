from __future__ import annotations

import copy
import pickle
from collections.abc import Iterable, Mapping, Sequence
from collections import defaultdict, deque
import json
import os
from typing import Any

import networkx as nx
import numpy as np

import analysis
import configuration
import platoon
import vehicle_group as vg
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


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
        return (f'[{self.__class__.__name__}] remaining|moved|coop_order = '
                f'{self._remaining_vehicles}|{self._dest_lane_vehicles}|'
                f'{self._cooperating_order}')

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


class VehicleStatesGraph:

    _empty_initial_state: configuration.QuantizedState = (0,)

    def __init__(self, n_platoon: int, has_fd: bool):
        n_vehicles = n_platoon + 2 + has_fd
        self.state_quantizer = StateQuantizer(n_vehicles, dx=9, dv=2)
        self.n_platoon = n_platoon
        self._has_fd = has_fd

        self.states_graph = nx.DiGraph()
        self._initial_states: set[configuration.QuantizedState] = set()
        self._initial_state_per_vehicle: \
            dict[int, configuration.QuantizedState] = dict()

    @staticmethod
    def load_from_file(n_platoon: int, has_fd: bool) -> VehicleStatesGraph:
        file_name = VehicleStatesGraph.get_graph_file_name(n_platoon, has_fd)
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 'vehicle_state_graphs',
                                 file_name + '.pickle')
        with open(file_path, 'rb') as f:
            vsg: VehicleStatesGraph = pickle.load(f)
        return vsg

    @staticmethod
    def get_graph_file_name(n_platoon: int, has_fd: bool) -> str:
        return f'graph_{n_platoon}_vehicles_with{"" if has_fd else "out"}_fd'

    @staticmethod
    def load_strategies(n_platoon: int, cost_name: str
                        ) -> configuration.StrategyMap:
        file_name = '_'.join(['min', cost_name, 'strategies_for',
                              str(n_platoon), 'vehicles.json'])
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 'strategy_maps', file_name)

        with open(file_path) as f:
            raw_data: list[dict] = json.load(f)
            strategy_map: configuration.StrategyMap = defaultdict(dict)
            for d in raw_data:
                key1 = tuple(d['root'])
                key2 = frozenset(d['first_mover_set'])
                lc_order = [set(i) for i in d['lc_order']]
                strategy_map[key1][key2] = ((lc_order, d['coop_order']),
                                            d[cost_name])
        return strategy_map

    def create_graph(self, vel_orig_lane: Sequence[float],
                     vel_ff_platoon: float, delta_v_dest_lane: Sequence[float]):

        delta_x_lo = 0.

        print(f'Creating graph ')

        visited_states = set()
        for i, v0_lo in enumerate(vel_orig_lane):
            print(f'  {i+1}/{len(vel_orig_lane)} vo')
            for j, v0_diff in enumerate(delta_v_dest_lane):
                print(f'    {j+1}/{len(delta_v_dest_lane)} delta_v')
                v0_ld = v0_lo + v0_diff
                v0_fd = v0_lo + v0_diff
                free_flow_speeds = self._order_values(v0_lo, vel_ff_platoon,
                                                      v0_ld, v0_fd)

                nodes: deque[tuple[PlatoonLCTracker, tuple]] = deque()
                # Create and add the initial states nodes (before any lane
                # change)
                initial_states = self._create_initial_states(
                    v0_lo, v0_lo, v0_ld, v0_fd, delta_x_lo)
                self._initial_states |= set(initial_states)

                print(f'    {len(initial_states)} roots')
                for x0 in initial_states:
                    self.states_graph.add_node(x0)
                    nodes.appendleft((PlatoonLCTracker(self.n_platoon), x0))
                    # print(np.array(x0).reshape([-1, 4]).transpose())

                # Explore the children of each initial state node in BFS mode
                self._explore_until_maneuver_completion(
                    nodes, visited_states, free_flow_speeds)

    def _add_new_tree_to_graph(self, root: configuration.QuantizedState, 
                               v0_lo: float, v0_ld: float, v0_fd: float = None):

        visited_states = set(self.states_graph.nodes)
        v_ff_platoon = 1.2 * v0_lo
        free_flow_speeds = self._order_values(v0_lo, v_ff_platoon,
                                              v0_ld, v0_fd)
        self.states_graph.add_node(root)
        nodes = deque([(PlatoonLCTracker(self.n_platoon), root)])
        self._explore_until_maneuver_completion(
            nodes, visited_states, free_flow_speeds)

    def set_maneuver_initial_state(
            self, ego_position_in_platoon: int, lo_states: Sequence[float],
            platoon_states: Iterable[float], ld_states: Sequence[float],
            fd_states: Sequence[float]):
        states = np.round(
            self._order_values(lo_states, platoon_states, ld_states, fd_states))
        quantized_states = self.state_quantizer.quantize_state(states)
        if quantized_states not in self._initial_states:
            # not int set(self.states_graph.nodes):
            print(f'State {quantized_states} not in graph. Adding it now...')
            self._add_new_tree_to_graph(quantized_states, lo_states[-1],
                                        ld_states[-1], fd_states[-1])
        self._initial_state_per_vehicle[ego_position_in_platoon] = (
            quantized_states
        )

    def set_empty_maneuver_initial_state(self, ego_position_in_platoon: int):
        self._initial_state_per_vehicle[ego_position_in_platoon] = (
            self._empty_initial_state)

    def set_first_mover_cost(self, vehicle_position: int, cost: float):
        dag = self.states_graph
        root = self._initial_state_per_vehicle[vehicle_position]
        node = None
        for neighbor in dag.successors(root):
            if dag[root][neighbor]['lc_vehicle'] == vehicle_position:
                node = neighbor
                break
        if node is not None:
            dag[root][node]['weight'] = cost
        elif cost < np.inf:
            raise ValueError('Graph does not offer the possibility of '
                             f'starting the maneuver by veh position '
                             f'{vehicle_position}')

    def find_minimum_cost_maneuver(
            self, cost_name: str) -> tuple[configuration.Strategy, float]:
        initial_states = set(self._initial_state_per_vehicle.values())
        initial_states.discard(self._empty_initial_state)

        all_costs = []  # for debugging
        all_strategies = []  # for debugging
        opt_cost = np.inf
        opt_strategy = None
        for root in initial_states:
            strategy, cost = self.find_minimum_cost_strategy_from_node(
                root, cost_name)
            all_costs.append(cost)
            all_strategies.append(strategy)
            if cost < opt_cost:
                opt_cost = cost
                opt_strategy = strategy
        return opt_strategy, opt_cost

    def find_minimum_cost_maneuver_order_given_first_mover(
            self, first_mover_platoon_positions: set[int], cost_name: str
    ) -> tuple[configuration.Strategy, float]:
        # Get the initial state as seen by the first mover group
        dag: nx.DiGraph = self.states_graph
        initial_state_candidates = {self._initial_state_per_vehicle[p]
                                    for p in first_mover_platoon_positions}
        if len(initial_state_candidates) > 1:
            # If multiple first-movers, they should all 'see' the same initial
            # state, i.e., the same position for ld
            raise RuntimeError('More than one possible initial state')
        initial_state = initial_state_candidates.pop()

        # Get the node after the first-movers have changed lanes
        first_move_state = None
        for node in dag.successors(initial_state):
            if (dag[initial_state][node]['lc_vehicles']
                    == first_mover_platoon_positions):
                first_move_state = node
                break
        if first_move_state is None:
            raise nx.NetworkXNoPath

        opt_strategy = ([first_mover_platoon_positions.copy()], [-1])
        opt_strategy_from_first, opt_cost_from_first = (
            self.find_minimum_cost_strategy_from_node(first_move_state,
                                                      cost_name)
        )
        opt_strategy[0].extend(opt_strategy_from_first[0])
        opt_strategy[1].extend(opt_strategy_from_first[1])
        opt_cost = opt_cost_from_first

        return opt_strategy, opt_cost

    def find_minimum_cost_maneuver_order_given_first_mover_2(
            self, first_mover_platoon_positions: set[int],
            strategy_map: configuration.StrategyMap
    ) -> tuple[configuration.Strategy, float]:
        # Get the initial state as seen by the first mover group
        initial_state_candidates = {self._initial_state_per_vehicle[p]
                                    for p in first_mover_platoon_positions}
        if len(initial_state_candidates) > 1:
            # If multiple first-movers, they should all 'see' the same initial
            # state, i.e., the same position for ld
            raise RuntimeError('More than one possible initial state')
        initial_state = initial_state_candidates.pop()

        opt_strategy = strategy_map[initial_state][
            frozenset(first_mover_platoon_positions)][0]
        opt_cost = strategy_map[initial_state][
            frozenset(first_mover_platoon_positions)][1]

        return opt_strategy, opt_cost

    def save_self_to_file(self):
        file_name = VehicleStatesGraph.get_graph_file_name(self.n_platoon,
                                                           self._has_fd)
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 'vehicle_state_graphs',
                                 file_name + '.pickle')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print(f'File {file_name} saved')

    def save_minimum_cost_strategies_to_json(self, cost_name: str):
        dag = self.states_graph
        strategy_map: list[dict] = []

        for root in self._initial_states:
            for next_node in dag.successors(root):
                first_mover_set = dag[root][next_node]['lc_vehicles'].copy()
                strategy = ([first_mover_set], [-1])
                strategy_from_node, cost = (
                    self.find_minimum_cost_strategy_from_node(
                        next_node, cost_name)
                )
                strategy[0].extend(strategy_from_node[0])
                strategy[1].extend(strategy_from_node[1])
                strategy_map.append(
                    {'root': [int(i) for i in root],
                     'first_mover_set': list(first_mover_set),
                     'lc_order': [list(s) for s in strategy[0]],
                     'coop_order': strategy[1], cost_name: cost}
                )

        json_data = json.dumps(strategy_map, indent=2)
        file_name = '_'.join(['min', cost_name, 'strategies_for',
                             str(self.n_platoon), 'vehicles.json'])
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 'strategy_maps', file_name)
        with open(file_path, 'w') as file:
            file.write(json_data)
            print('Saved file ', file_name)

    def _create_vehicle_group(self, include_ld: bool = True
                              ) -> vg.ShortSimulationVehicleGroup:
        base.BaseVehicle.reset_vehicle_counter()
        lo = fsv.ShortSimulationVehicle(False)
        lo.set_name('lo')
        platoon_vehicles = []
        for i in range(self.n_platoon):
            veh = fsv.ShortSimulationVehicle(True, is_connected=True)
            veh.set_name('p' + str(i + 1))
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
            ld.set_name('ld')
        else:
            ld = []
        if self._has_fd:
            fd = fsv.ShortSimulationVehicle(False)
            fd.set_name('fd')
        else:
            fd = []
        all_vehicles = self._order_values(lo, platoon_vehicles, ld, fd)
        vehicle_group = vg.ShortSimulationVehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)

        return vehicle_group

    def _order_values(self, lo_value: Any, platoon_value: Any, ld_value: Any,
                      fd_value: Any) -> np.ndarray:
        if np.isscalar(platoon_value):
            platoon_value = [platoon_value] * self.n_platoon
        if not self._has_fd:
            fd_value = []
        return np.hstack((lo_value, platoon_value, ld_value, fd_value))

    @staticmethod
    def split_values(values: Sequence[Any]) -> dict[str, Any]:
        return {'lo': values[0], 'platoon': values[1:-1], 'ld': values[-1]}

    def _create_initial_states(
            self, v0_lo: float, v0_platoon: float, v0_ld: float, v0_fd: float,
            delta_x_lo: float
    ) -> list[configuration.QuantizedState]:
        """
        The root node is the quantized states of all platoon vehicles and the
        two destination lane vehicles between which the platoon will move.
        We assume x0_p1 = 0
        :param v0_lo:
        :param v0_platoon:
        :param delta_x_lo:
        :return:
        """
        vehicle_group = self._create_vehicle_group()
        p1 = vehicle_group.get_vehicle_by_name('p1')

        # Origin lane leader (convention: x0_p1 = 0)
        x0_lo_min = p1.compute_lane_keeping_desired_gap(v0_platoon)
        x0_lo = x0_lo_min + delta_x_lo
        lo = vehicle_group.get_vehicle_by_name('lo')
        lo.set_initial_state(x0_lo, 0., 0., v0_lo)

        # Platoon vehicles
        x0_platoon_vehicle = x0_lo_min
        for i in range(self.n_platoon):
            p_i = vehicle_group.get_vehicle_by_name('p' + str(i+1))
            x0_platoon_vehicle -= p_i.compute_lane_keeping_desired_gap(
                v0_platoon)
            p_i.set_initial_state(x0_platoon_vehicle, 0., 0., v0_platoon)

        # Destination lane
        ld = vehicle_group.get_vehicle_by_name('ld')
        pN = vehicle_group.get_vehicle_by_name('p' + str(self.n_platoon))
        dx = self.state_quantizer.dx
        possible_ld_x = np.arange(pN.get_x(), lo.get_x() + 2*dx, dx)
        quantized_initial_states: list[configuration.QuantizedState] = []
        for ld_x in possible_ld_x:
            ld.set_initial_state(ld_x, configuration.LANE_WIDTH, 0., v0_ld)
            if self._has_fd:
                fd = vehicle_group.get_vehicle_by_name('fd')
                fd_safe_gap = fd.compute_lane_keeping_desired_gap(v0_fd)
                possible_fd_x = np.arange(
                    ld_x - fd_safe_gap, pN.get_x() - fd_safe_gap - dx, -dx)
                for fd_x in possible_fd_x:
                    fd.set_initial_state(fd_x, configuration.LANE_WIDTH, 0.,
                                         v0_fd)
                    initial_state = (
                        vehicle_group.get_full_initial_state_vector())
                    quantized_initial_states.append(
                        self.state_quantizer.quantize_state(initial_state))
            else:
                initial_state = vehicle_group.get_full_initial_state_vector()
                quantized_initial_states.append(
                    self.state_quantizer.quantize_state(initial_state))
        return quantized_initial_states

    def _explore_until_maneuver_completion(
            self, nodes: deque[tuple[PlatoonLCTracker, tuple]],
            visited_states: set[tuple], free_flow_speeds: Sequence[float]):
        """
        Given a list of nodes containing all the first move states, explore
        until maneuver completion
        :param nodes:
        :param free_flow_speeds:
        :return:
        """

        while len(nodes) > 0:
            tracker, quantized_state = nodes.pop()
            if quantized_state in visited_states:
                continue
            visited_states.add(quantized_state)
            remaining_vehicles = tracker.get_remaining_vehicles()
            if len(remaining_vehicles) == 0:
                self._mark_terminal_node(quantized_state)
                continue

            initial_state = self.state_quantizer.dequantize_state(
                quantized_state)
            # print(tracker)
            # print('x0=', np.array(initial_state).reshape(-1, 4).transpose())
            for next_pos_to_coop in tracker.get_possible_cooperative_vehicles():
                for starting_next_pos_to_move in remaining_vehicles:
                    next_positions_to_move = set()
                    for p in range(starting_next_pos_to_move, self.n_platoon):
                        if p not in remaining_vehicles:
                            continue
                        next_positions_to_move = next_positions_to_move | {p}
                        vehicle_group = self._create_vehicle_group()
                        vehicle_group.set_verbose(False)
                        vehicle_group.set_vehicles_initial_states_from_array(
                            initial_state)
                        vehicle_group.set_platoon_lane_change_order(
                            [next_positions_to_move], [next_pos_to_coop])
                        platoon_veh_ids = (
                            vehicle_group.get_lane_changing_vehicle_ids())
                        if next_pos_to_coop >= 0:
                            next_veh_to_coop = vehicle_group.vehicles[
                                platoon_veh_ids[next_pos_to_coop]]
                        else:
                            next_veh_to_coop = None
                        next_vehs_to_move = [
                            vehicle_group.vehicles[platoon_veh_ids[pos]]
                            for pos in next_positions_to_move]

                        # print('  Coop veh:', vehicle_group.vehicles[
                        #     platoon_veh_ids[next_pos_to_coop]].get_name() if
                        #       next_pos_to_coop >= 0 else str(-1))
                        # print('  Next to move:',
                        #       [veh.get_name() for veh in next_vehs_to_move])

                        success = self.simulate_till_lane_change(
                            vehicle_group, free_flow_speeds,
                            next_vehs_to_move, next_veh_to_coop)
                        # data = vehicle_group.to_dataframe()
                        # analysis.plot_trajectory(data)
                        # analysis.plot_platoon_lane_change(data)
                        if success:
                            next_tracker = copy.deepcopy(tracker)
                            next_tracker.move_vehicles(next_positions_to_move,
                                                       next_pos_to_coop)
                            next_quantized_state = (
                                self.state_quantizer.quantize_state(
                                    vehicle_group.get_current_state())
                            )
                            transition_time = vehicle_group.get_current_time()
                            accel_cost = (
                                vehicle_group.compute_acceleration_cost())
                            self._update_graphs(
                                quantized_state, next_quantized_state,
                                transition_time, accel_cost,
                                next_positions_to_move, next_pos_to_coop)
                            nodes.appendleft((next_tracker,
                                              next_quantized_state))

    def simulate_till_lane_change(
            self, vehicle_group: vg.ShortSimulationVehicleGroup,
            free_flow_speeds: Sequence[float],
            lc_vehicles: Iterable[fsv.ShortSimulationVehicle],
            next_to_coop: fsv.ShortSimulationVehicle) -> bool:

        dt = 1.0e-2
        tf = 20.
        time = np.arange(0, tf + dt, dt)

        # Set initial state and desired velocities
        # vehicle_group.set_vehicles_initial_states_from_array(
        #     initial_state)
        # Due to quantization, we could end up with initial vel above free
        # flow desired vel.
        adjusted_free_flow_speeds = []
        for i, veh in enumerate(vehicle_group.get_all_vehicles_in_order()):
            adjusted_free_flow_speeds.append(
                max(free_flow_speeds[i], veh.get_vel()))
        vehicle_group.set_free_flow_speeds(adjusted_free_flow_speeds)

        vehicle_group.prepare_to_start_simulation(len(time))
        [veh.set_lane_change_direction(1) for veh in lc_vehicles]
        vehicle_group.update_surrounding_vehicles()

        if next_to_coop is None:
            front_most_vehicle = max([veh for veh in lc_vehicles],
                                     key=lambda x: x.get_x())
            ld = vehicle_group.get_vehicle_by_name('ld')
            if ld.get_x() < front_most_vehicle.get_x():
                return False
            desired_leader_id = ld.get_id()
        else:
            rear_most_vehicle = min([veh for veh in lc_vehicles],
                                    key=lambda x: x.get_x())
            desired_leader_id = next_to_coop.get_origin_lane_leader_id()
            next_to_coop.set_incoming_vehicle_id(rear_most_vehicle.get_id())

        [veh.set_desired_dest_lane_leader_id(desired_leader_id) for veh
         in lc_vehicles]
        i = 0
        while i < len(time) - 1 and np.any(
                [veh.has_lane_change_intention() for veh in lc_vehicles]):
            vehicle_group.simulate_one_time_step(time[i + 1])
            i += 1

        vehicle_group.truncate_simulation_history()
        success = [not veh.has_lane_change_intention() for veh in lc_vehicles]
        return np.all(success)

    def _update_graphs(
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
        if n_nodes % 50 == 0:
            print(f'\t{n_nodes} nodes created')
        # percentage = n_nodes * 100 / self.n_nodes_per_root
        # if percentage % 10 < 0.5:
        #     print(f'{percentage:.1f}% done')

    def _mark_terminal_node(self, node):
        self.states_graph.nodes[node]['is_terminal'] = True

    def _add_all_first_mover_nodes(
            self, nodes: deque[tuple[PlatoonLCTracker, tuple]],
            platoon_veh_ids: Sequence[int],
            v0_ld: float, delta_x_ld: float):
        """
        Adds to nodes all the states after a single platoon vehicle has moved
        to the destination lane. The method puts the destination lane leader
        at a safe distance in front of the first platoon vehicle to move
        :param nodes:
        :param platoon_veh_ids:
        :param v0_ld:
        :param delta_x_ld:
        :return:
        """
        tracker, quantized_state = nodes.pop()
        root_state = self.state_quantizer.dequantize_state(quantized_state)
        # print(tracker)
        # print('x0=\n', np.array(quantized_state).reshape(
        #     -1, 4).transpose())
        for first_pos_to_move in tracker.get_remaining_vehicles():
            vehicle_group = self._create_vehicle_group()
            vehicle_group.set_vehicles_initial_states_from_array(root_state)
            # print('First to move:', vehicle_group.vehicles[
            #     platoon_veh_ids[first_pos_to_move]].get_name())
            next_quantized_state = self._create_first_mover_node(
                copy.deepcopy(vehicle_group), v0_ld, delta_x_ld,
                platoon_veh_ids[first_pos_to_move]
            )
            # print('x1=\n',
            #       np.array(next_quantized_state).reshape(-1, 4).transpose())
            next_tracker = copy.deepcopy(tracker)
            next_tracker.move_vehicles([first_pos_to_move], -1)
            nodes.appendleft((next_tracker, next_quantized_state))
            self._update_graphs(
                quantized_state, next_quantized_state, 1., 0.,
                [first_pos_to_move], -1)

    def _create_first_mover_node(
            self, vehicle_group: vg.VehicleGroup, v0_ld: float,
            delta_x_ld: float, first_to_move_id: int
    ) -> tuple:

        lc_vehicle = vehicle_group.get_vehicle_by_id(first_to_move_id)
        lc_vehicle_x = lc_vehicle.get_x()  # move veh to the dest lane
        lc_vehicle.set_initial_state(
            lc_vehicle_x, configuration.LANE_WIDTH, lc_vehicle.get_theta(),
            lc_vehicle.get_vel())

        # The dest lane leader is placed ahead of the lane changing vehicle
        lc_vehicle_ref_gap = lc_vehicle.compute_lane_keeping_desired_gap()
        x0_ld = (lc_vehicle_x + lc_vehicle_ref_gap + delta_x_ld)
        ld = vehicle_group.get_vehicle_by_name('ld')
        ld.set_initial_state(x0_ld, configuration.LANE_WIDTH, 0., v0_ld)

        vehicle_group.get_full_initial_state_vector()
        initial_state = vehicle_group.get_full_initial_state_vector()

        return self.state_quantizer.quantize_state(initial_state)

    def _create_first_mover_node_from_scratch(
            self, vehicle_group: vg.VehicleGroup, v0_lo: float,
            v0_platoon: Sequence[float], v0_ld: float, v0_fd: float,
            delta_x: Mapping[str, float], first_move_pos_in_platoon: int
    ) -> tuple:
        """
        Creates the state after one platoon vehicle has changed lane from basic
        parameters (as opposed to requiring that the vehicle group have an
        initial state)
        :return:
        """
        ref_gaps = vehicle_group.get_initial_desired_gaps(
            self._order_values(v0_lo, v0_platoon, v0_ld, v0_fd))
        x0_platoon = np.zeros(self.n_platoon)
        y0_platoon = np.zeros(self.n_platoon)
        # Loop goes from p_2 to p_N because p1's position is already set to zero
        for i in range(1, self.n_platoon):
            x0_platoon[i] = x0_platoon[i - 1] - ref_gaps[i]
        idx_p1 = 1  # platoon leader idx in the vehicle array
        idx_lc = first_move_pos_in_platoon + idx_p1  # idx in the vehicle array
        y0_platoon[first_move_pos_in_platoon] = configuration.LANE_WIDTH
        theta0_platoon = np.array([0.] * self.n_platoon)
        platoon_states = np.vstack((x0_platoon, y0_platoon, theta0_platoon,
                                    v0_platoon)).reshape(-1, order='F')
        # Ahead of the platoon in origin lane
        x0_lo = x0_platoon[0] + ref_gaps[idx_p1] + delta_x['lo']
        y0_lo = 0
        lo_states = np.hstack((x0_lo, y0_lo, 0., v0_lo))
        # Ahead of the platoon in dest lane
        x0_ld = (x0_platoon[first_move_pos_in_platoon]
                 + ref_gaps[idx_lc] + delta_x['ld'])
        y0_d = configuration.LANE_WIDTH
        ld_states = np.hstack((x0_ld, y0_d, 0., v0_ld))
        # Behind ld
        if self._has_fd:
            x0_fd = (x0_platoon[first_move_pos_in_platoon]
                     - ref_gaps[-1] + delta_x['ld'])
            fd_states = np.hstack((x0_fd, y0_d, 0., v0_fd))
        else:
            fd_states = []
        return self.state_quantizer.quantize_state(
            self._order_values(lo_states, platoon_states, ld_states, fd_states))

    def find_minimum_cost_strategy_from_node(
            self, starting_node: configuration.QuantizedState, cost_name: str
    ) -> tuple[configuration.Strategy, float]:
        dag: nx.DiGraph = self.states_graph
        all_costs = []  # for debugging
        all_strategies = []  # for debugging
        opt_cost = np.inf
        opt_strategy = None

        if dag.nodes[starting_node].get('is_terminal', False):
            return ([], []), 0

        for node in nx.descendants(dag, starting_node):
            if dag.nodes[node].get('is_terminal', False):
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
            lc_order.append(edge['lc_vehicles'])
            coop_order.append(edge['coop_vehicle'])
        return lc_order, coop_order


class StateQuantizer:
    """
    Class to manage quantization of states
    """

    # Possible changes for speed or maintainability:
    # - Transform everything in numpy arrays and try to vectorize the
    #  quantize and dequantize operations in case this becomes a bottleneck
    # - Make the methods work with vehicle objects if that's better
    # - Set n_vehicles in constructor

    def __init__(
            self, n_vehicles: int, dx: float, dv: float, dy: float = 4,
            dtheta: float = None, veh_type: type[base.BaseVehicle] = None):
        if veh_type is None:
            veh_type = fsv.FourStateVehicle
        self.dx, self.dv, self.dy, self.dtheta = dx, dv, dy, dtheta
        intervals = []
        shift = []
        # veh_type.create_state_vector()
        for state in veh_type.get_state_names():
            if state == 'x':
                intervals.append(dx)
                shift.append(0.)
            elif state == 'y':
                intervals.append(dy)
                shift.append(-2.)
            elif state == 'theta':
                intervals.append(dtheta if dtheta is not None else np.inf)
                shift.append(-np.pi / 2)
            elif state == 'v':
                intervals.append(dv)
                shift.append(0.)
            else:
                raise ValueError(f'Vehicle type {veh_type} has an unknown'
                                 f' state: {state}.')
        self._intervals = np.tile(intervals, n_vehicles)
        self._shift = np.tile(shift, n_vehicles)
        self._zero_idx = np.isinf(self._intervals)

    def quantize_state(self, full_system_state: Sequence[float]
                       ) -> configuration.QuantizedState:
        """
        Computes the quantized version of the system state (stack of all
        vehicle states)
        :param full_system_state:
        :return:
        """
        full_system_state = np.array(full_system_state)
        qx = (full_system_state - self._shift) // self._intervals
        return tuple(qx.astype(int))

    def dequantize_state(self, full_quantized_state: Sequence[float],
                         mode: str = 'mean') -> np.ndarray:
        """
        Estimates the continuous system state given the quantized state. The
        estimate is done based on the mode.
        :param full_quantized_state:
        :param mode:
        :return:
        """

        if mode == 'min':
            delta = 0
        elif mode == 'mean':
            delta = 0.5
        elif mode == 'max':
            delta = 1.
        else:
            raise ValueError('Parameter mode must be "min", "mean", or "max"')

        full_quantized_state = np.array(full_quantized_state)
        x = (full_quantized_state + delta) * self._intervals + self._shift
        x[self._zero_idx] = 0.
        return x

    # def quantize_free_flow_velocities(self, velocities):
    #     qv = []
    #     for v in velocities:
    #         qv.append(v - self.min_value)
