from __future__ import annotations

import copy
import pickle
from collections.abc import Iterable, Mapping, Sequence
from collections import deque
from typing import Any

import networkx as nx
import numpy as np

import analysis
import configuration
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
        self._lane_changing_order: list[int] = []
        self._cooperating_order: list[int] = []

    def __repr__(self):
        return (f'[{self.__class__.__name__}] remaining|lc_order|coop_order = '
                f'{self._remaining_vehicles}|{self._lane_changing_order}|'
                f'{self._cooperating_order}')

    def get_remaining_vehicles(self) -> set[int]:
        return self._remaining_vehicles

    def get_possible_cooperative_vehicles(self) -> list[int]:
        """
        We return all the platoon vehicles that have already changed lanes
        :return:
        """
        return (self._lane_changing_order if len(self._lane_changing_order) > 0
                else [-1])

    def get_maneuver_order(self) -> tuple[tuple[int], tuple[int]]:
        """
        Returns tuples, so they can become nodes in a graph
        :return: lane changing order and cooperating order
        """
        return (tuple(self._lane_changing_order),
                tuple(self._cooperating_order))

    def move_vehicle(self, position_in_platoon: int,
                     cooperative_vehicle_position: int):
        self._remaining_vehicles.remove(position_in_platoon)
        self._lane_changing_order.append(position_in_platoon)
        self._cooperating_order.append(cooperative_vehicle_position)

    def bring_back_vehicle(self, position_in_platoon: int):
        self._cooperating_order.pop()
        self._lane_changing_order.pop()
        self._remaining_vehicles.add(position_in_platoon)


class VehicleStatesGraph:


    def __init__(self, n_platoon: int):
        n_vehs = n_platoon + 2
        self.state_quantizer = StateQuantizer(n_vehs, dx=10, dv=2)
        self.n_platoon = n_platoon

        self.states_graph = nx.DiGraph()
        self._initial_state_per_vehicle: dict[int, tuple[int]] = dict()

        # There are several ways of getting the total number of nodes.
        # We could avoid iteration by using a formula:
        # N![sum_{i=0}^{N} i!/(N-i-1)!]
        # but this gives us less information to debug (and possible numerical
        # issues?)
        children_per_level = [(self.n_platoon - i) * i
                              for i in range(self.n_platoon)]
        children_per_level[0] = self.n_platoon
        nodes_per_level = [1]
        for n in children_per_level:
            nodes_per_level.append(nodes_per_level[-1] * n)
        self.n_nodes_per_root = sum(nodes_per_level)

    def create_graph(self):

        sample_vehicle_group = self._create_vehicle_group()
        platoon_veh_ids = [
            sample_vehicle_group.get_vehicle_by_name(
                'p' + str(i + 1)).get_id() for i in range(self.n_platoon)
        ]

        # Still missing: loops for speeds and distances
        delta_x_lo = 0
        possible_vel = [10, 20]

        self.n_nodes_per_root *= (len(possible_vel) * self.n_platoon)
        print(f'Creating graph with approx. {self.n_nodes_per_root} nodes')

        visited_states = set()
        # Desired speeds
        for v0_lo in possible_vel:
            # for v0_ld in possible_vel:
                # for v0_p in possible_vel:
            v0_ld = v0_lo
            v0_platoon = [v0_lo] * self.n_platoon
            v_ff_platoon = v0_lo * 1.2
            free_flow_speeds = self._order_values(v0_lo, v_ff_platoon,
                                                  v0_ld)

            nodes: deque[tuple[PlatoonLCTracker, tuple]] = deque()

            # Create and add the initial states nodes (before any lane
            # change)
            initial_states = self._create_initial_states(v0_lo, v0_platoon,
                                                         v0_ld, delta_x_lo)
            for x0 in initial_states:
                nodes.appendleft((PlatoonLCTracker(self.n_platoon), x0))
                # print(x0)

            # Next, add all the first mover nodes
            # self._add_all_first_mover_nodes(nodes, platoon_veh_ids, v0_ld,
            #                                 delta_x['ld'])
            # print('All first mover nodes created.')

            # Then, explore the children of each node in BFS mode
            self._explore_until_maneuver_completion(
                nodes, visited_states, platoon_veh_ids, free_flow_speeds)

    def set_maneuver_initial_state(
            self, ego_position_in_platoon: int, lo_states: Iterable[float],
            platoon_states: Iterable[float], ld_states: Iterable[float]):
        states = self._order_values(lo_states, platoon_states, ld_states)
        quantized_states = self.state_quantizer.quantize_state(states)
        if quantized_states not in set(self.states_graph.nodes):
            raise KeyError('Initial state not in graph')
        self._initial_state_per_vehicle[ego_position_in_platoon] = (
            quantized_states
        )

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

    def find_minimum_time_maneuver_order(
            self, first_mover_platoon_position: int
    ) -> tuple[list[int], list[int]]:
        dag: nx.DiGraph = self.states_graph
        initial_state = self._initial_state_per_vehicle[
            first_mover_platoon_position]
        root = None
        for node in dag.successors(initial_state):
            if (dag[initial_state][node]['lc_vehicle']
                    == first_mover_platoon_position):
                root = node
                break
        if root is None:
            raise nx.NetworkXNoPath

        costs = []
        paths = []
        opt_cost = np.inf
        opt_path = None
        for node in dag.nodes:
            if dag.out_degree(node) == 0:  # pycharm is tripping
                lc_order = [first_mover_platoon_position]
                coop_order = [-1]
                try:
                    c, p = nx.single_source_dijkstra(dag, root, node)
                    costs.append(c)
                    # Get the lc and coop sequences in the path
                    for source_node, target_node in nx.utils.pairwise(p):
                        edge = dag[source_node][target_node]
                        lc_order.append(edge['lc_vehicle'])
                        coop_order.append(edge['coop_vehicle'])
                    paths.append((lc_order, coop_order))
                    if c < opt_cost:
                        opt_cost = c
                        opt_path = paths[-1]
                except nx.exception.NetworkXNoPath:
                    continue
        return opt_path

    def save_to_file(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def _create_vehicle_group(self, include_ld: bool = True
                              ) -> vg.ShortSimulationVehicleGroup:
        base.BaseVehicle.reset_vehicle_counter()
        lo = fsv.ShortSimulationVehicle(False)
        lo.set_name('lo')
        platoon_vehicles = []
        # platoon_veh_ids = []
        for i in range(self.n_platoon):
            veh = fsv.ShortSimulationVehicle(True)
            veh.set_name('p' + str(i + 1))
            platoon_vehicles.append(veh)
            # platoon_veh_ids.append(veh.get_id())
        if include_ld:
            ld = fsv.ShortSimulationVehicle(False)
            ld.set_name('ld')
        else:
            ld = []
        all_vehicles = self._order_values(lo, platoon_vehicles, ld)
        vehicle_group = vg.ShortSimulationVehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)

        return vehicle_group

    def _order_values(self, lo_value: Any, platoon_value: Any, ld_value: Any
                      ) -> np.ndarray:
        if np.isscalar(platoon_value):
            platoon_value = [platoon_value] * self.n_platoon
        return np.hstack((lo_value, platoon_value, ld_value))

    @staticmethod
    def split_values(values: Sequence[Any]) -> dict[str, Any]:
        return {'lo': values[0], 'platoon': values[1:-1], 'ld': values[-1]}

    def _create_initial_states(
            self, v0_lo: float, v0_platoon: Sequence[float], v0_ld: float,
            delta_x_lo: float
    ) -> list[tuple]:
        """
        The root node is the quantized states of all platoon vehicles and the
        vehicle ahead the platoon on the origin lane. We assume x0_p1 = 0
        The leader on the destination lane is only defined after we set which
        platoon vehicle moves first.
        :param v0_lo:
        :param v0_platoon:
        :param delta_x_lo:
        :return:
        """
        vehicle_group = self._create_vehicle_group()
        p1 = vehicle_group.get_vehicle_by_name('p1')

        # Origin lane leader (convention: x0_p1 = 0)
        x0_lo_min = p1.compute_lane_keeping_desired_gap(v0_platoon[0])
        x0_lo = x0_lo_min + delta_x_lo
        lo = vehicle_group.get_vehicle_by_name('lo')
        lo.set_initial_state(x0_lo, 0., 0., v0_lo)

        # Platoon vehicles and possible locations of the dest lane leader
        possible_ld_x = []  # safe distance from each platoon vehicle
        x0_platoon_vehicle = x0_lo_min
        for i in range(self.n_platoon):
            possible_ld_x.append(x0_platoon_vehicle)
            p_i = vehicle_group.get_vehicle_by_name('p' + str(i+1))
            v0_pi = v0_platoon[i]
            x0_platoon_vehicle -= p_i.compute_lane_keeping_desired_gap(v0_pi)
            p_i.set_initial_state(x0_platoon_vehicle, 0., 0., v0_pi)

        # Dest lane leader
        quantized_initial_states = []
        ld = vehicle_group.get_vehicle_by_name('ld')
        for ld_x in possible_ld_x:
            ld.set_initial_state(ld_x, configuration.LANE_WIDTH, 0., v0_ld)
            initial_state = vehicle_group.get_full_initial_state_vector()
            quantized_initial_states.append(
                self.state_quantizer.quantize_state(initial_state))
        return quantized_initial_states

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
            next_tracker.move_vehicle(first_pos_to_move, -1)
            nodes.appendleft((next_tracker, next_quantized_state))
            self._update_graphs(
                quantized_state, next_quantized_state,  # tracker, next_tracker,
                1, first_pos_to_move, -1)

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

    def _explore_until_maneuver_completion(
            self, nodes: deque[tuple[PlatoonLCTracker, tuple]],
            visited_states: set[tuple],
            platoon_veh_ids: Sequence[int], free_flow_speeds: Sequence[float]):
        """
        Given a list of nodes containing all the first move states, explore
        until maneuver completion
        :param nodes:
        :param platoon_veh_ids:
        :param free_flow_speeds:
        :return:
        """
        # visited_states: set[tuple] = set()
        while len(nodes) > 0:
            tracker, quantized_state = nodes.pop()
            remaining_vehicles = tracker.get_remaining_vehicles()
            # Separated if conditions just for debugging
            if quantized_state in visited_states:
                continue
            if len(remaining_vehicles) == 0:
                continue
            visited_states.add(quantized_state)
            initial_state = self.state_quantizer.dequantize_state(
                quantized_state)
            # print(tracker)
            # print('x0=', np.array(initial_state).reshape(-1, 4).transpose())
            for next_pos_to_coop in tracker.get_possible_cooperative_vehicles():
                for next_pos_to_move in remaining_vehicles:
                    vehicle_group = self._create_vehicle_group()
                    vehicle_group.set_verbose(False)
                    if next_pos_to_coop >= 0:
                        next_veh_to_coop = vehicle_group.vehicles[
                            platoon_veh_ids[next_pos_to_coop]]
                    else:
                        next_veh_to_coop = None
                    next_veh_to_move = vehicle_group.vehicles[
                        platoon_veh_ids[next_pos_to_move]]
                    # print('  Coop veh:', vehicle_group.vehicles[
                    #     platoon_veh_ids[next_pos_to_coop]].get_name())
                    # print('  Next to move:', vehicle_group.vehicles[
                    #     platoon_veh_ids[next_pos_to_move]].get_name())
                    success = self.simulate_till_lane_change(
                        vehicle_group, free_flow_speeds, initial_state,
                        next_veh_to_move, next_veh_to_coop)
                    # data = vehicle_group.to_dataframe()
                    # analysis.plot_trajectory(data)
                    # analysis.plot_platoon_lane_change(data)
                    next_tracker = copy.deepcopy(tracker)
                    next_tracker.move_vehicle(next_pos_to_move,
                                              next_pos_to_coop)
                    if success:
                        next_quantized_state = (
                            self.state_quantizer.quantize_state(
                                vehicle_group.get_current_state())
                        )
                        nodes.appendleft((next_tracker, next_quantized_state))
                        transition_time = vehicle_group.get_current_time()
                        self._update_graphs(
                            quantized_state, next_quantized_state,
                            transition_time, next_pos_to_move, next_pos_to_coop)
                    # else:
                    #     print(f'# Transition from '
                    #           f'{tracker.get_maneuver_order()} to '
                    #           f'{next_tracker.get_maneuver_order()} failed #')
        print('done')

    def simulate_till_lane_change(
            self, vehicle_group: vg.ShortSimulationVehicleGroup,
            free_flow_speeds: Sequence[float], initial_state: np.ndarray,
            lc_vehicle: fsv.ShortSimulationVehicle,
            next_to_coop: fsv.ShortSimulationVehicle):

        dt = 1.0e-2
        tf = 20.
        time = np.arange(0, tf + dt, dt)

        # Set initial state and desired velocities
        vehicle_group.set_vehicles_initial_states_from_array(
            initial_state)
        # Due to quantization, we could end up with initial vel above free
        # flow desired vel.
        adjusted_free_flow_speeds = []
        for i, veh in enumerate(vehicle_group.get_all_vehicles_in_order()):
            adjusted_free_flow_speeds.append(
                max(free_flow_speeds[i], veh.get_vel()))
        vehicle_group.set_free_flow_speeds(adjusted_free_flow_speeds)

        vehicle_group.prepare_to_start_simulation(len(time))
        lc_vehicle.set_lane_change_direction(1)
        vehicle_group.update_surrounding_vehicles()

        if next_to_coop is None:
            ld = vehicle_group.get_vehicle_by_name('ld')
            if ld.get_x() <= lc_vehicle.get_x():
                return False
            desired_leader_id = ld.get_id()
        else:
            desired_leader_id = next_to_coop.get_origin_lane_leader_id()
            next_to_coop.set_incoming_vehicle_id(lc_vehicle.get_id())

        lc_vehicle.set_desired_dest_lane_leader_id(desired_leader_id)
        i = 0
        while i < len(time) - 1 and lc_vehicle.has_lane_change_intention():
            vehicle_group.simulate_one_time_step(time[i + 1])
            i += 1

        vehicle_group.truncate_simulation_history()
        success = not lc_vehicle.has_lane_change_intention()
        return success

    def _update_graphs(
            self, source_state: tuple[int], dest_state: tuple[int],
            # source_tracker: PlatoonLCTracker, dest_tracker: PlatoonLCTracker,
            weight: float, lc_vehicle: int, coop_vehicle: int):
        """
        Updates all tracking graphs with the same weight
        """
        self.states_graph.add_edge(source_state, dest_state, weight=weight,
                                   lc_vehicle=lc_vehicle,
                                   coop_vehicle=coop_vehicle)
        # self.maneuver_order_graph.add_edge(
        #     source_tracker.get_maneuver_order(),
        #     dest_tracker.get_maneuver_order(), weight=weight)

        n_nodes = self.states_graph.number_of_nodes()
        percentage = n_nodes * 100 / self.n_nodes_per_root
        if percentage % 10 < 0.5:
            print(f'{percentage:.1f}% done')

    def _create_first_mover_node_from_scratch(
            self, vehicle_group: vg.VehicleGroup, v0_lo: float,
            v0_platoon: Sequence[float], v0_ld: float,
            delta_x: Mapping[str, float], first_move_pos_in_platoon: int
    ) -> tuple:
        """
        Creates the state after one platoon vehicle has changed lane from basic
        parameters (as opposed to requiring that the vehicle group have an
        initial state)
        :return:
        """
        ref_gaps = vehicle_group.get_initial_desired_gaps(
            self._order_values(v0_lo, v0_platoon, v0_ld))
        x0_platoon = np.zeros(self.n_platoon)
        y0_platoon = np.zeros(self.n_platoon)
        # Loop goes from p_2 to p_N because p1's position is already set to zero
        for i in range(1, self.n_platoon):
            x0_platoon[i] = x0_platoon[i - 1] - ref_gaps[i]
        idx_p1 = 1  # platoon leader idx in the vehicle array
        idx_lc = first_move_pos_in_platoon + idx_p1  # idx in the vehicle array
        y0_platoon[first_move_pos_in_platoon] = configuration.LANE_WIDTH
        theta0_platoon = np.array([0.] * self.n_platoon)
        # Ahead of the platoon in origin lane
        x0_lo = x0_platoon[0] + ref_gaps[idx_p1] + delta_x['lo']
        y0_lo = 0
        theta0_lo = 0.
        # Ahead of the platoon in dest lane
        x0_ld = (x0_platoon[first_move_pos_in_platoon]
                 + ref_gaps[idx_lc] + delta_x['ld'])
        y0_ld = configuration.LANE_WIDTH
        theta0_ld = 0.

        # Get single column platoon states
        platoon_states = np.vstack((x0_platoon, y0_platoon, theta0_platoon,
                                    v0_platoon)).reshape(-1, order='F')
        lo_states = np.hstack((x0_lo, y0_lo, theta0_lo, v0_lo))
        ld_states = np.hstack((x0_ld, y0_ld, theta0_ld, v0_ld))
        return self.state_quantizer.quantize_state(
            self._order_values(lo_states, platoon_states, ld_states))


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
        intervals = []
        min_value = []
        for state in veh_type.get_state_names():
            if state == 'x':
                intervals.append(dx)
                min_value.append(0.)
            elif state == 'y':
                intervals.append(dy)
                min_value.append(-2.)
            elif state == 'theta':
                intervals.append(dtheta if dtheta is not None else np.inf)
                min_value.append(-np.pi / 2)
            elif state == 'v':
                intervals.append(dv)
                min_value.append(0.)
            else:
                raise ValueError(f'Vehicle type {veh_type} has an unknown'
                                 f' state: {state}.')
        self._intervals = np.tile(intervals, n_vehicles)
        self._min_value = np.tile(min_value, n_vehicles)
        self._zero_idx = np.isinf(self._intervals)

    def quantize_state(self, full_system_state: Sequence[float]) -> tuple[int]:
        """
        Computes the quantized version of the system state (stack of all
        vehicle states)
        :param full_system_state:
        :return:
        """
        # i = 0
        # n = len(full_system_state)
        # qx = np.zeros(n, dtype=int)
        # while i < n:
        #     for j in range(len(self._intervals)):
        #         qx[i] = ((full_system_state[i] - self._min_value[j])
        #                  // self._intervals[j])
        #         i += 1
        full_system_state = np.array(full_system_state)
        qx = (full_system_state - self._min_value) // self._intervals
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

        # i = 0
        # n = len(full_quantized_state)  # time is the last state
        # x = np.zeros(n)
        # while i < n:
        #     for j in range(len(self._intervals)):
        #         if self._intervals[j] == np.inf:
        #             x[i] = 0.
        #         else:
        #             x[i] = ((full_quantized_state[i] + delta)
        #                     * self._intervals[j] + self._min_value[j])
        #         i += 1
        full_quantized_state = np.array(full_quantized_state)
        x = (full_quantized_state + delta) * self._intervals + self._min_value
        x[self._zero_idx] = 0.
        return x

    # def quantize_free_flow_velocities(self, velocities):
    #     qv = []
    #     for v in velocities:
    #         qv.append(v - self.min_value)