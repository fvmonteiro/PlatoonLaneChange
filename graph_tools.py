import copy
from collections.abc import Mapping, Sequence
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
        return self._lane_changing_order

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
        self.state_quantizer = StateQuantizer(10, 2)
        self.n_platoon = n_platoon

        # The graphs contain overlapping information (we could obtain the order
        # from the states) but:
        # - it's easier to retrieve the chosen ordering from the order graph
        # - we may need the states graph later on
        self.maneuver_order_graph = nx.DiGraph()
        self.states_graph = nx.DiGraph()

        # TODO: wrong number
        self.expected_nodes = sum(
            [np.math.factorial(self.n_platoon - i) * np.math.factorial(i)
             for i in range(self.n_platoon + 1)])

    def create_vehicle_group(self, include_ld: bool = True):
        base.BaseVehicle.reset_vehicle_counter()
        veh_type = fsv.ShortSimulationVehicle
        lo = veh_type(False, False)
        lo.set_name('lo')
        platoon_vehs = []
        for i in range(self.n_platoon):
            veh = veh_type(True, True)
            veh.set_name('p' + str(i + 1))
            platoon_vehs.append(veh)
        if include_ld:
            ld = veh_type(False, False)
            ld.set_name('ld')
        else:
            ld = []
        all_vehicles = self.order_values(lo, platoon_vehs, ld)
        vehicle_group: vg.VehicleGroup = vg.VehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)

        return vehicle_group

    def order_values(self, lo_value: Any, platoon_value: Any, ld_value: Any
                     ) -> np.ndarray:
        if np.isscalar(platoon_value):
            platoon_value = [platoon_value] * self.n_platoon
        return np.hstack((lo_value, platoon_value, ld_value))

    @staticmethod
    def split_values(values: Sequence[Any]) -> dict[str, Any]:
        return {'lo': values[0], 'platoon': values[1:-1], 'ld': values[-1]}

    def create_graph(self):

        print(f'Creating graph with approx. {self.expected_nodes} nodes')

        # Still missing: loops for speeds and distances
        delta_x = {'lo': 0., 'ld': 0.}
        possible_vel = 10

        # Desired speeds
        v0_lo = possible_vel
        v0_ld = possible_vel
        v0_platoon = [possible_vel] * self.n_platoon
        v_ff_platoon = possible_vel * 1.2
        free_flow_speeds = self.order_values(v0_lo, v_ff_platoon, v0_ld)

        # Create and add the root node (before any lane change)
        vehicle_group = self.create_vehicle_group()
        platoon_veh_ids = [
            vehicle_group.get_vehicle_by_name('p' + str(i + 1)).get_id()
            for i in range(self.n_platoon)
        ]
        root_state = self.create_root_node(vehicle_group, v0_lo, v0_platoon,
                                           delta_x['lo'])

        # Next, add all the first mover nodes
        nodes: deque[tuple[PlatoonLCTracker, tuple]] = deque()
        nodes.appendleft((PlatoonLCTracker(self.n_platoon), root_state))
        self.add_all_first_mover_nodes(nodes, platoon_veh_ids, v0_ld,
                                       delta_x['ld'])
        print('All first mover nodes created.')

        # Then, explore the children of each node in BFS mode
        self.explore_until_maneuver_completion(nodes, platoon_veh_ids,
                                               free_flow_speeds)

    def create_root_node(
            self, vehicle_group: vg.VehicleGroup,
            v0_lo: float, v0_platoon: Sequence[float], delta_x_lo: float
    ) -> tuple:
        """
        The root node is the quantized states of all platoon vehicles and the
        vehicle ahead the platoon on the origin lane. We assume x0_p1 = 0
        The leader on the destination lane is only defined after we set which
        platoon vehicle moves first.
        :param vehicle_group: Vehicle group containing lo and platoon vehicles.
         Gets modified in place.
        :param v0_lo:
        :param v0_platoon:
        :param delta_x_lo:
        :return:
        """
        p1 = vehicle_group.get_vehicle_by_name('p1')
        # We start at x0 = h.v_p1 + c knowing it gets subtracted by the same
        # amount in the first iteration of the loop. So x0_p1 = 0
        x0_platoon_vehicle = p1.compute_lane_keeping_desired_gap(v0_platoon[0])
        for i in range(self.n_platoon):
            p_i = vehicle_group.get_vehicle_by_name('p' + str(i+1))
            v0_pi = v0_platoon[i]
            x0_platoon_vehicle -= p_i.compute_lane_keeping_desired_gap(v0_pi)
            p_i.set_initial_state(x0_platoon_vehicle, 0., 0., v0_pi)

        # Ahead of the platoon in origin lane
        x0_lo = p1.get_x() + p1.compute_lane_keeping_desired_gap() + delta_x_lo

        # Set initial state of other vehicles
        lo = vehicle_group.get_vehicle_by_name('lo')
        lo.set_initial_state(x0_lo, 0., 0., v0_lo)
        ld = vehicle_group.get_vehicle_by_name('ld')
        ld.set_initial_state(0., 0., 0., 0.)  # ld 'empty' for now
        initial_state = vehicle_group.get_full_initial_state_vector()
        quantized_state = self.state_quantizer.quantize_state(initial_state)
        return quantized_state

    def add_all_first_mover_nodes(
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
            vehicle_group = self.create_vehicle_group()
            vehicle_group.set_vehicles_initial_states_from_array(root_state)
            # print('First to move:', vehicle_group.vehicles[
            #     platoon_veh_ids[first_pos_to_move]].get_name())
            next_quantized_state = self.create_first_mover_node(
                copy.deepcopy(vehicle_group), v0_ld, delta_x_ld,
                platoon_veh_ids[first_pos_to_move]
            )
            # print('x1=\n',
            #       np.array(next_quantized_state).reshape(-1, 4).transpose())
            next_tracker = copy.deepcopy(tracker)
            next_tracker.move_vehicle(first_pos_to_move, -1)
            nodes.appendleft((next_tracker, next_quantized_state))
            self.update_graphs(
                quantized_state, next_quantized_state,
                tracker, next_tracker, weight=1)  # TODO: np.inf, 0 or 1?

    def create_first_mover_node(
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

    def explore_until_maneuver_completion(
            self, nodes: deque[tuple[PlatoonLCTracker, tuple]],
            platoon_veh_ids: Sequence[int], free_flow_speeds: Sequence[float]):
        """
        Given a list of nodes containing all the first move states, explore
        until maneuver completion
        :param nodes:
        :param platoon_veh_ids:
        :param free_flow_speeds:
        :return:
        """
        visited_states: set[tuple] = set()
        while len(nodes) > 0:
            tracker, quantized_state = nodes.pop()
            remaining_vehicles = tracker.get_remaining_vehicles()
            if (quantized_state in visited_states
                    or len(remaining_vehicles) == 0):
                continue
            visited_states.add(quantized_state)
            initial_state = self.state_quantizer.dequantize_state(
                quantized_state)
            # print(tracker)
            # print('x0=', np.array(initial_state).reshape(-1, 4).transpose())
            for next_pos_to_coop in tracker.get_possible_cooperative_vehicles():
                for next_pos_to_move in remaining_vehicles:
                    vehicle_group = self.create_vehicle_group()
                    vehicle_group.set_verbose(False)
                    next_id_to_coop = vehicle_group.vehicles[
                        platoon_veh_ids[next_pos_to_coop]].get_id()
                    next_id_to_move = vehicle_group.vehicles[
                        platoon_veh_ids[next_pos_to_move]].get_id()
                    # print('  Coop veh:', vehicle_group.vehicles[
                    #     platoon_veh_ids[next_pos_to_coop]].get_name())
                    # print('  Next to move:', vehicle_group.vehicles[
                    #     platoon_veh_ids[next_pos_to_move]].get_name())
                    success = self.simulate_till_lane_change(
                        vehicle_group, free_flow_speeds, initial_state,
                        next_id_to_move, next_id_to_coop)
                    # data = vehicle_group.to_dataframe()
                    # analysis.plot_trajectory(data)
                    # analysis.plot_platoon_lane_change(data)
                    if success:
                        next_quantized_state = (
                            self.state_quantizer.quantize_state(
                                vehicle_group.get_current_state())
                        )
                        next_tracker = copy.deepcopy(tracker)
                        next_tracker.move_vehicle(next_pos_to_move,
                                                  next_pos_to_coop)
                        nodes.appendleft((next_tracker, next_quantized_state))
                        transition_time = vehicle_group.get_current_time()
                        self.update_graphs(
                            quantized_state, next_quantized_state,
                            tracker, next_tracker, transition_time)
                    else:
                        print('### Failed ###')
        print('done')

    def simulate_till_lane_change(
            self, vehicle_group: vg.VehicleGroup,
            free_flow_speeds: Sequence[float], initial_state: np.ndarray,
            next_to_move: int, next_to_cooperate: int):

        dt = 1.0e-2
        tf = 20.
        time = np.arange(0, tf + dt, dt)

        vehicle_group.set_free_flow_speeds(free_flow_speeds)
        vehicle_group.set_vehicles_initial_states_from_array(
            initial_state)
        vehicle_group.prepare_to_start_simulation(len(time))
        vehicle_group.update_surrounding_vehicles()

        lc_vehicle = vehicle_group.get_vehicle_by_id(next_to_move)
        lc_vehicle.set_lane_change_direction(1)
        if next_to_cooperate is not None:
            coop_vehicle = vehicle_group.get_vehicle_by_id(next_to_cooperate)
            desired_leader_id = coop_vehicle.get_origin_lane_leader_id()
            coop_vehicle.set_incoming_vehicle_id(lc_vehicle.get_id())
        else:
            desired_leader_id = lc_vehicle.get_destination_lane_leader_id()
        lc_vehicle.set_desired_dest_lane_leader_id(desired_leader_id)
        i = 0
        while i < len(time) - 1 and lc_vehicle.has_lane_change_intention():
            vehicle_group.simulate_one_time_step(time[i + 1])
            i += 1

        vehicle_group.truncate_simulation_history()
        success = not lc_vehicle.has_lane_change_intention()
        return success

    def update_graphs(
            self, source_state: tuple[int], dest_state: tuple[int],
            source_tracker: PlatoonLCTracker, dest_tracker: PlatoonLCTracker,
            weight: float):
        """
        Updates all tracking graphs with the same weight
        """
        self.states_graph.add_edge(source_state, dest_state, weight=weight)
        self.maneuver_order_graph.add_edge(
            source_tracker.get_maneuver_order(),
            dest_tracker.get_maneuver_order(), weight=weight)
        # TODO: print periodically
        # if self.states_graph.number_of_nodes() /

    def get_minimum_time_maneuver_order(
            self, first_mover_platoon_position: int = None):
        dag = self.maneuver_order_graph
        if first_mover_platoon_position is None:
            root = list(dag.nodes)[0]
        else:
            roots = [node for node in dag.nodes
                     if (first_mover_platoon_position,) == node[0]]
            if len(roots) > 1:
                raise NotImplementedError('Not ready for multiple possible '
                                          'roots')
            root = roots[0]
        costs = []
        paths = []
        opt_cost = np.inf
        opt_path = None
        for node in dag.nodes:
            if dag.out_degree(node) == 0:  # pycharm is tripping
                try:
                    c, p = nx.single_source_dijkstra(dag, root, node)
                    print(p[-1], c)
                    costs.append(c)
                    paths.append(p[-1])
                    if c < opt_cost:
                        opt_cost = c
                        opt_path = p[-1]
                except nx.exception.NetworkXNoPath:
                    continue
        return opt_cost, opt_path

    def create_first_mover_node_from_scratch(
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
            self.order_values(v0_lo, v0_platoon, v0_ld))
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
            self.order_values(lo_states, platoon_states, ld_states))


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
            self, dx: float, dv: float, dy: float = 4, dtheta: float = None,
            veh_type: type[base.BaseVehicle] = fsv.FourStateVehicle):
        self.intervals = []
        self.min_value = []
        for state in veh_type.get_state_names():
            if state == 'x':
                self.intervals.append(dx)
                self.min_value.append(0.)
            elif state == 'y':
                self.intervals.append(dy)
                self.min_value.append(-2.)
            elif state == 'theta':
                self.intervals.append(dtheta if dtheta is not None else np.inf)
                self.min_value.append(-np.pi / 2)
            elif state == 'v':
                self.intervals.append(dv)
                self.min_value.append(0.)
            else:
                raise ValueError(f'Vehicle type {veh_type} has an unknown'
                                 f' state: {state}.')

    def quantize_state(self, full_system_state: Sequence[float]) -> tuple[int]:
        """
        Computes the quantized version of the system state (stack of all
        vehicle states)
        :param full_system_state:
        :return:
        """
        i = 0
        n = len(full_system_state)  # time is the last state
        qx = np.zeros(n, dtype=int)
        while i < n:
            for j in range(len(self.intervals)):
                qx[i] = ((full_system_state[i] - self.min_value[j])
                         // self.intervals[j])
                i += 1
        return tuple(qx)

    def dequantize_state(self, full_quantized_state: Sequence[float],
                         mode: str = 'mean') -> np.ndarray:
        """
        Estimates the continuous system state given the quantized state. The
        estimate is done based on the mode.
        :param full_quantized_state:
        :param mode:
        :return:
        """
        i = 0
        n = len(full_quantized_state)  # time is the last state
        x = np.zeros(n)
        if mode == 'min':
            delta = 0
        elif mode == 'mean':
            delta = 0.5
        elif mode == 'max':
            delta = 1.
        else:
            raise ValueError('Parameter mode must be "min", "mean", or "max"')

        while i < n:
            for j in range(len(self.intervals)):
                if self.intervals[j] == np.inf:
                    x[i] = 0.
                else:
                    x[i] = ((full_quantized_state[i] + delta)
                            * self.intervals[j] + self.min_value[j])
                i += 1
        return x
