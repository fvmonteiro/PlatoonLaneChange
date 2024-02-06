from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
import time
import warnings

import networkx as nx
import numpy as np

import configuration
import graph_tools
import platoon as plt
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


class StrategyGenerator:
    """
    Creates all vehicle merging orders and allows vehicles to try each
    at a time
    """

    def __init__(self):
        self.counter = 0

    def get_all_orders(self, n_vehicles: int,
                       starting_veh_positions: Iterable[int]
                       ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generates all merging and cooperation orders given the number of
        lane changing vehicles and which ones may start the maneuver
        :param n_vehicles:
        :param starting_veh_positions:
        :return:
        """
        all_positions = set([i for i in range(n_vehicles)])
        merging_orders = []
        coop_orders = []
        for i in starting_veh_positions:
            remaining_vehicles = all_positions
            for m_order, c_order in self.generate_order_all(
                    i, [], [], remaining_vehicles):
                merging_orders.append(m_order[:])
                coop_orders.append(c_order[:])
        return merging_orders, coop_orders

    def generate_order_all(
            self, veh_position: int, merging_order: list[int],
            cooperating_order: list[int], remaining_vehicles: set[int]
    ) -> tuple[list[int], list[int]]:
        # At every iteration, any of the vehicles still in the origin lane
        # (the remaining_vehicles) can be the next to merge. And this vehicle
        # can merge in front any of the vehicles already in the destination lane
        # or behind all of them (-1).
        # This generates a total of (N!)^2 where N is the number of platoon
        # vehicles, and we assume any platoon vehicle can be the first to merge.
        # If only M platoon vehicles cna be the first to merge,
        # then M.(N-1)!.N!

        # We can merge in front of any vehicle that has already changed lanes
        # or behind all of them [-1]
        gap_choices = [-1] + merging_order
        merging_order.append(veh_position)
        remaining_vehicles.remove(veh_position)
        for dest_lane_veh in gap_choices:
            # Choose gap (i.e. choose who cooperates with us)
            cooperating_order.append(dest_lane_veh)
            if len(remaining_vehicles) == 0:
                # Prepare to iterate over the current merging order
                # self._prepare_to_start_merging_order(merging_order,
                #                                      cooperating_order)
                self.counter += 1
                yield merging_order, cooperating_order
            for veh in remaining_vehicles:
                yield from self.generate_order_all(
                    veh, merging_order, cooperating_order,
                    remaining_vehicles.copy())
            cooperating_order.pop()
        merging_order.pop()
        remaining_vehicles.add(veh_position)


class LaneChangeStrategy(ABC):
    """
    Lane change strategies for platoons of vehicles controlled by feedback
    control laws. We can use the output of a simulation using a given strategy
    as initial guess for an optimal controller.
    """

    _id: int
    _name: str

    def __init__(self, platoon: plt.Platoon):
        # TODO: make the strategy have the platoon as a member. This will make
        #  it clearer that the strategy has access to updated platoon info
        self.platoon = platoon
        self._decision_time = 0.  # only used by Graph-Based.
        self._platoon_dest_lane_leader_id = -1

    @classmethod
    def get_id(cls) -> int:
        return cls._id

    @classmethod
    def get_name(cls) -> str:
        return cls._name

    def get_decision_time(self) -> float:
        return self._decision_time

    def get_platoon_destination_lane_leader(self) -> int:
        return self._platoon_dest_lane_leader_id

    @abstractmethod
    def get_lane_change_order(self) -> configuration.LCOrder:
        pass

    @abstractmethod
    def get_cooperation_order(self) -> configuration.CoopOrder:
        pass

    @abstractmethod
    def set_maneuver_order(
            self, lane_changing_order: configuration.LCOrder,
            cooperating_order: configuration.CoopOrder) -> None:
        pass

    def get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        """
        Defines sequence of leaders during a coordinated lane change maneuver.
        Only effective if platoon vehicles have a closed loop acceleration
        policy, i.e., not optimal control
        :param ego_position:
        :return:
        """
        # Coding the strategies becomes complicated when we want to control
        # when each vehicle increases the desired time headway to its leader.
        ego_veh = self.platoon.vehicles[ego_position]
        if not ego_veh.has_lane_change_intention():
            return -1
        return self._get_desired_dest_lane_leader_id(ego_position)

    def get_incoming_vehicle_id(self, ego_position: int) -> int:
        """
        Defines with which other platoon vehicle the ego vehicle cooperates
        after the ego has finished its lane change maneuver
        :param ego_position:
        :return:
        """
        ego_veh = self.platoon.vehicles[ego_position]
        if ego_veh.has_lane_change_intention():
            return -1
        return self._get_incoming_vehicle_id(ego_position)

    @abstractmethod
    def can_start_lane_change(self, ego_position: int,
                              vehicles: Mapping[int, base.BaseVehicle]) -> bool:
        """
        Unrelated to safety. This method checks if the vehicle is
        authorized by the strategy to start its maneuver
        """
        pass

    @abstractmethod
    def check_maneuver_step_done(
            self, lane_changing_vehs: list[fsv.FourStateVehicle]) -> None:
        pass

    @abstractmethod
    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        pass

    @abstractmethod
    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        pass


class TemplateStrategy(LaneChangeStrategy):
    """
    Strategy class for which we must provide the lane changing order and
    the cooperation order.
    Any sequential one-by-one maneuver strategy can be described this way.
    """

    _id = 2
    _name = 'Template'

    _idx: int
    # Order in which platoon vehicles change lanes
    _lane_changing_order: configuration.LCOrder
    # Defines which platoon vehicle cooperates with the lane changing platoon
    # vehicle at the same index
    _cooperating_order: configuration.CoopOrder
    # Index of the last (further behind) platoon vehicle that is already
    # at the destination lane
    _last_dest_lane_vehicle_idx: int

    def __init__(self, platoon: plt.Platoon):
        super().__init__(platoon)
        self._is_initialized = False

    def get_lane_change_order(self) -> configuration.LCOrder:
        return self._lane_changing_order

    def get_cooperation_order(self) -> configuration.CoopOrder:
        return self._cooperating_order

    def set_maneuver_order(
            self, lane_changing_order: configuration.LCOrder = None,
            cooperating_order: configuration.CoopOrder = None):
        self._idx = 0
        self._lane_changing_order = lane_changing_order
        self._cooperating_order = cooperating_order
        self._last_dest_lane_vehicle_idx = (
            self._get_rearmost_lane_changing_vehicle_position())
        self._is_initialized = True
        # print(f'Chosen LC/coop order: {lane_changing_order}, '
        #       f'{cooperating_order}')

    def can_start_lane_change(self, ego_position: int,
                              vehicles: Mapping[int, base.BaseVehicle]) -> bool:
        if (not self._is_initialized
                # not self.platoon.vehicles[0].has_started_lane_change()
                and ego_position == 0):
            self._decide_lane_change_order(vehicles)
        if not self._is_initialized:
            return False
        if self._idx >= len(self._lane_changing_order):
            warnings.warn('Template strategy unexpected behavior. '
                          'Come check')
            return False

        next_in_line = self._lane_changing_order[self._idx]
        next_vehs_to_maneuver = [self.platoon.vehicles[i] for i in next_in_line]
        self.check_maneuver_step_done(next_vehs_to_maneuver)
        is_my_turn = ego_position in next_in_line
        if is_my_turn:
            for veh in next_vehs_to_maneuver:
                if (veh.get_desired_destination_lane_leader_id()
                        != veh.get_destination_lane_leader_id()):
                    return False
                if not veh.get_is_lane_change_safe():
                    return False
            return True
        return False

    def check_maneuver_step_done(
            self, lane_changing_vehs: list[fsv.FourStateVehicle]) -> None:
        are_done = [not veh.has_lane_change_intention() for veh
                    in lane_changing_vehs]
        if np.all(are_done):
            if self._cooperating_order[self._idx] == -1:
                # If the vehicle completed a maneuver behind all others (no
                # coop), it is now the last vehicle
                # self._last_dest_lane_vehicle_idx = next_in_line[-1]
                self._last_dest_lane_vehicle_idx = (
                    self._get_rearmost_lane_changing_vehicle_position()
                )
            self._idx += 1

    def _decide_lane_change_order(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        pass

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if not self._is_initialized or self._is_lane_change_done():
            return -1
        if ego_position not in self._lane_changing_order[self._idx]:
            return -1

        coop_veh_id = self._cooperating_order[self._idx]
        if coop_veh_id == -1:
            # The first vehicles to simultaneously change lanes do so behind the
            # destination lane leader of the front-most vehicle (same thing for
            # single vehicle lane change)
            if self._idx == 0:
                front_most_veh = max([self.platoon.vehicles[i]
                                      for i in self._lane_changing_order[0]],
                                     key=lambda x: x.get_x())
                dest_lane_leader_id = (
                    front_most_veh.get_suitable_destination_lane_leader_id())
                self._platoon_dest_lane_leader_id = dest_lane_leader_id
            else:
                # Merge behind the platoon vehicle farther back in the dest lane
                dest_lane_leader_id = self.platoon.vehicles[
                    self._last_dest_lane_vehicle_idx].get_id()
        else:
            # Get the vehicle ahead the vehicle which helps generate the gap
            dest_lane_leader_id = self.platoon.vehicles[
                coop_veh_id].get_origin_lane_leader_id()

        return dest_lane_leader_id

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        # We don't have to check whether the vehicle is already at the
        # destination lane because the cooperating order already takes care
        # of that
        if (not self._is_initialized or self._is_lane_change_done()
                or ego_position != self._cooperating_order[self._idx]):
            return -1
        rear_most_pos = self._get_rearmost_lane_changing_vehicle_position()
        return self.platoon.vehicles[rear_most_pos].get_id()

    def _get_rearmost_lane_changing_vehicle_position(self):
        """

        :return:
        """
        min_x = np.inf
        rear_most_vehicle_position_in_platoon = 0
        for i in self._lane_changing_order[self._idx]:
            veh = self.platoon.vehicles[i]
            if veh.get_x() < min_x:
                min_x = veh.get_x()
                rear_most_vehicle_position_in_platoon = i
        return rear_most_vehicle_position_in_platoon

    def _is_lane_change_done(self):
        return self._idx >= len(self.platoon.vehicles)

    @staticmethod
    def _to_list_of_sets(value: Sequence[int]) -> list[set[int]]:
        return [{i} for i in value]


# TODO: poor naming
class GraphLaneChangeApproach(TemplateStrategy):
    _id = 4
    _name = 'Graph-based'

    _lane_change_graph: graph_tools.VehicleStatesGraph
    _strategy_map = configuration.StrategyMap
    _cost_name: str

    def __init__(self, platoon: plt.Platoon):
        super().__init__(platoon)
        self._is_data_loaded = False

    @classmethod
    def get_name(cls) -> str:
        return '_'.join([cls._name, cls._cost_name])

    def _load_data(self):
        n = len(self.platoon.vehicles)
        self._lane_change_graph = (
            graph_tools.VehicleStatesGraph.load_from_file(
                n, has_fd=False))
        self._strategy_map = (
            graph_tools.VehicleStatesGraph.load_strategies(n,
                                                           self._cost_name))
        self._is_data_loaded = True

    def _decide_lane_change_order(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:

        if not self._is_data_loaded:
            self._load_data()

        self._set_maneuver_initial_state_for_all_vehicles(vehicles)
        start_time = time.time()
        opt_strategy_from_graph = self._find_best_strategy_from_graph()
        if opt_strategy_from_graph is not None:
            final_time = time.time()
            self._decision_time = final_time - start_time
            print(f'Graph min cost took '
                  f'{self._decision_time:.2e} seconds')
            self.set_maneuver_order(opt_strategy_from_graph[0],
                                    opt_strategy_from_graph[1])

        # Testing
        opt_strategy_from_map = self._find_best_strategy_from_map()
        if opt_strategy_from_map != opt_strategy_from_graph:
            print(f'Different strategies.\n'
                  f'From graph: {opt_strategy_from_graph}\n'
                  f'From map: {opt_strategy_from_map}')

    def _set_maneuver_initial_state_for_all_vehicles(
            self, all_vehicles: Mapping[int, base.BaseVehicle]) -> None:
        """
        Method should only be called by the platoon leader and if the
        lane change has not yet started.
        :param all_vehicles:
        :return:
        """

        # print(f"[GraphApproach] t={all_vehicles[0].get_current_time():.2f} "
        #       f"setting all possible initial states")

        platoon_leader = self.platoon.get_platoon_leader()
        if platoon_leader.has_origin_lane_leader():
            lo = all_vehicles[platoon_leader.get_origin_lane_leader_id()]
            lo_states = lo.get_states()
        else:
            lo_states = []

        for pos, veh in enumerate(self.platoon.vehicles):
            if veh.get_is_lane_change_gap_suitable():
                if veh.has_destination_lane_leader():
                    ld = all_vehicles[veh.get_destination_lane_leader_id()]
                    ld_states = ld.get_states()
                else:
                    ld_states = []
                if veh.has_destination_lane_follower():
                    fd = all_vehicles[veh.get_destination_lane_follower_id()]
                    fd_states = fd.get_states()
                else:
                    fd_states = []
                self._set_maneuver_initial_state(pos, lo_states,
                                                 ld_states, fd_states)
            else:
                self._set_empty_maneuver_initial_state(veh.get_id())

    def _set_maneuver_initial_state(
            self, ego_position_in_platoon: int, lo_states: Sequence[float],
            ld_states: Sequence[float], fd_states: Sequence[float]) -> None:

        # print("[GraphApproach] set_maneuver_initial_state for veh at",
        #       ego_position_in_platoon)

        p1 = self.platoon.get_platoon_leader()
        # TODO: lazy workaround. We need to include the no leader
        #  possibilities in the graph
        if len(lo_states) == 0:
            lo_states = p1.get_states().copy()
            lo_states[0] += p1.compute_non_connected_reference_gap()
        else:
            lo_states = np.copy(lo_states)
        if len(ld_states) == 0:
            ld_states = lo_states.copy()
            ld_states[1] = p1.get_target_y()
        else:
            ld_states = np.copy(ld_states)
        if len(fd_states) == 0:
            pN = self.platoon.get_last_platoon_vehicle()
            fd_states = pN.get_states().copy()
            fd_states[0] -= pN.compute_non_connected_reference_gap()
            fd_states[1] = pN.get_target_y()
        else:
            fd_states = np.copy(fd_states)

        # We center all around the leader
        leader_x = p1.get_x()
        leader_y = p1.get_y()

        # TODO: avoid hard coding array indices
        platoon_states = []
        for veh in self.platoon.vehicles:
            veh_states = veh.get_states().copy()
            veh_states[0] -= leader_x
            veh_states[1] -= leader_y
            platoon_states.extend(veh_states)

        lo_states[0] -= leader_x
        lo_states[1] -= leader_y
        ld_states[0] -= leader_x
        ld_states[1] -= leader_y
        fd_states[0] -= leader_x
        fd_states[1] -= leader_y

        self._lane_change_graph.set_maneuver_initial_state(
            ego_position_in_platoon, lo_states, platoon_states, ld_states,
            fd_states)

    def _set_empty_maneuver_initial_state(self, ego_position_in_platoon: int):
        """
        When the vehicle at the given position cannot be in the first group
        to move to the destination lane, we must set its maneuver initial state
        to an empty value
        :param ego_position_in_platoon:
        :return:
        """
        # if not self._is_data_loaded:
        #     self._load_data()
        if not self._is_initialized:
            self._lane_change_graph.set_empty_maneuver_initial_state(
                ego_position_in_platoon)

    def _find_best_strategy_from_graph(self) -> configuration.Strategy:
        opt_strategy = None
        opt_cost = np.inf

        # First, we check if any vehicles are already at safe position to
        # start the maneuver
        all_costs = []
        all_strategies = []
        for pos1 in range(len(self.platoon.vehicles)):
            first_movers = set()
            pos2 = pos1
            while (pos2 < len(self.platoon.vehicles)
                   and self.platoon.vehicles[pos2].get_is_lane_change_safe()):
                first_movers.add(pos2)
                try:
                    strategy, cost = (
                        self._lane_change_graph.
                        find_minimum_cost_maneuver_order_given_first_mover(
                            first_movers, self._cost_name))
                    all_costs.append(cost)
                    all_strategies.append(strategy)
                except nx.NetworkXNoPath:
                    continue
                pos2 += 1
                if cost < opt_cost:
                    opt_cost = cost
                    opt_strategy = strategy

        # If there are no vehicles at safe positions, we check if any are
        # close to a suitable gap
        if opt_strategy is None:
            for veh_pos in range(len(self.platoon.vehicles)):
                veh = self.platoon.vehicles[veh_pos]
                if veh.get_is_lane_change_gap_suitable():
                    try:
                        strategy, cost = (
                            self._lane_change_graph.
                            find_minimum_cost_maneuver_order_given_first_mover(
                                {veh_pos}, self._cost_name))
                    except nx.NetworkXNoPath:
                        continue
                    if cost < opt_cost:
                        opt_cost = cost
                        opt_strategy = strategy
        return opt_strategy

    def _find_best_strategy_from_map(self) -> configuration.Strategy:
        opt_strategy = None
        opt_cost = np.inf
        # First, we check if any vehicles are already at safe position to
        # start the maneuver
        all_costs_from_map = []
        all_strategies_from_map = []
        for pos1 in range(len(self.platoon.vehicles)):
            first_movers = set()
            pos2 = pos1
            while (pos2 < len(self.platoon.vehicles)
                   and self.platoon.vehicles[pos2].get_is_lane_change_safe()):
                first_movers.add(pos2)
                strategy, cost = (
                    self._lane_change_graph.
                    find_minimum_cost_maneuver_order_given_first_mover_2(
                        first_movers, self._strategy_map))
                all_costs_from_map.append(opt_cost)
                all_strategies_from_map.append(opt_strategy)
                pos2 += 1
                if cost < opt_cost:
                    opt_cost = cost
                    opt_strategy = strategy

        # If there are no vehicles at safe positions, we check if any are
        # close to a suitable gap
        if opt_strategy is None:
            for veh_pos in range(len(self.platoon.vehicles)):
                veh = self.platoon.vehicles[veh_pos]
                if veh.get_is_lane_change_gap_suitable():
                    strategy, cost = (
                        self._lane_change_graph.
                        find_minimum_cost_maneuver_order_given_first_mover_2(
                            {veh_pos}, self._strategy_map))
                    if cost < opt_cost:
                        opt_cost = cost
                        opt_strategy = strategy
        return opt_strategy

    def _decide_lane_change_order_from_root(self, cost_name: str):
        # To be used if we include fd's state in the graph node's state
        # representation.
        path, min_cost = self._lane_change_graph.find_minimum_cost_maneuver(
            cost_name=cost_name
        )
        if path is not None:
            self.set_maneuver_order(path[0], path[1])
            # self._is_initialized = True
            print(f'Path chosen from graph: {path[0]}, {path[1]}')


class GraphLaneChangeApproachMinTime(GraphLaneChangeApproach):
    _id = 5
    _cost_name = 'time'


class GraphLaneChangeApproachMinAccel(GraphLaneChangeApproach):
    _id = 6
    _cost_name = 'accel'


graphStrategyIds = {GraphLaneChangeApproach.get_id(),
                    GraphLaneChangeApproachMinTime.get_id(),
                    GraphLaneChangeApproachMinAccel.get_id()}


# ========================= Heuristic Strategies ============================= #

class IndividualStrategy(TemplateStrategy):
    """"Vehicles behave without platoon coordination"""

    _id = 9
    _name = 'Individual strategy'

    def _decide_lane_change_order(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        # lc and coop order are not used by this strategy
        lane_changing_order = [{i for i in range(len(self.platoon.vehicles))}]
        cooperating_order = [-1] * len(self.platoon.vehicles)
        self.set_maneuver_order(lane_changing_order, cooperating_order)

    def can_start_lane_change(self, ego_position: int,
                              vehicles: Mapping[int, base.BaseVehicle]) -> bool:
        return True

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        ego_veh = self.platoon.vehicles[ego_position]
        return ego_veh.get_destination_lane_leader_id()


class SynchronousStrategy(TemplateStrategy):
    _id = 10
    _name = 'Synchronous'

    def _decide_lane_change_order(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        lane_changing_order = [{i for i in range(len(self.platoon.vehicles))}]
        cooperating_order = [-1] * len(self.platoon.vehicles)
        self.set_maneuver_order(lane_changing_order, cooperating_order)


class LeaderFirstStrategy(TemplateStrategy):
    _id = 11
    _name = 'Leader First'

    def _decide_lane_change_order(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        lane_changing_order = [i for i in range(len(self.platoon.vehicles))]
        cooperating_order = [-1] * len(self.platoon.vehicles)
        self.set_maneuver_order(
            TemplateStrategy._to_list_of_sets(lane_changing_order),
            cooperating_order)


class LastFirstStrategy(TemplateStrategy):
    _id = 12
    _name = 'Last First'

    def _decide_lane_change_order(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        lane_changing_order = [i for i in range(len(self.platoon.vehicles))]
        lane_changing_order.reverse()
        cooperating_order = [-1] + lane_changing_order[:-1]
        self.set_maneuver_order(
            TemplateStrategy._to_list_of_sets(lane_changing_order),
            cooperating_order)


class LeaderFirstReverseStrategy(TemplateStrategy):
    _id = 13
    _name = 'Leader First Reverse'

    def _decide_lane_change_order(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> None:
        lane_changing_order = [i for i in range(len(self.platoon.vehicles))]
        cooperating_order = [-1] + lane_changing_order[:-1]
        self.set_maneuver_order(
            TemplateStrategy._to_list_of_sets(lane_changing_order),
            cooperating_order)


# =========================== OLD ============================ #
# These strategies were 'hard coded' in the sense that they don't use the
# template strategy format

class HardCodedStrategy(LaneChangeStrategy, ABC):
    def check_maneuver_step_done(
            self, lane_changing_vehs: list[fsv.FourStateVehicle]) -> None:
        pass

    def get_lane_change_order(self):
        raise AttributeError('Only instances of the TemplateStrategy class '
                             'have the lane_change_order attribute.')

    def get_cooperation_order(self):
        raise AttributeError('Only instances of the TemplateStrategy class '
                             'have the cooperation_order attribute.')

    def set_maneuver_order(
            self, lane_changing_order: configuration.LCOrder,
            cooperating_order: configuration.CoopOrder) -> None:
        raise AttributeError('Only instances of the TemplateStrategy class '
                             'can have their lane change order set.')

    def can_start_lane_change(self, ego_position: int,
                              vehicles: Mapping[int, base.BaseVehicle]) -> bool:
        return self._implement_can_start_lane_change(ego_position)

    @abstractmethod
    def _implement_can_start_lane_change(self, ego_position: int) -> bool:
        pass


class IndividualStrategyHardCoded(HardCodedStrategy):
    """"Vehicles behave without platoon coordination"""

    _id = 0
    _name = 'Individual strategy'

    def _implement_can_start_lane_change(self, ego_position: int) -> bool:
        return True

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        ego_veh = self.platoon.vehicles[ego_position]
        return ego_veh.get_destination_lane_leader_id()

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        return -1


class SynchronousStrategyHardCoded(HardCodedStrategy):
    """
    All platoon vehicles change lanes at the same time
    """

    _id = 1
    _name = 'Synchronous'

    def _implement_can_start_lane_change(self, ego_position: int) -> bool:
        # Check if lane change is safe for all vehicles
        for veh in self.platoon.vehicles:
            if not veh.get_is_lane_change_safe():
                return False
        return True

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        ego_veh = self.platoon.vehicles[ego_position]
        if ego_position == 0:
            return ego_veh.get_destination_lane_leader_id()
        return -1  # self.get_preceding_vehicle_id(ego_id)

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        # ego_veh = self.vehicles[ego_position]
        if ego_position == 0:
            return -1
        return -1  # self.get_preceding_vehicle_id(ego_id)


class LeaderFirstStrategyHardCoded(HardCodedStrategy):
    _id = 101
    _name = 'Leader First'

    def _implement_can_start_lane_change(self, ego_position: int) -> bool:
        # Check if preceding vehicle has finished its lane change
        if ego_position == 0:
            return True
        preceding_veh = self.platoon.vehicles[ego_position - 1]
        return not preceding_veh.has_lane_change_intention()

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if ego_position == 0:
            ego_veh = self.platoon.vehicles[ego_position]
            return ego_veh.get_destination_lane_leader_id()
        return self.platoon.vehicles[ego_position - 1].get_id()

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if ego_position == 0:
            return -1
        return -1


class LastFirstStrategyHardCoded(HardCodedStrategy):
    _id = 102
    _name = 'Last First'

    def _implement_can_start_lane_change(self, ego_position: int) -> bool:
        # Check if following vehicle has finished its lane change
        if ego_position == len(self.platoon.vehicles) - 1:
            return True
        follower = self.platoon.vehicles[ego_position + 1]
        return not follower.has_lane_change_intention()

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if ego_position == len(self.platoon.vehicles) - 1:
            ego_veh = self.platoon.vehicles[ego_position]
            return ego_veh.get_destination_lane_leader_id()
        # If the follower has completed the lane change, then we want to
        # merge between the follower and the vehicle ahead of it (which
        # is the follower's current lane leader). Otherwise, we don't have
        # a target vehicle at the destination lane
        follower = self.platoon.vehicles[ego_position + 1]
        if follower.has_lane_change_intention():
            return -1
        follower_lo = follower.get_origin_lane_leader_id()
        return follower_lo

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if ego_position == 0:
            return -1
        # In theory, we only need to cooperate with the preceding vehicle
        # if it has not yet completed the lane change. But it will make
        # no difference returning the id here independent of that
        return self.platoon.vehicles[ego_position - 1].get_id()


class LeaderFirstReverseStrategyHardCoded(HardCodedStrategy):
    _id = 103
    _name = 'Leader First Reverse'

    def _implement_can_start_lane_change(self, ego_position: int) -> bool:
        if ego_position == 0:
            return True
        # Check if we have overtaken the former preceding vehicle
        ego_veh = self.platoon.vehicles[ego_position]
        return (ego_veh.get_destination_lane_follower_id()
                == self.platoon.vehicles[ego_position - 1].get_id())

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if ego_position == 0:
            ego_veh = self.platoon.vehicles[ego_position]
            return ego_veh.get_destination_lane_leader_id()
        # If the preceding veh has completed the lane change, then we want
        # to merge between the preceding veh and the vehicle ahead of it
        # (which is the preceding veh's current lane leader). Otherwise,
        # we don't have a target vehicle at the destination lane
        preceding = self.platoon.vehicles[ego_position - 1]
        if preceding.has_lane_change_intention():
            return -1
        leader_lo = preceding.get_origin_lane_leader_id()
        return leader_lo

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if ego_position == len(self.platoon.vehicles) - 1:
            return -1
        # Similarly to the last veh first case, we don't need to check whether
        # the (former) follower has completed the lane change
        follower = self.platoon.vehicles[ego_position + 1]
        if follower.has_lane_change_intention():
            return follower.get_id()
        return -1


strategy_map: dict[int, type[LaneChangeStrategy]] = {
    IndividualStrategy.get_id(): IndividualStrategy,
    SynchronousStrategy.get_id(): SynchronousStrategy,
    TemplateStrategy.get_id(): TemplateStrategy,
    GraphLaneChangeApproach.get_id(): GraphLaneChangeApproach,
    GraphLaneChangeApproachMinTime.get_id(): GraphLaneChangeApproachMinTime,
    GraphLaneChangeApproachMinAccel.get_id(): GraphLaneChangeApproachMinAccel,
    LeaderFirstStrategy.get_id(): LeaderFirstStrategy,
    LastFirstStrategy.get_id(): LastFirstStrategy,
    LeaderFirstReverseStrategy.get_id(): LeaderFirstReverseStrategy,
    LeaderFirstStrategyHardCoded.get_id(): LeaderFirstStrategyHardCoded,
    LastFirstStrategyHardCoded.get_id(): LastFirstStrategyHardCoded,
    LeaderFirstReverseStrategyHardCoded.get_id():
        LeaderFirstReverseStrategyHardCoded,
}
