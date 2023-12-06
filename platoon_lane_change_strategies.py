from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import warnings

import networkx as nx
import numpy as np

import configuration
import graph_tools
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

    def __init__(self, platoon_vehicles: list[fsv.FourStateVehicle]):
        # TODO: make the strategy have the platoon as a member. This will make
        #  it clearer that the strategy has access to updated platoon info
        self.platoon_vehicles = platoon_vehicles

    @classmethod
    def get_id(cls) -> int:
        return cls._id

    @classmethod
    def get_name(cls) -> str:
        return cls._name

    def get_lane_change_order(self):
        raise AttributeError('Only instances of the TemplateStrategy class '
                             'have a given lane change order.')

    def get_cooperation_order(self):
        raise AttributeError('Only instances of the TemplateStrategy class '
                             'have a given cooperation order.')

    def set_maneuver_order(
            self, lane_changing_order: list[set[int]] = None,
            cooperating_order: Sequence[int] = None):
        raise AttributeError('Only instances of the TemplateStrategy class '
                             'can have their lane change order set.')

    def set_states_graph(self, states_graph: graph_tools.VehicleStatesGraph):
        raise AttributeError('Only instances of the GraphStrategy class '
                             'can have their states graph set.')

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
        ego_veh = self.platoon_vehicles[ego_position]
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
        ego_veh = self.platoon_vehicles[ego_position]
        if ego_veh.has_lane_change_intention():
            return -1
        return self._get_incoming_vehicle_id(ego_position)

    @abstractmethod
    def can_start_lane_change(self, ego_position: int) -> bool:
        """
        Unrelated to safety. This method checks if the vehicle is
        authorized by the strategy to start its maneuver
        """
        pass

    def set_maneuver_initial_state(
            self, ego_position_in_platoon: int, lo_states: Iterable[float],
            platoon_states: Iterable[float], ld_states: Iterable[float],
            fd_states: Iterable[float]):
        pass

    def set_empty_maneuver_initial_state(self, ego_position_in_platoon: int):
        """
        When the vehicle at the given position cannot be in the first group
        to move to the destination lane, we must set its maneuver initial state
        to an empty value
        :param ego_position_in_platoon:
        :return:
        """
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
    _lane_changing_order: list[set[int]]
    # Defines which platoon vehicle cooperates with the lane changing platoon
    # vehicle at the same index
    _cooperating_order: np.ndarray
    # Index of the last (further behind) platoon vehicle that is already
    # at the destination lane
    _last_dest_lane_vehicle_idx: int

    def __init__(self, platoon_vehicles: list[fsv.FourStateVehicle]):
        super().__init__(platoon_vehicles)
        self._is_initialized = False

    def get_lane_change_order(self) -> list[set[int]]:
        return self._lane_changing_order

    def get_cooperation_order(self) -> np.ndarray:
        return self._cooperating_order

    def set_maneuver_order(
            self, lane_changing_order: list[set[int]] = None,
            cooperating_order: Sequence[int] = None):
        self._idx = 0
        self._lane_changing_order = lane_changing_order
        self._cooperating_order = np.array(cooperating_order, dtype=int)
        self._last_dest_lane_vehicle_idx = (
            self._get_rearmost_lane_changing_vehicle_position())
        self._is_initialized = True

    def can_start_lane_change(self, ego_position: int) -> bool:
        if self._idx >= len(self._lane_changing_order):
            warnings.warn('Template strategy unexpected behavior. '
                          'Come check')
            return False
        next_in_line = self._lane_changing_order[self._idx]
        next_vehs_to_maneuver = [self.platoon_vehicles[i] for i in next_in_line]

        # TODO: change where this check happens. The current lc vehicles should
        #  call a new method when they finish their maneuvers. In this new
        #  method we advance the idx
        # Check if next_in_line has finished its lane change
        are_done = [not veh.has_lane_change_intention() for veh
                    in next_vehs_to_maneuver]
        if np.all(are_done):
            if self._cooperating_order[self._idx] == -1:
                # If the vehicle completed a maneuver behind all others (no
                # coop), it is now the last vehicle
                # self._last_dest_lane_vehicle_idx = next_in_line[-1]
                self._last_dest_lane_vehicle_idx = (
                    self._get_rearmost_lane_changing_vehicle_position()
                )
            self._idx += 1

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
        # return is_my_turn and are_all_on_the_right_gap and are_all_safe

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        # First check if it's our turn to change lanes
        if ego_position not in self._lane_changing_order[self._idx]:
            return -1
        # The first vehicles to simultaneously change lanes do so behind the
        # destination lane leader of the front-most vehicle (same thing for
        # single vehicle lane change)
        if self._idx == 0:
            front_most_veh = max([self.platoon_vehicles[i]
                                  for i in self._lane_changing_order[0]],
                                 key=lambda x: x.get_x())
            return front_most_veh.get_suitable_destination_lane_leader_id()

        coop_veh_id = self._cooperating_order[self._idx]
        if coop_veh_id == -1:
            # Merge behind the platoon vehicle farther back in the dest lane
            return self.platoon_vehicles[
                self._last_dest_lane_vehicle_idx].get_id()
        else:
            # Get the vehicle ahead the vehicle which helps generate the gap
            return self.platoon_vehicles[self._cooperating_order[
                self._idx]].get_origin_lane_leader_id()

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        # We don't have to check whether the vehicle is already at the
        # destination lane because the cooperating order already takes care
        # of that
        if ego_position == self._cooperating_order[self._idx]:
            rear_most_pos = self._get_rearmost_lane_changing_vehicle_position()
            return self.platoon_vehicles[rear_most_pos].get_id()
        return -1

    def _get_rearmost_lane_changing_vehicle_position(self):
        """

        :return:
        """
        min_x = np.inf
        rear_most_vehicle_position_in_platoon = 0
        for i in self._lane_changing_order[self._idx]:
            veh = self.platoon_vehicles[i]
            if veh.get_x() < min_x:
                min_x = veh.get_x()
                rear_most_vehicle_position_in_platoon = i
        return rear_most_vehicle_position_in_platoon

    @staticmethod
    def _to_list_of_sets(value: Sequence[int]) -> list[set[int]]:
        return [{i} for i in value]


# TODO: poor naming
class GraphStrategy(TemplateStrategy):
    _id = 4
    _name = 'Graph-based'

    _lane_change_graph: graph_tools.VehicleStatesGraph

    def set_states_graph(self, states_graph: graph_tools.VehicleStatesGraph):
        self._lane_change_graph = states_graph

    def set_maneuver_initial_state(
            self, ego_position_in_platoon: int, lo_states: Sequence[float],
            platoon_states: Iterable[float], ld_states: Sequence[float],
            fd_states: Sequence[float]):
        if not self._is_initialized:
            self._lane_change_graph.set_maneuver_initial_state(
                ego_position_in_platoon, lo_states, platoon_states, ld_states,
                fd_states)

    def set_empty_maneuver_initial_state(self, ego_position_in_platoon: int):
        """
        When the vehicle at the given position cannot be in the first group
        to move to the destination lane, we must set its maneuver initial state
        to an empty value
        :param ego_position_in_platoon:
        :return:
        """
        if not self._is_initialized:
            self._lane_change_graph.set_empty_maneuver_initial_state(
                ego_position_in_platoon)

    def can_start_lane_change(self, ego_position: int) -> bool:
        if not self._is_initialized:
            self._decide_lane_change_order(ego_position)
            # self._decide_lane_change_order_new()
        if not self._is_initialized:
            return False
        return super().can_start_lane_change(ego_position)

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if not self._is_initialized:
            return -1
        return super()._get_desired_dest_lane_leader_id(ego_position)

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if not self._is_initialized:
            return -1
        return super()._get_incoming_vehicle_id(ego_position)

    def _decide_lane_change_order(self, ego_position: int):
        opt_path = None
        opt_cost = np.inf

        # TODO: we may be able to simplify this if the graph states include
        #  the dest lane follower (see _decide_lane_change_order_new), but
        #  I was not able to avoid quantization issues yet

        # First, we check if any vehicles are already at safe position to
        # start the maneuver
        for pos1 in range(len(self.platoon_vehicles)):
            first_movers = set()
            pos2 = pos1
            while (pos2 < len(self.platoon_vehicles)
                   and self.platoon_vehicles[pos2].get_is_lane_change_safe()):
                first_movers.add(pos2)
                try:
                    path, cost = (
                        self._lane_change_graph.
                        find_minimum_time_maneuver_order_given_first_mover(
                            first_movers))
                except nx.NetworkXNoPath:
                    continue
                pos2 += 1
                if cost < opt_cost:
                    opt_cost = cost
                    opt_path = path

        # If there are no vehicles at safe positions, we check if any are
        # close to a suitable gap
        if opt_path is None:
            for veh_pos in range(len(self.platoon_vehicles)):
                veh = self.platoon_vehicles[veh_pos]
                if veh.get_is_lane_change_gap_suitable():
                    try:
                        path, cost = (
                            self._lane_change_graph.
                            find_minimum_time_maneuver_order_given_first_mover(
                                {veh_pos}))
                    except nx.NetworkXNoPath:
                        continue
                    if cost < opt_cost:
                        opt_cost = cost
                        opt_path = path

        if opt_path is not None:
            self.set_maneuver_order(opt_path[0], opt_path[1])
            self._is_initialized = True
            print(f'Path chosen from graph: {opt_path[0]}, {opt_path[1]}')

    def _decide_lane_change_order_new(self):
        path, cost = self._lane_change_graph.find_minimum_time_maneuver()
        if path is not None:
            self.set_maneuver_order(path[0], path[1])
            self._is_initialized = True
            print(f'Path chosen from graph: {path[0]}, {path[1]}')


# ========================= Heuristic Strategies ============================= #

class IndividualStrategy(TemplateStrategy):
    """"Vehicles behave without platoon coordination"""

    _id = 9
    _name = 'Individual strategy'

    def set_maneuver_order(self, lane_changing_order: Sequence[int] = None,
                           cooperating_order: Sequence[int] = None):
        # lc and coop order are not used by this strategy
        lane_changing_order = [{i for i in range(len(self.platoon_vehicles))}]
        cooperating_order = [-1] * len(self.platoon_vehicles)
        super().set_maneuver_order(lane_changing_order, cooperating_order)

    def can_start_lane_change(self, ego_position: int) -> bool:
        return True

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        ego_veh = self.platoon_vehicles[ego_position]
        return ego_veh.get_destination_lane_leader_id()


class SynchronousStrategy(TemplateStrategy):
    _id = 10
    _name = 'Synchronous'

    def set_maneuver_order(self, lane_changing_order: Sequence[int] = None,
                           cooperating_order: Sequence[int] = None):
        # lc and coop order are not used by this strategy
        lane_changing_order = [{i for i in range(len(self.platoon_vehicles))}]
        cooperating_order = [-1] * len(self.platoon_vehicles)
        super().set_maneuver_order(lane_changing_order, cooperating_order)


class LeaderFirstStrategy(TemplateStrategy):
    _id = 11
    _name = 'Leader First'

    def set_maneuver_order(self, lane_changing_order: Sequence[int] = None,
                           cooperating_order: Sequence[int] = None):
        lane_changing_order = [i for i in range(len(self.platoon_vehicles))]
        cooperating_order = [-1] * len(self.platoon_vehicles)
        super().set_maneuver_order(
            TemplateStrategy._to_list_of_sets(lane_changing_order),
            cooperating_order)


class LastFirstStrategy(TemplateStrategy):
    _id = 12
    _name = 'Last First'

    def set_maneuver_order(self, lane_changing_order: Sequence[int] = None,
                           cooperating_order: Sequence[int] = None):
        lane_changing_order = [i for i in range(len(self.platoon_vehicles))]
        lane_changing_order.reverse()
        cooperating_order = [-1] + lane_changing_order[:-1]
        super().set_maneuver_order(
            TemplateStrategy._to_list_of_sets(lane_changing_order),
            cooperating_order)


class LeaderFirstReverseStrategy(TemplateStrategy):
    _id = 13
    _name = 'Leader First Reverse'

    def set_maneuver_order(self, lane_changing_order: Sequence[int] = None,
                           cooperating_order: Sequence[int] = None):
        lane_changing_order = [i for i in range(len(self.platoon_vehicles))]
        cooperating_order = [-1] + lane_changing_order[:-1]
        super().set_maneuver_order(
            TemplateStrategy._to_list_of_sets(lane_changing_order),
            cooperating_order)


# =========================== OLD ============================ #
# These strategies were 'hard coded' in the sense that they don't use the
# template strategy format

class IndividualStrategyHardCoded(LaneChangeStrategy):
    """"Vehicles behave without platoon coordination"""

    _id = 0
    _name = 'Individual strategy'

    def can_start_lane_change(self, ego_position: int) -> bool:
        return True

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        ego_veh = self.platoon_vehicles[ego_position]
        return ego_veh.get_destination_lane_leader_id()

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        return -1


class SynchronousStrategyHardCoded(LaneChangeStrategy):
    """
    All platoon vehicles change lanes at the same time
    """

    _id = 1
    _name = 'Synchronous'

    def can_start_lane_change(self, ego_position: int) -> bool:
        # Check if lane change is safe for all vehicles
        for veh in self.platoon_vehicles:
            if not veh.get_is_lane_change_safe():
                return False
        return True

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        ego_veh = self.platoon_vehicles[ego_position]
        if ego_position == 0:
            return ego_veh.get_destination_lane_leader_id()
        return -1  # self.get_preceding_vehicle_id(ego_id)

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        # ego_veh = self.vehicles[ego_position]
        if ego_position == 0:
            return -1
        return -1  # self.get_preceding_vehicle_id(ego_id)


class LeaderFirstStrategyHardCoded(LaneChangeStrategy):
    _id = 101
    _name = 'Leader First'

    def can_start_lane_change(self, ego_position: int) -> bool:
        # Check if preceding vehicle has finished its lane change
        if ego_position == 0:
            return True
        preceding_veh = self.platoon_vehicles[ego_position - 1]
        return not preceding_veh.has_lane_change_intention()

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if ego_position == 0:
            ego_veh = self.platoon_vehicles[ego_position]
            return ego_veh.get_destination_lane_leader_id()
        return self.platoon_vehicles[ego_position - 1].get_id()

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if ego_position == 0:
            return -1
        return -1


class LastFirstStrategyHardCoded(LaneChangeStrategy):
    _id = 102
    _name = 'Last First'

    def can_start_lane_change(self, ego_position: int) -> bool:
        # Check if following vehicle has finished its lane change
        if ego_position == len(self.platoon_vehicles) - 1:
            return True
        follower = self.platoon_vehicles[ego_position + 1]
        return not follower.has_lane_change_intention()

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if ego_position == len(self.platoon_vehicles) - 1:
            ego_veh = self.platoon_vehicles[ego_position]
            return ego_veh.get_destination_lane_leader_id()
        # If the follower has completed the lane change, then we want to
        # merge between the follower and the vehicle ahead of it (which
        # is the follower's current lane leader). Otherwise, we don't have
        # a target vehicle at the destination lane
        follower = self.platoon_vehicles[ego_position + 1]
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
        return self.platoon_vehicles[ego_position - 1].get_id()


class LeaderFirstReverseStrategyHardCoded(LaneChangeStrategy):
    _id = 103
    _name = 'Leader First Reverse'

    def can_start_lane_change(self, ego_position: int) -> bool:
        if ego_position == 0:
            return True
        # Check if we have overtaken the former preceding vehicle
        ego_veh = self.platoon_vehicles[ego_position]
        return (ego_veh.get_destination_lane_follower_id()
                == self.platoon_vehicles[ego_position - 1].get_id())

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if ego_position == 0:
            ego_veh = self.platoon_vehicles[ego_position]
            return ego_veh.get_destination_lane_leader_id()
        # If the preceding veh has completed the lane change, then we want
        # to merge between the preceding veh and the vehicle ahead of it
        # (which is the preceding veh's current lane leader). Otherwise,
        # we don't have a target vehicle at the destination lane
        preceding = self.platoon_vehicles[ego_position - 1]
        if preceding.has_lane_change_intention():
            return -1
        leader_lo = preceding.get_origin_lane_leader_id()
        return leader_lo

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if ego_position == len(self.platoon_vehicles) - 1:
            return -1
        # Similarly to the last veh first case, we don't need to check whether
        # the (former) follower has completed the lane change
        follower = self.platoon_vehicles[ego_position + 1]
        if follower.has_lane_change_intention():
            return follower.get_id()
        return -1


strategy_map: dict[int, type[LaneChangeStrategy]] = {
    IndividualStrategy.get_id(): IndividualStrategy,
    SynchronousStrategy.get_id(): SynchronousStrategy,
    TemplateStrategy.get_id(): TemplateStrategy,
    GraphStrategy.get_id(): GraphStrategy,
    LeaderFirstStrategy.get_id(): LeaderFirstStrategy,
    LastFirstStrategy.get_id(): LastFirstStrategy,
    LeaderFirstReverseStrategy.get_id(): LeaderFirstReverseStrategy,
    LeaderFirstStrategyHardCoded.get_id(): LeaderFirstStrategyHardCoded,
    LastFirstStrategyHardCoded.get_id(): LastFirstStrategyHardCoded,
    LeaderFirstReverseStrategyHardCoded.get_id():
        LeaderFirstReverseStrategyHardCoded,
}
