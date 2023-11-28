from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import warnings

import networkx as nx
import numpy as np

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

    def set_lane_change_order(self, merging_order: Sequence[int] = None,
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
            platoon_states: Iterable[float], ld_states: Iterable[float]):
        pass

    @abstractmethod
    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        pass

    @abstractmethod
    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        pass


class IndividualStrategy(LaneChangeStrategy):
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


class SynchronousStrategy(LaneChangeStrategy):
    """
    All platoon vehicles change lanes at the same time
    """

    _id = 1
    _name = 'Synchronous strategy'

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


class TemplateStrategy(LaneChangeStrategy):
    """
    Strategy class for which we must provide the lane changing order and
    the cooperation order.
    Any sequential one-by-one maneuver strategy can be described this way.
    """

    _id = 2
    _name = 'Template strategy'

    _idx: int
    # Order in which platoon vehicles change lanes
    _lane_changing_order: Sequence[int]
    # Defines which platoon vehicle cooperates with the lane changing platoon
    # vehicle at the same index
    _cooperating_order: Sequence[int]
    # Index of the last (further behind) platoon vehicle that is already
    # at the destination lane
    _last_dest_lane_vehicle_idx: int

    def __init__(self, platoon_vehicles: list[fsv.FourStateVehicle]):
        super().__init__(platoon_vehicles)
        self._is_initialized = False

    def set_lane_change_order(self, lane_changing_order: Sequence[int] = None,
                              cooperating_order: Sequence[int] = None):
        self._idx = 0
        self._lane_changing_order = lane_changing_order
        self._cooperating_order = cooperating_order
        self._last_dest_lane_vehicle_idx = self._lane_changing_order[0]
        self._is_initialized = True

    def can_start_lane_change(self, ego_position: int) -> bool:
        if self._idx >= len(self._lane_changing_order):
            warnings.warn('BruteForce strategy unexpected behavior. '
                          'Come check')
            return False
        next_in_line = self._lane_changing_order[self._idx]
        next_veh_to_maneuver = self.platoon_vehicles[next_in_line]
        is_my_turn = ego_position == next_in_line
        # Check if next_in_line has finished its lane change
        if not next_veh_to_maneuver.has_lane_change_intention():
            if self._cooperating_order[self._idx] == -1:
                # If the vehicle completed a maneuver behind all others (no
                # coop), it is now the last vehicle
                self._last_dest_lane_vehicle_idx = next_in_line
            self._idx += 1
        return (is_my_turn and
                (next_veh_to_maneuver.get_desired_destination_lane_leader_id()
                 == next_veh_to_maneuver.get_destination_lane_leader_id()))

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        # First check if it's our turn to change lanes
        if ego_position != self._lane_changing_order[self._idx]:
            return -1
        # The first vehicle to change lanes does so behind its destination
        # lane leader
        if self._idx == 0:
            return self.platoon_vehicles[self._lane_changing_order[
                self._idx]].get_suitable_destination_lane_leader_id()

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
            return self.platoon_vehicles[
                self._lane_changing_order[self._idx]].get_id()
        return -1


class TemplateRelaxedStrategy(TemplateStrategy):
    """
    Similar to TemplateStrategy but allows vehicles to move as soon as it is
    safe
    There are some issues to address:
    - How to ensure vehicles stay together? We keep the desired dest lane
     leader to current dest lane leader comparison
    - How to ensure cooperation once the next vehicle starts its lane change
    """

    _id = 3
    _name = 'Relaxed brute force'

    def can_start_lane_change(self, ego_position: int) -> bool:
        if self._idx >= len(self._lane_changing_order):
            return True
        next_in_line = self._lane_changing_order[self._idx]
        next_veh_to_maneuver = self.platoon_vehicles[next_in_line]
        is_my_turn = ego_position == next_in_line
        # If it's my turn, and I start moving, I can allow the next vehicle
        # to move too.
        if is_my_turn and next_veh_to_maneuver.get_is_lane_change_safe():
            if self._cooperating_order[self._idx] == -1:
                # If the vehicle started a maneuver behind all others (no
                # coop), it is now the last vehicle
                self._last_dest_lane_vehicle_idx = next_in_line
            self._idx += 1

        return (is_my_turn and
                (next_veh_to_maneuver.get_desired_destination_lane_leader_id()
                 == next_veh_to_maneuver.get_destination_lane_leader_id()))

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        # First check if it's our turn to change lanes
        if (self._idx >= len(self.platoon_vehicles)  # all platoon moved
                or ego_position != self._lane_changing_order[self._idx]):
            return -1
        # The first vehicle to change lanes does so behind its destination
        # lane leader
        if self._idx == 0:
            return self.platoon_vehicles[self._lane_changing_order[
                self._idx]].get_destination_lane_leader_id()

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
        # TODO: cooperation is ending too soon
        # if (self._idx < len(self.platoon_vehicles)
        idx = min(self._idx, len(self.platoon_vehicles) - 1)
        if ego_position == self._cooperating_order[idx]:
            return self.platoon_vehicles[
                self._lane_changing_order[idx]].get_id()
        return -1


# TODO: poor naming
class GraphStrategy(TemplateStrategy):
    _id = 4
    _name = 'Graph strategy'

    _lane_change_graph: graph_tools.VehicleStatesGraph

    def set_states_graph(self, states_graph: graph_tools.VehicleStatesGraph):
        self._lane_change_graph = states_graph

    def set_maneuver_initial_state(
            self, ego_position_in_platoon: int, lo_states: Iterable[float],
            platoon_states: Iterable[float], ld_states: Iterable[float]):
        self._lane_change_graph.set_maneuver_initial_state(
            ego_position_in_platoon, lo_states, platoon_states, ld_states)

    def can_start_lane_change(self, ego_position: int) -> bool:
        if not self._is_initialized:
            self._decide_lane_change_order(ego_position)
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
        for veh_pos in range(len(self.platoon_vehicles)):
            veh = self.platoon_vehicles[veh_pos]
            if veh.get_is_lane_change_safe():
            #     self._lane_change_graph.set_first_mover_cost(veh_pos, 0.)
            # else:
            #     self._lane_change_graph.set_first_mover_cost(veh_pos, np.inf)
                try:
                    path, cost = (
                        self._lane_change_graph.
                        find_minimum_time_maneuver_order(veh_pos))
                    if cost < opt_cost:
                        opt_path = path
                except nx.NetworkXNoPath:
                    continue

        self.set_lane_change_order(opt_path[0], opt_path[1])
        self._is_initialized = True

        print(f'Path chosen from graph: {opt_path[0]}, {opt_path[1]}')


# ========================= Heuristic STRATEGIES ============================= #
class LeaderFirstStrategy(TemplateStrategy):
    _id = 11
    _name = 'Leader First strategy'

    def set_lane_change_order(self, lane_changing_order: Sequence[int] = None,
                              cooperating_order: Sequence[int] = None):
        lane_changing_order = [i for i in range(len(self.platoon_vehicles))]
        cooperating_order = [-1] * len(self.platoon_vehicles)
        super().set_lane_change_order(lane_changing_order, cooperating_order)


class LastFirstStrategy(TemplateStrategy):
    _id = 12
    _name = 'Last First strategy'

    def set_lane_change_order(self, lane_changing_order: Sequence[int] = None,
                              cooperating_order: Sequence[int] = None):
        lane_changing_order = [i for i in range(len(self.platoon_vehicles))]
        lane_changing_order.reverse()
        cooperating_order = [-1] + lane_changing_order[:-1]
        super().set_lane_change_order(lane_changing_order, cooperating_order)


class LeaderFirstReverseStrategy(TemplateStrategy):
    _id = 13
    _name = 'Leader First Reverse strategy'

    def set_lane_change_order(self, lane_changing_order: Sequence[int] = None,
                              cooperating_order: Sequence[int] = None):
        lane_changing_order = [i for i in range(len(self.platoon_vehicles))]
        cooperating_order = [-1] + lane_changing_order[:-1]
        super().set_lane_change_order(lane_changing_order, cooperating_order)


# =========================== OLD ============================ #
# These strategies were 'hard coded' in the sense that they don't use the
# template strategy format

class LeaderFirstStrategyHardCoded(LaneChangeStrategy):
    _id = 101
    _name = 'Leader First strategy'

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
    _name = 'Last First strategy'

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
    _name = 'Leader First Reverse strategy'

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


strategy_map = {
    IndividualStrategy.get_id(): IndividualStrategy,
    SynchronousStrategy.get_id(): SynchronousStrategy,
    TemplateStrategy.get_id(): TemplateStrategy,
    TemplateRelaxedStrategy.get_id(): TemplateRelaxedStrategy,
    GraphStrategy.get_id(): GraphStrategy,
    LeaderFirstStrategy.get_id(): LeaderFirstStrategy,
    LastFirstStrategy.get_id(): LastFirstStrategy,
    LeaderFirstReverseStrategy.get_id(): LeaderFirstReverseStrategy,
    LeaderFirstStrategyHardCoded.get_id(): LeaderFirstStrategyHardCoded,
    LastFirstStrategyHardCoded.get_id(): LastFirstStrategyHardCoded,
    LeaderFirstReverseStrategyHardCoded.get_id():
        LeaderFirstReverseStrategyHardCoded,
}
