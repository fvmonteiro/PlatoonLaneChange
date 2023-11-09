from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import warnings

# import bisect
import numpy as np

import controllers.optimal_controller as opt_ctrl
import vehicle_models.four_state_vehicles as fsv


class Platoon:
    vehicles: list[fsv.FourStateVehicle]
    lane_change_strategy: LaneChangeStrategy
    current_inputs: dict[int, float]

    _counter: int = 0

    def __init__(self):
        self._id: int = Platoon._counter
        Platoon._counter += 1
        self._id_to_position_map: dict[int, int] = {}

    def get_platoon_leader(self) -> fsv.FourStateVehicle:
        """
        Gets the front-most vehicle in the platoon. This method should not be
        used during a lane change maneuver in which platoon vehicles may change
        order
        :return:
        """
        return self.vehicles[0]

    def get_platoon_leader_id(self) -> int:
        return self.get_platoon_leader().get_id()

    def get_last_platoon_vehicle(self) -> fsv.FourStateVehicle:
        """
        Gets the vehicle at the end of the platoon. This method should not be
        used during a lane change maneuver in which platoon vehicles may change
        order
        :return:
        """
        return self.vehicles[-1]

    # def get_platoon_last_vehicle_name(self) -> str:
    #     return self.get_last_platoon_vehicle().get_name()

    def get_vehicle_ids(self) -> list[int]:
        return [veh.get_id() for veh in self.vehicles]

    def get_preceding_vehicle_id(self, veh_id: int) -> int:
        preceding_position = self._id_to_position_map[veh_id] - 1
        if preceding_position >= 0:
            return self.vehicles[preceding_position].get_id()
        else:
            return -1

    def get_following_vehicle_id(self, veh_id: int) -> int:
        following_position = self._id_to_position_map[veh_id] + 1
        if following_position < len(self.vehicles):
            return self.vehicles[following_position].get_id()
        else:
            return -1

    def get_desired_dest_lane_leader_id(self, ego_id) -> int:
        """
        Defines sequence of leaders during a coordinated lane change maneuver.
        Only effective if platoon vehicles have a closed loop acceleration
        policy, i.e., not optimal control
        :param ego_id:
        :return:
        """
        # Coding the strategies becomes complicated when we want to control
        # when each vehicle increases the desired time headway to its leader.
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.get_desired_dest_lane_leader_id(
            ego_position)

    def get_incoming_vehicle_id(self, ego_id) -> int:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.get_incoming_vehicle_id(ego_position)

    def set_strategy(self, lane_change_strategy: int):
        self.lane_change_strategy = strategy_map[lane_change_strategy](
            self.vehicles)

    def add_vehicle(self, new_vehicle: fsv.FourStateVehicle):
        """
        Adds the vehicle to the platoon. The new vehicle does not have to
        be behind all platoon vehicles
        :param new_vehicle: Vehicle being added to the platoon
        :return:
        """
        if len(self.vehicles) == 0:
            self._id_to_position_map[new_vehicle.get_id()] = 0
            self.vehicles.append(new_vehicle)
        elif new_vehicle.get_x() < self.vehicles[-1].get_x():
            self._id_to_position_map[new_vehicle.get_id()] = len(self.vehicles)
            self.vehicles.append(new_vehicle)
        else:
            # bisect.insort(self.vehicles, new_vehicle, key=lambda v: v.get_x())
            idx = np.searchsorted([veh.get_x() for veh in self.vehicles],
                                  new_vehicle.get_x())
            # Possibly slow, but irrelevant for the total run time
            self.vehicles = (self.vehicles[:idx] + [new_vehicle]
                             + self.vehicles[idx:])
            for i, veh in enumerate(self.vehicles):
                self._id_to_position_map[veh.get_id()] = i


class OptimalPlatoon(Platoon):

    def __init__(self, first_vehicle: fsv.OptimalControlVehicle,
                 lane_change_strategy: int):
        super().__init__()

        # Vehicles and their ids sorted by position (first is front-most)
        self.vehicles: list[fsv.OptimalControlVehicle] = []
        self.add_vehicle(first_vehicle)
        self.set_strategy(lane_change_strategy)

    def get_platoon_leader(self) -> fsv.OptimalControlVehicle:
        return self.vehicles[0]

    def get_optimal_controller(self) -> opt_ctrl.VehicleOptimalController:
        return self.get_platoon_leader().get_opt_controller()

    def add_vehicle(self, new_vehicle: fsv.OptimalControlVehicle):
        super().add_vehicle(new_vehicle)
        new_vehicle.set_centralized_controller(self.get_optimal_controller())

    # def guess_mode_sequence(self, initial_mode_sequence: som.ModeSequence):
    #     return self.strategy.create_mode_sequence(initial_mode_sequence)
    

class ClosedLoopPlatoon(Platoon):
    def __init__(self, first_vehicle: fsv.ClosedLoopVehicle,
                 lane_change_strategy: int,
                 strategy_parameters: tuple[list[int], list[int]] = None):
        super().__init__()

        # Vehicles and their ids sorted by position (first is front-most)
        self.vehicles: list[fsv.ClosedLoopVehicle] = []
        self.add_vehicle(first_vehicle)
        self.set_strategy(lane_change_strategy)
        if strategy_parameters:
            self.lane_change_strategy.set_parameters(strategy_parameters[0],
                                                     strategy_parameters[1])

    # def get_platoon_leader(self) -> fsv.ClosedLoopVehicle:
    #     return self.vehicles[0]

    def add_vehicle(self, new_vehicle: fsv.ClosedLoopVehicle):
        super().add_vehicle(new_vehicle)

    def can_start_lane_change(self, ego_id: int) -> bool:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.can_start_lane_change(ego_position)


class LaneChangeStrategy(ABC):
    """
    Lane change strategies for platoons of vehicles controlled by feedback
    control laws. We can use the output of a simulation using a given strategy
    as initial guess for an optimal controller.
    """

    _id: int
    _name: str

    def __init__(self, platoon_vehicles: list[fsv.FourStateVehicle]):
        self.platoon_vehicles = platoon_vehicles

    @classmethod
    def get_id(cls) -> int:
        return cls._id

    @classmethod
    def get_name(cls) -> str:
        return cls._name

    def set_parameters(self, merging_order: list[int] = None,
                       cooperating_order: list[int] = None):
        pass

    def get_desired_dest_lane_leader_id(self, ego_position) -> int:
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

    def get_incoming_vehicle_id(self, ego_position) -> int:
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
        Unrelated to safety. This is just about if the vehicle is
        authorized by the strategy to start its maneuver
        """
        pass

    @abstractmethod
    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        pass

    @abstractmethod
    def _get_incoming_vehicle_id(self, ego_position) -> int:
        pass

    # @abstractmethod
    # def create_mode_sequence(self, mode_sequence: som.ModeSequence):
    #     """
    #     Modifies the given mode sequence in place. The new mode sequence is
    #     a guess of how modes will change during the maneuver. It is not a
    #     prediction of the mode sequence.
    #     :param mode_sequence:
    #     :return:
    #     """
    #     #  if we create the notion of leader for optimal control vehicles,
    #     #  we must change everywhere in the virtual methods where 'lo' is set
    #     pass


class IndividualStrategy(LaneChangeStrategy):
    """"Vehicles behave without platoon coordination"""

    _id = 0
    _name = 'Individual strategy'

    def can_start_lane_change(self, ego_position) -> bool:
        return True

    def _get_desired_dest_lane_leader_id(self, ego_position) -> int:
        ego_veh = self.platoon_vehicles[ego_position]
        return ego_veh.get_dest_lane_leader_id()

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        return -1

    # def create_mode_sequence(self, mode_sequence: som.ModeSequence):
    #     # We create a mode where each platoon vehicle is between its
    #     # respective follower and leader in the destination lane
    #     changes = {}
    #     for veh in self.vehicles:
    #         changes.update(
    #             veh.get_surrounding_vehicle_changes_after_lane_change())
    #     time = mode_sequence.get_latest_switch_time()
    #     switch_time = time + self.switch_dt
    #     mode_sequence.alter_and_add_mode(switch_time, changes)


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
            return ego_veh.get_dest_lane_leader_id()
        return -1  # self.get_preceding_vehicle_id(ego_id)

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        # ego_veh = self.vehicles[ego_position]
        if ego_position == 0:
            return -1
        return -1  # self.get_preceding_vehicle_id(ego_id)

    # def create_mode_sequence(self, mode_sequence: som.ModeSequence):
    #     p1 = self.vehicles[0]
    #     sv_ids = p1.get_relevant_surrounding_vehicle_ids()
    #     changes = {}
    #     if sv_ids['lo'] >= 0 or sv_ids['ld'] >= 0:
    #         changes[p1.get_id()] = {'lo': sv_ids['ld']}
    #     if sv_ids['fd'] >= 0:
    #         last_id = self.vehicles[-1].get_id()
    #         changes[sv_ids['fd']] = {'lo': last_id, 'leader': last_id}
    #     time = mode_sequence.get_latest_switch_time()
    #     switch_time = time + self.switch_dt
    #     mode_sequence.alter_and_add_mode(switch_time, changes)


class LeaderFirstStrategy(LaneChangeStrategy):

    _id = 2
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
            return ego_veh.get_dest_lane_leader_id()
        return self.platoon_vehicles[ego_position - 1].get_id()

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if ego_position == 0:
            return -1
        return -1  # self.get_preceding_vehicle_id(ego_id)

    # def create_mode_sequence(self, mode_sequence: som.ModeSequence):
    #     time = mode_sequence.get_latest_switch_time()
    #     switch_time = time
    #     # Vehicles in front of the platoon in the origin lane and at
    #     # the dest lane, and vehicle behind at the dest lane
    #     platoon_leader_sv_ids = (
    #         self.vehicles[0].get_relevant_surrounding_vehicle_ids())
    #     ld = platoon_leader_sv_ids['ld']
    #     for i in range(len(self.vehicles)):
    #         changes = {}
    #         veh = self.vehicles[i]
    #         veh_id = veh.get_id()
    #         switch_time += self.switch_dt
    #         # Veh i will be behind its dest lane leader and in front of
    #         # the platoon's dest lane follower
    #         changes[veh_id] = {'lo': ld}
    #         if platoon_leader_sv_ids['fd'] >= 0:
    #             changes[platoon_leader_sv_ids['fd']] = {'lo': veh_id,
    #                                                     'leader': veh_id}
    #         # When veh i is at the dest lane, veh i+1 sees the vehicle in
    #         # front of the platoon at the origin lane as its origin lane
    #         # leader. Moreover, vehicle i becomes the dest lane leader of i+1
    #         if i + 1 < len(self.vehicles):
    #             ld = veh_id
    #             following_veh_id = self.vehicles[i + 1].get_id()
    #             changes[following_veh_id] = {
    #                 'lo': platoon_leader_sv_ids['lo'], 'ld': ld,
    #                 'fd': platoon_leader_sv_ids['fd']}
    #         mode_sequence.alter_and_add_mode(switch_time, changes)


class LastFirstStrategy(LaneChangeStrategy):

    _id = 3
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
            return ego_veh.get_dest_lane_leader_id()
        # If the follower has completed the lane change, then we want to
        # merge between the follower and the vehicle ahead of it (which
        # is the follower's current lane leader). Otherwise, we don't have
        # a target vehicle at the destination lane
        follower = self.platoon_vehicles[ego_position + 1]
        if follower.has_lane_change_intention():
            return -1
        follower_lo = follower.get_orig_lane_leader_id()
        return follower_lo

    def _get_incoming_vehicle_id(self, ego_position: int) -> int:
        if ego_position == 0:
            return -1
        # In theory, we only need to cooperate with the preceding vehicle
        # if it has not yet completed the lane change. But it will make
        # no difference returning the id here independent of that
        return self.platoon_vehicles[ego_position - 1].get_id()

    # def create_mode_sequence(self, mode_sequence: som.ModeSequence):
    #     time = mode_sequence.get_latest_switch_time()
    #     switch_time = time
    #     last_veh_sv_ids = (
    #         self.vehicles[-1].get_relevant_surrounding_vehicle_ids())
    #     # We assume no platoon vehicle changes 'lo'. This promotes cooperation
    #     # The last platoon veh moves in front of its dest lane follower
    #     changes = {}
    #     switch_time += self.switch_dt
    #     if last_veh_sv_ids['fd'] >= 0:
    #         veh_id = self.vehicles[-1].get_id()
    #         changes[last_veh_sv_ids['fd']] = {'lo': veh_id, 'leader': veh_id}
    #         mode_sequence.alter_and_add_mode(switch_time, changes)
    #     for i in range(len(self.vehicles) - 2, -1, -1):
    #         changes = {self.vehicles[i].get_id(): {
    #             'fd': self.vehicles[i + 1].get_id(),
    #             'ld': last_veh_sv_ids['ld']
    #         }}
    #         switch_time += self.switch_dt
    #         mode_sequence.alter_and_add_mode(switch_time, changes)


class LeaderFirstReverseStrategy(LaneChangeStrategy):

    _id = 4
    _name = 'Leader First Reverse strategy'

    def can_start_lane_change(self, ego_position: int) -> bool:
        if ego_position == 0:
            return True
        # Check if we have overtaken the former preceding vehicle
        ego_veh = self.platoon_vehicles[ego_position]
        return (ego_veh.get_dest_lane_follower_id()
                == self.platoon_vehicles[ego_position - 1].get_id())

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        if ego_position == 0:
            ego_veh = self.platoon_vehicles[ego_position]
            return ego_veh.get_dest_lane_leader_id()
        # If the preceding veh has completed the lane change, then we want
        # to merge between the preceding veh and the vehicle ahead of it
        # (which is the preceding veh's current lane leader). Otherwise,
        # we don't have a target vehicle at the destination lane
        preceding = self.platoon_vehicles[ego_position - 1]
        if preceding.has_lane_change_intention():
            return -1
        leader_lo = preceding.get_orig_lane_leader_id()
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

    # def create_mode_sequence(self, mode_sequence: som.ModeSequence):
    #     pass


class TemplateStrategy(LaneChangeStrategy):

    _id = 5
    _name = 'Brute Force strategy'

    _idx: int
    # Order in which platoon vehicles change lanes
    _lane_changing_order: list[int]
    # Defines which platoon vehicle cooperates with the lane changing platoon
    # vehicle at the same index
    _cooperating_order: list[int]
    # Index of the last (further behind) platoon vehicle that is already
    # at the destination lane
    _last_dest_lane_vehicle_idx: int

    def set_parameters(self, merging_order: list[int] = None,
                       cooperating_order: list[int] = None):
        self._idx = 0
        self._lane_changing_order = merging_order
        self._cooperating_order = cooperating_order
        self._last_dest_lane_vehicle_idx = self._lane_changing_order[0]

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
                (next_veh_to_maneuver.get_desired_dest_lane_leader_id()
                 == next_veh_to_maneuver.get_dest_lane_leader_id()))

    def _get_desired_dest_lane_leader_id(self, ego_position: int) -> int:
        # First check if it's our turn to change lanes
        if ego_position != self._lane_changing_order[self._idx]:
            return -1
        # The first vehicle to change lanes does so behind its destination
        # lane leader
        if self._idx == 0:
            return self.platoon_vehicles[self._lane_changing_order[
                self._idx]].get_dest_lane_leader_id()

        coop_veh_id = self._cooperating_order[self._idx]
        if coop_veh_id == -1:
            # Merge behind the platoon vehicle farther back in the dest lane
            return self.platoon_vehicles[
                self._last_dest_lane_vehicle_idx].get_id()
        else:
            # Get the vehicle ahead the vehicle which helps generate the gap
            return self.platoon_vehicles[self._cooperating_order[
                self._idx]].get_orig_lane_leader_id()

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        # We don't have to check whether the vehicle is already at the
        # destination lane because the cooperating order already takes care
        # of that
        if ego_position == self._cooperating_order[self._idx]:
            return self.platoon_vehicles[
                self._lane_changing_order[self._idx]].get_id()
        return -1


class StrategyGenerator:
    """
    Creates all vehicle merging orders and allows vehicles to try each
    at a time
    """

    def __init__(self):
        self.counter = 0

    def get_all_orders(self, n_vehicles: int,
                       starting_veh_positions: list[int]
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
                    copy.deepcopy(remaining_vehicles))
            cooperating_order.pop()
        merging_order.pop()
        remaining_vehicles.add(veh_position)


strategy_map = {
    IndividualStrategy.get_id(): IndividualStrategy,
    SynchronousStrategy.get_id(): SynchronousStrategy,
    LeaderFirstStrategy.get_id(): LeaderFirstStrategy,
    LastFirstStrategy.get_id(): LastFirstStrategy,
    LeaderFirstReverseStrategy.get_id(): LeaderFirstReverseStrategy,
    TemplateStrategy.get_id(): TemplateStrategy
}
