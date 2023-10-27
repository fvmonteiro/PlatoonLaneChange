from __future__ import annotations

from abc import ABC, abstractmethod
# import bisect
import numpy as np

import constants
import controllers.optimal_controller as opt_ctrl
import operating_modes.system_operating_mode as som
import vehicle_models.four_state_vehicles as fsv


class Platoon:
    current_inputs: dict[int, float]

    _counter: int = 0

    def __init__(self, first_vehicle: fsv.PlatoonVehicle):
        self._id: int = Platoon._counter
        Platoon._counter += 1

        # Vehicles and their ids sorted by position (first is front-most)
        self.vehicles: list[fsv.PlatoonVehicle] = []
        self._id_to_position_map: dict[int, int] = {}
        self.add_vehicle(first_vehicle)

        strategy_number = constants.Configuration.platoon_strategy
        self.strategy = _strategy_map[strategy_number](self.vehicles)

    def get_platoon_leader(self):
        return self.vehicles[0]

    def get_platoon_leader_id(self) -> int:
        return self.get_platoon_leader().get_id()

    def get_platoon_last_vehicle_name(self) -> str:
        return self.vehicles[-1].get_name()

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

    def get_optimal_controller(self) -> opt_ctrl.VehicleOptimalController:
        return self.get_platoon_leader().opt_controller

    def add_vehicle(self, new_vehicle: fsv.PlatoonVehicle):
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

    def get_dest_lane_leader_id(self, ego_id) -> int:
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
        return self.strategy.get_dest_lane_leader_id(ego_position)

    def get_incoming_vehicle_id(self, ego_id) -> int:
        ego_position = self._id_to_position_map[ego_id]
        return self.strategy.get_incoming_vehicle_id(ego_position)

    def guess_mode_sequence(self, initial_mode_sequence: som.ModeSequence):
        return self.strategy.create_mode_sequence(initial_mode_sequence)
    

# class OptimalPlatoon(Platoon):
#     def __init__


class LaneChangeStrategy(ABC):

    # When guessing modes, the time between each mode change
    switch_dt = 1.0  # TODO: might have to play with this value

    def __init__(self, platoon_vehicles: list[fsv.PlatoonVehicle]):
        self.vehicles = platoon_vehicles

    def get_dest_lane_leader_id(self, ego_position) -> int:
        """
        Defines sequence of leaders during a coordinated lane change maneuver.
        Only effective if platoon vehicles have a closed loop acceleration
        policy, i.e., not optimal control
        :param ego_position:
        :return:
        """
        # Coding the strategies becomes complicated when we want to control
        # when each vehicle increases the desired time headway to its leader.
        ego_veh = self.vehicles[ego_position]
        if not ego_veh.has_lane_change_intention():
            return -1
        return self._get_dest_lane_leader_id(ego_position)

    def get_incoming_vehicle_id(self, ego_position) -> int:
        """
        Defines with which other platoon vehicle the ego vehicle cooperates
        after the ego has finished its lane change maneuver
        :param ego_position:
        :return:
        """
        ego_veh = self.vehicles[ego_position]
        if ego_veh.has_lane_change_intention():
            return -1
        return self._get_incoming_vehicle_id(ego_position)

    @abstractmethod
    def _get_dest_lane_leader_id(self, ego_position: int):
        pass

    @abstractmethod
    def _get_incoming_vehicle_id(self, ego_position) -> int:
        pass

    @abstractmethod
    def create_mode_sequence(self, mode_sequence: som.ModeSequence):
        """
        Modifies the given mode sequence in place. The new mode sequence is
        a guess of how modes will change during the maneuver. It is not a
        prediction of the mode sequence.
        :param mode_sequence:
        :return:
        """
        # TODO: if we create the notion of leader for optimal control vehicles,
        #  we must change everywhere in the virtual methods where 'lo' is set
        pass


class IndividualStrategy(LaneChangeStrategy):
    """"Vehicles behave without platoon coordination"""

    def _get_dest_lane_leader_id(self, ego_position) -> int:
        ego_veh = self.vehicles[ego_position]
        return ego_veh.get_dest_lane_leader_id()

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        return -1

    def create_mode_sequence(self, mode_sequence: som.ModeSequence):
        # We create a mode where each platoon vehicle is between its
        # respective follower and leader in the destination lane
        changes = {}
        for veh in self.vehicles:
            changes.update(
                veh.get_surrounding_vehicle_changes_after_lane_change())
        time = mode_sequence.get_latest_switch_time()
        switch_time = time + self.switch_dt
        mode_sequence.alter_and_add_mode(switch_time, changes)


class SynchronousStrategy(LaneChangeStrategy):
    """
    All platoon vehicles change lanes at the same time
    """

    def _get_dest_lane_leader_id(self, ego_position) -> int:
        ego_veh = self.vehicles[ego_position]
        if ego_position == 0:
            return ego_veh.get_dest_lane_leader_id()
        # Strictly speaking, platoon vehicles in this strategy don't
        # need a virtual leader at the dest lane, but setting it here
        # helps them maintain a safe gap in case the leader changes lanes
        # first
        return -1  # self.get_preceding_vehicle_id(ego_id)

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        # ego_veh = self.vehicles[ego_position]
        if ego_position == 0:
            return -1
        # Strictly speaking, platoon vehicles in this strategy don't
        # need to cooperate with anyone, but the setting here prevents them
        # from accelerating in case they change lanes before their leaders
        return -1  # self.get_preceding_vehicle_id(ego_id)

    def create_mode_sequence(self, mode_sequence: som.ModeSequence):
        p1 = self.vehicles[0]
        sv_ids = p1.get_relevant_surrounding_vehicle_ids()
        changes = {}
        if sv_ids['lo'] >= 0 or sv_ids['ld'] >= 0:
            changes[p1.get_id()] = {'lo': sv_ids['ld']}
        if sv_ids['fd'] >= 0:
            last_id = self.vehicles[-1].get_id()
            changes[sv_ids['fd']] = {'lo': last_id, 'leader': last_id}
        time = mode_sequence.get_latest_switch_time()
        switch_time = time + self.switch_dt
        mode_sequence.alter_and_add_mode(switch_time, changes)


class LeaderFirstStrategy(LaneChangeStrategy):

    def _get_dest_lane_leader_id(self, ego_position) -> int:
        if ego_position == 0:
            ego_veh = self.vehicles[ego_position]
            return ego_veh.get_dest_lane_leader_id()
        return self.vehicles[ego_position - 1].get_id()

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        if ego_position == 0:
            return -1
        # Strictly speaking, platoon vehicles in this strategy don't
        # need to cooperate with anyone, but the setting here prevents them
        # from accelerating in case they change lanes before their leaders
        return -1  # self.get_preceding_vehicle_id(ego_id)

    def create_mode_sequence(self, mode_sequence: som.ModeSequence):
        time = mode_sequence.get_latest_switch_time()
        switch_time = time
        # Vehicles in front of the platoon in the origin lane and at
        # the dest lane, and vehicle behind at the dest lane
        platoon_leader_sv_ids = (
            self.vehicles[0].get_relevant_surrounding_vehicle_ids())
        ld = platoon_leader_sv_ids['ld']
        for i in range(len(self.vehicles)):
            changes = {}
            veh = self.vehicles[i]
            veh_id = veh.get_id()
            switch_time += self.switch_dt
            # Veh i will be behind its dest lane leader and in front of
            # the platoon's dest lane follower
            changes[veh_id] = {'lo': ld}
            if platoon_leader_sv_ids['fd'] >= 0:
                changes[platoon_leader_sv_ids['fd']] = {'lo': veh_id,
                                                        'leader': veh_id}
            # When veh i is at the dest lane, veh i+1 sees the vehicle in front
            # of the platoon at the origin lane as its origin lane leader.
            # Moreover, vehicle i becomes the dest lane leader of i+1
            if i + 1 < len(self.vehicles):
                ld = veh_id
                following_veh_id = self.vehicles[i + 1].get_id()
                changes[following_veh_id] = {'lo': platoon_leader_sv_ids['lo'],
                                             'ld': ld,
                                             'fd': platoon_leader_sv_ids['fd']}
            mode_sequence.alter_and_add_mode(switch_time, changes)


class LastFirstStrategy(LaneChangeStrategy):
    def _get_dest_lane_leader_id(self, ego_position) -> int:
        if ego_position == len(self.vehicles) - 1:
            ego_veh = self.vehicles[ego_position]
            return ego_veh.get_dest_lane_leader_id()
        # If the follower has completed the lane change, then we want to
        # merge between the follower and the vehicle ahead of it (which
        # is the follower's current lane leader). Otherwise, we don't have
        # a target vehicle at the destination lane
        follower = self.vehicles[ego_position + 1]
        if follower.has_lane_change_intention():
            return -1
        follower_lo = follower.get_orig_lane_leader_id()
        return follower_lo

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        if ego_position == 0:
            return -1
        # In theory, we only need to cooperate with the preceding vehicle
        # if it has not yet completed the lane change. But it will make
        # no difference returning the id here independent of that
        return self.vehicles[ego_position - 1].get_id()

    def create_mode_sequence(self, mode_sequence: som.ModeSequence):
        time = mode_sequence.get_latest_switch_time()
        switch_time = time
        last_veh_sv_ids = (
            self.vehicles[-1].get_relevant_surrounding_vehicle_ids())
        # We assume no platoon vehicle changes 'lo'. This promotes cooperation
        # The last platoon veh moves in front of its dest lane follower
        changes = {}
        switch_time += self.switch_dt
        if last_veh_sv_ids['fd'] >= 0:
            veh_id = self.vehicles[-1].get_id()
            changes[last_veh_sv_ids['fd']] = {'lo': veh_id, 'leader': veh_id}
            mode_sequence.alter_and_add_mode(switch_time, changes)
        for i in range(len(self.vehicles) - 2, -1, -1):
            changes = {self.vehicles[i].get_id(): {
                'fd': self.vehicles[i + 1].get_id(),
                'ld': last_veh_sv_ids['ld']
            }}
            switch_time += self.switch_dt
            mode_sequence.alter_and_add_mode(switch_time, changes)


class LeaderFirstReverseStrategy(LaneChangeStrategy):
    def _get_dest_lane_leader_id(self, ego_position) -> int:
        ego_veh = self.vehicles[ego_position]
        if not ego_veh.has_lane_change_intention():
            return -1

        if ego_position == 0:
            return ego_veh.get_dest_lane_leader_id()
        # If the preceding veh has completed the lane change, then we want
        # to merge between the preceding veh and the vehicle ahead of it
        # (which is the preceding veh's current lane leader). Otherwise,
        # we don't have a target vehicle at the destination lane
        preceding = self.vehicles[ego_position - 1]
        if preceding.has_lane_change_intention():
            return -1
        leader_lo = preceding.get_orig_lane_leader_id()
        return leader_lo

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        ego_veh = self.vehicles[ego_position]
        if ego_veh.has_lane_change_intention():
            return -1
        if ego_position == len(self.vehicles) - 1:
            return -1
        follower = self.vehicles[ego_position + 1]
        if follower.has_lane_change_intention():
            return -1
        # Similarly to the previous case, we don't need to check whether
        # the (former) follower has completed the lane change
        return self.vehicles[ego_position + 1].get_id()

    def create_mode_sequence(self, mode_sequence: som.ModeSequence):
        pass


_strategy_map = {
    0: IndividualStrategy, 1: SynchronousStrategy, 2: LeaderFirstStrategy,
    3: LastFirstStrategy, 4: LeaderFirstReverseStrategy
}
