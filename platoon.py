from __future__ import annotations

from abc import ABC, abstractmethod
# import bisect
from enum import Enum
from typing import Dict, List

import numpy as np

import constants
import controllers.optimal_controller as opt_ctrl
import vehicle_models.four_state_vehicles as fsv


class Platoon:
    current_inputs: Dict[int, float]

    _counter: int = 0

    def __init__(self, first_vehicle: fsv.PlatoonVehicle):
        self._id: int = Platoon._counter
        Platoon._counter += 1

        # Vehicles and their ids sorted by position (first is front-most)
        self.vehicles: List[fsv.PlatoonVehicle] = []
        self._id_to_position_map: Dict[int, int] = {}
        self.add_vehicle(first_vehicle)

        self.strategy = _strategy_map[constants.Configuration.platoon_strategy](
            self.vehicles
        )

    def get_platoon_leader(self):
        return self.vehicles[0]

    def get_platoon_leader_id(self):
        return self.get_platoon_leader().get_id()

    def get_platoon_last_vehicle_name(self):
        return self.vehicles[-1].get_name()

    def get_vehicle_ids(self):
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
            opt_control = self.get_optimal_controller()
            new_vehicle.set_centralized_controller(opt_control)
        else:
            # bisect.insort(self.vehicles, new_vehicle, key=lambda v: v.get_x())
            idx = np.searchsorted([veh.get_x() for veh in self.vehicles],
                                  new_vehicle.get_x())
            # Possibly slow, but irrelevant for the total run time
            self.vehicles = (self.vehicles[:idx] + [new_vehicle]
                             + self.vehicles[idx:])
            for i, veh in enumerate(self.vehicles):
                self._id_to_position_map[veh.get_id()] = i
            opt_control = self.get_optimal_controller()
            new_vehicle.set_centralized_controller(opt_control)

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


class LaneChangeStrategy(ABC):

    def __init__(self, platoon_vehicles: List[fsv.PlatoonVehicle]):
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
    def _get_incoming_vehicle_id(self, ego_position) -> int:
        pass

    @abstractmethod
    def _get_dest_lane_leader_id(self, ego_position: int):
        pass


class SynchronousStrategy(LaneChangeStrategy):
    def _get_dest_lane_leader_id(self, ego_position) -> int:
        """
        Defines sequence of leaders during a coordinated lane change maneuver.
        Only effective if platoon vehicles have a closed loop acceleration
        policy, i.e., not optimal control
        :param ego_position:
        :return:
        """
        ego_veh = self.vehicles[ego_position]
        if ego_position == 0:
            return ego_veh.get_dest_lane_leader_id()
        # Strictly speaking, platoon vehicles in this strategy don't
        # need a virtual leader at the dest lane, but setting it here
        # helps them maintain a safe gap in case the leader changes lanes
        # first
        return -1  # self.get_preceding_vehicle_id(ego_id)

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        ego_veh = self.vehicles[ego_position]
        if ego_veh.has_lane_change_intention():
            return -1
        if ego_position == 0:
            return -1
        # Strictly speaking, platoon vehicles in this strategy don't
        # need to cooperate with anyone, but the setting here prevents them
        # from accelerating in case they change lanes before their leaders
        return -1  # self.get_preceding_vehicle_id(ego_id)


class LeaderFirstStrategy(LaneChangeStrategy):

    def _get_dest_lane_leader_id(self, ego_position) -> int:
        """
        Defines sequence of leaders during a coordinated lane change maneuver.
        Only effective if platoon vehicles have a closed loop acceleration
        policy, i.e., not optimal control
        :param ego_position:
        :return:
        """
        ego_veh = self.vehicles[ego_position]
        if not ego_veh.has_lane_change_intention():
            return -1
        if ego_position == 0:
            return ego_veh.get_dest_lane_leader_id()
        return self.vehicles[ego_position - 1].get_id()

    def _get_incoming_vehicle_id(self, ego_position) -> int:
        ego_veh = self.vehicles[ego_position]
        if ego_veh.has_lane_change_intention():
            return -1
        if ego_position == 0:
            return -1
        # Strictly speaking, platoon vehicles in this strategy don't
        # need to cooperate with anyone, but the setting here prevents them
        # from accelerating in case they change lanes before their leaders
        return -1  # self.get_preceding_vehicle_id(ego_id)


class LastFirstStrategy(LaneChangeStrategy):
    def _get_dest_lane_leader_id(self, ego_position) -> int:
        """
        Defines sequence of leaders during a coordinated lane change maneuver.
        Only effective if platoon vehicles have a closed loop acceleration
        policy, i.e., not optimal control
        :param ego_position:
        :return:
        """
        """
        Coding the strategies becomes complicated when we want to control
        when each vehicle increases the desired time headway to its leader.
        """
        ego_veh = self.vehicles[ego_position]
        if not ego_veh.has_lane_change_intention():
            return -1

        if ego_position == len(self.vehicles) - 1:
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
        ego_veh = self.vehicles[ego_position]
        if ego_veh.has_lane_change_intention():
            return -1
        if ego_position == 0:
            return -1
        # In theory, we only need to cooperate with the preceding vehicle
        # if it has not yet completed the lane change. But it will make
        # no difference returning the id here independent of that
        return self.vehicles[ego_position - 1].get_id()


class LeaderFirstReverseStrategy(LaneChangeStrategy):
    def _get_dest_lane_leader_id(self, ego_position) -> int:
        """
        Defines sequence of leaders during a coordinated lane change maneuver.
        Only effective if platoon vehicles have a closed loop acceleration
        policy, i.e., not optimal control
        :param ego_position:
        :return:
        """
        """
        Coding the strategies becomes complicated when we want to control
        when each vehicle increases the desired time headway to its leader.
        """
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


_strategy_map = {1: SynchronousStrategy, 2: LeaderFirstStrategy,
                 3: LastFirstStrategy, 4: LeaderFirstReverseStrategy}
