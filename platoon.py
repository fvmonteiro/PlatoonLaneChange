from __future__ import annotations

from collections.abc import Mapping, Sequence
# import bisect
import numpy as np

import configuration
import platoon_lane_change_strategies as lc_strategies
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


class Platoon:

    lane_change_strategy: lc_strategies.LaneChangeStrategy
    current_inputs: dict[int, float]
    _counter: int = 0

    def __init__(self):
        self._id: int = Platoon._counter
        Platoon._counter += 1
        self.vehicles: list[fsv.FourStateVehicle] = []
        self._id_to_position_map: dict[int, int] = {}
        self._lc_start_time = np.inf

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

    def get_vehicle_desired_dest_lane_leader_id(self, ego_id) -> int:
        """
        Defines the desired destination lane leader for vehicle ego_id based
        on the platoon lane change strategy.
        :param ego_id:
        :return:
        """
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.get_desired_dest_lane_leader_id(
            ego_position)

    def get_aided_vehicle_id(self, ego_id) -> int:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.get_incoming_vehicle_id(ego_position)

    def get_platoon_desired_dest_lane_leader_id(self) -> int:
        # TODO: looks wrong... We want the non-platoon dest lane leader and
        #  this may return a platoon vehicle.
        for veh in self.vehicles:
            if veh.get_desired_destination_lane_leader_id() > -1:
                return veh.get_desired_destination_lane_leader_id()
        print('Note: no platoon vehicle has a desired dest lane leader')
        return -1

    def get_strategy(self) -> lc_strategies.LaneChangeStrategy:
        return self.lane_change_strategy

    def set_lc_start_time(self, time: float) -> None:
        self._lc_start_time = time

    def set_strategy(self, lane_change_strategy: int) -> None:
        self.lane_change_strategy = lc_strategies.strategy_map[
            lane_change_strategy](self.vehicles)

    def set_lane_change_parameters(self) -> None:
        self.lane_change_strategy.set_parameters()

    def set_lane_change_order(
            self, strategy_parameters: configuration.Strategy
    ) -> None:
        self.lane_change_strategy.set_maneuver_order(
            strategy_parameters[0], strategy_parameters[1])

    def add_vehicle(self, new_vehicle: fsv.FourStateVehicle) -> None:
        """
        Adds the vehicle to the platoon. The new vehicle does not have to
        be behind all platoon vehicles. The method checks the longitudinal
        positions and inserts the vehicle properly.
        :param new_vehicle: Vehicle being added to the platoon
        :return:
        """
        if (len(self.vehicles) == 0
                or new_vehicle.get_x() < self.vehicles[-1].get_x()):
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

    def append_vehicle(self, new_vehicle: fsv.FourStateVehicle) -> None:
        """
        Adds the vehicle as the last platoon vehicle, even if it is not
        longitudinally behind the rear most platoon vehicle
        :param new_vehicle:
        :return:
        """
        self._id_to_position_map[new_vehicle.get_id()] = len(self.vehicles)
        self.vehicles.append(new_vehicle)

    def has_lane_change_started(self) -> bool:
        return self._lc_start_time < np.inf


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

    def get_optimal_controller(self):  # -> opt_ctrl.VehicleOptimalController:
        return self.get_platoon_leader().get_opt_controller()

    def add_vehicle(self, new_vehicle: fsv.OptimalControlVehicle) -> None:
        super().add_vehicle(new_vehicle)
        new_vehicle.set_centralized_controller(self.get_optimal_controller())

    # def guess_mode_sequence(self, initial_mode_sequence: som.ModeSequence):
    #     return self.strategy.create_mode_sequence(initial_mode_sequence)
    

class ClosedLoopPlatoon(Platoon):
    def __init__(self, first_vehicle: fsv.ClosedLoopVehicle,
                 lane_change_strategy: int,
                 ):
        super().__init__()

        # Vehicles and their ids sorted by position (first is front-most)
        self.vehicles: list[fsv.ClosedLoopVehicle] = []
        self.add_vehicle(first_vehicle)
        self.set_strategy(lane_change_strategy)
        # if lane_changing_order is not None:

    # def get_platoon_leader(self) -> fsv.ClosedLoopVehicle:
    #     return self.vehicles[0]

    def add_vehicle(self, new_vehicle: fsv.ClosedLoopVehicle):
        super().add_vehicle(new_vehicle)

    def set_maneuver_initial_state_for_all_vehicles(
            self, all_vehicles: Mapping[int, base.BaseVehicle]) -> None:

        if self.has_lane_change_started():
            return

        platoon_leader = self.get_platoon_leader()
        if platoon_leader.has_origin_lane_leader():
            lo = all_vehicles[platoon_leader.get_origin_lane_leader_id()]
            lo_states = lo.get_states()
        else:
            lo_states = []

        for veh in self.vehicles:
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
                self.set_maneuver_initial_state(veh.get_id(), lo_states,
                                                ld_states, fd_states)
            else:
                self.lane_change_strategy.set_empty_maneuver_initial_state(
                    veh.get_id())

    def set_maneuver_initial_state(
            self, ego_id: int, lo_states: Sequence[float],
            ld_states: Sequence[float], fd_states: Sequence[float]) -> None:
        # TODO: avoid hard coding array indices

        p1 = self.get_platoon_leader()
        # TODO: lazy workaround. We need to include the no leader
        #  possibilities in the graph
        if len(lo_states) == 0:
            lo_states = p1.get_states().copy()
            lo_states[0] += p1.compute_lane_keeping_desired_gap()
        else:
            lo_states = np.copy(lo_states)
        if len(ld_states) == 0:
            ld_states = lo_states.copy()
            ld_states[1] = p1.get_target_y()
        else:
            ld_states = np.copy(ld_states)
        if len(fd_states) == 0:
            pN = self.get_last_platoon_vehicle()
            fd_states = pN.get_states().copy()
            fd_states[0] -= pN.compute_safe_lane_change_gap()
            fd_states[1] = pN.get_target_y()
        else:
            fd_states = np.copy(fd_states)

        # We center all around the leader
        leader_x = p1.get_x()
        leader_y = p1.get_y()

        platoon_states = []
        for veh in self.vehicles:
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

        ego_position = self._id_to_position_map[ego_id]
        self.lane_change_strategy.set_maneuver_initial_state(
            ego_position, lo_states, platoon_states, ld_states, fd_states)

    def can_start_lane_change(self, ego_id: int) -> bool:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.can_start_lane_change(ego_position)
