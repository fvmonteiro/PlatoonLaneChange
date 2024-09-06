from __future__ import annotations

from collections.abc import Mapping, Sequence
# import bisect
import numpy as np

import configuration
import platoon_functionalities.platoon_lane_change_strategies as lc_strategies
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
        self._dest_lane_leader_id = -1  # the non-platoon vehicle behind which
        # the entire platoon move

    def get_id(self) -> int:
        return self._id

    def get_size(self) -> int:
        """
        Gets the number of vehicles
        :return:
        """
        return len(self.vehicles)

    def get_vehicle_by_position(self, pos: int) -> fsv.FourStateVehicle:
        """
        Gets the vehicle at the given position in the platoon. The vehicle
        position in the platoon is set at creation time. The vehicle order in
        the platoon's vehicle list is not updated even if vehicles change
        order during the lane change.
        :param pos:
        :return:
        """
        return self.vehicles[pos]

    def get_platoon_leader(self) -> fsv.FourStateVehicle:
        """
        Gets the front-most vehicle in the platoon. This method should not be
        used during a lane change maneuver in which platoon vehicles may change
        order
        :return:
        """
        return self.get_vehicle_by_position(0)

    def get_platoon_leader_id(self) -> int:
        return self.get_platoon_leader().id

    def get_last_platoon_vehicle(self) -> fsv.FourStateVehicle:
        """
        Gets the vehicle at the end of the platoon. This method should not be
        used during a lane change maneuver in which platoon vehicles may change
        order
        :return:
        """
        return self.get_vehicle_by_position(-1)

    def get_vehicle_ids(self) -> list[int]:
        return [veh.id for veh in self.vehicles]

    def get_centered_vehicles_states(self) -> dict[str, np.ndarray]:
        """
        Returns the vehicles states centered around the platoon leader
        :return:
        """
        p1 = self.get_platoon_leader()
        leader_x = p1.get_x()
        leader_y = p1.get_y()
        x_idx = fsv.FourStateVehicle.get_idx_of_state("x")
        y_idx = fsv.FourStateVehicle.get_idx_of_state("y")
        states = dict()
        for veh in self.vehicles:
            veh_states = veh.get_states().copy()
            veh_states[x_idx] -= leader_x
            veh_states[y_idx] -= leader_y
            states[veh.name] = veh_states
        return states

    def get_preceding_vehicle_id(self, veh_id: int) -> int:
        preceding_position = self._id_to_position_map[veh_id] - 1
        if preceding_position >= 0:
            return self.vehicles[preceding_position].id
        else:
            return -1

    def get_following_vehicle_id(self, veh_id: int) -> int:
        following_position = self._id_to_position_map[veh_id] + 1
        if following_position < len(self.vehicles):
            return self.vehicles[following_position].id
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
        veh_dest_lane_leader_id = (
            self.lane_change_strategy.get_desired_dest_lane_leader_id(
                ego_position)
        )
        if self._dest_lane_leader_id < 0:
            self._dest_lane_leader_id = veh_dest_lane_leader_id
        return veh_dest_lane_leader_id

    def get_aided_vehicle_id(self, ego_id) -> int:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.get_incoming_vehicle_id(ego_position)

    def get_platoon_desired_dest_lane_leader_id(self) -> int:
        return self._dest_lane_leader_id
        # return self.lane_change_strategy.get_platoon_destination_lane_leader()

        # for veh in self.vehicles:
        #     if veh.has_lane_change_intention():
        #         desired_ld_id = self.get_vehicle_desired_dest_lane_leader_id(
        #                 veh.id)
        #         if desired_ld_id > -1:
        #             return desired_ld_id
        # print('Note: no platoon vehicle has a desired dest lane leader')
        # return -1

    def get_strategy(self) -> lc_strategies.LaneChangeStrategy:
        return self.lane_change_strategy

    def get_system_states(
            self, veh_pos: int, all_vehicles: Mapping[int, base.BaseVehicle]
    ) -> dict[str, np.ndarray]:
        """
        Creates system states around the vehicle at veh_pos
        :return:
        """
        states = self.get_centered_vehicles_states()
        platoon_leader = self.get_platoon_leader()
        leader_x = platoon_leader.get_x()
        leader_y = platoon_leader.get_y()
        x_idx = fsv.FourStateVehicle.get_idx_of_state("x")
        y_idx = fsv.FourStateVehicle.get_idx_of_state("y")
        if platoon_leader.has_origin_lane_leader():
            lo = all_vehicles[platoon_leader.get_origin_lane_leader_id()]
            lo_states = lo.get_states()
        else:
            lo_states = platoon_leader.get_states()
            lo_states[x_idx] += configuration.MAX_DISTANCE
        lo_states[x_idx] -= leader_x
        lo_states[y_idx] -= leader_y
        states["lo"] = lo_states

        veh = self.get_vehicle_by_position(veh_pos)
        if veh.has_destination_lane_leader():
            ld = all_vehicles[veh.get_destination_lane_leader_id()]
            ld_states = ld.get_states()
        else:
            ld_states = platoon_leader.get_states()
            ld_states[x_idx] += configuration.MAX_DISTANCE
            ld_states[y_idx] = platoon_leader.get_target_y()
        ld_states[x_idx] -= leader_x
        ld_states[y_idx] -= leader_y
        states["ld"] = ld_states
        return states

    def set_lc_start_time(self, time: float) -> None:
        self._lc_start_time = time

    def set_strategy(self, lane_change_strategy: lc_strategies.StrategyMap
                     ) -> None:
        self.lane_change_strategy = lane_change_strategy.get_implementation()(
            self)

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
            self._id_to_position_map[new_vehicle.id] = len(self.vehicles)
            self.vehicles.append(new_vehicle)
        else:
            # bisect.insort(self.vehicles, new_vehicle, key=lambda v: v.get_x())
            idx = np.searchsorted([veh.get_x() for veh in self.vehicles],
                                  new_vehicle.get_x())
            # Possibly slow, but irrelevant for the total run time
            self.vehicles = (self.vehicles[:idx] + [new_vehicle]
                             + self.vehicles[idx:])
            for i, veh in enumerate(self.vehicles):
                self._id_to_position_map[veh.id] = i

    def append_vehicle(self, new_vehicle: fsv.FourStateVehicle) -> None:
        """
        Adds the vehicle as the last platoon vehicle, even if it is not
        longitudinally behind the rear most platoon vehicle
        :param new_vehicle:
        :return:
        """
        self._id_to_position_map[new_vehicle.id] = len(self.vehicles)
        self.vehicles.append(new_vehicle)

    def has_lane_change_started(self) -> bool:
        return self._lc_start_time < np.inf

    def contains_vehicle(self, veh_id: int):
        return veh_id in self._id_to_position_map


class OptimalPlatoon(Platoon):

    def __init__(self, first_vehicle: fsv.OptimalControlVehicle,
                 lane_change_strategy: lc_strategies.StrategyMap):
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
    

class ClosedLoopPlatoon(Platoon):
    def __init__(self, first_vehicle: fsv.ClosedLoopVehicle,
                 lane_change_strategy: lc_strategies.StrategyMap,
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

    def can_start_lane_change(self, ego_id: int,
                              vehicles: Mapping[int, base.BaseVehicle]) -> bool:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.can_start_lane_change(
            ego_position, vehicles)
