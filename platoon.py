from __future__ import annotations

from collections.abc import Sequence
# import bisect
import numpy as np

import graph_tools
# import controllers.optimal_controller as opt_ctrl
import platoon_lane_change_strategies as lc_strategy
import vehicle_models.four_state_vehicles as fsv


class Platoon:

    lane_change_strategy: lc_strategy.LaneChangeStrategy
    current_inputs: dict[int, float]
    _counter: int = 0

    def __init__(self):
        self._id: int = Platoon._counter
        Platoon._counter += 1
        self.vehicles: list[fsv.FourStateVehicle] = []
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

    def get_aided_vehicle_id(self, ego_id) -> int:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.get_incoming_vehicle_id(ego_position)

    def set_strategy(self, lane_change_strategy: int):
        self.lane_change_strategy = lc_strategy.strategy_map[
            lane_change_strategy](self.vehicles)

    def set_strategy_parameters(
            self, strategy_parameters: tuple[list[int], list[int]] = None
    ) -> None:
        if strategy_parameters:
            self.lane_change_strategy.set_lane_change_order(
                strategy_parameters[0], strategy_parameters[1])
        else:
            self.lane_change_strategy.set_lane_change_order()

    def set_strategy_states_graph(self,
                                  state_graph: graph_tools.VehicleStatesGraph):
        self.lane_change_strategy.set_states_graph(state_graph)

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

    def get_optimal_controller(self):  # -> opt_ctrl.VehicleOptimalController:
        return self.get_platoon_leader().get_opt_controller()

    def add_vehicle(self, new_vehicle: fsv.OptimalControlVehicle):
        super().add_vehicle(new_vehicle)
        new_vehicle.set_centralized_controller(self.get_optimal_controller())

    # def guess_mode_sequence(self, initial_mode_sequence: som.ModeSequence):
    #     return self.strategy.create_mode_sequence(initial_mode_sequence)
    

class ClosedLoopPlatoon(Platoon):
    def __init__(self, first_vehicle: fsv.ClosedLoopVehicle,
                 lane_change_strategy: int,
                 # lane_changing_order: tuple[list[int], list[int]] = None
                 ):
        super().__init__()

        # Vehicles and their ids sorted by position (first is front-most)
        self.vehicles: list[fsv.ClosedLoopVehicle] = []
        self.add_vehicle(first_vehicle)
        self.set_strategy(lane_change_strategy)
        # if lane_changing_order is not None:
        #     self.lane_change_strategy.set_parameters(lane_changing_order[0],
        #                                              lane_changing_order[1])

    # def get_platoon_leader(self) -> fsv.ClosedLoopVehicle:
    #     return self.vehicles[0]

    def add_vehicle(self, new_vehicle: fsv.ClosedLoopVehicle):
        super().add_vehicle(new_vehicle)

    def set_maneuver_initial_state(
            self, ego_id: int, lo_states: Sequence[float],
            ld_states: Sequence[float]):
        ego_position = self._id_to_position_map[ego_id]
        # We center all around the leader TODO: avoid hard coding array indices
        leader_x = self.get_platoon_leader().get_x()
        leader_y = self.get_platoon_leader().get_y()

        platoon_states = []
        for veh in self.vehicles:
            veh_states = veh.get_states()
            veh_states[0] -= leader_x
            veh_states[1] -= leader_y
            platoon_states.extend(veh_states)
        if lo_states is not None:
            lo_states[0] -= leader_x
            lo_states[1] -= leader_y
        if ld_states is not None:
            ld_states[0] -= leader_x
            ld_states[1] -= leader_y
        self.lane_change_strategy.set_maneuver_initial_state(
            ego_position, lo_states, platoon_states, ld_states)

    def can_start_lane_change(self, ego_id: int) -> bool:
        ego_position = self._id_to_position_map[ego_id]
        return self.lane_change_strategy.can_start_lane_change(ego_position)
