from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar, Union

import numpy as np
import pandas as pd

import controllers.optimal_controller as opt_ctrl
import helper
import platoon_functionalities.platoon_lane_change_strategies as lc_strategies
from platoon_functionalities import vehicle_platoon
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv
import operating_modes.system_operating_mode as som

V = TypeVar('V', bound=base.BaseVehicle)


class CollisionException(RuntimeError):
    pass


class VehicleGroup:
    """ Class to help manage groups of vehicles """

    _front_most_dest_lane_vehicle: base.BaseVehicle
    _rear_most_dest_lane_vehicle: base.BaseVehicle
    _front_most_lc_vehicle: base.BaseVehicle
    _rear_most_lc_vehicle: base.BaseVehicle

    forced_state: Mapping[str, np.ndarray]

    _platoon_lane_change_strategy: lc_strategies.StrategyMap

    def __init__(self):
        self.vehicles: dict[int, base.BaseVehicle] = {}
        # Often, we need to iterate over all vehicles in the order they were
        # created. The list below makes that easy
        self.sorted_vehicle_ids: list[int] = []
        self.lane_changing_vehicle_ids: list[int] = []
        self.name_to_id: dict[str, int] = {}
        # The full system (all vehicles) mode is defined by follower/leader
        # pairs.
        self.mode_sequence: som.ModeSequence = som.ModeSequence()
        # self._platoon_lane_change_strategy = None
        # self._vehicle_states_graph = None
        # self._strategy_map = None
        self._maneuver_order = None
        self._is_verbose = True
        self._ids_must_change_lanes = []

    def get_n_vehicles(self) -> int:
        return len(self.vehicles)

    def get_n_states(self) -> int:
        # time is the last system state
        return sum(veh.get_n_states() for veh in self.vehicles.values()) + 1

    def get_n_inputs(self) -> int:
        return sum(veh.get_n_inputs() for veh in self.vehicles.values())

    def get_current_mode(self) -> som.SystemMode:
        try:
            return self.mode_sequence.get_latest_mode()
        except IndexError:
            return som.SystemMode({})

    def get_all_vehicles(self) -> Iterable[base.BaseVehicle]:
        return self.vehicles.values()

    def get_all_vehicles_in_order(self) -> list[base.BaseVehicle]:
        return [self.vehicles[veh_id] for veh_id in self.sorted_vehicle_ids]

    def get_free_flow_speeds(self) -> np.ndarray:
        v_ff = np.zeros(self.get_n_vehicles())
        for veh_id in self.sorted_vehicle_ids:
            v_ff[veh_id] = self.vehicles[veh_id].free_flow_speed
        return v_ff

    def yield_vehicles_in_order(self) -> Iterable[base.BaseVehicle]:
        for veh_id in self.sorted_vehicle_ids:
            yield self.vehicles[veh_id]

    def get_full_initial_state_vector(self) -> np.ndarray:
        initial_state = []
        for veh_id in self.sorted_vehicle_ids:
            initial_state.extend(self.vehicles[veh_id].get_initial_state())
        return np.array(initial_state)

    def get_initial_state_by_vehicle(self) -> dict[str, np.ndarray]:
        initial_state = dict()
        for veh in self.vehicles.values():
            initial_state[veh.name] = veh.get_initial_state()
        return initial_state

    def get_state(self) -> np.ndarray:
        states = []
        for veh_id in self.sorted_vehicle_ids:
            states.append(self.vehicles[veh_id].get_states())
        return np.hstack(states)

    def get_state_by_vehicle(self) -> dict[str, np.ndarray]:
        initial_state = dict()
        for veh in self.vehicles.values():
            initial_state[veh.name] = veh.get_states()
        return initial_state

    def get_current_time(self) -> float:
        return self.vehicles[self.sorted_vehicle_ids[0]].get_current_time()

    def get_simulated_time(self) -> np.ndarray:
        return self.vehicles[self.sorted_vehicle_ids[0]].get_simulated_time()

    def get_all_states(self) -> np.ndarray:
        states = []
        for veh_id in self.sorted_vehicle_ids:
            states.append(self.vehicles[veh_id].get_state_history())
            if veh_id == self.sorted_vehicle_ids[-1]:
                states.append(self.vehicles[veh_id].get_simulated_time())
        return np.vstack(states)

    def get_current_inputs(self) -> np.ndarray:
        inputs = []
        for veh_id in self.sorted_vehicle_ids:
            inputs.append(self.vehicles[veh_id].get_inputs())
        return np.hstack(inputs)

    def get_all_inputs(self, selected_vehicles_ids: Iterable[int] = None
                       ) -> np.ndarray:
        inputs = []
        for veh_id in self.sorted_vehicle_ids:
            if (selected_vehicles_ids is None
                    or veh_id in selected_vehicles_ids):
                inputs.append(self.vehicles[veh_id].get_input_history())
        return np.vstack(inputs)

    def get_selected_inputs(
            self, vehicle_inputs_map: Mapping[int, Iterable[str]] = None
    ) -> np.ndarray:
        inputs = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            # input_history = vehicle.get_input_history()
            if vehicle_inputs_map is None:
                inputs.append(vehicle.get_input_history())
            elif veh_id in vehicle_inputs_map:
                for input_name in vehicle_inputs_map[veh_id]:
                    inputs.append(vehicle.get_an_input_history(input_name))
                    # inputs.append(input_history[
                    #                   vehicle.get_idx_of_input(input_name)])
        return np.vstack(inputs)

    def get_mode_sequence(self) -> som.ModeSequence:
        return self.mode_sequence

    def get_vehicle_by_id(self, veh_id: int) -> base.BaseVehicle:
        return self.vehicles[veh_id]

    def get_vehicle_id_by_name(self, name: str) -> int:
        """
        Returns the id of the vehicle with given name. Returns -1 if the name
        is not found.
        """
        return self.name_to_id.get(name, -1)

    def get_vehicle_by_name(self, name: str) -> base.BaseVehicle:
        return self.vehicles[self.name_to_id[name]]

    def get_platoon_vehicle_by_position(self, pos: int) -> base.BaseVehicle:
        """
        Gets the vehicle at the given position in the platoon
        :param pos:
        :return:
        """
        cl_vehicles = self.get_closed_loop_control_vehicles()

        for veh in cl_vehicles:
            if veh.is_in_a_platoon():
                return veh.get_platoon().get_vehicle_by_position(pos)
        raise RuntimeError("There are not platoons in this vehicle group.")

    def get_optimal_control_vehicles(self) -> list[fsv.OptimalControlVehicle]:
        return self.get_vehicles_of_type(fsv.OptimalControlVehicle)

    def get_closed_loop_control_vehicles(self) -> list[fsv.ClosedLoopVehicle]:
        return self.get_vehicles_of_type(fsv.ClosedLoopVehicle)

    def get_vehicles_of_type(self, vehicle_type: type[base.BaseVehicle]
                             ) -> list[V]:
        selected_vehicles: list[vehicle_type] = []
        for veh_id in self.sorted_vehicle_ids:
            veh = self.vehicles[veh_id]
            if isinstance(veh, vehicle_type):
                selected_vehicles.append(veh)
        return selected_vehicles

    def get_lane_changing_vehicle_ids(self) -> list[int]:
        return self.lane_changing_vehicle_ids

    def get_platoon_lane_change_strategy(
            self) -> Union[lc_strategies.LaneChangeStrategy, None]:
        # TODO: messy. Probably should create a vehicle group class that only
        #  has four state vehicles
        cl_vehicles = self.get_closed_loop_control_vehicles()

        for veh in cl_vehicles:
            if veh.is_in_a_platoon():
                return veh.get_platoon().get_strategy()
        return None

    def get_lc_end_times(self) -> list[float]:
        final_times = []
        for veh_id in self.sorted_vehicle_ids:
            final_times.append(self.vehicles[veh_id].get_lc_end_time())
        return final_times

    def get_decision_time(self) -> float:
        for veh in self.vehicles.values():
            if veh.is_in_a_platoon():
                return veh.get_platoon_strategy_decision_time()

    def set_platoon_lane_change_strategy(
            self, strategy: lc_strategies.StrategyMap) -> None:
        self._platoon_lane_change_strategy = strategy

    def set_predefined_lane_change_order(
            self, lane_change_order: list[set[int]],
            cooperation_order: list[int]) -> None:
        self._maneuver_order = (lane_change_order, cooperation_order)

    def set_platoon_lane_change_order(
            self, lane_change_order: list[Union[set[int], frozenset[int]]],
            cooperation_order: list[int]) -> None:
        p1 = self.get_vehicle_by_name("p1")
        p1.set_platoon_lane_change_order((lane_change_order, cooperation_order))

    def set_verbose(self, value: bool) -> None:
        self._is_verbose = value
        for vehicle in self.vehicles.values():
            vehicle.set_verbose(value)

    def set_a_vehicle_free_flow_speed(self, veh_id, v_ff) -> None:
        self.vehicles[veh_id].set_free_flow_speed(v_ff)

    def set_free_flow_speeds(self, values: Union[float, Sequence[float]]
                             ) -> None:
        if np.isscalar(values):
            values = [values] * self.get_n_vehicles()
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.set_free_flow_speed(values[veh_id])

    def set_free_flow_speeds_by_name(self, values: Mapping[str, float]) -> None:
        for vehicle in self.vehicles.values():
            vehicle.set_free_flow_speed(values[vehicle.name])

    def set_vehicles_initial_states(
            self, x0: Sequence[float], y0: Sequence[float],
            theta0: Sequence[float], v0: Sequence[float]) -> None:
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.set_initial_state(x0[veh_id], y0[veh_id],
                                      theta0[veh_id], v0[veh_id])

    def set_vehicles_initial_states_from_array(
            self, full_state: np.ndarray) -> None:
        start = 0
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            end = start + vehicle.get_n_states()
            vehicle.set_initial_state(full_state=full_state[start:end])
            start = end

    def set_vehicles_lane_change_direction(
            self, lc_direction: Union[int, Sequence[int]],
            ids_or_names: Sequence[Union[int, str]] = None) -> None:
        if ids_or_names is None or len(ids_or_names) == 0:
            ids_or_names = [veh_id for veh_id, veh in self.vehicles.items()
                            if veh.name[0] == 'p']
        if np.isscalar(lc_direction):
            lc_direction = [lc_direction] * len(ids_or_names)
        for i in range(len(ids_or_names)):
            self.set_single_vehicle_lane_change_direction(ids_or_names[i],
                                                          lc_direction[i])

    def set_single_vehicle_lane_change_direction(
            self, veh_id_or_name: Union[int, str], lc_direction: int) -> None:
        if isinstance(veh_id_or_name, str):
            veh_id = self.name_to_id[veh_id_or_name]
        else:
            veh_id = veh_id_or_name
        self.vehicles[veh_id].set_lane_change_direction(lc_direction)

    def set_vehicle_names(self, names: Sequence[str]) -> None:
        for i in range(len(self.sorted_vehicle_ids)):
            veh_id = self.sorted_vehicle_ids[i]
            vehicle = self.vehicles[veh_id]
            vehicle.set_name(names[i])
            self.name_to_id[names[i]] = veh_id

    def has_vehicle_with_name(self, veh_name: str) -> bool:
        return veh_name in self.name_to_id

    def make_all_connected(self) -> None:
        for vehicle in self.vehicles.values():
            vehicle.make_connected()

    def force_state(self, states_by_vehicle: Mapping[str, np.ndarray]) -> None:
        self.forced_state = states_by_vehicle
        for veh_name, state in states_by_vehicle.items():
            self.get_vehicle_by_name(veh_name).force_state(state)
        # for veh_id, veh in self.vehicles.items():
        #     veh.force_state(states_by_vehicle[veh.get_name()])

    def map_values_to_names(self, values) -> dict[str, Any]:
        """
        Receives variables ordered in the same order as the vehicles were
        created and returns the variables in a dictionary with vehicle names as
        keys
        """
        d = {}
        for veh_id in self.sorted_vehicle_ids:
            d[self.vehicles[veh_id].name] = values[veh_id]
        return d

    def prepare_to_start_simulation(
            self, n_samples: int,
            platoon_lc_strategy: lc_strategies.StrategyMap = None
    ) -> None:
        """
        Sets all internal states, inputs and other simulation-related variables
        to zero.
        """
        self.mode_sequence = som.ModeSequence()
        for vehicle in self.vehicles.values():
            vehicle.prepare_to_start_simulation(n_samples)
        self.initialize_platoons(platoon_lc_strategy)

        dest_lane_vehicles = [veh for veh in self.vehicles.values()
                              if veh.get_current_lane() > 0]
        self._front_most_dest_lane_vehicle = max(dest_lane_vehicles,
                                                 key=lambda x: x.get_x())
        self._rear_most_dest_lane_vehicle = min(dest_lane_vehicles,
                                                key=lambda x: x.get_x())
        lane_changing_vehicles = [self.vehicles[veh_id] for veh_id
                                  in self.lane_changing_vehicle_ids]
        self._front_most_lc_vehicle = max(lane_changing_vehicles,
                                          key=lambda x: x.get_x())
        self._rear_most_lc_vehicle = min(lane_changing_vehicles,
                                         key=lambda x: x.get_x())

    def fill_vehicle_array(self, vehicles: Iterable[base.BaseVehicle]) -> None:
        if self.get_n_vehicles() > 0:
            raise AttributeError("Trying to create a vehicle array in a "
                                 "vehicle group that was already initialized")
        self.vehicles = {}
        self.sorted_vehicle_ids = []
        for veh in vehicles:
            self.add_vehicle(veh)

    def add_vehicle(self, new_vehicle: base.BaseVehicle) -> None:
        veh_id = new_vehicle.id
        self.sorted_vehicle_ids.append(veh_id)
        self.vehicles[veh_id] = new_vehicle
        self.name_to_id[new_vehicle.name] = veh_id
        if new_vehicle.can_change_lanes:
            self.lane_changing_vehicle_ids.append(veh_id)

    def populate_with_open_loop_copies(
            self, vehicles: Mapping[int, base.BaseVehicle],
            controlled_vehicle_ids: set[int],
            initial_state_per_vehicle: Mapping[int, np.ndarray] = None
    ) -> None:
        self.populate_with_copies(vehicles, controlled_vehicle_ids, True,
                                  initial_state_per_vehicle)

    def populate_with_closed_loop_copies(
            self, vehicles: Mapping[int, base.BaseVehicle],
            controlled_vehicle_ids: set[int],
            initial_state_per_vehicle: Mapping[int, np.ndarray] = None
    ) -> None:
        self.populate_with_copies(vehicles, controlled_vehicle_ids, False,
                                  initial_state_per_vehicle)

    def populate_with_copies(
            self, vehicles: Mapping[int, base.BaseVehicle],
            controlled_vehicle_ids: set[int], are_copies_open_loop: bool,
            initial_state_per_vehicle: Mapping[int, np.ndarray] = None
    ) -> None:
        """
        Creates copies of existing vehicles and groups them in this instance.
        This is useful for simulations that happen inside the optimal
        controller
        :param vehicles: All vehicles in the simulation
        :param controlled_vehicle_ids: Ids of vehicles being controlled by the
         optimal controller running the simulation
        :param are_copies_open_loop: If true, we need to provide open loop
         controls to the controlled vehicles. If false, the controlled vehicles
         adopt feedback control laws
        :param initial_state_per_vehicle: Mapping from vehicle id to initial
         state
        :return:
        """
        # if self.get_n_vehicles() > 0:
        #     raise AttributeError("Cannot set vehicles to a vehicle group "
        #                          "that was already initialized")
        copied_vehicles = []
        for veh_id in sorted(vehicles.keys()):
            if initial_state_per_vehicle:
                initial_state = initial_state_per_vehicle[veh_id]
            else:
                initial_state = None

            if veh_id not in controlled_vehicle_ids:
                vehicle = vehicles[veh_id].make_reset_copy(initial_state)
            elif are_copies_open_loop:
                vehicle = vehicles[veh_id].make_open_loop_copy(
                    initial_state)
            else:
                vehicle = vehicles[veh_id].make_closed_loop_copy(
                    initial_state)
            copied_vehicles.append(vehicle)
            # self.add_vehicle(vehicle)
        self.fill_vehicle_array(copied_vehicles)

    def create_full_state_vector(self, x, y, theta, v=None) -> np.ndarray:
        """
        Creates a single state vector.

        :param x: Longitudinal position of each vehicle
        :param y: Lateral position of each vehicle
        :param theta: Orientation of each vehicle
        :param v: Initial speed of each vehicle. Only used if speed is one of
         the model states
        :return: The array with the states of all vehicles
        """

        full_state = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            full_state.extend(vehicle.create_state_vector(
                x[veh_id], y[veh_id], theta[veh_id], v[veh_id]))
        return np.array(full_state)

    def reset_platoons(self) -> None:
        for veh in self.vehicles.values():
            veh.reset_platoon()

    # TODO [Apr 5]: check if this function needs a default param or if we
    #  only call it when there is a strategy
    def initialize_platoons(
            self, platoon_lc_strategy: lc_strategies.StrategyMap = None
    ) -> None:
        """
        Based on vehicles parameters and initial position, create platoons
        of vehicles
        :return:
        """
        # First, we must check which vehicles are physically close to each other
        self.update_surrounding_vehicles()

        # Then, we run their internal algorithms to form platoons
        for veh_id, veh in self.vehicles.items():
            veh.initialize_platoons(
                self.vehicles, platoon_lc_strategy)

        # TODO: poor organization. The platoon strategy only has to be set once
        #  per platoon. The code is also confusing cause self._maneuver_order
        #  is sometimes None (by design)
        # Strategy parameters depend on the number of vehicles in the
        # platoon. So, we can only set them after forming the platoons
        for veh in self.vehicles.values():
            if self._maneuver_order is not None:
                veh.set_platoon_lane_change_order(self._maneuver_order)
            # else:
            #     veh.set_platoon_lane_change_parameters()

    def simulate_one_time_step(
            self, new_time: float,
            open_loop_controls: Mapping[int, np.ndarray] = None,
            detect_collision: bool = True
    ) -> None:
        """
        Advances the simulation by one time step
        :param new_time:
        :param open_loop_controls:
        :param detect_collision: If true, raises an error when there is a
         collision. Otherwise, collisions happen silently.
        :param must_change_lanes: List of vehicle that must start longitudinal
         adjustments for lane change at the beginning of the simulation
         independent of safety criteria. This should only be used when expanding
         a root node.
        """
        if open_loop_controls is None:
            open_loop_controls = {}

        # The separate for loops ensure that all vehicles use information from
        # the same time step

        # We check which vehicles are physically close to each other
        # and, given that, each vehicle decides who's its target leader and
        # whether it is safe to perform a lane change
        self.update_surrounding_vehicles(detect_collision)
        for veh in self.vehicles.values():
            veh.update_virtual_leader(self.vehicles)
            if veh.has_lane_change_intention():
                veh.check_surrounding_gaps_safety(self.vehicles)
                if veh.id in self._ids_must_change_lanes:
                    veh._is_lane_change_gap_suitable = True

        # Then, we check the new system mode
        new_mode = som.SystemMode(self.vehicles)
        if self.get_current_mode() != new_mode:
            time = self.vehicles[0].get_current_time()
            if self._is_verbose and not self.mode_sequence.is_empty():
                print("t={:.2f}. Mode update\nold: {}\nnew: {}".format(
                    time, self.get_current_mode(), new_mode))
            self.mode_sequence.add_mode(time, new_mode)

        # Next, the dynamics and controls are computed
        for veh_id, veh in self.vehicles.items():
            # veh.update_platoons(
            #     self.vehicles, self._platoon_lane_change_strategy)
            veh.update_mode(self.vehicles)
            veh.determine_inputs(open_loop_controls.get(veh_id, []),
                                 self.vehicles)
            veh.compute_derivatives()
            veh.update_states(new_time)

        # Last, we update the internal iteration counters
        for veh in self.vehicles.values():
            veh.update_iteration_counter()

    def is_any_vehicle_maneuvering(self) -> bool:
        """
        Returns true if any platoon vehicle has lane intention or non-zero
        acceleration
        """
        for veh_id in self.lane_changing_vehicle_ids:
            veh = self.vehicles[veh_id]
            if (veh.has_lane_change_intention()
                    or not veh.is_at_lane_center()
                    or np.abs(veh.get_an_input_by_name('a') > 1.0e-10)):
                return True
        return False

    def is_platoon_out_of_range(self) -> bool:
        if (self._rear_most_lc_vehicle.get_x()
                >= self._front_most_dest_lane_vehicle.get_x()
            or self._front_most_lc_vehicle.get_x()
                <= self._rear_most_dest_lane_vehicle.get_x()):
            return True

        # Not necessary form here one, but helps terminate earlier
        if (self._platoon_lane_change_strategy in
                [lc_strategies.StrategyMap.last_vehicle_first,
                 lc_strategies.StrategyMap.single_body_platoon]):
            if (self._rear_most_lc_vehicle.get_x()
                    < self._rear_most_dest_lane_vehicle.get_x()):
                return True
        if (self._platoon_lane_change_strategy in
                [lc_strategies.StrategyMap.leader_first_and_reverse,
                 lc_strategies.StrategyMap.single_body_platoon]):
            if (self._front_most_lc_vehicle.get_x()
                    > self._front_most_dest_lane_vehicle.get_x()):
                return True

    def write_vehicle_states(self, time,
                             state_vectors: Mapping[int, np.ndarray],
                             optimal_inputs: Mapping[int, np.ndarray]) -> None:
        """
        Directly sets vehicle states and inputs when they were computed
        by the optimal control solver.
        :param time:
        :param state_vectors:
        :param optimal_inputs:
        :return:
        """
        for veh_id in self.sorted_vehicle_ids:
            self.vehicles[veh_id].write_state_and_input(
                time, state_vectors[veh_id], optimal_inputs[veh_id])

    def update_surrounding_vehicles(self, detect_collision: bool = False
                                    ) -> None:
        for ego_vehicle in self.vehicles.values():
            ego_vehicle.update_surrounding_vehicles(self.vehicles)
            if detect_collision and ego_vehicle.detect_collision():
                # TODO: raise error, warning?
                veh1 = ego_vehicle.name
                veh2 = self.vehicles[ego_vehicle.get_origin_lane_leader_id()
                                     ].name
                print(f' ==== COLLISION DETECTED ====\n'
                      f't={self.get_current_time():.2f} between vehicles '
                      f'{veh1} and {veh2}')
                # raise CollisionException

    def update_states(self, new_time) -> None:
        for vehicle in self.vehicles.values():
            vehicle.update_states(new_time)

    def check_lane_change_success(self) -> bool:
        for vehicle in self.vehicles.values():
            if vehicle.has_lane_change_intention():
                if self._is_verbose:
                    print(
                        f'Vehicle {vehicle.name} did not finish the lane '
                        f'change.\n(target lane: {vehicle.get_target_lane()}, '
                        f'current lane: {vehicle.get_current_lane()}.)')
                return False
        return True

    def are_all_at_target_lane_center(self) -> bool:
        # for veh_id in self.lane_changing_vehicle_ids:
        #     if not self.vehicles[veh_id].is_at_lane_center():
        #         return False
        # return True
        return np.all([self.vehicles[veh_id].is_at_lane_center()
                      for veh_id in self.lane_changing_vehicle_ids])

    def truncate_simulation_history(self) -> None:
        """
        Truncates the matrices with state and inputs history so that their size
        matches the simulation length. Useful when simulations may end before
        the initially set final time.
        :return:
        """
        for veh in self.vehicles.values():
            veh.truncate_simulation_history()

    def centralize_control(self) -> None:
        centralized_controller = opt_ctrl.VehicleOptimalController()
        ocv = self.get_optimal_control_vehicles()
        for vehicle in ocv:
            vehicle.set_centralized_controller(centralized_controller)

    def compute_acceleration_cost(
            self, include_followers: bool = False,
            include_post_maneuver_adjustments: bool = False) -> float:
        """
        Returns the sum of the acceleration costs (integral of squared
        acceleration) of platoon vehicles and, possibly, immediate followers
        at the origin and destination lanes
        :return:
        """
        lc_vehicle_ids = set(self.lane_changing_vehicle_ids)
        if include_followers:
            # Get all vehicles that, at any point, followed one of the platoon
            # vehicles.
            # This should return fo, fd and most of the platoon vehicles.
            follower_ids = {veh_id for veh_id, veh in self.vehicles.items()
                            if not lc_vehicle_ids.isdisjoint(
                    set(veh.get_origin_lane_leader_id_history()))}
            relevant_ids = lc_vehicle_ids | follower_ids
        else:
            relevant_ids = lc_vehicle_ids
        accel_cost = 0
        if include_post_maneuver_adjustments:
            final_time = self.get_simulated_time()[-1]
        else:
            final_time = np.max(self.get_lc_end_times())
        indices = self.get_simulated_time() <= final_time
        for veh_id in relevant_ids:
            time = self.vehicles[veh_id].get_simulated_time()[indices]
            accel = self.vehicles[veh_id].get_an_input_history('a')[indices]
            accel_cost += np.trapz(np.square(accel), time)
        return accel_cost

    def to_dataframe(self) -> pd.DataFrame:
        """

        :return:
        """
        data_per_vehicle = []
        for vehicle in self.vehicles.values():
            vehicle_df = vehicle.to_dataframe()
            data_per_vehicle.append(vehicle_df)
        all_data = pd.concat(data_per_vehicle).reset_index(drop=True)
        return all_data.fillna(0)


class ShortSimulationVehicleGroup(VehicleGroup):
    """
    Contains only ShortSimulationVehicle instances; used for simulations
    between nodes in a graph of quantized states
    """

    def __init__(self):
        super().__init__()
        self.vehicles: dict[int, fsv.ShortSimulationVehicle] = {}

    @staticmethod
    def create_platoon_scenario(
            n_platoon: int, initial_state: np.ndarray,
            next_platoon_positions_to_move: frozenset[int],
            next_platoon_position_to_coop: int,
            has_ld: bool = True, has_fd: bool = False
    ) -> ShortSimulationVehicleGroup:
        # TODO: must make the method name clearer
        base.BaseVehicle.reset_vehicle_counter()
        lo = fsv.ShortSimulationVehicle(False)
        lo.set_name("lo")
        platoon_vehicles = []
        for i in range(n_platoon):
            veh = fsv.ShortSimulationVehicle(True, is_connected=True)
            veh.set_name("p" + str(i + 1))
            platoon_vehicles.append(veh)

        leader_platoon = vehicle_platoon.ClosedLoopPlatoon(
            platoon_vehicles[0], lc_strategies.StrategyMap.template)
        platoon_vehicles[0].set_platoon(leader_platoon)
        for i in range(1, len(platoon_vehicles)):
            leader_platoon.append_vehicle(platoon_vehicles[i])
            platoon_vehicles[i].set_platoon(leader_platoon)

        if has_ld:
            ld = fsv.ShortSimulationVehicle(False)
            ld.set_name("ld")
            leader_platoon._dest_lane_leader_id = ld.id
        else:
            ld = []
        if has_fd:
            fd = fsv.ShortSimulationVehicle(False)
            fd.set_name("fd")
        else:
            fd = None
        all_vehicles = helper.order_values(lo, platoon_vehicles,
                                           ld, fd)
        vehicle_group = ShortSimulationVehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)
        vehicle_group.set_platoon_lane_change_strategy(
            lc_strategies.StrategyMap.template)

        vehicle_group.set_verbose(False)
        vehicle_group.set_vehicles_initial_states_from_array(
            initial_state)
        vehicle_group.set_platoon_lane_change_order(
            [next_platoon_positions_to_move],
            [next_platoon_position_to_coop])
        return vehicle_group

    def get_vehicle_by_id(self, veh_id: int) -> fsv.ShortSimulationVehicle:
        return self.vehicles[veh_id]

    def get_next_vehicles_to_change_lane(
            self, vehicles_positions_in_platoon: Iterable[int]
    ) -> list[fsv.ShortSimulationVehicle]:
        # We're working with scenarios where only platoon vehicles can change
        # lanes
        platoon_veh_ids = self.get_lane_changing_vehicle_ids()
        return [self.vehicles[platoon_veh_ids[pos]]
                for pos in vehicles_positions_in_platoon]

    def get_next_vehicle_to_cooperate(
            self, vehicle_position_in_platoon: int
    ) -> Union[None, fsv.ShortSimulationVehicle]:
        # We're working with scenarios where only platoon vehicles can change
        # lanes
        platoon_veh_ids = self.get_lane_changing_vehicle_ids()
        return (
            None if vehicle_position_in_platoon < 0
            else self.vehicles[platoon_veh_ids[vehicle_position_in_platoon]]
        )

    def count_completed_lane_changes(self):
        """
        Counts how many platoon vehicles are at the destination lane
        """
        # origin_lane = self.get_vehicle_by_name("lo").get_current_lane()
        dest_lane = self.get_vehicle_by_name("ld").get_current_lane()
        count = 0
        for veh in self.vehicles.values():
            if veh.name[0] == "p" and veh.get_current_lane() == dest_lane:
                count += 1
        return count

    def set_ids_must_change_lanes(self, veh_ids: Iterable[int]):
        self._ids_must_change_lanes = veh_ids
