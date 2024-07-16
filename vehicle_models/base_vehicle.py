from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, TypeVar, Type, Union
import warnings

import numpy as np
import pandas as pd

import configuration as config
from vehicle_models import dynamics
import operating_modes.system_operating_mode as som
import operating_modes.base_operating_modes as modes


class BaseVehicle(ABC):
    _counter = 0

    _initial_state: np.ndarray
    _free_flow_speed: float
    _target_lane: int
    _state_names: list[str]
    _input_names: list[str]
    _n_states: int
    _n_inputs: int
    _state_idx: dict[str, int]
    _input_idx: dict[str, int]
    _can_change_lanes: bool
    _time: np.ndarray
    # _states: np.ndarray
    _inputs: np.ndarray
    _iter_counter: int
    _states_history: np.ndarray
    _inputs_history: np.ndarray
    _origin_leader_id: np.ndarray  # vehicle physically ahead
    _origin_follower_id: np.ndarray  # vehicle physically behind
    _destination_leader_id: np.ndarray  # vehicle physically ahead in dest lane
    _destination_follower_id: np.ndarray  # vehicle physically behind in dest
    # lane
    _desired_destination_lane_leader_id: np.ndarray  # vehicle behind which we
    # want to move
    _aided_vehicle_id: np.ndarray  # vehicle with which we cooperate
    _virtual_leader_id: np.ndarray  # whichever of aided vehicle or desired
    # dest lane leader is valid
    _leader_id: np.ndarray  # vehicle used to determine current accel
    _derivatives: np.ndarray
    _mode: modes.VehicleMode
    _ocp_interface_type: Type[BaseVehicleInterface]
    _desired_future_follower_id: int
    _lc_start_time: float
    _lc_end_time: float

    # Used when we need to copy a vehicle
    static_attribute_names = {
        '_free_flow_speed', '_id', '_name', 'lr', 'lf', 'wheelbase',
        'phi_max', 'brake_max', 'accel_max',
        'h_safe_lk', 'h_ref_lk', 'h_safe_lc', 'h_ref_lc', 'c',
        '_is_connected', '_is_verbose',
    }

    def __init__(self, is_connected: bool = False):
        """

        """

        # Some parameters
        self._id = BaseVehicle._counter
        BaseVehicle._counter += 1
        self._name = str(self._id)  # default
        self.lr = 2  # dist from C.G. to rear wheel
        self.lf = 1  # dist from C.G. to front wheel
        self.wheelbase = self.lr + self.lf
        self.phi_max = 0.1  # max steering wheel angle
        self.brake_max = -6.0
        self.accel_max = 2.0
        self.brake_comfort_max = -2.0
        self._desired_future_follower_id = -1
        # Note: safe time headway values are used to linearly overestimate
        # the nonlinear safe gaps
        self.h_safe_origin_leader = 0
        self.h_ref_origin_leader = 0
        self.h_safe_destination_leader = 0
        self.h_ref_virtual_leader = 0
        self.h_safe_destination_follower = 0
        self.c = config.STANDSTILL_DISTANCE

        self._is_connected = is_connected
        self._is_lane_change_safe = False
        self._is_lane_change_gap_suitable = False
        self._is_verbose = True

    def __repr__(self):
        return f'{self.__class__.__name__} id={self._id}, name={self._name}'

    @staticmethod
    def reset_vehicle_counter() -> None:
        BaseVehicle._counter = 0

    @staticmethod
    def copy_from_other(vehicle_to: BaseVehicle, vehicle_from: BaseVehicle
                        ) -> None:
        for attr_name in vehicle_from.static_attribute_names:
            setattr(vehicle_to, attr_name, getattr(vehicle_from, attr_name))
        vehicle_to.copy_initial_state(vehicle_from.get_states())

    @classmethod
    def get_idx_of_input(cls, name: str) -> int:
        return cls._input_idx[name]

    @classmethod
    def get_idx_of_state(cls, name: str) -> int:
        return cls._state_idx[name]

    def get_id(self) -> int:
        return self._id

    def get_name(self) -> str:
        return self._name

    def get_is_connected(self) -> bool:
        return self._is_connected

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls._state_names)

    @classmethod
    def get_n_inputs(cls) -> int:
        return len(cls._input_names)

    @classmethod
    def get_state_names(cls) -> list[str]:
        return cls._state_names

    @classmethod
    def get_input_names(cls) -> list[str]:
        return cls._input_names

    def get_can_change_lanes(self) -> bool:
        return self._can_change_lanes

    def get_ocp_interface(self) -> BaseVehicleInterface:
        return self._ocp_interface_type(self)

    def get_free_flow_speed(self) -> float:
        return self._free_flow_speed

    def get_initial_state(self) -> np.ndarray:
        return self._initial_state.copy()

    @abstractmethod
    def get_has_open_loop_acceleration(self) -> bool:
        pass

    def get_current_time(self) -> float:
        return self._time[self._iter_counter]

    def get_simulated_time(self) -> np.ndarray:
        return self._time

    def get_x(self) -> float:
        return self.get_a_state_by_name('x')

    def get_y(self) -> float:
        return self.get_a_state_by_name('y')

    def get_theta(self) -> float:
        return self.get_a_state_by_name('theta')

    def get_a_state_by_name(self, state_name) -> float:
        return self.get_states()[self._state_idx[state_name]]

    def get_an_input_by_name(self, input_name) -> float:
        return self.get_inputs()[self._input_idx[input_name]]

    def get_current_lane(self) -> int:
        return round(self.get_y() / config.LANE_WIDTH)

    def get_target_lane(self) -> int:
        return self._target_lane

    def get_target_y(self) -> float:
        return self._target_lane * config.LANE_WIDTH

    def get_states(self) -> np.ndarray:
        try:
            return self._states_history[:, self._iter_counter].copy()
        except AttributeError:  # lazy coding
            return self._initial_state.copy()

    def get_x_history(self) -> np.ndarray:
        return self.get_a_state_history('x')

    def get_y_history(self) -> np.ndarray:
        return self.get_a_state_history('y')

    def get_theta_history(self) -> np.ndarray:
        return self.get_a_state_history('theta')

    def get_a_state_history(self, state_name: str) -> np.ndarray:
        return self.get_state_history()[self._state_idx[state_name], :]

    def get_state_history(self) -> np.ndarray:
        return self._states_history

    def get_inputs(self) -> np.ndarray:
        return self._inputs_history[:, self._iter_counter]

    def get_an_input_history(self, input_name: str) -> np.ndarray:
        return self.get_input_history()[self._input_idx[input_name], :]

    def get_input_history(self) -> np.ndarray:
        return self._inputs_history

    @abstractmethod
    def get_external_input_idx(self) -> dict[str, int]:
        pass

    @abstractmethod
    def get_optimal_input_history(self) -> np.ndarray:
        pass

    def get_derivatives(self) -> np.ndarray:
        return self._derivatives

    def get_origin_lane_leader_id_history(self) -> np.ndarray:
        return self._origin_leader_id

    def get_origin_lane_follower_id_history(self) -> np.ndarray:
        return self._origin_follower_id

    def get_destination_lane_leader_id_history(self) -> np.ndarray:
        return self._destination_leader_id

    def get_destination_lane_follower_id_history(self) -> np.ndarray:
        return self._destination_follower_id

    def get_incoming_vehicle_id_history(self) -> np.ndarray:
        return self._aided_vehicle_id

    def get_origin_lane_leader_id(self) -> int:
        return self._origin_leader_id[self._iter_counter]

    def get_origin_lane_follower_id(self) -> int:
        return self._origin_follower_id[self._iter_counter]

    def get_destination_lane_leader_id(self) -> int:
        return self._destination_leader_id[self._iter_counter]

    def get_destination_lane_follower_id(self) -> int:
        return self._destination_follower_id[self._iter_counter]

    def get_aided_vehicle_id(self) -> int:
        return self._aided_vehicle_id[self._iter_counter]

    def get_is_lane_change_safe(self) -> bool:
        return self._is_lane_change_safe

    def get_is_lane_change_gap_suitable(self) -> bool:
        """
        True if the gap between destination lane follower and leader is
        greater than the sum of the destination lane follower safe gap and the
        ego vehicle's safe gap
        :return:
        """
        return self._is_lane_change_gap_suitable

    # def get_active_brake_max(self) -> float:
    #     if self.get_current_leader_id() == self.get_origin_lane_leader_id():
    #         return self.brake_max
    #     else:
    #         return self.brake_comfort_max

    def get_virtual_leader_id(self) -> int:
        return self._virtual_leader_id[self._iter_counter]

    def get_relevant_surrounding_vehicle_ids(self) -> dict[str, int]:
        """
        Returns the IDs relevant vehicles
        :return:
        """
        return {'lo': self.get_origin_lane_leader_id(),
                'ld': self.get_destination_lane_leader_id(),
                'fd': self.get_destination_lane_follower_id(),
                'leader': self.get_current_leader_id()}

    def get_possible_target_leader_ids(self) -> list[int]:
        """
        Returns the IDs of all vehicles possibly used as target leaders
        :return:
        """
        candidates = [
            # For safety, we always need to be aware of the vehicle physically
            # ahead on the same lane
            self.get_origin_lane_leader_id(),
            # The vehicle behind which we want to merge. When in a platoon,
            # this may be different from current destination lane leader
            self.get_desired_destination_lane_leader_id(),
            # A vehicle that needs our cooperation to merge in front of us. The
            # value can be determined by a cooperation request or by a platoon
            # lane changing strategy
            self.get_aided_vehicle_id()
        ]
        return candidates

    def get_desired_destination_lane_leader_id(self) -> int:
        return self._desired_destination_lane_leader_id[self._iter_counter]

    def get_desired_future_follower_id(self) -> int:
        return self._desired_future_follower_id

    # def get_lane_changing_time_headway(self, are_connected: bool) -> float:
    #     h = self.get_h_safe_lk()
    #     return h + (0.2 if config.Configuration.increase_lc_time_headway
    #                 else 0.)

    def get_current_leader_id(self) -> int:
        """
        The 'current' leader is the vehicle being used to define this vehicle's
         acceleration
        :return:
        """
        try:
            return self._leader_id[self._iter_counter]
        except IndexError:
            # Some vehicles never set a target leader because they do not
            # have closed-loop acceleration control
            return self._origin_leader_id[self._iter_counter]

    def get_lc_end_time(self) -> float:
        return (self._lc_end_time if self._lc_end_time < np.inf
                else self.get_current_time())

    def get_platoon(self):
        return None

    def get_platoon_strategy_decision_time(self) -> float:
        pass  # TODO: make abstract or refactor code

    # def _set_time_headways(self, are_connected: bool) -> None:
    #     self.h_safe_lk = (config.SAFE_CONNECTED_TIME_HEADWAY if are_connected
    #                       else config.SAFE_TIME_HEADWAY)
    #     self.h_safe_lc = config.get_lane_changing_time_headway(are_connected)

    def set_free_flow_speed(self, v_ff: float) -> None:
        self._free_flow_speed = v_ff

    def set_name(self, value: str) -> None:
        self._name = value

    def set_verbose(self, value: bool) -> None:
        self._is_verbose = value

    def set_initial_state(self, x: float = None, y: float = None,
                          theta: float = None, v: float = None,
                          full_state: np.ndarray = None) -> None:
        """
        We can either provide each state individually or the full state vector
        directly. In the later case, the caller must ensure the states are in
        the right order
        :param x:
        :param y:
        :param theta:
        :param v:
        :param full_state:
        :return:
        """
        if full_state is None:
            self._initial_state = self.create_state_vector(x, y, theta, v)
        else:
            self._initial_state = full_state
        # self._states = self._initial_state
        # self._states_history[:, 0] = self._initial_state
        self._target_lane = self.get_current_lane()

    def target_origin_lane_leader(self):
        self._leader_id[self._iter_counter] = self.get_origin_lane_leader_id()

    def target_virtual_leader(self):
        self._leader_id[self._iter_counter] = self.get_virtual_leader_id()

    def force_state(self, states: np.ndarray):
        # self._states = states
        self._states_history[:, self._iter_counter] = states

    def make_reset_copy(self, initial_state: np.ndarray = None) -> V:
        """
        Creates copies of vehicles used in internal iterations of our optimal
        controller. For vehicles without optimal control, the method returns
        an "empty" copy of the vehicle (without any state history). For
        vehicles with optimal control, the method returns the equivalent
        open loop type vehicle.
        :param initial_state: If None, sets the new vehicle's initial state
        equal to the most recent state of this instance
        :return:
        """
        new_vehicle_type = type(self)
        new_vehicle = new_vehicle_type()
        self.copy_attributes(new_vehicle, initial_state)
        return new_vehicle

    def copy_attributes(self, new_vehicle: BaseVehicle,
                        initial_state: np.ndarray = None) -> None:
        for attr_name in self.static_attribute_names:
            setattr(new_vehicle, attr_name, getattr(self, attr_name))
        if initial_state is None:
            initial_state = self.get_states()
        new_vehicle.set_initial_state(full_state=initial_state)
        new_vehicle._target_lane = self._target_lane

    def make_open_loop_copy(self, initial_state: np.ndarray = None) -> V:
        """
        Creates copies of vehicles used in internal iterations of our optimal
        controller. For vehicles without optimal control, the method returns
        an "empty" copy of the vehicle (without any state history). For
        vehicles with optimal control, the method returns the equivalent
        open loop type vehicle.
        :return:
        """
        return self.make_reset_copy(initial_state)

    def make_closed_loop_copy(self, initial_state: np.ndarray = None) -> V:
        return self.make_reset_copy(initial_state)

    def _reset_copied_vehicle(self, new_vehicle: BaseVehicle,
                              initial_state: np.ndarray = None) -> None:
        if initial_state is None:
            initial_state = self.get_states()
        new_vehicle.set_initial_state(full_state=initial_state)
        new_vehicle._target_lane = self._target_lane
        new_vehicle.reset_simulation_logs()

    def copy_initial_state(self, initial_state: np.ndarray) -> None:
        if len(initial_state) != self._n_states:
            raise ValueError('Wrong size of initial state vector ({} instead'
                             'of {})'.format(len(initial_state),
                                             self._n_states))
        self._initial_state = initial_state
        # self._states_history[:, 0] = initial_state
        # self._states = self._initial_state
        # self._target_lane = self.get_current_lane()

    def set_mode(self, mode: modes.VehicleMode) -> None:
        if self._is_verbose:
            try:
                print("t={:.2f}, veh {}. From: {} to {}".format(
                    self.get_current_time(), self._id, self._mode, mode))
            except AttributeError:  # when the mode is set for the first time
                # print("t=0.0, veh {}. Initial op mode: {}".format(
                #     self._id, mode))
                pass
        self._mode = mode
        self._mode.set_ego_vehicle(self)

    def set_lane_change_direction(self, lc_direction) -> None:
        self._target_lane = self.get_current_lane() + lc_direction

    def write_state_and_input(self, time: float, states: np.ndarray,
                              optimal_inputs: np.ndarray) -> None:
        """
        Used when vehicle states were computed externally. Do not mix use of
        this method and update_states.
        :param time:
        :param states:
        :param optimal_inputs:
        :return:
        """
        self._iter_counter += 1
        self._time[self._iter_counter] = time
        # self._states = states
        self._states_history[:, self._iter_counter] = states
        self._inputs = np.zeros(self._n_inputs)
        if len(optimal_inputs) > 0:
            self._write_optimal_inputs(optimal_inputs)
        self._inputs_history[:, self._iter_counter] = self._inputs

    def prepare_to_start_simulation(self, n_samples: int) -> None:
        """
        Erases all simulation data and allows the vehicle trajectory to be
        simulated again
        """
        self.initialize_simulation_logs(n_samples)
        self._lc_start_time = np.inf
        self._lc_end_time = 0.

        # Initial state
        self._time[self._iter_counter] = 0.0
        self._states_history[:, self._iter_counter] = self._initial_state
        self._inputs_history[:, self._iter_counter] = np.zeros(self._n_inputs)

    def initialize_simulation_logs(self, n_samples: int) -> None:
        self._iter_counter = 0
        self._time = np.zeros(n_samples)
        self._states_history = np.zeros([self._n_states, n_samples])
        self._inputs_history = np.zeros([self._n_inputs, n_samples])
        self._origin_leader_id = -np.ones(n_samples, dtype=int)
        self._origin_follower_id = -np.ones(n_samples, dtype=int)
        self._destination_leader_id = -np.ones(n_samples, dtype=int)
        self._destination_follower_id = -np.ones(n_samples, dtype=int)
        self._aided_vehicle_id = -np.ones(n_samples, dtype=int)
        self._desired_destination_lane_leader_id = -np.ones(n_samples,
                                                            dtype=int)
        self._virtual_leader_id = -np.ones(n_samples, dtype=int)
        self._leader_id = -np.ones(n_samples, dtype=int)

    def reset_simulation_logs(self) -> None:
        self.initialize_simulation_logs(0)

    def truncate_simulation_history(self):
        """
        Truncates all the data matrices so that their size matches the
        simulation length. Useful when simulations may end before the initially
        set final time.
        :return:
        """
        try:
            self._time = self._time[:self._iter_counter + 1]
            self._states_history = self._states_history[
                                   :, :self._iter_counter + 1]
            self._inputs_history = self._inputs_history[
                                   :, :self._iter_counter + 1]
            self._origin_leader_id = self._origin_leader_id[
                                     :self._iter_counter + 1]
            self._origin_follower_id = self._destination_follower_id[
                                     :self._iter_counter + 1]
            self._destination_leader_id = self._destination_leader_id[
                                          :self._iter_counter + 1]
            self._destination_follower_id = self._destination_follower_id[
                                            :self._iter_counter + 1]
            self._aided_vehicle_id = self._aided_vehicle_id[
                                     :self._iter_counter + 1]
            self._virtual_leader_id = self._virtual_leader_id[
                                      :self._iter_counter + 1]
            self._leader_id = self._leader_id[:self._iter_counter + 1]
            self._desired_destination_lane_leader_id = (
                self._desired_destination_lane_leader_id[
                    :self._iter_counter + 1]
            )
        except IndexError:
            # The matrices already have the right size
            pass

    def has_origin_lane_leader(self) -> bool:
        try:
            return self.get_origin_lane_leader_id() >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(orig lane leader id) before simulation start")
            return False

    def has_destination_lane_leader(self) -> bool:
        try:
            return self.get_destination_lane_leader_id() >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(dest lane leader id) before simulation start")
            return False

    def has_destination_lane_follower(self) -> bool:
        try:
            return self.get_destination_lane_follower_id() >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(dest lane follower id) before simulation start")
            return False

    def is_cooperating(self) -> bool:
        return self.get_aided_vehicle_id() >= 0

    def has_virtual_leader(self) -> bool:
        return self.get_virtual_leader_id() >= 0

    # def has_changed_leader(self) -> bool:
    #     try:
    #         return (self._leader_id[self._iter_counter - 1]
    #                 != self._leader_id[self._iter_counter])
    #     except IndexError:
    #         warnings.warn("Warning: trying to check if leader has changed too"
    #                       "early (before two simulation steps)")
    #         return False

    def has_lane_change_intention(self) -> bool:
        return self.get_current_lane() != self._target_lane

    def is_at_lane_center(self) -> bool:
        return np.abs(self.get_y() - self.get_target_y()) <= 1.0e-4

    def has_requested_cooperation(self) -> bool:
        return self.get_desired_future_follower_id() >= 0

    def make_connected(self) -> None:
        self._is_connected = True

    def is_in_a_platoon(self) -> bool:
        return not (self.get_platoon() is None)

    def has_started_lane_change(self) -> bool:
        return self.get_current_time() >= self._lc_start_time

    def has_origin_lane_leader_changed(self) -> bool:
        if self._iter_counter > 0:
            return self.get_origin_lane_leader_id() != self._origin_leader_id[
                self._iter_counter - 1]
        return True

    def update_surrounding_vehicles(self, vehicles: Mapping[int, BaseVehicle]):
        self.find_origin_lane_vehicles(vehicles)
        self.find_destination_lane_vehicles(vehicles)
        # self.find_cooperation_requests(vehicles)

    def find_origin_lane_vehicles(self, vehicles: Mapping[int, BaseVehicle]
                                  ) -> None:
        """
        Finds and sets the ids of the origin lane leader and follower.
        """
        ego_x = self.get_x()
        orig_lane_leader_x = np.inf
        orig_lane_follower_x = -np.inf
        new_orig_leader_id = -1
        new_orig_follower_id = -1
        for other_id, other_vehicle in vehicles.items():
            other_x = other_vehicle.get_x()
            is_on_same_lane = (
                    self.get_current_lane() == other_vehicle.get_current_lane()
                    or self._is_other_cutting_in(other_vehicle))
            if is_on_same_lane and ego_x < other_x < orig_lane_leader_x:
                orig_lane_leader_x = other_x
                new_orig_leader_id = other_id
            elif is_on_same_lane and orig_lane_follower_x < other_x < ego_x:
                orig_lane_follower_x = other_x
                new_orig_follower_id = other_id
        if (new_orig_leader_id >= 0
                and (self._iter_counter == 0
                     or (new_orig_leader_id
                         != self._origin_leader_id[self._iter_counter - 1]))):
            self._update_origin_lane_time_headway(vehicles[new_orig_leader_id])

        self._origin_leader_id[self._iter_counter] = new_orig_leader_id
        self._origin_follower_id[self._iter_counter] = new_orig_follower_id

    def _is_other_cutting_in(self, other_vehicle: BaseVehicle):
        # return False
        return (other_vehicle.has_lane_change_intention()
                and other_vehicle.get_target_lane() == self.get_current_lane()
                and (other_vehicle.get_id()
                     != self._aided_vehicle_id[self._iter_counter-1])
                and np.abs(other_vehicle.get_an_input_by_name('phi') > 1e-3))

    def find_destination_lane_vehicles(self, vehicles: Mapping[int, BaseVehicle]
                                       ) -> None:
        new_dest_leader_id = -1
        new_dest_follower_id = -1
        if self.has_lane_change_intention():
            y_target_lane = self._target_lane * config.LANE_WIDTH
            ego_x = self.get_x()
            dest_lane_follower_x = -np.inf
            dest_lane_leader_x = np.inf
            for other_id, other_vehicle in vehicles.items():
                other_x = other_vehicle.get_x()
                other_y = other_vehicle.get_y()
                if np.abs(other_y - y_target_lane) < config.LANE_WIDTH / 2:
                    if ego_x <= other_x < dest_lane_leader_x:
                        dest_lane_leader_x = other_x
                        new_dest_leader_id = other_id
                    elif dest_lane_follower_x < other_x < ego_x:
                        dest_lane_follower_x = other_x
                        new_dest_follower_id = other_id

        if (new_dest_leader_id >= 0
                and (self._iter_counter > 0
                     or (new_dest_leader_id != self._destination_leader_id[
                            self._iter_counter - 1]))):
            self._update_destination_lane_time_headway(
                vehicles[new_dest_leader_id])
        if (new_dest_follower_id >= 0
                and (self._iter_counter > 0
                     or (new_dest_follower_id != self._destination_follower_id[
                            self._iter_counter - 1]))):
            self._update_future_follower_time_headway(
                vehicles[new_dest_follower_id])

        self._destination_leader_id[self._iter_counter] = new_dest_leader_id
        self._destination_follower_id[self._iter_counter] = new_dest_follower_id

    def find_cooperation_requests(self, vehicles: Iterable[BaseVehicle]
                                  ) -> None:
        new_incoming_vehicle_id = -1
        incoming_veh_x = np.inf
        if self._is_connected:
            for other_vehicle in vehicles:
                other_request = other_vehicle.get_desired_future_follower_id()
                other_x = other_vehicle.get_x()
                if other_request == self._id and other_x < incoming_veh_x:
                    new_incoming_vehicle_id = other_vehicle._id
                    incoming_veh_x = other_x
        self._aided_vehicle_id[self._iter_counter] = new_incoming_vehicle_id

    def find_desired_destination_lane_leader(self):
        self._desired_destination_lane_leader_id[self._iter_counter] = (
            self.get_destination_lane_leader_id())

    def detect_collision(self) -> bool:
        if (self._iter_counter > 0
                and self.has_origin_lane_leader()
                and (self.get_origin_lane_leader_id()
                     == self._origin_follower_id[self._iter_counter - 1])):
            return True
        return False

    def _update_origin_lane_time_headway(self, new_leader: BaseVehicle):
        self.h_safe_origin_leader = self._get_time_headway(new_leader)
        self.h_ref_origin_leader = (self.h_safe_origin_leader
                                    + config.TIME_HEADWAY_MARGIN)

    def _update_destination_lane_time_headway(self, new_leader: BaseVehicle):
        self.h_safe_destination_leader = self._get_time_headway(new_leader)

    def _update_future_follower_time_headway(self, new_follower: BaseVehicle):
        self.h_safe_destination_follower = self._get_time_headway(new_follower)

    def _update_virtual_leader_time_headway(self, new_leader: BaseVehicle):
        pass

    def _get_time_headway(self, other_vehicle):
        if self._is_connected and other_vehicle.get_is_connected():
            return config.SAFE_CONNECTED_TIME_HEADWAY
        else:
            return config.SAFE_TIME_HEADWAY

    def check_surrounding_gaps_safety(
            self, vehicles: Mapping[int, BaseVehicle],
            ignore_orig_lane_leader: bool = False) -> None:
        margin = 1e-1

        if ignore_orig_lane_leader:
            is_safe_to_orig_lane_leader = True
        else:
            orig_leader_safe_gap = self.compute_reference_gap(
                self.h_safe_origin_leader)
            if self.has_origin_lane_leader():
                orig_lane_leader = vehicles[self.get_origin_lane_leader_id()]
                gap_to_lo = BaseVehicle.compute_a_gap(orig_lane_leader, self)
            else:
                gap_to_lo = np.inf
            is_safe_to_orig_lane_leader = (gap_to_lo + margin
                                           >= orig_leader_safe_gap)

        if self.has_destination_lane_leader():
            dest_lane_leader = vehicles[self.get_destination_lane_leader_id()]
            gap_to_ld = BaseVehicle.compute_a_gap(dest_lane_leader, self)
            dest_lane_leader_vel = dest_lane_leader.get_vel()
            dest_leader_safe_gap = self.compute_safe_lane_change_gap(
                self.h_safe_destination_leader, dest_lane_leader_vel,
                dest_lane_leader.brake_max, is_other_ahead=True)
        else:
            gap_to_ld = np.inf
            dest_lane_leader_vel = np.inf
            dest_leader_safe_gap = 0
        is_safe_to_dest_lane_leader = gap_to_ld + margin >= dest_leader_safe_gap

        # is_safe_to_dest_lane_follower = True
        if self.has_destination_lane_follower():
            dest_lane_follower = vehicles[
                self.get_destination_lane_follower_id()]
            gap_from_fd = BaseVehicle.compute_a_gap(self, dest_lane_follower)
            fd_safe_gap = dest_lane_follower.compute_safe_lane_change_gap(
                self.h_safe_destination_follower, dest_lane_follower.get_vel(),
                dest_lane_follower.brake_max, is_other_ahead=False)
        else:
            gap_from_fd = np.inf
            fd_safe_gap = 0.
        is_safe_to_dest_lane_follower = gap_from_fd + margin >= fd_safe_gap

        self._is_lane_change_safe = (is_safe_to_orig_lane_leader
                                     and is_safe_to_dest_lane_leader
                                     and is_safe_to_dest_lane_follower)
        # A suitable gap is a gap that the vehicle can reach by decelerating
        # and that is large enough for a lane change.
        lc_speed = min(self.get_vel(), dest_lane_leader_vel)

        # Because the gap below is computed with the min speed of both vehicles,
        # we don't need to call compute_safe_lane_change_gap (the nonlinear
        # term will be zero anyway)
        ego_safe_gap_after_deceleration = self.compute_reference_gap(
            self.h_safe_destination_leader, lc_speed)
        min_gap_to_decelerate = (
            (self.get_vel()**2 - dest_lane_leader_vel**2)
            / 2 / np.abs(self.brake_comfort_max))
        self._is_lane_change_gap_suitable = (
                self._is_lane_change_safe
                or (
                    is_safe_to_dest_lane_follower
                    and gap_to_ld >= min_gap_to_decelerate
                    and (gap_to_ld + gap_from_fd + margin
                         >= fd_safe_gap + ego_safe_gap_after_deceleration)
                )
        )

    # @staticmethod
    # def is_gap_safe_for_lane_change(leading_vehicle: BaseVehicle,
    #                                 following_vehicle: BaseVehicle):
    #     margin = 1e-2
    #     gap = BaseVehicle.compute_a_gap(leading_vehicle, following_vehicle)
    #     safe_gap = following_vehicle.compute_safe_lane_change_gap(
    #         following_vehicle.get_vel())
    #     return gap + margin >= safe_gap

    def reset_platoon(self):
        pass

    def initialize_platoons(
            self, vehicles: Mapping[int, BaseVehicle],
            platoon_lane_change_strategy=None
    ) -> None:
        pass

    def set_platoon_lane_change_order(
            self, strategy_order: config.Strategy
    ) -> None:
        pass

    def request_cooperation(self) -> None:
        if self._is_connected:
            self._desired_future_follower_id = (
                self.get_destination_lane_follower_id())

    def receive_cooperation_request(self, other_id) -> None:
        if self._is_connected:
            self._aided_vehicle_id[self._iter_counter] = other_id

    def update_virtual_leader(self, vehicles: Mapping[int, BaseVehicle]
                              ) -> None:
        self.find_cooperation_requests(vehicles.values())
        self.find_desired_destination_lane_leader()

        if (self.get_aided_vehicle_id() >= 0
                and self.get_desired_destination_lane_leader_id() >= 0):
            warnings.warn(
                "[BaseVehicle] Vehicles has a valid desired destination lane "
                "leader and a valid assisted vehicle. Setting virtual leader "
                "as the desired destination lane leader.")
            new_virtual_leader_id = (
                self.get_desired_destination_lane_leader_id())
        elif self.get_aided_vehicle_id() >= 0:
            new_virtual_leader_id = (
                self.get_aided_vehicle_id())
        else:
            new_virtual_leader_id = (
                    self.get_desired_destination_lane_leader_id())

        if (new_virtual_leader_id >= 0
                and (self._iter_counter == 0
                     or (new_virtual_leader_id
                         != self._virtual_leader_id[self._iter_counter - 1]))):
            self._update_virtual_leader_time_headway(
                vehicles[new_virtual_leader_id])

        self._virtual_leader_id[self._iter_counter] = new_virtual_leader_id

    @staticmethod
    def compute_a_gap(leading_vehicle: BaseVehicle,
                      following_vehicle: BaseVehicle) -> float:
        return (leading_vehicle.get_x()
                - following_vehicle.get_x())

    # def compute_free_flow_desired_gap(self) -> float:
    #     h_ref = self.h_ref_origin_leader
    #     return self.compute_desired_gap(self._free_flow_speed, h_ref)

    # def compute_safe_lane_change_gap(self, v_ego=None) -> float:
    #     if v_ego is None:
    #         v_ego = self.get_vel()
    #     return self.h_safe_lc * v_ego + self.c

    # def compute_lane_keeping_desired_gap(self, vel: float = None) -> float:
    #     return self._compute_reference_gap(self.h_ref_lk, vel)

    def compute_non_connected_reference_gap(self, vel: float = None) -> float:
        h_ref = config.SAFE_TIME_HEADWAY + config.TIME_HEADWAY_MARGIN
        return self.compute_reference_gap(h_ref, vel)

    def compute_initial_reference_gap_to(self, other_vehicle: BaseVehicle,
                                         vel: float = None) -> float:
        """
        Computes the reference gap between self and other vehicle before
        the vehicles are included in the simulation. Used to define initial
        states
        :param other_vehicle:
        :param vel:
        :return:
        """
        h_ref = (self._get_time_headway(other_vehicle)
                 + config.TIME_HEADWAY_MARGIN)
        return self.compute_reference_gap(h_ref, vel)

    def compute_reference_gap(self, h_ref: float, vel: float = None
                              ) -> float:
        if vel is None:
            vel = self.get_vel()
        return h_ref * vel + self.c

    def compute_safe_lane_change_gap(
            self, h_safe: float, other_vel: float, other_max_brake: float,
            is_other_ahead: bool) -> float:
        g_ref = self.compute_reference_gap(h_safe)
        if is_other_ahead:
            leader_vel, leader_brake = other_vel, other_max_brake
            follower_vel, follower_brake = self.get_vel(), self.brake_max
        else:
            leader_vel, leader_brake = self.get_vel(), self.brake_max
            follower_vel, follower_brake = other_vel, other_max_brake
        rel_vel_term = (follower_vel ** 2 / 2 / abs(follower_brake)
                        - leader_vel ** 2 / 2 / abs(leader_brake))
        return g_ref + max(rel_vel_term, 0)

    def compute_derivatives(self) -> None:
        self._derivatives = np.zeros(self._n_states)
        theta = self.get_theta()
        phi = self.get_an_input_by_name('phi')
        vel = self.get_vel()
        self._compute_derivatives(vel, theta, phi)

    def determine_inputs(self, open_loop_controls: np.ndarray,
                         vehicles: Mapping[int, BaseVehicle]) -> None:
        """
        Sets the open loop controls and computes the closed loop controls.
        :param open_loop_controls: dictionary whose keys are the input name.
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs = np.zeros(self._n_inputs)
        self._determine_inputs(open_loop_controls, vehicles)
        self._inputs_history[:, self._iter_counter] = self._inputs

    def update_states(self, next_time) -> None:
        dt = next_time - self.get_current_time()
        # self._states = self._states + self._derivatives * dt
        self._states_history[:, self._iter_counter + 1] = (
            self._states_history[:, self._iter_counter] + self._derivatives * dt
        )
        self._time[self._iter_counter + 1] = next_time
        # self._iter_counter += 1

    def update_iteration_counter(self):
        self._iter_counter += 1

    @abstractmethod
    def update_mode(self, vehicles: Mapping[int, BaseVehicle]) -> None:
        # only vehicles with controllers update their discrete states (modes)
        pass

    def _position_derivative_longitudinal_only(self, vel) -> None:
        dx, dy, dtheta = dynamics.position_derivative_longitudinal_only(vel)
        self._derivatives[self._state_idx['x']] = dx
        self._derivatives[self._state_idx['y']] = dy
        self._derivatives[self._state_idx['theta']] = dtheta

    def _position_derivative_cg(self, vel: float, theta: float, phi: float
                                ) -> None:
        dx, dy, dtheta = dynamics.position_derivative_cg(vel, theta, phi,
                                                         self.lf, self.lr)
        self._derivatives[self._state_idx['x']] = dx
        self._derivatives[self._state_idx['y']] = dy
        self._derivatives[self._state_idx['theta']] = dtheta

    def _position_derivative_rear_wheels(self, vel: float, theta: float,
                                         phi: float) -> None:
        dx, dy, dtheta = dynamics.position_derivative_rear_wheels(
            vel, theta, phi, self.wheelbase
        )
        self._derivatives[self._state_idx['x']] = dx
        self._derivatives[self._state_idx['y']] = dy
        self._derivatives[self._state_idx['theta']] = dtheta

    # TODO: duplicated at interface
    def create_state_vector(self, x: float, y: float, theta: float,
                            v: float = None):
        state_vector = np.zeros(self._n_states)
        state_vector[self._state_idx['x']] = x
        state_vector[self._state_idx['y']] = y
        state_vector[self._state_idx['theta']] = theta
        self._set_speed(v, state_vector)
        return state_vector

    @classmethod
    def create_state_vector_2(cls, x: float, y: float, theta: float,
                              v: float = None):
        state_vector = np.zeros(cls._n_states)
        state_vector[cls.get_idx_of_state('x')] = x
        state_vector[cls.get_idx_of_state('y')] = y
        state_vector[cls.get_idx_of_state('theta')] = theta
        cls._set_speed(v, state_vector)
        return state_vector

    def to_dataframe(self) -> pd.DataFrame:
        data = np.concatenate([self._time.reshape(1, -1),
                               self._states_history, self._inputs_history])
        columns = (['t'] + [s for s in self._state_names]
                   + [i for i in self._input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self._id
        df['name'] = self._name
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'orig_lane_leader_id', self._origin_leader_id)
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'dest_lane_leader_id', self._destination_leader_id)
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'dest_lane_follower_id', self._destination_follower_id)
        return df

    def prepare_for_lane_keeping_start(self) -> None:
        self._lc_start_time = np.inf
        # we assume at most one lane change per simulation
        self._lc_end_time = self.get_current_time()

    def prepare_for_longitudinal_adjustments_start(self) -> None:
        self.request_cooperation()
        self._lc_end_time = np.inf

    def prepare_for_lane_change_start(self) -> None:
        self._lc_start_time = self.get_current_time()
        self._set_up_lane_change_control()

    # @classmethod
    # def _set_model(cls) -> None:
    #     """
    #     Must be called in the constructor of every derived class to set the
    #     variables that define which vehicle model is implemented.
    #     :return:
    #     """
    #     cls._n_states = len(cls._state_names)
    #     cls._n_inputs = len(cls._input_names)
    #     cls._state_idx = {cls._state_names[i]: i for i
    #                       in range(cls._n_states)}
    #     cls._input_idx = {cls._input_names[i]: i for i
    #                       in range(cls._n_inputs)}

    @staticmethod
    def _set_model(state_names, input_names
                   ) -> tuple[int, int, dict[str, int], dict[str, int]]:
        """
        Must be called by every derived class to set their class variables
        :param state_names:
        :param input_names:
        :return:
        """
        _n_states = len(state_names)
        _n_inputs = len(input_names)
        _state_idx = {state_names[i]: i for i in range(_n_states)}
        _input_idx = {input_names[i]: i for i in range(_n_inputs)}
        return _n_states, _n_inputs, _state_idx, _input_idx
    # def follow_origin_lane_leader(self) -> None:
    #     self._set_current_leader_id(self.get_origin_lane_leader_id())

    @abstractmethod
    def get_vel(self) -> float:
        pass

    @staticmethod
    def _set_surrounding_vehicles_ids_to_df(df, col_name, col_value) -> None:
        if len(col_value) == 1:
            df[col_name] = col_value[0]
        else:
            df[col_name] = col_value

    @classmethod
    @abstractmethod
    def _set_speed(cls, v0, state) -> None:
        """
        Sets the proper element in array state equal to v0
        :param v0: speed to write
        :param state: state vector being modified
        :return: nothing, modifies the state array
        """
        pass

    @abstractmethod
    def _compute_derivatives(self, vel, theta, phi):
        """ Computes the derivatives of x, y, and theta, and stores them in the
         derivatives array """
        pass

    @abstractmethod
    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Mapping[int, BaseVehicle]):
        """
        Sets the open loop controls and computes the closed loop controls.
        :param open_loop_controls: dictionary whose keys are the input name.
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        pass

    def _set_up_lane_change_control(self):
        pass

    def _write_optimal_inputs(self, optimal_inputs):
        """
        Used when vehicle states were computed externally.
        :return: Nothing
        """
        pass

    # def _set_current_leader_id(self, veh_id):
    #     """
    #     Sets which vehicle used to determine this vehicle's accel. The
    #     definition of leader ids for all vehicles in a vehicle group
    #     determines the operating mode.
    #     :param veh_id: leading vehicle's id. Use -1 to designate no leader
    #     :return:
    #     """
    #     self._leader_id[self._iter_counter] = veh_id


# TODO: still must figure out a way to prevent so much repeated code between
#  vehicles and their interfaces with the opc solver
class BaseVehicleInterface(ABC):
    _state_names: list[str]
    n_states: int
    _input_names: list[str]
    n_inputs: int
    state_idx: dict[str, int]
    optimal_input_idx: dict[str, int]

    def __init__(self, vehicle: BaseVehicle):
        """

        """
        self.base_vehicle = vehicle
        self._time: float = 0.

        self._origin_leader_id: int = vehicle.get_origin_lane_leader_id()
        self._destination_leader_id: int = (
            vehicle.get_destination_lane_leader_id())
        self._destination_follower_id: int = (
            vehicle.get_destination_lane_follower_id())
        # TODO: we must copy other surrounding vehicles' ids here too
        self._leader_id: int = vehicle.get_current_leader_id()
        self.target_lane: int = vehicle.get_target_lane()
        # The vehicle's current state is the starting point for the ocp
        self._initial_state = vehicle.get_states()

        # Only set for some vehicle types
        self._interval_number = 0
        # TODO: lump all these in a single dictionary
        self.ocp_mode_switch_times: list[float] = []
        self.ocp_origin_leader_sequence: list[int] = []
        self.ocp_destination_leader_sequence: list[int] = []
        self.ocp_destination_follower_sequence: list[int] = []
        self.ocp_target_leader_sequence: list[int] = []

    def __repr__(self):
        return (self.__class__.__name__ + ': id=' + str(self.get_id())
                + "V_f=" + str(self.get_free_flow_speed()))

    def _set_model(self):
        """
        Must be called in the constructor of every derived class to set the
        variables that define which vehicle model is implemented.
        :return:
        """
        self.n_states = len(self._state_names)
        self.n_inputs = len(self._input_names)
        self.state_idx = {self._state_names[i]: i for i in range(self.n_states)}
        self.optimal_input_idx = {self._input_names[i]: i for i in
                                  range(self.n_inputs)}

    @classmethod
    def get_state_names(cls):
        return cls._state_names

    def get_input_names(self):
        return self._input_names

    def select_state_from_vector(self, states: Union[np.ndarray, list],
                                 state_name: str) -> float:
        return states[self.state_idx[state_name]]

    def select_input_from_vector(self, optimal_inputs: Union[np.ndarray, list],
                                 input_name: str) -> float:
        return optimal_inputs[self.optimal_input_idx[input_name]]

    def select_vel_from_vector(self, states: Union[np.ndarray, list],
                               inputs: Union[np.ndarray, list]):
        try:
            return states[self.state_idx['v']]
        except KeyError:
            return inputs[self.optimal_input_idx['v']]

    def get_phi(self, optimal_inputs):
        return self.select_input_from_vector(optimal_inputs, 'phi')

    def get_id(self) -> int:
        return self.base_vehicle.get_id()

    def get_name(self) -> str:
        return self.base_vehicle.get_name()

    def get_free_flow_speed(self) -> float:
        return self.base_vehicle.get_free_flow_speed()

    def get_phi_max(self) -> float:
        return self.base_vehicle.phi_max

    def get_accel_max(self) -> float:
        return self.base_vehicle.accel_max

    def get_brake_max(self) -> float:
        return self.base_vehicle.brake_max

    def get_initial_state(self):
        return self._initial_state

    def get_x0(self):
        return self.select_state_from_vector(self._initial_state, 'x')

    def get_y0(self):
        return self.select_state_from_vector(self._initial_state, 'y')

    def get_origin_lane_leader_id(self, time: float):
        if len(self.ocp_origin_leader_sequence) == 0:
            return self._origin_leader_id
        self.set_time_interval(time)
        return self.ocp_origin_leader_sequence[self._interval_number]

    def get_destination_lane_leader_id(self, time: float):
        if len(self.ocp_destination_leader_sequence) == 0:
            return self._destination_leader_id
        self.set_time_interval(time)
        return self.ocp_destination_leader_sequence[self._interval_number]

    def get_destination_lane_follower_id(self, time: float):
        if len(self.ocp_destination_follower_sequence) == 0:
            return self._destination_follower_id
        self.set_time_interval(time)
        return self.ocp_destination_follower_sequence[self._interval_number]

    def get_current_leader_id(self, time: float) -> int:
        """
        The 'current' leader is the vehicle being used to define this vehicle's
         acceleration
        :return:
        """
        if len(self.ocp_target_leader_sequence) == 0:
            return self._leader_id
        self.set_time_interval(time)
        return self.ocp_target_leader_sequence[self._interval_number]

    def has_origin_lane_leader(self, time: float) -> bool:
        return self.get_origin_lane_leader_id(time) >= 0

    def has_destination_lane_leader(self, time: float) -> bool:
        return self.get_destination_lane_leader_id(time) >= 0

    def has_destination_lane_follower(self, time: float) -> bool:
        return self.get_destination_lane_follower_id(time) >= 0

    def has_leader(self, time: float) -> bool:
        return self.get_current_leader_id(time) >= 0

    def get_target_y(self) -> float:
        return self.target_lane * config.LANE_WIDTH

    def is_long_control_optimal(self) -> bool:
        return 'a' in self._input_names or 'v' in self._input_names

    def is_lat_control_optimal(self) -> bool:
        return 'phi' in self._input_names

    def has_lane_change_intention(self) -> bool:
        return self.base_vehicle.has_lane_change_intention()

    def set_leader_sequence(self, leader_sequence: som.SVSequence
                            ) -> None:
        self._interval_number = 0
        self.ocp_mode_switch_times: list[float] = []
        self.ocp_origin_leader_sequence: list[int] = []
        self.ocp_destination_leader_sequence: list[int] = []
        self.ocp_destination_follower_sequence: list[int] = []
        self.ocp_target_leader_sequence: list[int] = []
        for t, l_id in leader_sequence:
            self.ocp_mode_switch_times.append(t)
            self.ocp_origin_leader_sequence.append(l_id['lo'])
            self.ocp_destination_leader_sequence.append(l_id['ld'])
            self.ocp_destination_follower_sequence.append(l_id['fd'])
            self.ocp_target_leader_sequence.append(l_id['leader'])

    def set_time_interval(self, time: float) -> None:
        if (np.abs(self._time - time)
                >= config.Configuration.discretization_step):
            self._time = time
            idx = np.searchsorted(self.ocp_mode_switch_times, time,
                                  side='right')
            self._interval_number = idx - 1

    def create_state_vector(self, x: float, y: float, theta: float,
                            v: float = None) -> np.ndarray:
        state_vector = np.zeros(self.n_states)
        state_vector[self.state_idx['x']] = x
        state_vector[self.state_idx['y']] = y
        state_vector[self.state_idx['theta']] = theta
        self._set_speed(v, state_vector)
        return state_vector

    def shift_initial_state(self, shift: dict[str, float]) -> None:
        """
        Shifts the initial state based on the given values
        :param shift: dictionary with state name and shift value
        :return:
        """
        for state_name, value in shift.items():
            original = self._initial_state[self.state_idx[state_name]]
            shifted = np.round(original + value, 4)
            self._initial_state[self.state_idx[state_name]] = shifted

    def compute_lane_keeping_safe_gap(self, v_ego: float) -> float:
        return self.compute_safe_gap(v_ego, self.base_vehicle.get_h_safe_lk())

    def compute_lane_changing_safe_gap(self, v_ego: float) -> float:
        return self.compute_safe_gap(v_ego, self.base_vehicle.get_h_safe_lc())

    def compute_safe_gap(self, v_ego: float, safe_h: float) -> float:
        return safe_h * v_ego + self.base_vehicle.c

    def compute_error_to_safe_gap(
            self, x_ego: float, v_ego: float,
            x_leader: float, has_lane_change_intention: bool) -> float:
        gap = x_leader - x_ego
        if has_lane_change_intention:
            safe_gap = self.compute_lane_changing_safe_gap(v_ego)
        else:
            safe_gap = self.compute_lane_keeping_safe_gap(v_ego)
        return gap - safe_gap

    def compute_derivatives(self, ego_states, inputs, leader_states):
        dxdt = np.zeros(self.n_states)
        theta = self.select_state_from_vector(ego_states, 'theta')
        vel = self.select_vel_from_vector(ego_states, inputs)
        accel = self.get_accel(ego_states, inputs, leader_states)
        phi = self.get_phi(inputs)
        self._compute_derivatives(vel, theta, phi, accel, dxdt)
        return dxdt

    def _position_derivative_longitudinal_only(self, vel, derivatives):
        dx, dy, dtheta = dynamics.position_derivative_longitudinal_only(vel)
        derivatives[self.state_idx['x']] = dx
        derivatives[self.state_idx['y']] = dy
        derivatives[self.state_idx['theta']] = dtheta

    def _position_derivative_cg(self, vel: float, theta: float, phi: float,
                                derivatives) -> None:
        dx, dy, dtheta = dynamics.position_derivative_cg(
            vel, theta, phi, self.base_vehicle.lf, self.base_vehicle.lr)
        derivatives[self.state_idx['x']] = dx
        derivatives[self.state_idx['y']] = dy
        derivatives[self.state_idx['theta']] = dtheta

    def _position_derivative_rear_wheels(self, vel: float, theta: float,
                                         phi: float, derivatives):
        dx, dy, dtheta = dynamics.position_derivative_rear_wheels(
            vel, theta, phi, self.base_vehicle.wheelbase)
        derivatives[self.state_idx['x']] = dx
        derivatives[self.state_idx['y']] = dy
        derivatives[self.state_idx['theta']] = dtheta

    def to_dataframe(self, time, states, inputs):
        data = np.concatenate([time.reshape(1, -1), states, inputs])
        columns = (['t'] + [s for s in self._state_names]
                   + [i for i in self._input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self.get_id()
        df['name'] = self.get_name()
        df['orig_lane_leader_id'] = self._origin_leader_id
        df['dest_lane_leader_id'] = self._destination_leader_id
        df['dest_lane_follower_id'] = self._destination_follower_id
        return df

    @abstractmethod
    def _set_speed(self, v0, state):
        """
        Sets the proper element in array state equal to v0
        :param v0: speed to write
        :param state: state vector being modified
        :return: nothing, modifies the state array
        """
        pass

    @abstractmethod
    def get_desired_input(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_input_limits(self) -> (list[float], list[float]):
        """

        :return: Tuple with lower limits first and upper limits second
        """
        pass

    @abstractmethod
    def _compute_derivatives(self, vel, theta, phi, accel, derivatives):
        """ Computes the derivatives of x, y, and theta, and stores them in the
         derivatives array """
        pass

    @abstractmethod
    def get_accel(self, ego_states, inputs, leader_states) -> float:
        pass

    # def _set_model_old(self):
    #     """
    #     Must be called in the constructor of every derived class to set the
    #     variables that define which vehicle model is implemented.
    #     :return:
    #     """
    #     self.n_states = len(self._state_names)
    #     # self.state_names = state_names
    #     self.n_inputs = len(self._input_names)
    #     # self.input_names = input_names
    #     self.state_idx = {self._state_names[i]: i for i
    #                       in range(self.n_states)}
    #     self.optimal_input_idx = {self._input_names[i]: i for i in
    #                               range(self.n_inputs)}


V = TypeVar('V', bound=BaseVehicle)
