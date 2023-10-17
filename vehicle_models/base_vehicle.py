from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Dict, Iterable, List, TypeVar, Type, Union
import warnings

import numpy as np
import pandas as pd

import constants as const
import system_operating_mode as som
import vehicle_operating_modes.base_operating_modes as modes


class BaseVehicle(ABC):
    _counter = 0

    free_flow_speed: float
    initial_state: np.ndarray
    target_lane: int
    state_names: List[str]
    input_names: List[str]
    _n_states: int
    _n_inputs: int
    _state_idx: Dict[str, int]
    _input_idx: Dict[str, int]
    _time: np.ndarray
    _states: np.ndarray
    _inputs: np.ndarray
    _iter_counter: int
    _states_history: np.ndarray
    _inputs_history: np.ndarray
    _orig_leader_id: np.ndarray[int]
    _destination_leader_id: np.ndarray[int]
    _destination_follower_id: np.ndarray[int]
    _incoming_vehicle_id: np.ndarray[int]
    _leader_id: np.ndarray[int]  # vehicle used to determine current accel
    _derivatives: np.ndarray[float]
    _mode: modes.VehicleMode
    _ocp_interface: Type[BaseVehicleInterface]
    _desired_future_follower_id: int
    _long_adjust_start_time: float
    _lc_start_time: float

    # Used when we need to copy a vehicle
    static_attribute_names = {
        'free_flow_speed', '_id', '_name', 'lr', 'lf', '_wheelbase',
        'phi_max', 'h', 'safe_h', 'h', 'c', '_is_connected', '_is_verbose',
    }

    def __init__(self):
        """

        """

        # self.free_flow_speed = None
        # self.initial_state = None
        # self.target_lane = None
        # self.target_y = None
        # self.state_names, self.input_names = None, None
        # self._n_states, self._n_inputs = None, None
        # self._state_idx, self._input_idx = {}, {}

        # Some parameters
        self._id: int = BaseVehicle._counter
        BaseVehicle._counter += 1
        self._name: str = str(self._id)  # default
        self.lr = 2  # dist from C.G. to rear wheel
        self.lf = 1  # dist from C.G. to front wheel
        self._wheelbase = self.lr + self.lf
        self.phi_max = 0.1  # max steering wheel angle
        # self._lateral_gain = 1
        self._lc_duration = 5  # [s]
        # Note: h and safe_h are a simplification of the system. The time
        # headway used for vehicle following is computed in a way to
        # overestimate the nonlinear safe distance. In the above, we just
        # assume the safe distance is also linear and with a smaller h.
        self.h = const.TIME_HEADWAY
        self.safe_h = const.SAFE_LC_TIME_HEADWAY
        self.c = const.STANDSTILL_DISTANCE

        self._is_connected = False
        self._is_verbose = True

    def __repr__(self):
        return self.__class__.__name__ + ' id=' + str(self._id)

    def __str__(self):
        return self.__class__.__name__ + ": " + str(self._name)

    @staticmethod
    def reset_vehicle_counter():
        BaseVehicle._counter = 0

    @staticmethod
    def copy_from_other(vehicle_to: BaseVehicle, vehicle_from: BaseVehicle):
        for attr_name in vehicle_from.static_attribute_names:
            setattr(vehicle_to, attr_name, getattr(vehicle_from, attr_name))
        vehicle_to.copy_initial_state(vehicle_from.get_states())

    def get_id(self) -> int:
        return self._id

    def get_name(self) -> str:
        return self._name

    def get_ocp_interface(self) -> BaseVehicleInterface:
        return self._ocp_interface(self)

    def get_current_time(self) -> float:
        return self._time[self._iter_counter]

    def get_x(self) -> float:
        return self.get_a_state_by_name('x')

    def get_y(self) -> float:
        return self.get_a_state_by_name('y')

    def get_theta(self) -> float:
        return self.get_a_state_by_name('theta')

    def get_a_state_by_name(self, state_name) -> float:
        return self._states[self._state_idx[state_name]]

    def get_an_input_by_name(self, input_name) -> float:
        return self._inputs[self._input_idx[input_name]]

    def get_current_lane(self) -> int:
        return round(self.get_y() / const.LANE_WIDTH)

    def get_target_y(self) -> float:
        return self.target_lane * const.LANE_WIDTH

    def get_states(self) -> np.ndarray:
        return self._states.copy()

    def get_state_history(self) -> np.ndarray:
        return self._states_history

    def get_inputs(self) -> np.ndarray:
        return self._inputs.copy()

    def get_input_history(self) -> np.ndarray:
        return self._inputs_history

    def get_derivatives(self) -> np.ndarray:
        return self._derivatives

    def get_orig_lane_leader_id(self) -> int:
        return self._orig_leader_id[self._iter_counter]

    def get_dest_lane_leader_id(self) -> int:
        return self._destination_leader_id[self._iter_counter]

    def get_dest_lane_follower_id(self) -> int:
        return self._destination_follower_id[self._iter_counter]

    def get_incoming_vehicle_id(self) -> int:
        return self._incoming_vehicle_id[self._iter_counter]

    def get_relevant_surrounding_vehicle_ids(self):
        """
        Returns the IDs relevant vehicles
        :return:
        """
        return {'lo': self.get_orig_lane_leader_id(),
                'ld': self.get_dest_lane_leader_id(),
                'fd': self.get_dest_lane_follower_id(),
                'leader': self.get_current_leader_id()}

    def get_possible_target_leader_ids(self) -> List[int]:
        """
        Returns the IDs of all vehicles possibly used as target leaders
        :return:
        """
        return [self.get_orig_lane_leader_id(),
                self.get_dest_lane_leader_id(),
                self.get_incoming_vehicle_id()]

    def get_desired_future_follower_id(self) -> int:
        return self._desired_future_follower_id

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
            return self._orig_leader_id[self._iter_counter]

    def get_platoon(self):
        return None

    # TODO: figure out more descriptive name. This is a copy used for iterative
    #  simulations
    def make_reset_copy(self, initial_state=None,
                        new_vehicle_type: Type[BaseVehicle] = None) -> V:
        """
        Creates copies of vehicles used in internal iterations of our optimal
        controller. For vehicles without optimal control, the method returns
        an "empty" copy of the vehicle (without any state history). For
        vehicles with optimal control, the method returns the equivalent
        open loop type vehicle.
        :param initial_state: If None, sets the new vehicle's initial state
        equal to the most recent state of this instance
        :param new_vehicle_type: If None, the new vehicle is of the same type
        as this instance
        :return:
        """
        # new_vehicle = copy.deepcopy(self)
        # self._reset_copied_vehicle(new_vehicle, initial_state)
        if new_vehicle_type is None:
            new_vehicle_type = type(self)
        new_vehicle = new_vehicle_type()
        for attr_name in self.static_attribute_names:
            setattr(new_vehicle, attr_name, getattr(self, attr_name))
        if initial_state is None:
            initial_state = self.get_states()
        new_vehicle.copy_initial_state(initial_state)
        new_vehicle.target_lane = self.target_lane
        return new_vehicle

    def make_open_loop_copy(self, initial_state=None) -> V:
        """
        Creates copies of vehicles used in internal iterations of our optimal
        controller. For vehicles without optimal control, the method returns
        an "empty" copy of the vehicle (without any state history). For
        vehicles with optimal control, the method returns the equivalent
        open loop type vehicle.
        :return:
        """
        return self.make_reset_copy(initial_state, type(self))

    def _reset_copied_vehicle(self, new_vehicle: BaseVehicle,
                              initial_state=None):
        if initial_state is None:
            initial_state = self.get_states()
        # x = initial_state[self._state_idx['x']]
        # y = initial_state[self._state_idx['y']]
        # theta = initial_state[self._state_idx['theta']]
        # try:
        #     v = initial_state[self._state_idx['v']]
        # except KeyError:
        #     v = None
        new_vehicle.copy_initial_state(initial_state)
        new_vehicle.target_lane = self.target_lane
        new_vehicle.reset_simulation_logs()

    def set_free_flow_speed(self, v_ff: float):
        self.free_flow_speed = v_ff

    def set_name(self, value: str):
        self._name = value

    def set_verbose(self, value: bool):
        self._is_verbose = value

    def set_initial_state(self, x: float, y: float, theta: float,
                          v: float = None):
        self.initial_state = self.create_state_vector(x, y, theta, v)
        self._states = self.initial_state
        self.target_lane = self.get_current_lane()

    def copy_initial_state(self, initial_state: np.ndarray):
        if len(initial_state) != self._n_states:
            raise ValueError('Wrong size of initial state vector ({} instead'
                             'of {})'.format(len(initial_state),
                                             self._n_states))
        self.initial_state = initial_state
        self._states = self.initial_state

    def set_mode(self, mode: modes.VehicleMode):
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

    def set_lane_change_direction(self, lc_direction):
        self.target_lane = self.get_current_lane() + lc_direction

    def write_state_and_input(self, time: float, states: np.ndarray,
                              optimal_inputs: np.ndarray):
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
        self._states = states
        self._states_history[:, self._iter_counter] = self._states
        self._inputs = np.zeros(self._n_inputs)
        if len(optimal_inputs) > 0:
            self._write_optimal_inputs(optimal_inputs)
        self._inputs_history[:, self._iter_counter] = self._inputs

    def prepare_to_start_simulation(self, n_samples: int):
        """
        Erases all simulation data and allows the vehicle trajectory to be
        simulated again
        """
        self.initialize_simulation_logs(n_samples)
        # TODO: maybe all the settings below can be in the constructor
        #  It depends on whether we need to "restart" vehicles
        self._desired_future_follower_id = -1
        self._long_adjust_start_time = -np.inf
        self._lc_start_time = -np.inf

        # Initial state
        self._time[self._iter_counter] = 0.0
        self._states_history[:, self._iter_counter] = self.initial_state

    def initialize_simulation_logs(self, n_samples: int):
        self._iter_counter = 0
        self._time = np.zeros(n_samples)
        self._states_history = np.zeros([self._n_states, n_samples])
        self._inputs_history = np.zeros([self._n_inputs, n_samples])
        self._orig_leader_id = -np.ones(n_samples, dtype=int)
        self._destination_leader_id = -np.ones(n_samples, dtype=int)
        self._destination_follower_id = -np.ones(n_samples, dtype=int)
        self._incoming_vehicle_id = -np.ones(n_samples, dtype=int)
        self._leader_id = -np.ones(n_samples, dtype=int)

    def reset_simulation_logs(self):
        self.initialize_simulation_logs(0)

    def reset_lane_change_start_time(self):
        self._lc_start_time = -np.inf

    def has_orig_lane_leader(self):
        try:
            return self.get_orig_lane_leader_id() >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(orig lane leader id) before simulation start")
            return False

    def has_dest_lane_leader(self):
        try:
            return self.get_dest_lane_leader_id() >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(dest lane leader id) before simulation start")
            return False

    def has_dest_lane_follower(self):
        try:
            return self.get_dest_lane_follower_id() >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(dest lane follower id) before simulation start")
            return False

    def is_cooperating(self):
        return self.get_incoming_vehicle_id() >= 0

    def has_leader(self):
        return self.get_current_leader_id() >= 0

    def has_changed_leader(self):
        try:
            return (self._leader_id[self._iter_counter - 1]
                    != self._leader_id[self._iter_counter])
        except IndexError:
            warnings.warn("Warning: trying to check if leader has changed too"
                          "early (before two simulation steps")
            return False

    def has_lane_change_intention(self):
        return self.get_current_lane() != self.target_lane

    def has_requested_cooperation(self):
        return self.get_desired_future_follower_id() >= 0

    def make_connected(self):
        self._is_connected = True

    def is_in_a_platoon(self) -> bool:
        return not (self.get_platoon() is None)

    def find_orig_lane_leader(self, vehicles: Iterable[BaseVehicle]):
        ego_x = self.get_x()
        ego_y = self.get_y()
        orig_lane_leader_x = np.inf
        new_orig_leader_id = -1
        for other_vehicle in vehicles:
            other_x = other_vehicle.get_x()
            other_y = other_vehicle.get_y()
            if (np.abs(other_y - ego_y) < const.LANE_WIDTH / 2  # same lane
                    and ego_x < other_x < orig_lane_leader_x):
                orig_lane_leader_x = other_x
                new_orig_leader_id = other_vehicle._id
        self._orig_leader_id[self._iter_counter] = new_orig_leader_id

    def find_dest_lane_vehicles(self, vehicles: Iterable[BaseVehicle]):
        new_dest_leader_id = -1
        new_dest_follower_id = -1
        if self.has_lane_change_intention():
            y_target_lane = self.target_lane * const.LANE_WIDTH
            ego_x = self.get_x()
            dest_lane_follower_x = -np.inf
            dest_lane_leader_x = np.inf
            for other_vehicle in vehicles:
                other_x = other_vehicle.get_x()
                other_y = other_vehicle.get_y()
                if np.abs(other_y - y_target_lane) < const.LANE_WIDTH / 2:
                    if ego_x < other_x < dest_lane_leader_x:
                        dest_lane_leader_x = other_x
                        new_dest_leader_id = other_vehicle._id
                    elif dest_lane_follower_x < other_x < ego_x:
                        dest_lane_follower_x = other_x
                        new_dest_follower_id = other_vehicle._id
        self._destination_leader_id[self._iter_counter] = new_dest_leader_id
        self._destination_follower_id[self._iter_counter] = new_dest_follower_id

    def find_cooperation_requests(self, vehicles: Iterable[BaseVehicle]):
        new_incoming_vehicle_id = -1
        incoming_veh_x = np.inf
        if self._is_connected:
            for other_vehicle in vehicles:
                other_request = other_vehicle._desired_future_follower_id
                other_x = other_vehicle.get_x()
                if other_request == self._id and other_x < incoming_veh_x:
                    new_incoming_vehicle_id = other_vehicle._id
                    incoming_veh_x = other_x
        self._incoming_vehicle_id[self._iter_counter] = new_incoming_vehicle_id

    def analyze_platoons(self, vehicles: Dict[int, BaseVehicle]):
        pass

    def join_centrally_controlled_platoon(self,
                                          vehicles: Dict[int, BaseVehicle]):
        pass

    def request_cooperation(self):
        if self._is_connected:
            self._desired_future_follower_id = self.get_dest_lane_follower_id()

    def receive_cooperation_request(self, other_id):
        if self._is_connected:
            self._incoming_vehicle_id = other_id

    def update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Defines which surrounding vehicle should be used to determine this
        vehicle's own acceleration
        :return:
        """
        self._update_target_leader(vehicles)

    @staticmethod
    def compute_a_gap(leading_vehicle: BaseVehicle,
                      following_vehicle: BaseVehicle):
        return (leading_vehicle.get_x()
                - following_vehicle.get_x())

    def compute_safe_gap(self, v_ego=None):
        if v_ego is None:
            v_ego = self.get_vel()
        return self.safe_h * v_ego + self.c

    def compute_free_flow_desired_gap(self):
        return self.compute_desired_gap(self.free_flow_speed)

    def compute_desired_gap(self, vel: float = None):
        if vel is None:
            vel = self.get_vel()
        return self.h * vel + self.c

    def compute_derivatives(self):
        self._derivatives = np.zeros(self._n_states)
        theta = self.get_theta()
        phi = self.get_an_input_by_name('phi')
        vel = self.get_vel()
        self._compute_derivatives(vel, theta, phi)

    def determine_inputs(self, open_loop_controls: np.ndarray,
                         vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop controls and computes the closed loop controls.
        :param open_loop_controls: Dictionary whose keys are the input name.
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs = np.zeros(self._n_inputs)
        self._determine_inputs(open_loop_controls, vehicles)
        self._inputs_history[:, self._iter_counter] = self._inputs

    def update_states(self, next_time):
        dt = next_time - self.get_current_time()
        self._states = self._states + self._derivatives * dt
        self._states_history[:, self._iter_counter + 1] = self._states
        self._iter_counter += 1
        self._time[self._iter_counter] = next_time

    @abstractmethod
    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        # only vehicles with controllers update their discrete states (modes)
        pass

    # TODO: duplicated at interface
    def _position_derivative_longitudinal_only(self, vel):
        self._derivatives[self._state_idx['x']] = vel
        self._derivatives[self._state_idx['y']] = 0.
        self._derivatives[self._state_idx['theta']] = 0.

    # TODO: duplicated at interface
    def _position_derivative_cg(self, vel: float, theta: float, phi: float
                                ) -> None:

        beta = np.arctan(self.lr * np.tan(phi) / (self.lf + self.lr))
        self._derivatives[self._state_idx['x']] = vel * np.cos(theta + beta)
        self._derivatives[self._state_idx['y']] = vel * np.sin(theta + beta)
        self._derivatives[self._state_idx['theta']] = (vel * np.sin(beta)
                                                       / self.lr)

    # TODO: duplicated at interface
    def _position_derivative_rear_wheels(self, vel: float, theta: float,
                                         phi: float):
        self._derivatives[self._state_idx['x']] = vel * np.cos(theta)
        self._derivatives[self._state_idx['y']] = vel * np.sin(theta)
        self._derivatives[self._state_idx['theta']] = (vel * np.tan(phi)
                                                       / self._wheelbase)

    # TODO: duplicated at interface
    def create_state_vector(self, x: float, y: float, theta: float,
                            v: float = None):
        state_vector = np.zeros(self._n_states)
        state_vector[self._state_idx['x']] = x
        state_vector[self._state_idx['y']] = y
        state_vector[self._state_idx['theta']] = theta
        self._set_speed(v, state_vector)
        return state_vector

    def to_dataframe(self) -> pd.DataFrame:
        data = np.concatenate([self._time.reshape(1, -1),
                               self._states_history, self._inputs_history])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self._id
        df['name'] = self._name
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'orig_lane_leader_id', self._orig_leader_id)
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'dest_lane_leader_id', self._destination_leader_id)
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'dest_lane_follower_id', self._destination_follower_id)
        return df

    def prepare_for_longitudinal_adjustments_start(
            self, vehicles: Dict[int, BaseVehicle]):
        self._long_adjust_start_time = self.get_current_time()
        self._set_up_longitudinal_adjustments_control(vehicles)

    def prepare_for_lane_change_start(self):
        self._lc_start_time = self.get_current_time()
        self._set_up_lane_change_control()

    def _set_model(self, state_names: List[str], input_names: List[str]):
        """
        Must be called in the constructor of every derived class to set the
        variables that define which vehicle model is implemented.
        :param state_names: Names of the state variables
        :param input_names: Names of the input variables
        :return:
        """
        self._n_states = len(state_names)
        self.state_names = state_names
        self._n_inputs = len(input_names)
        self.input_names = input_names
        self._state_idx = {state_names[i]: i for i in range(self._n_states)}
        self._input_idx = {input_names[i]: i for i in range(self._n_inputs)}
        # self._derivatives = np.zeros(self._n_states)

    def follow_orig_lane_leader(self):
        self._set_current_leader_id(self.get_orig_lane_leader_id())

    @abstractmethod
    def get_vel(self):
        pass

    @staticmethod
    def _set_surrounding_vehicles_ids_to_df(df, col_name, col_value):
        if len(col_value) == 1:
            df[col_name] = col_value[0]
        else:
            df[col_name] = col_value

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
    def _compute_derivatives(self, vel, theta, phi):
        """ Computes the derivatives of x, y, and theta, and stores them in the
         derivatives array """
        pass

    @abstractmethod
    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop controls and computes the closed loop controls.
        :param open_loop_controls: Dictionary whose keys are the input name.
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        pass

    @abstractmethod
    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, BaseVehicle]):
        pass

    @abstractmethod
    def _set_up_lane_change_control(self):
        pass

    @abstractmethod
    def _update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        pass

    def _write_optimal_inputs(self, optimal_inputs):
        """
        Used when vehicle states were computed externally.
        :return: Nothing
        """
        pass

    def _set_current_leader_id(self, veh_id):
        """
        Sets which vehicle used to determine this vehicle's accel. The
        definition of leader ids for all vehicles in a vehicle group determines
        the operating mode.
        :param veh_id: leading vehicle's id. Use -1 to designate no leader
        :return:
        """
        self._leader_id[self._iter_counter] = veh_id


# TODO: still must figure out a way to prevent so much repeated code between
#  vehicles and their interfaces with the opc solver
class BaseVehicleInterface(ABC):

    state_names: List[str]
    n_states: int
    input_names: List[str]
    n_inputs: int
    state_idx: Dict[str, int]
    optimal_input_idx: Dict[str, int]

    def __init__(self, vehicle: BaseVehicle):
        """

        """
        # self.state_names, self.n_states = None, None
        # self.input_names, self.n_inputs = None, None
        # self.state_idx, self.optimal_input_idx = {}, {}
        self._time: float = 0.

        # Copy values from the vehicle
        # for attr_name in vehicle.static_attribute_names:
        #     setattr(self, attr_name, getattr(vehicle, attr_name))
        self._id: int = vehicle.get_id()
        self._name: str = vehicle.get_name()
        self.lr: float = vehicle.lr  # dist from C.G. to rear wheel
        self.lf: float = vehicle.lf  # dist from C.G. to front wheel
        self._wheelbase: float = self.lr + self.lf
        self.phi_max: float = vehicle.phi_max  # max steering wheel angle
        self.safe_h: float = vehicle.safe_h
        self.c: float = vehicle.c  # standstill distance [m]
        self.free_flow_speed: float = vehicle.free_flow_speed
        self._origin_leader_id: int = vehicle.get_orig_lane_leader_id()
        self._destination_leader_id: int = vehicle.get_dest_lane_leader_id()
        self._destination_follower_id: int = vehicle.get_dest_lane_follower_id()
        self._leader_id: int = vehicle.get_current_leader_id()
        self.target_lane: int = vehicle.target_lane
        # The vehicle's current state is the starting point for the ocp
        self._initial_state = vehicle.get_states()

        # Only set for some vehicle types
        self._interval_number = 0
        # TODO: lump all these in a single dictionary
        self.ocp_mode_switch_times: List[float] = []
        self.ocp_origin_leader_sequence: List[int] = []
        self.ocp_destination_leader_sequence: List[int] = []
        self.ocp_destination_follower_sequence: List[int] = []
        self.ocp_target_leader_sequence: List[int] = []

    def __repr__(self):
        return self.__class__.__name__ + ' id=' + str(self._id)

    def __str__(self):
        return (self.__class__.__name__ + ": id=" + str(self._id)
                + "V_f=" + str(self.free_flow_speed))

    def get_id(self) -> int:
        return self._id

    def get_name(self) -> str:
        return self._name

    def get_initial_state(self):
        return self._initial_state

    def get_x0(self):
        return self.select_state_from_vector(self._initial_state, 'x')

    def get_y0(self):
        return self.select_state_from_vector(self._initial_state, 'y')

    def get_orig_lane_leader_id(self, time: float):
        if len(self.ocp_origin_leader_sequence) == 0:
            return self._origin_leader_id
        self.set_time_interval(time)
        return self.ocp_origin_leader_sequence[self._interval_number]

    def get_dest_lane_leader_id(self, time: float):
        if len(self.ocp_destination_leader_sequence) == 0:
            return self._destination_leader_id
        self.set_time_interval(time)
        return self.ocp_destination_leader_sequence[self._interval_number]

    def get_dest_lane_follower_id(self, time: float):
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

    def has_orig_lane_leader(self, time: float) -> bool:
        return self.get_orig_lane_leader_id(time) >= 0

    def has_dest_lane_leader(self, time: float) -> bool:
        return self.get_dest_lane_leader_id(time) >= 0

    def has_dest_lane_follower(self, time: float) -> bool:
        return self.get_dest_lane_follower_id(time) >= 0

    def has_leader(self, time: float) -> bool:
        return self.get_current_leader_id(time) >= 0

    def get_target_y(self) -> float:
        return self.target_lane * const.LANE_WIDTH

    def is_long_control_optimal(self) -> bool:
        return 'a' in self.input_names or 'v' in self.input_names

    def set_leader_sequence(self, leader_sequence: som.SVSequence
                            ) -> None:
        self._interval_number = 0
        self.ocp_mode_switch_times: List[float] = []
        self.ocp_origin_leader_sequence: List[int] = []
        self.ocp_destination_leader_sequence: List[int] = []
        self.ocp_destination_follower_sequence: List[int] = []
        self.ocp_target_leader_sequence: List[int] = []
        for t, l_id in leader_sequence:
            self.ocp_mode_switch_times.append(t)
            self.ocp_origin_leader_sequence.append(l_id['lo'])
            self.ocp_destination_leader_sequence.append(l_id['ld'])
            self.ocp_destination_follower_sequence.append(l_id['fd'])
            self.ocp_target_leader_sequence.append(l_id['leader'])

    # TODO: maybe most of these methods could be class methods since they don't
    #  depend on any 'internal' value of the instance
    def select_state_from_vector(self, states: Union[np.ndarray, List],
                                 state_name: str) -> float:
        return states[self.state_idx[state_name]]

    def select_input_from_vector(self, optimal_inputs: Union[np.ndarray, List],
                                 input_name: str) -> float:
        return optimal_inputs[self.optimal_input_idx[input_name]]

    def select_vel_from_vector(self, states: Union[np.ndarray, List],
                               inputs: Union[np.ndarray, List]):
        try:
            return states[self.state_idx['v']]
        except KeyError:
            return inputs[self.optimal_input_idx['v']]

    def set_time_interval(self, time: float) -> None:
        # TODO: figure out what's wrong with the new approach
        # n_switches = len(self.ocp_mode_switch_times) - 1
        # # Reset interval number when the simulation is restarted
        # if np.isclose(time, self.ocp_mode_switch_times[0]):
        #     self._interval_number = 0
        # # Updated interval number as time advances
        # if (n_switches > self._interval_number
        #         and time >= self.ocp_mode_switch_times[
        #             self._interval_number + 1]):
        #     self._interval_number += 1
        # Old approach: search every time
        if not np.isclose(self._time, time):
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

    def shift_initial_state(self, shift: Dict[str, float]) -> None:
        """
        Shifts the initial state based on the given values
        :param shift: Dictionary with state name and shift value
        :return:
        """
        for state_name, value in shift.items():
            original = self._initial_state[self.state_idx[state_name]]
            shifted = np.round(original + value, 4)
            self._initial_state[self.state_idx[state_name]] = shifted

    def compute_safe_gap(self, v_ego) -> float:
        return self.safe_h * v_ego + self.c

    def compute_error_to_safe_gap(self, ego_states, leader_x) -> float:
        gap = leader_x - self.select_state_from_vector(ego_states, 'x')
        safe_gap = self.compute_safe_gap(self.select_state_from_vector(
            ego_states, 'v'))
        return gap - safe_gap

    # def lane_changing_safety_constraint(self, ego_states, other_states, phi):
    #     if not other_states:  # no risk
    #         return 0
    #     if is_other_behind:
    #         follower_id, leader_id = other_id, lc_veh_id
    #     else:
    #         follower_id, leader_id = lc_veh_id, other_id

    def compute_derivatives(self, ego_states, inputs, leader_states):
        dxdt = np.zeros(self.n_states)
        theta = self.select_state_from_vector(ego_states, 'theta')
        phi = self.select_input_from_vector(inputs, 'phi')
        vel = self.select_vel_from_vector(ego_states, inputs)
        accel = self.compute_acceleration(ego_states, inputs, leader_states)
        self._compute_derivatives(vel, theta, phi, accel, dxdt)
        return dxdt

    # @abstractmethod
    # def determine_inputs(self, ego_states, optimal_inputs, leader_states
    #                      ) -> Tuple[float, float]:
    #     """
    #     Returns accel and phi, which are computed internally by a feedback
    #     law or read from the optimal inputs vector depending on the vehicle
    #     type
    #     """
    #     pass

    def _position_derivative_longitudinal_only(self, vel, derivatives):
        derivatives[self.state_idx['x']] = vel
        derivatives[self.state_idx['y']] = 0.
        derivatives[self.state_idx['theta']] = 0.

    def _position_derivative_cg(self, vel: float, theta: float, phi: float,
                                derivatives) -> None:
        beta = np.arctan(self.lr * np.tan(phi) / (self.lf + self.lr))
        derivatives[self.state_idx['x']] = vel * np.cos(theta + beta)
        derivatives[self.state_idx['y']] = vel * np.sin(theta + beta)
        derivatives[self.state_idx['theta']] = (vel * np.sin(beta)
                                                / self.lr)

    def _position_derivative_rear_wheels(self, vel: float, theta: float,
                                         phi: float, derivatives):
        derivatives[self.state_idx['x']] = vel * np.cos(theta)
        derivatives[self.state_idx['y']] = vel * np.sin(theta)
        derivatives[self.state_idx['theta']] = (vel * np.tan(phi)
                                                / self._wheelbase)

    def to_dataframe(self, time, states, inputs):
        data = np.concatenate([time.reshape(1, -1), states, inputs])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self._id
        df['name'] = self._name
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
    def get_initial_input_guess(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_input_limits(self) -> (List[float], List[float]):
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
    def compute_acceleration(self, ego_states, inputs, leader_states):
        pass

    def _set_model(self, state_names: List[str], input_names: List[str]):
        """
        Must be called in the constructor of every derived class to set the
        variables that define which vehicle model is implemented.
        :param state_names: Names of the state variables
        :param input_names: Names of the input variables
        :return:
        """
        self.n_states = len(state_names)
        self.state_names = state_names
        self.n_inputs = len(input_names)
        self.input_names = input_names
        self.state_idx = {state_names[i]: i for i in range(self.n_states)}
        self.optimal_input_idx = {input_names[i]: i for i in
                                  range(self.n_inputs)}
        self._states = np.zeros(self.n_states)
        self._derivatives = np.zeros(self.n_states)


V = TypeVar('V', bound=BaseVehicle)
