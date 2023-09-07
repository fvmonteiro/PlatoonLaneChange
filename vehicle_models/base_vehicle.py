from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple, Type
import warnings

import numpy as np
import pandas as pd

import constants as const
import vehicle_operating_modes.base_operating_modes as modes
# import vehicle_ocp_interface as vi


class BaseVehicle(ABC):
    _counter = 0

    time: np.ndarray
    _states: np.ndarray
    _inputs: np.ndarray
    _iter_counter: int
    _states_history: np.ndarray
    _inputs_history: np.ndarray
    _derivatives: np.ndarray
    _mode: modes.VehicleMode
    _ocp_interface: Type[BaseVehicleInterface]

    def __init__(self):
        """

        """

        self.free_flow_speed = None
        self.initial_state = None
        self.target_lane = None
        self.target_y = None
        self.state_names, self.input_names = None, None
        self._n_states, self._n_inputs = None, None
        self._state_idx, self._input_idx = {}, {}
        self._orig_leader_id: List[int] = []
        self._destination_leader_id: List[int] = []
        self._destination_follower_id: List[int] = []
        self._incoming_vehicle_id: List[int] = []
        # Vehicle used to determine current accel
        self._leader_id: List[int] = []
        self._desired_future_follower_id = -1
        self._polynomial_lc_coeffs = None
        self._long_adjust_start_time = -np.inf
        self._lc_start_time = -np.inf
        self._platoon_id = -1

        # For when we want to predefine who should be followed. Used in
        # optimal control problems
        self.ocp_leader_switch_times: List[float] = []
        self.ocp_leader_sequence: List[int] = []

        # Some parameters
        self.id: int = BaseVehicle._counter
        BaseVehicle._counter += 1
        self.name: str = str(self.id)  # default
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

    def __repr__(self):
        return self.__class__.__name__ + ' id=' + str(self.id)

    def __str__(self):
        return self.__class__.__name__ + ": " + str(self.name)

    @staticmethod
    def reset_vehicle_counter():
        BaseVehicle._counter = 0

    def get_ocp_interface(self):
        return self._ocp_interface(self)

    def get_current_time(self):
        return self.time[self._iter_counter]

    def get_x(self):
        return self.get_a_state_by_name('x')

    def get_y(self):
        return self.get_a_state_by_name('y')

    def get_theta(self):
        return self.get_a_state_by_name('theta')

    def get_a_state_by_name(self, state_name):
        return self._states[self._state_idx[state_name]]

    def get_an_input_by_name(self, input_name):
        return self._inputs[self._input_idx[input_name]]

    def get_current_lane(self):
        return round(self.get_y() / const.LANE_WIDTH)

    def get_states(self):
        return self._states

    def get_state_history(self):
        return self._states_history

    def get_input_history(self):
        return self._inputs_history

    def get_derivatives(self):
        return self._derivatives

    def get_orig_lane_leader_id(self):
        return self._orig_leader_id[-1]

    def get_dest_lane_leader_id(self):
        return self._destination_leader_id[-1]

    def get_dest_lane_follower_id(self):
        return self._destination_follower_id[-1]

    def get_incoming_vehicle_id(self):
        return self._incoming_vehicle_id[-1]

    def get_desired_future_follower_id(self):
        return self._desired_future_follower_id

    def get_current_leader_id(self):
        """
        The 'current' leader is the vehicle being used to define this vehicle's
         acceleration
        :return:
        """
        if len(self.ocp_leader_sequence) == 0:
            try:
                return self._leader_id[-1]
            except IndexError:
                # Some vehicles never set a target leader because they do not
                # have closed-loop acceleration control
                return self._orig_leader_id[-1]
        else:
            time = self.get_current_time()
            idx = np.searchsorted(self.ocp_leader_switch_times, time,
                                  side='right')
            return self.ocp_leader_sequence[idx-1]

    def get_platoon_id(self):
        return self._platoon_id

    def set_free_flow_speed(self, v_ff: float):
        self.free_flow_speed = v_ff

    def set_initial_state(self, x: float, y: float, theta: float,
                          v: float = None):
        self.initial_state = self.create_state_vector(x, y, theta, v)
        self._states = self.initial_state
        self.target_lane = round(y / const.LANE_WIDTH)
        self.target_y = y

    def set_mode(self, mode: modes.VehicleMode):
        try:
            print("t={:.2f}, veh {}. From: {} to {}".format(
                self.get_current_time(), self.id, self._mode, mode))
        except AttributeError:  # when the mode is set for the first time
            print("t=0.0, veh {}. Initial op mode: {}".format(
                self.id, mode))
        self._mode = mode
        self._mode.set_ego_vehicle(self)

    def set_lane_change_direction(self, lc_direction):
        self.target_lane = (
                    round(self.get_y() / const.LANE_WIDTH)
                    + lc_direction)

    def set_current_leader_id(self, veh_id):
        """
        Sets which vehicle used to determine this vehicle's accel. The
        definition of leader ids for all vehicles in a vehicle group determines
        the operating mode.
        :param veh_id: leading vehicle's id. Use -1 to designate no leader
        :return:
        """
        self._leader_id.append(veh_id)

    def set_ocp_leader_sequence(self, leader_sequence: List[Tuple[float, int]]):
        """
        For optimal control problems, sets the desired/tested
        sequence of leaders throughout the solver horizon
        """

        for t, l_id in leader_sequence:
            self.ocp_leader_switch_times.append(t)
            self.ocp_leader_sequence.append(l_id)

    def initialize_simulation_logs(self, n_samples):
        self._iter_counter = 0
        self.time = np.zeros(n_samples)
        self.time[self._iter_counter] = 0.0
        self._states_history = np.zeros([self._n_states, n_samples])
        self._states_history[:, self._iter_counter] = self.initial_state
        self._inputs_history = np.zeros([self._n_inputs, n_samples])
        # the first input is computed, not given

    def update_target_y(self):
        self.target_y = self.target_lane * const.LANE_WIDTH

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
            return self._leader_id[-2] != self._leader_id[-1]
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

    def is_in_a_platoon(self):
        return self.get_platoon_id() >= 0

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
                new_orig_leader_id = other_vehicle.id
        self._orig_leader_id.append(new_orig_leader_id)

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
                        new_dest_leader_id = other_vehicle.id
                    elif dest_lane_follower_x < other_x < ego_x:
                        dest_lane_follower_x = other_x
                        new_dest_follower_id = other_vehicle.id
        self._destination_leader_id.append(new_dest_leader_id)
        self._destination_follower_id.append(new_dest_follower_id)

    def find_cooperation_requests(self, vehicles: Iterable[BaseVehicle]):
        new_incoming_vehicle_id = -1
        incoming_veh_x = np.inf
        if self._is_connected:
            for other_vehicle in vehicles:
                other_request = other_vehicle._desired_future_follower_id
                other_x = other_vehicle.get_x()
                if other_request == self.id and other_x < incoming_veh_x:
                    new_incoming_vehicle_id = other_vehicle.id
                    incoming_veh_x = other_x
        self._incoming_vehicle_id.append(new_incoming_vehicle_id)

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
        self.time[self._iter_counter] = next_time

    @abstractmethod
    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        # only vehicles with controllers update their discrete states (modes)
        pass

    def _position_derivative_cg(self, vel: float, theta: float, phi: float
                                ) -> None:

        beta = np.arctan(self.lr * np.tan(phi) / (self.lf + self.lr))
        self._derivatives[self._state_idx['x']] = vel * np.cos(theta + beta)
        self._derivatives[self._state_idx['y']] = vel * np.sin(theta + beta)
        self._derivatives[self._state_idx['theta']] = (vel * np.sin(beta)
                                                       / self.lr)

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
        data = np.concatenate([self.time.reshape(1, -1),
                               self._states_history, self._inputs_history])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self.id
        df['name'] = self.name
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
        self._derivatives = np.zeros(self._n_states)

    def follow_orig_lane_leader(self):
        self.set_current_leader_id(self.get_orig_lane_leader_id())

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


class BaseVehicleInterface(ABC):

    def __init__(self, vehicle: BaseVehicle):
        """

        """
        self.state_names, self.n_states = None, None
        self.input_names, self.n_inputs = None, None
        self.state_idx, self.input_idx = {}, {}

        # Copy values from the vehicle
        self.id = vehicle.id
        self.name = vehicle.name
        self.lr = vehicle.lr  # dist from C.G. to rear wheel
        self.lf = vehicle.lf  # dist from C.G. to front wheel
        self.wheelbase = self.lr + self.lf
        self.phi_max = vehicle.phi_max  # max steering wheel angle
        self.safe_h = vehicle.safe_h
        self.c = vehicle.c  # standstill distance [m]
        self.free_flow_speed = vehicle.free_flow_speed
        self._orig_leader_id: int = vehicle.get_orig_lane_leader_id()
        self._destination_leader_id: int = vehicle.get_dest_lane_leader_id()
        self._destination_follower_id: int = vehicle.get_dest_lane_follower_id()
        self._leader_id = vehicle.get_current_leader_id()
        self.target_lane = vehicle.target_lane
        self.target_y = vehicle.target_y
        # The vehicle's current state is the starting point for the ocp
        self.initial_state = vehicle.get_states()

        # Only set for some vehicle types
        self.ocp_leader_switch_times: List[float] = []
        self.ocp_leader_sequence: List[int] = []

    def __repr__(self):
        return self.__class__.__name__ + ' id=' + str(self.id)

    def __str__(self):
        return (self.__class__.__name__ + ": id=" + str(self.id)
                + "V_f=" + str(self.free_flow_speed))

    # TODO: maybe most of these methods could be class methods since they don't
    #  depend on any 'internal' value of the instance
    def select_state_from_vector(self, states: List, state_name: str) -> float:
        return states[self.state_idx[state_name]]

    def select_input_from_vector(self, inputs: List, input_name: str) -> float:
        return inputs[self.input_idx[input_name]]

    def select_vel_from_vector(self, states, inputs):
        try:
            return states[self.state_idx['v']]
        except KeyError:
            return inputs[self.input_idx['v']]

    def get_current_leader_id(self, time):
        """
        The 'current' leader is the vehicle being used to define this vehicle's
         acceleration
        :return:
        """
        if len(self.ocp_leader_sequence) == 0:
            return self._leader_id
        idx = np.searchsorted(self.ocp_leader_switch_times, time, side='right')
        return self.ocp_leader_sequence[idx-1]

    def has_leader(self, time):
        return self.get_current_leader_id(time) >= 0

    def create_state_vector(self, x: float, y: float, theta: float,
                            v: float = None):
        state_vector = np.zeros(self.n_states)
        state_vector[self.state_idx['x']] = x
        state_vector[self.state_idx['y']] = y
        state_vector[self.state_idx['theta']] = theta
        self._set_speed(v, state_vector)
        return state_vector

    def compute_safe_gap(self, v_ego):
        return self.safe_h * v_ego + self.c

    def compute_error_to_safe_gap(self, ego_states, leader_states):
        gap = (self.select_state_from_vector(leader_states, 'x')
               - self.select_state_from_vector(ego_states, 'x'))
        safe_gap = self.compute_safe_gap(self.select_state_from_vector(
            ego_states, 'v'))
        return gap - safe_gap

    def compute_derivatives(self, ego_states, inputs, leader_states
                            ) -> np.ndarray:
        dxdt = np.zeros(self.n_states)

        theta = self.select_state_from_vector(ego_states, 'theta')
        phi = self.select_input_from_vector(inputs, 'phi')
        vel = self.select_vel_from_vector(ego_states, inputs)
        accel = self.compute_acceleration(ego_states, inputs, leader_states)
        self._compute_derivatives(vel, theta, phi, accel, dxdt)
        return dxdt

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
                                                / self.wheelbase)

    def to_dataframe(self, time, states, inputs):
        data = np.concatenate([time.reshape(1, -1), states, inputs])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self.id
        df['name'] = self.name
        df['orig_lane_leader_id'] = self._orig_leader_id
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
    def get_desired_input(self) -> List[float]:
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
        self.input_idx = {input_names[i]: i for i in range(self.n_inputs)}
        self._states = np.zeros(self.n_states)
        self._derivatives = np.zeros(self.n_states)

