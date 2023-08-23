from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Iterable
import warnings

import numpy as np
import pandas as pd

from constants import lane_width
import operating_modes as om


class BaseVehicle(ABC):

    _counter = 0

    time: np.ndarray
    _iter_counter: int
    _states: np.ndarray
    _inputs: np.ndarray  # TODO: manage input array!!
    _derivatives: np.ndarray

    def __init__(self):
        """

        """
        self.state_names, self.n_states = None, None
        self.input_names, self.n_inputs = None, None
        self.state_idx, self.input_idx = {}, {}
        self.free_flow_speed = None
        self.initial_state = None
        self._orig_leader_id: List[int] = []
        self._destination_leader_id: List[int] = []
        self._destination_follower_id: List[int] = []
        self._leader_id = []  # vehicle used to determine current accel
        self.polynomial_lc_coeffs = None
        self.lc_start_time = -np.inf
        self.mode = om.LaneKeepingMode()
        self.mode.set_ego_vehicle(self)
        self.target_lane = None
        self.target_y = None

        # Some parameters
        self.id = BaseVehicle._counter
        BaseVehicle._counter += 1
        self.lr = 2  # dist from C.G. to rear wheel
        self.lf = 1  # dist from C.G. to front wheel
        self.wheelbase = self.lr + self.lf
        self.phi_max = 0.1  # max steering wheel angle
        self.lateral_gain = 1
        self.lane_change_duration = 5  # [s]
        self.safe_h = 1.0
        self.c = 1.0  # standstill distance [m]

    def __repr__(self):
        return self.__class__.__name__ + ' id=' + str(self.id)

    def __str__(self):
        return (self.__class__.__name__ + ": id=" + str(self.id)
                + "V_f=" + str(self.free_flow_speed)
                + ", x0=" + str(self.initial_state))

    @staticmethod
    def reset_vehicle_counter():
        BaseVehicle._counter = 0

    def select_input_from_vector(self, inputs: List[float], input_name: str
                                 ) -> float:
        return inputs[self.input_idx[input_name]]

    def get_current_time(self):
        return self.time[self._iter_counter]

    def get_a_current_state(self, state_name):
        return self._states[self.state_idx[state_name], self._iter_counter]

    def get_a_current_input(self, input_name):
        return self._inputs[self.state_idx[input_name], self._iter_counter]

    def get_current_lane(self):
        return round(self.get_a_current_state('y') / lane_width)

    def get_derivatives(self):
        return self._derivatives

    def get_orig_lane_leader_id(self):
        return self._orig_leader_id[-1]

    def get_dest_lane_leader_id(self):
        return self._destination_leader_id[-1]

    def get_dest_lane_follower_id(self):
        return self._destination_follower_id[-1]

    def get_current_leader_id(self):
        """
        The 'current' leader is the vehicle being used to define this vehicle's
         acceleration
        :return:
        """
        return self._leader_id[-1]

    def set_free_flow_speed(self, v_ff: float):
        self.free_flow_speed = v_ff

    def set_initial_state(self, x: float, y: float, theta: float,
                          v: float = None):
        self.initial_state = self.create_state_vector(x, y, theta, v)
        self.target_lane = round(y / lane_width)
        self.target_y = y

    def set_mode(self, mode: om.VehicleMode):
        print("t={}, veh {},. From mode: {} to {}".format(
            self.get_current_time(), self.id, self.mode, mode))
        self.mode = mode
        self.mode.set_ego_vehicle(self)

    def set_lane_change_direction(self, lc_direction):
        self.target_lane = (round(self.get_a_current_state('y') / lane_width)
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

    def initialize_states(self, n_samples):
        self._iter_counter = 0
        self.time = np.zeros(n_samples)
        self.time[self._iter_counter] = 0.0
        self._states = np.zeros([self.n_states, n_samples])
        self._states[:, self._iter_counter] = self.initial_state
        self._inputs = np.zeros([self.n_inputs, n_samples])
        # the first input is computed, not given

    def update_target_y(self):
        self.target_y = self.target_lane * lane_width

    def reset_lane_change_start_time(self):
        self.lc_start_time = -np.inf

    def has_orig_lane_leader(self):
        try:
            return self._orig_leader_id[-1] >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(orig lane leader id) before simulation start")
            return False

    def has_dest_lane_leader(self):
        try:
            return self._destination_leader_id[-1] >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(dest lane leader id) before simulation start")
            return False

    def has_dest_lane_follower(self):
        try:
            return self._destination_follower_id[-1] >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(dest lane follower id) before simulation start")
            return False

    def has_leader(self):
        try:
            return self._leader_id[-1] >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(leader id) before simulation start")
            return False

    def has_lane_change_intention(self):
        return self.get_current_lane() != self.target_lane

    def find_orig_lane_leader(self, vehicles: Iterable[BaseVehicle]):
        ego_x = self.get_a_current_state('x')
        ego_y = self.get_a_current_state('y')
        orig_lane_leader_x = np.inf
        new_orig_leader_id = -1
        for other_vehicle in vehicles:
            other_x = other_vehicle.get_a_current_state('x')
            other_y = other_vehicle.get_a_current_state('y')
            if (np.abs(other_y - ego_y) < lane_width / 2  # same lane
                    and ego_x < other_x < orig_lane_leader_x):
                orig_lane_leader_x = other_x
                new_orig_leader_id = other_vehicle.id
        self._orig_leader_id.append(new_orig_leader_id)

    def find_dest_lane_vehicles(self, vehicles: Iterable[BaseVehicle]):
        new_dest_leader_id = -1
        new_dest_follower_id = -1
        if self.has_lane_change_intention():
            y_target_lane = self.target_lane * lane_width
            ego_x = self.get_a_current_state('x')
            dest_lane_follower_x = -np.inf
            dest_lane_leader_x = np.inf
            for other_vehicle in vehicles:
                other_x = other_vehicle.get_a_current_state('x')
                other_y = other_vehicle.get_a_current_state('y')
                if np.abs(other_y - y_target_lane) < lane_width / 2:
                    if ego_x < other_x < dest_lane_leader_x:
                        dest_lane_leader_x = other_x
                        new_dest_leader_id = other_vehicle.id
                    elif dest_lane_follower_x < other_x < ego_x:
                        dest_lane_follower_x = other_x
                        new_dest_follower_id = other_vehicle.id
        self._destination_leader_id.append(new_dest_leader_id)
        self._destination_follower_id.append(new_dest_follower_id)

    def update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Defines which surrounding vehicle should be used to determine this
        vehicle's own acceleration
        :return:
        """
        self.set_current_leader_id(self.get_orig_lane_leader_id())

    def is_lane_change_safe(self, vehicles: Dict[int, BaseVehicle]):
        is_safe_to_orig_lane_leader = True
        if self.has_orig_lane_leader():
            orig_lane_leader = vehicles[self._orig_leader_id[-1]]
            is_safe_to_orig_lane_leader = (
                BaseVehicle.is_gap_safe_for_lane_change(orig_lane_leader, self))

        is_safe_to_dest_lane_leader = True
        if self.has_dest_lane_leader():
            dest_lane_leader = vehicles[self._destination_leader_id[-1]]
            is_safe_to_dest_lane_leader = (
                BaseVehicle.is_gap_safe_for_lane_change(dest_lane_leader, self))

        is_safe_to_dest_lane_follower = True
        if self.has_dest_lane_follower():
            dest_lane_follower = vehicles[self._destination_follower_id[-1]]
            is_safe_to_dest_lane_follower = (
                BaseVehicle.is_gap_safe_for_lane_change(
                    self, dest_lane_follower))

        return (is_safe_to_orig_lane_leader
                and is_safe_to_dest_lane_leader
                and is_safe_to_dest_lane_follower)

    def is_lane_change_complete(self):
        return (np.abs(self.get_a_current_state('y') - self.target_y) < 1e-2
                and np.abs(self.get_a_current_state('theta')) < 1e-3)

    @staticmethod
    def is_gap_safe_for_lane_change(leading_vehicle: BaseVehicle,
                                    following_vehicle: BaseVehicle):
        margin = 1e-2
        gap = (leading_vehicle.get_a_current_state('x')
               - following_vehicle.get_a_current_state('x'))
        safe_gap = following_vehicle.compute_safe_gap(
            following_vehicle.get_a_current_state('v'))
        return gap + margin >= safe_gap

    def compute_safe_gap(self, v_ego=None):
        if v_ego is None:
            v_ego = self.get_a_current_state('v')
        return self.safe_h * v_ego + self.c

    def compute_derivatives(self, inputs, vehicles: Dict[int, BaseVehicle]):
        self._derivatives = np.zeros(self.n_states)

        theta = self.get_a_current_state('theta')
        phi = self.select_input_from_vector(inputs, 'phi')
        vel = self.get_a_current_state('v')
        accel = self.compute_acceleration(inputs, vehicles)
        self._compute_derivatives(vel, theta, phi, accel)
        # self.derivatives = dxdt

    def update_states(self, next_time):
        dt = next_time - self.get_current_time()
        self._states[:, self._iter_counter + 1] = (
                self._states[:, self._iter_counter] + self._derivatives * dt)
        self._iter_counter += 1
        self.time[self._iter_counter] = next_time

    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        if self.has_lane_change_intention():
            self.mode.handle_lane_changing_intention(vehicles)
        else:
            self.mode.handle_lane_keeping_intention(vehicles)

    def _position_derivative_cg(self, vel: float, theta: float, phi: float
                                ) -> None:

        beta = np.arctan(self.lr * np.tan(phi) / (self.lf + self.lr))
        self._derivatives[self.state_idx['x']] = vel * np.cos(theta + beta)
        self._derivatives[self.state_idx['y']] = vel * np.sin(theta + beta)
        self._derivatives[self.state_idx['theta']] = (vel * np.sin(beta)
                                                      / self.lr)

    def _position_derivative_rear_wheels(self, vel: float, theta: float,
                                         phi: float):
        self._derivatives[self.state_idx['x']] = vel * np.cos(theta)
        self._derivatives[self.state_idx['y']] = vel * np.sin(theta)
        self._derivatives[self.state_idx['theta']] = (vel * np.tan(phi)
                                                      / self.wheelbase)

    def create_state_vector(self, x: float, y: float, theta: float,
                            v: float = None):
        state_vector = np.zeros(self.n_states)
        state_vector[self.state_idx['x']] = x
        state_vector[self.state_idx['y']] = y
        state_vector[self.state_idx['theta']] = theta
        self._set_speed(v, state_vector)
        return state_vector

    def to_dataframe(self, inputs: np.ndarray) -> pd.DataFrame:
        data = np.concatenate([self.time.reshape(1, -1), self._states, inputs])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self.id
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'orig_lane_leader_id', self._orig_leader_id)
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'dest_lane_leader_id', self._destination_leader_id)
        BaseVehicle._set_surrounding_vehicles_ids_to_df(
            df, 'dest_lane_follower_id', self._destination_follower_id)
        return df

    @staticmethod
    def _set_surrounding_vehicles_ids_to_df(df, col_name, col_value):
        if len(col_value) == 1:
            df[col_name] = col_value[0]
        else:
            df[col_name] = col_value

    def set_lane_change_maneuver_parameters(self):
        self.lc_start_time = self.get_current_time()
        self.update_target_y()
        self.compute_polynomial_lc_trajectory()

    def compute_polynomial_lc_trajectory(self):
        y0 = self.get_a_current_state('y')
        vy0 = 0
        ay0 = 0
        yf = self.target_y
        vyf = vy0
        ayf = ay0

        tf = self.lane_change_duration
        A = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 2, 0, 0, 0],
                      [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                      [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                      [0, 0, 2, 6 * tf, 12 * tf * 2, 20 * tf ** 3]])
        b = np.array([[y0], [vy0], [ay0], [yf], [vyf], [ayf]])
        self.polynomial_lc_coeffs = np.linalg.solve(A, b)

    def compute_desired_slip_angle(self):
        if self.is_lane_change_complete():
            return 0.0

        delta_t = self.get_current_time() - self.lc_start_time
        if delta_t <= self.lane_change_duration:
            yr = sum([self.polynomial_lc_coeffs[i] * delta_t ** i
                      for i in range(len(self.polynomial_lc_coeffs))])
            vyr = sum([i * self.polynomial_lc_coeffs[i] * delta_t ** (i - 1)
                       for i in range(1, len(self.polynomial_lc_coeffs))])
        else:
            yr = self.target_y
            vyr = 0

        ey = yr - self.get_a_current_state('y')
        theta = self.get_a_current_state('theta')
        vel = self.get_a_current_state('v')
        return ((vyr + self.lateral_gain * ey) / (vel * np.cos(theta))
                - np.tan(theta))

    def compute_steering_wheel_angle(self):
        return np.arctan(self.lr / (self.lf + self.lr)
                         * np.tan(self.compute_desired_slip_angle()))

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
    def _compute_derivatives(self, vel, theta, phi, accel):
        """ Computes the derivatives of x, y, and theta, and stores them in the
         derivatives array """
        pass

    @abstractmethod
    def compute_acceleration(self, inputs, vehicles: Dict[int, BaseVehicle]):
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
        self._derivatives = np.zeros(self.n_states)


class ThreeStateVehicle(BaseVehicle, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)

    def _set_speed(self, v0, state):
        # Does nothing because velocity is an input for this model
        pass

    def compute_acceleration(self, inputs, vehicles: Dict[int, BaseVehicle]):
        # Does nothing because velocity is an input for this model
        pass

    def get_desired_input(self) -> List[float]:
        return [self.free_flow_speed, 0]

    def get_input_limits(self) -> (List[float], List[float]):
        return [0, -self.phi_max], [self.free_flow_speed + 5, self.phi_max]


class ThreeStateVehicleRearWheel(ThreeStateVehicle):
    """ From the library's example.
    States: [x, y, theta], inputs: [v, phi], centered at the rear wheels """

    def __init__(self):
        super().__init__()

    def _compute_derivatives(self, vel, theta, phi, accel):
        self._position_derivative_rear_wheels(vel, theta, phi)


class ThreeStateVehicleCG(ThreeStateVehicle):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self):
        super().__init__()

    def _compute_derivatives(self, vel, theta, phi, accel):
        self._position_derivative_cg(vel, theta, phi)


class FourStateVehicle(BaseVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G. """

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)
        self.brake_max = -4
        self.accel_max = 2

    def _set_speed(self, v0, state):
        state[self.state_idx['v']] = v0

    def _compute_derivatives(self, vel, theta, phi, accel):
        self._position_derivative_cg(vel, theta, phi)
        self._derivatives[self.state_idx['v']] = accel

    def compute_acceleration(self, inputs, vehicles: Dict[int, BaseVehicle]):
        return self.select_input_from_vector(inputs, 'a')

    def get_desired_input(self) -> List[float]:
        return [0] * self.n_inputs

    def get_input_limits(self) -> (List[float], List[float]):
        return [self.brake_max, -self.phi_max], [self.accel_max, self.phi_max]


class FourStateVehicleAccelFB(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [phi], centered at the C.G.
     and accel is computed by a feedback law"""

    _input_names = ['phi']

    def __init__(self):
        super().__init__()

        # Controller parameters
        self.h = 1.0  # time headway [s]
        self.kg = 0.5
        self.kv = 0.5
        # Note: h and safe_h are a simplification of the system. The time
        # headway used for vehicle following is computed in a way to
        # overestimate the nonlinear safe distance. In the above, we just
        # assume the safe distance is also linear and with a smaller h.

    def get_input_limits(self) -> (List[float], List[float]):
        return [-self.phi_max], [self.phi_max]

    def compute_acceleration(self, inputs, vehicles: Dict[int, BaseVehicle]
                             ) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.get_a_current_state('v')
        if not self.has_leader():
            return self.compute_velocity_control(v_ego)
        else:
            leader = vehicles[self.get_current_leader_id()]
            gap = (leader.get_a_current_state('x')
                   - self.get_a_current_state('x'))
            v_leader = leader.get_a_current_state('v')
            accel = self.compute_gap_control(gap, v_ego, v_leader)
            if v_ego >= self.free_flow_speed and accel > 0:
                return self.compute_velocity_control(v_ego)
            return accel

    def update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        if self.has_orig_lane_leader():
            orig_lane_leader = vehicles[self.get_orig_lane_leader_id()]
            orig_lane_accel = self._compute_acceleration(orig_lane_leader)
        else:
            orig_lane_accel = np.inf
        if self.has_dest_lane_leader():
            dest_lane_leader = vehicles[self.get_dest_lane_leader_id()]
            dest_lane_accel = self._compute_acceleration(dest_lane_leader)
        else:
            dest_lane_accel = np.inf

        if orig_lane_accel <= dest_lane_accel:
            self._leader_id.append(self.get_orig_lane_leader_id())
        else:
            self._leader_id.append(self.get_dest_lane_leader_id())

    # TODO: rename; figure out whole external vs internal states ordeal
    def _compute_acceleration(self, a_leader: BaseVehicle):
        gap = a_leader.get_a_current_state('x') - self.get_a_current_state('x')
        v_ego = self.get_a_current_state('v')
        v_leader = a_leader.get_a_current_state('v')
        return self.compute_gap_control(gap, v_ego, v_leader)

    def compute_velocity_control(self, v_ego: float) -> float:
        return self.kv * (self.free_flow_speed - v_ego)

    def compute_gap_control(self, gap: float, v_ego: float,
                            v_leader: float) -> float:
        return (self.kg * (gap - self.h * v_ego - self.c)
                + self.kv * (v_leader - v_ego))


class LongitudinalVehicle(FourStateVehicleAccelFB):
    """ Vehicle that does not perform lane change and computes its own
    acceleration """

    _input_names = []

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)

    def select_input_from_vector(self, inputs: List, input_name: str) -> float:
        return 0.0  # all inputs are computed internally

    def get_desired_input(self) -> List[float]:
        return []

    def get_input_limits(self) -> (List[float], List[float]):
        return [], []
