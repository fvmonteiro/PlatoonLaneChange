from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

from constants import lane_width
import operating_modes as om


class BaseVehicle(ABC):

    _counter = 0

    def __init__(self):
        """

        """
        self.state_names, self.n_states = None, None
        self.input_names, self.n_inputs = None, None
        self.state_idx, self.input_idx = {}, {}
        self.free_flow_speed = None
        self.initial_state = None
        self.leader_id = -1
        self.destination_leader_id = -1
        self.destination_follower_id = -1
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

        # Dynamics
        self.time = 0.0
        self._states = None
        self._derivatives = None

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return (self.__class__.__name__ + ": V_f=" + str(self.free_flow_speed)
                + ", x0=" + str(self.initial_state))

    @staticmethod
    def reset_vehicle_counter():
        BaseVehicle._counter = 0

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

    def get_a_current_state(self, state_name):
        return self._states[self.state_idx[state_name]]

    def get_derivatives(self):
        return self._derivatives

    def set_free_flow_speed(self, v_ff: float):
        self.free_flow_speed = v_ff

    def set_initial_state(self, x: float, y: float, theta: float,
                          v: float = None):
        self.initial_state = self.create_state_vector(x, y, theta, v)
        self._states = self.initial_state
        self.target_lane = y // lane_width
        self.target_y = y

    def set_states(self, ego_states):
        self._states = ego_states

    def set_mode(self, mode: om.VehicleMode):
        print("t={}, veh {},. From mode: {} to {}".format(
            self.time, self.id, self.mode, mode))
        self.mode = mode
        self.mode.set_ego_vehicle(self)

    def set_lane_change_direction(self, lc_direction):
        self.target_lane = (self.get_a_current_state('y') // lane_width
                            + lc_direction)

    def update_target_y(self):
        self.target_y = self.target_lane * lane_width

    def reset_lane_change_start_time(self):
        self.lc_start_time = -np.inf

    def has_leader(self):
        return self.leader_id >= 0

    def has_dest_lane_leader(self):
        return self.destination_leader_id >= 0

    def has_dest_lane_follower(self):
        return self.destination_follower_id >= 0

    def has_lane_change_intention(self):
        current_lane = self.get_a_current_state('y') // lane_width
        return current_lane != self.target_lane

    def is_lane_change_safe(self, vehicles: Dict[int, BaseVehicle]):
        is_safe_to_leader = True
        if self.has_leader():
            leader = vehicles[self.leader_id]
            is_safe_to_leader = BaseVehicle.is_gap_safe_for_lane_change(
                leader, self)

        is_safe_to_dest_lane_leader = True
        if self.has_dest_lane_leader():
            dest_lane_leader = vehicles[self.destination_leader_id]
            is_safe_to_dest_lane_leader = (
                BaseVehicle.is_gap_safe_for_lane_change(dest_lane_leader, self))

        is_safe_to_dest_lane_follower = True
        if self.has_dest_lane_follower():
            dest_lane_follower = vehicles[self.destination_follower_id]
            is_safe_to_dest_lane_follower = (
                BaseVehicle.is_gap_safe_for_lane_change(
                    self, dest_lane_follower))

        return (is_safe_to_leader
                and is_safe_to_dest_lane_leader
                and is_safe_to_dest_lane_follower)

    def is_lane_change_complete(self):
        return (np.abs(self.get_a_current_state('y') - self.target_y) < 1e-2
                and np.abs(self.get_a_current_state('theta')) < 1e-3)

    @staticmethod
    def is_gap_safe_for_lane_change(leader: BaseVehicle, follower: BaseVehicle):
        margin = 0.1
        gap = (leader.get_a_current_state('x')
               - follower.get_a_current_state('x'))
        safe_gap = follower.compute_safe_gap(follower.get_a_current_state('v'))
        return gap + margin >= safe_gap

    def compute_safe_gap(self, v_ego):
        return v_ego + 1  # Default value for vehicles without accel feedback

    def compute_gap_error(self, ego_states, leader_states):
        gap = (self.select_state_from_vector(leader_states, 'x')
               - self.select_state_from_vector(ego_states, 'x'))
        safe_gap = self.compute_safe_gap(self.select_state_from_vector(
            ego_states, 'v'))
        return gap - safe_gap

    def compute_derivatives(self, ego_states, inputs, leader_states):
        self._derivatives = np.zeros(self.n_states)

        theta = self.select_state_from_vector(ego_states, 'theta')
        phi = self.select_input_from_vector(inputs, 'phi')
        vel = self.select_vel_from_vector(ego_states, inputs)
        accel = self.compute_acceleration(ego_states, inputs, leader_states)
        self._compute_derivatives(vel, theta, phi, accel)
        # self.derivatives = dxdt

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

    def to_dataframe(self, time: np.ndarray,
                     states: np.ndarray, inputs: np.ndarray) -> pd.DataFrame:
        data = np.concatenate([time.reshape(1, -1), states, inputs])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        df = pd.DataFrame(data=np.transpose(data), columns=columns)
        df['id'] = self.id
        df['leader_id'] = self.leader_id
        df['dest_lane_leader_id'] = self.destination_leader_id
        df['dest_lane_follower_id'] = self.destination_follower_id
        return df

    def set_lane_change_maneuver_parameters(self):
        self.lc_start_time = self.time
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

        delta_t = self.time - self.lc_start_time
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

    def compute_acceleration(self, ego_states, inputs, leader_states):
        # Does nothing because velocity is an input for this model
        pass

    # def get_vel(self, states, inputs):
    #     return self.get_input(inputs, 'v')

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

    def compute_acceleration(self, ego_states, inputs, leader_states):
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
        self.safe_h = 1.0
        self.c = 1.0  # standstill distance [m]
        self.kg = 0.5
        self.kv = 0.5
        # Note: h and safe_h are a simplification of the system. The time
        # headway used for vehicle following is computed in a way to
        # overestimate the nonlinear safe distance. In the above, we just
        # assume the safe distance is also linear and with a smaller h.

    def compute_acceleration(self, ego_states, inputs, leader_states) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.select_state_from_vector(ego_states, 'v')
        if leader_states is None or len(leader_states) == 0:
            return self.compute_velocity_control(v_ego)
        else:
            gap = (self.select_state_from_vector(leader_states, 'x')
                   - self.select_state_from_vector(ego_states, 'x'))
            v_leader = self.select_state_from_vector(leader_states, 'v')
            accel = self.compute_gap_control(gap, v_ego, v_leader)
            if v_ego >= self.free_flow_speed and accel > 0:
                return self.compute_velocity_control(v_ego)
            return accel

    def get_input_limits(self) -> (List[float], List[float]):
        return [-self.phi_max], [self.phi_max]

    def compute_safe_gap(self, v_ego: float) -> float:
        return self.safe_h * v_ego + self.c

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
