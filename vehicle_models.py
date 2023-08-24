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
    _states: np.ndarray
    _inputs: np.ndarray
    _iter_counter: int
    _states_history: np.ndarray
    _inputs_history: np.ndarray
    _derivatives: np.ndarray

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
        # Vehicle used to determine current accel
        self._leader_id: List[int] = []
        self._polynomial_lc_coeffs = None
        self._lc_start_time = -np.inf
        self.mode = om.LaneKeepingMode()
        self.mode.set_ego_vehicle(self)

        # Some parameters
        self.id = BaseVehicle._counter
        BaseVehicle._counter += 1
        self.lr = 2  # dist from C.G. to rear wheel
        self.lf = 1  # dist from C.G. to front wheel
        self._wheelbase = self.lr + self.lf
        self.phi_max = 0.1  # max steering wheel angle
        self._lateral_gain = 1
        self._lane_change_duration = 5  # [s]
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

    def get_current_time(self):
        return self.time[self._iter_counter]

    def get_a_state_by_name(self, state_name):
        return self._states[self._state_idx[state_name]]

    def get_an_input_by_name(self, input_name):
        return self._inputs[self._input_idx[input_name]]

    def get_current_lane(self):
        return round(self.get_a_state_by_name('y') / lane_width)

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
        self._states = self.initial_state
        self.target_lane = round(y / lane_width)
        self.target_y = y

    def set_mode(self, mode: om.VehicleMode):
        print("t={}, veh {},. From mode: {} to {}".format(
            self.get_current_time(), self.id, self.mode, mode))
        self.mode = mode
        self.mode.set_ego_vehicle(self)

    def set_lane_change_direction(self, lc_direction):
        self.target_lane = (round(self.get_a_state_by_name('y') / lane_width)
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

    def initialize_simulation_logs(self, n_samples):
        self._iter_counter = 0
        self.time = np.zeros(n_samples)
        self.time[self._iter_counter] = 0.0
        self._states_history = np.zeros([self._n_states, n_samples])
        self._states_history[:, self._iter_counter] = self.initial_state
        self._inputs_history = np.zeros([self._n_inputs, n_samples])
        # the first input is computed, not given

    def update_target_y(self):
        self.target_y = self.target_lane * lane_width

    def reset_lane_change_start_time(self):
        self._lc_start_time = -np.inf

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
        ego_x = self.get_a_state_by_name('x')
        ego_y = self.get_a_state_by_name('y')
        orig_lane_leader_x = np.inf
        new_orig_leader_id = -1
        for other_vehicle in vehicles:
            other_x = other_vehicle.get_a_state_by_name('x')
            other_y = other_vehicle.get_a_state_by_name('y')
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
            ego_x = self.get_a_state_by_name('x')
            dest_lane_follower_x = -np.inf
            dest_lane_leader_x = np.inf
            for other_vehicle in vehicles:
                other_x = other_vehicle.get_a_state_by_name('x')
                other_y = other_vehicle.get_a_state_by_name('y')
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
        return (np.abs(self.get_a_state_by_name('y') - self.target_y) < 1e-2
                and np.abs(self.get_a_state_by_name('theta')) < 1e-3)

    @staticmethod
    def is_gap_safe_for_lane_change(leading_vehicle: BaseVehicle,
                                    following_vehicle: BaseVehicle):
        margin = 1e-2
        gap = (leading_vehicle.get_a_state_by_name('x')
               - following_vehicle.get_a_state_by_name('x'))
        safe_gap = following_vehicle.compute_safe_gap(
            following_vehicle.get_vel())
        return gap + margin >= safe_gap

    def compute_safe_gap(self, v_ego=None):
        if v_ego is None:
            v_ego = self.get_vel()
        return self.safe_h * v_ego + self.c

    def compute_derivatives(self):
        self._derivatives = np.zeros(self._n_states)
        theta = self.get_a_state_by_name('theta')
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

    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        if self.has_lane_change_intention():
            self.mode.handle_lane_changing_intention(vehicles)
        else:
            self.mode.handle_lane_keeping_intention(vehicles)

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
        self._lc_start_time = self.get_current_time()
        self.update_target_y()
        self.compute_polynomial_lc_trajectory()

    def compute_polynomial_lc_trajectory(self):
        y0 = self.get_a_state_by_name('y')
        vy0 = 0
        ay0 = 0
        yf = self.target_y
        vyf = vy0
        ayf = ay0

        tf = self._lane_change_duration
        a = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 2, 0, 0, 0],
                      [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                      [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                      [0, 0, 2, 6 * tf, 12 * tf * 2, 20 * tf ** 3]])
        b = np.array([[y0], [vy0], [ay0], [yf], [vyf], [ayf]])
        self._polynomial_lc_coeffs = np.linalg.solve(a, b)

    def compute_desired_slip_angle(self):
        if self.is_lane_change_complete():
            return 0.0

        delta_t = self.get_current_time() - self._lc_start_time
        if delta_t <= self._lane_change_duration:
            yr = sum([self._polynomial_lc_coeffs[i] * delta_t ** i
                      for i in range(len(self._polynomial_lc_coeffs))])
            vyr = sum([i * self._polynomial_lc_coeffs[i] * delta_t ** (i - 1)
                       for i in range(1, len(self._polynomial_lc_coeffs))])
        else:
            yr = self.target_y
            vyr = 0

        ey = yr - self.get_a_state_by_name('y')
        theta = self.get_a_state_by_name('theta')
        vel = self.get_vel()
        return ((vyr + self._lateral_gain * ey) / (vel * np.cos(theta))
                - np.tan(theta))

    def compute_steering_wheel_angle(self):
        return np.arctan(self.lr / (self.lf + self.lr)
                         * np.tan(self.compute_desired_slip_angle()))

    @abstractmethod
    def get_vel(self):
        pass

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


class OpenLoopVehicle(BaseVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Does not compute any inputs. """

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)
        self.brake_max = -4
        self.accel_max = 2

    def get_vel(self):
        return self.get_a_state_by_name('v')

    def _set_speed(self, v0, state):
        state[self._state_idx['v']] = v0

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_cg(vel, theta, phi)
        self._derivatives[self._state_idx['v']] = self.get_an_input_by_name('a')

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = open_loop_controls[
            self._input_idx['a']]
        self._inputs[self._input_idx['phi']] = open_loop_controls[
            self._input_idx['phi']]


class SafeAccelOptimalLCVehicle(OpenLoopVehicle):
    """ Safe acceleration (internally computed) and optimal control computed
    (open-loop) lane changes.
     States: [x, y, theta, v], external input: [phi], centered at the C.G.
     and accel is computed by a feedback law"""

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

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop control and phi (steering wheel angle) and
        computes the acceleration
        :param open_loop_controls: Dictionary containing 'phi' value
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = self.compute_acceleration(vehicles)
        self._inputs[self._input_idx['phi']] = open_loop_controls

    def compute_acceleration(self, vehicles: Dict[int, BaseVehicle]) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.get_vel()
        if not self.has_leader():
            accel = self.compute_velocity_control(v_ego)
        else:
            leader = vehicles[self.get_current_leader_id()]
            gap = (leader.get_a_state_by_name('x')
                   - self.get_a_state_by_name('x'))
            v_leader = leader.get_vel()
            accel = self.compute_gap_control(gap, v_ego, v_leader)
            if v_ego >= self.free_flow_speed and accel > 0:
                accel = self.compute_velocity_control(v_ego)
        return accel

    def update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Compares the acceleration if following the origin or destination lane
        leaders, and chooses as leader the vehicle which causes the lesser
        acceleration
        :param vehicles:
        :return:
        """
        x_ego = self.get_a_state_by_name('x')
        v_ego = self.get_vel()
        if self.has_orig_lane_leader():
            orig_lane_leader = vehicles[self.get_orig_lane_leader_id()]
            gap = orig_lane_leader.get_a_state_by_name('x') - x_ego
            orig_lane_accel = self.compute_gap_control(
                gap, v_ego, orig_lane_leader.get_vel())
        else:
            orig_lane_accel = np.inf

        if self.has_dest_lane_leader():
            dest_lane_leader = vehicles[self.get_dest_lane_leader_id()]
            gap = dest_lane_leader.get_a_state_by_name('x') - x_ego
            dest_lane_accel = self.compute_gap_control(
                gap, v_ego, dest_lane_leader.get_vel())
        else:
            dest_lane_accel = np.inf

        if orig_lane_accel <= dest_lane_accel:
            self._leader_id.append(self.get_orig_lane_leader_id())
        else:
            self._leader_id.append(self.get_dest_lane_leader_id())

    def compute_velocity_control(self, v_ego: float) -> float:
        return self.kv * (self.free_flow_speed - v_ego)

    def compute_gap_control(self, gap: float, v_ego: float,
                            v_leader: float) -> float:
        return (self.kg * (gap - self.h * v_ego - self.c)
                + self.kv * (v_leader - v_ego))


class ClosedLoopVehicle(SafeAccelOptimalLCVehicle):
    """ Vehicle that computes all of its inputs by feedback laws.
     States: [x, y, theta, v], external input: None, centered at the C.G. """

    def __init__(self):
        super().__init__()

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Computes the acceleration and phi (steering wheel angle)
        :param open_loop_controls: irrelevant
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self.compute_acceleration(vehicles))
        self._inputs[self._input_idx['phi']] = (
            self.compute_steering_wheel_angle())


# =========================== Three-State Vehicles =========================== #
# Three-state vehicles are used in initial tests with the optimization tool
# since they are simpler and were used in the tool's example.
class ThreeStateVehicle(BaseVehicle, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)

    def get_vel(self):
        try:
            return self.get_an_input_by_name('v')
        except AttributeError:
            # trying to read vehicle's speed before any input is computed
            return self.free_flow_speed

    def _set_speed(self, v0, state):
        # Does nothing because velocity is an input for this model
        pass

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop controls v (velocity) and phi (steering wheel
        :param open_loop_controls: Dictionary containing 'v' and 'phi' values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['v']] = open_loop_controls[
            self._input_idx['v']]
        self._inputs[self._input_idx['phi']] = open_loop_controls[
            self._input_idx['phi']]


class ThreeStateVehicleRearWheel(ThreeStateVehicle):
    """ From the library's example.
    States: [x, y, theta], inputs: [v, phi], centered at the rear wheels """

    def __init__(self):
        super().__init__()

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_rear_wheels(vel, theta, phi)


class ThreeStateVehicleCG(ThreeStateVehicle):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self):
        super().__init__()

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_cg(vel, theta, phi)
