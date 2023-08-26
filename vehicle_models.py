from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Iterable
import warnings

import numpy as np
import pandas as pd
import control as ct
import control.optimal as opt
import control.flatsys as flat
from scipy.optimize import NonlinearConstraint

import vehicle_group_ocp_interface as vgi
import constants as const
import vehicle_operating_modes as modes


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
        self._incoming_vehicle_id: int = -1  # TODO: must reset every time
                                             #  (make list?)
        # Vehicle used to determine current accel
        self._leader_id: List[int] = []
        self._polynomial_lc_coeffs = None
        self._long_adjust_start_time = -np.inf
        self._lc_start_time = -np.inf

        # Some parameters
        self.id = BaseVehicle._counter
        BaseVehicle._counter += 1
        self.name: str = str(self.id)  # default
        self.lr = 2  # dist from C.G. to rear wheel
        self.lf = 1  # dist from C.G. to front wheel
        self._wheelbase = self.lr + self.lf
        self.phi_max = 0.1  # max steering wheel angle
        self._lateral_gain = 1
        self._lane_change_duration = 5  # [s]
        self.safe_h = const.safe_time_headway
        self.c = const.standstill_distance

        self._is_cooperative = False

    def __repr__(self):
        return self.__class__.__name__ + ' id=' + str(self.id)

    def __str__(self):
        return self.__class__.__name__ + ": " + str(self.name)

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
        return round(self.get_a_state_by_name('y') / const.lane_width)

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
        return self._incoming_vehicle_id

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
        self.target_lane = round(y / const.lane_width)
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
                    round(self.get_a_state_by_name('y') / const.lane_width)
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
        self.target_y = self.target_lane * const.lane_width

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
        return self._incoming_vehicle_id >= 0

    def has_leader(self):
        try:
            return self._leader_id[-1] >= 0
        except IndexError:
            warnings.warn("Warning: trying to access vehicle data "
                          "(leader id) before simulation start")
            return False

    def has_lane_change_intention(self):
        return self.get_current_lane() != self.target_lane

    def make_cooperative(self):
        self._is_cooperative = True

    def find_orig_lane_leader(self, vehicles: Iterable[BaseVehicle]):
        ego_x = self.get_a_state_by_name('x')
        ego_y = self.get_a_state_by_name('y')
        orig_lane_leader_x = np.inf
        new_orig_leader_id = -1
        for other_vehicle in vehicles:
            other_x = other_vehicle.get_a_state_by_name('x')
            other_y = other_vehicle.get_a_state_by_name('y')
            if (np.abs(other_y - ego_y) < const.lane_width / 2  # same lane
                    and ego_x < other_x < orig_lane_leader_x):
                orig_lane_leader_x = other_x
                new_orig_leader_id = other_vehicle.id
        self._orig_leader_id.append(new_orig_leader_id)

    def find_dest_lane_vehicles(self, vehicles: Iterable[BaseVehicle]):
        new_dest_leader_id = -1
        new_dest_follower_id = -1
        if self.has_lane_change_intention():
            y_target_lane = self.target_lane * const.lane_width
            ego_x = self.get_a_state_by_name('x')
            dest_lane_follower_x = -np.inf
            dest_lane_leader_x = np.inf
            for other_vehicle in vehicles:
                other_x = other_vehicle.get_a_state_by_name('x')
                other_y = other_vehicle.get_a_state_by_name('y')
                if np.abs(other_y - y_target_lane) < const.lane_width / 2:
                    if ego_x < other_x < dest_lane_leader_x:
                        dest_lane_leader_x = other_x
                        new_dest_leader_id = other_vehicle.id
                    elif dest_lane_follower_x < other_x < ego_x:
                        dest_lane_follower_x = other_x
                        new_dest_follower_id = other_vehicle.id
        self._destination_leader_id.append(new_dest_leader_id)
        self._destination_follower_id.append(new_dest_follower_id)

    def request_cooperation(self, vehicles: Dict[int, BaseVehicle]):
        # We should avoid allowing one vehicle to modify another, but this is
        # the simplest solution. We can consider modifying later [Aug 2023]
        if self.has_dest_lane_follower():
            vehicles[
                self.get_dest_lane_follower_id()
            ].receive_cooperation_request(self.id)

    def receive_cooperation_request(self, other_id):
        if self._is_cooperative:
            self._incoming_vehicle_id = other_id

    def update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Defines which surrounding vehicle should be used to determine this
        vehicle's own acceleration
        :return:
        """
        self._update_target_leader(vehicles)

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


class FourStateVehicle(BaseVehicle, ABC):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    """

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)
        self.brake_max = -4
        self.accel_max = 2

        # Controller parameters
        self.h = const.veh_following_time_headway
        self.kg = 0.5
        self.kv = 0.5
        # Note: h and safe_h are a simplification of the system. The time
        # headway used for vehicle following is computed in a way to
        # overestimate the nonlinear safe distance. In the above, we just
        # assume the safe distance is also linear and with a smaller h.

    def get_vel(self):
        return self.get_a_state_by_name('v')

    def _set_speed(self, v0, state):
        state[self._state_idx['v']] = v0

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_cg(vel, theta, phi)
        self._derivatives[self._state_idx['v']] = self.get_an_input_by_name('a')

    def _compute_acceleration(self, vehicles: Dict[int, BaseVehicle]) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.get_vel()
        if not self.has_leader():
            accel = self._compute_velocity_control(v_ego)
        else:
            leader = vehicles[self.get_current_leader_id()]
            gap = (leader.get_a_state_by_name('x')
                   - self.get_a_state_by_name('x'))
            v_leader = leader.get_vel()
            accel = self._compute_gap_control(gap, v_ego, v_leader)
            if v_ego >= self.free_flow_speed and accel > 0:
                accel = self._compute_velocity_control(v_ego)
        return accel

    def _compute_velocity_control(self, v_ego: float) -> float:
        return self.kv * (self.free_flow_speed - v_ego)

    def _compute_gap_control(self, gap: float, v_ego: float,
                             v_leader: float) -> float:
        return (self.kg * (gap - self.h * v_ego - self.c)
                + self.kv * (v_leader - v_ego))

    def _compute_steering_wheel_angle(self, slip_angle: float):
        return np.arctan(self.lr / (self.lf + self.lr)
                         * np.tan(slip_angle))

    def _compute_lane_keeping_slip_angle(self):
        lane_center = self.get_current_lane() * const.lane_width
        return self._compute_cbf_slip_angle(lane_center, 0.0)

    def _compute_cbf_slip_angle(self, y_ref: float, vy_ref: float):
        lat_error = y_ref - self.get_a_state_by_name('y')
        theta = self.get_a_state_by_name('theta')
        vel = self.get_vel()
        return ((vy_ref + self._lateral_gain * lat_error)
                / (vel * np.cos(theta)) - np.tan(theta))

    def _choose_min_accel_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Compares the acceleration if following the origin or destination lane
        leaders, and chooses as leader the vehicle which causes the lesser
        acceleration
        :param vehicles:
        :return:
        """
        relevant_ids = [self.get_orig_lane_leader_id(),
                        self.get_dest_lane_leader_id(),
                        self.get_incoming_vehicle_id()]
        candidate_accel = {
            veh_id: self._compute_accel_to_a_leader(veh_id, vehicles)
            for veh_id in relevant_ids
        }

        self._leader_id.append(min(candidate_accel, key=candidate_accel.get))

        # x_ego = self.get_a_state_by_name('x')
        # v_ego = self.get_vel()
        # if self.has_orig_lane_leader():
        #     orig_lane_leader = vehicles[self.get_orig_lane_leader_id()]
        #     gap = orig_lane_leader.get_a_state_by_name('x') - x_ego
        #     orig_lane_accel = self._compute_gap_control(
        #         gap, v_ego, orig_lane_leader.get_vel())
        # else:
        #     orig_lane_accel = np.inf
        #
        # if self.has_dest_lane_leader():
        #     dest_lane_leader = vehicles[self.get_dest_lane_leader_id()]
        #     gap = dest_lane_leader.get_a_state_by_name('x') - x_ego
        #     dest_lane_accel = self._compute_gap_control(
        #         gap, v_ego, dest_lane_leader.get_vel())
        # else:
        #     dest_lane_accel = np.inf
        #
        # if self.is_cooperating():
        #     incoming_vehicle = vehicles[self.get_incoming_vehicle_id()]
        #     gap = incoming_vehicle.get_a_state_by_name('x') - x_ego
        #     coop_accel = self._compute_gap_control(
        #         gap, v_ego, incoming_vehicle.get_vel())
        # else:
        #     coop_accel = np.inf
        #
        # if orig_lane_accel <= dest_lane_accel:
        #     self._leader_id.append(self.get_orig_lane_leader_id())
        # else:
        #     self._leader_id.append(self.get_dest_lane_leader_id())

    def _compute_accel_to_a_leader(self, other_id, vehicles):
        if other_id >= 0:
            x_ego = self.get_a_state_by_name('x')
            v_ego = self.get_vel()
            other_vehicle = vehicles[other_id]
            gap = other_vehicle.get_a_state_by_name('x') - x_ego
            return self._compute_gap_control(
                gap, v_ego, other_vehicle.get_vel())
        else:
            return np.inf


class OpenLoopVehicle(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Does not compute any inputs internally. This class and its derivatives
    are useful for testing inputs computed by a centralized controller """

    def __init__(self):
        super().__init__()

    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        pass

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

    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, BaseVehicle]):
        pass

    def _set_up_lane_change_control(self):
        pass

    def _update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Does nothing, since this vehicle class does not have autonomous
        longitudinal control
        """
        pass


class SafeAccelOpenLoopLCVehicle(OpenLoopVehicle):
    """ Safe acceleration (internally computed) and externally determined phi.
    States: [x, y, theta, v], external input: [phi], centered at the C.G.
    and accel is computed by a feedback law"""

    def __init__(self):
        super().__init__()

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = self._compute_acceleration(
            vehicles)
        self._inputs[self._input_idx['phi']] = open_loop_controls[0]

    def _update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        self._choose_min_accel_leader(vehicles)


class OptimalControlVehicle(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Accel and phi computed by optimal control when there is lane change
    intention. Otherwise, zero accel and lane keeping phi. """

    _ocp_interface: vgi.VehicleGroupInterface
    _ocp_initial_state: np.ndarray
    _ocp_desired_state: np.ndarray
    _ocp_solver_max_iter = 300
    _solver_wait_time = 20.0  # [s] time between attempts to solve an ocp
    _n_optimal_inputs = 2

    def __init__(self):
        super().__init__()
        self.set_mode(modes.OCPLaneKeepingMode())
        self._n_feedback_inputs = self._n_inputs - self._n_optimal_inputs
        self._ocp_has_solution = False
        self._ocp_horizon = 10.0  # [s]
        self._solver_attempt_time = -np.inf

    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

    def find_lane_change_trajectory(self, vehicles: Dict[int, BaseVehicle]):
        self._ocp_interface = vgi.VehicleGroupInterface(vehicles)
        self._set_ocp_dynamics()
        self._ocp_initial_state = self._ocp_interface.get_initial_state()
        self._ocp_desired_state = self._ocp_interface.create_desired_state(
            self._ocp_horizon)
        self._set_ocp_costs()
        self._set_constraints()
        self._solve_ocp()

    def can_start_lane_change_with_cooperation(
            self, vehicles: Dict[int, BaseVehicle]) -> bool:
        # TODO: not working. Where do we update the leaders?
        self.request_cooperation(vehicles)
        self._solver_attempt_time = -np.inf  # reset
        return self.can_start_lane_change(vehicles)

    def can_start_lane_change(self, vehicles: Dict[int, BaseVehicle]) -> bool:
        if (self.get_current_time() - self._solver_attempt_time
                >= OptimalControlVehicle._solver_wait_time):
            self._solver_attempt_time = self.get_current_time()
            self.find_lane_change_trajectory(vehicles)
        return self._ocp_has_solution

    def is_lane_changing(self):
        delta_t = self.get_current_time() - self._lc_start_time
        return delta_t <= self._ocp_horizon

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        if self.is_lane_changing():
            self._inputs = self._get_optimal_input()
        else:
            self._inputs[self._input_idx['a']] = 0.0
            slip_angle = self._compute_lane_keeping_slip_angle()
            self._inputs[self._input_idx['phi']] = (
                self._compute_steering_wheel_angle(slip_angle))

    def _update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Does nothing, since this vehicle class does not have autonomous
        longitudinal control
        """
        pass

    def _get_optimal_input(self):
        delta_t = self.get_current_time() - self._long_adjust_start_time
        ego_inputs = self._ocp_interface.get_vehicle_inputs_vector_by_id(
            self.id, self._ocp_inputs)
        current_inputs = np.zeros(self._n_optimal_inputs)
        for i in range(self._n_optimal_inputs):
            current_inputs[i] = np.interp(
                delta_t, self._ocp_times, ego_inputs[i])
        return current_inputs

    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, BaseVehicle]) -> None:
        self.update_target_y()

    def _set_ocp_dynamics(self):
        params = {'vehicle_group': self._ocp_interface}
        input_names = self._ocp_interface.create_input_names()
        output_names = self._ocp_interface.create_output_names()
        n_states = self._ocp_interface.n_states
        # Define the vehicle dynamics as an input/output system
        self.dynamic_system = ct.NonlinearIOSystem(
            vgi.vehicles_derivatives, vgi.vehicle_output,
            params=params, states=n_states, name='vehicle_group',
            inputs=input_names, outputs=output_names)

    def _set_ocp_costs(self):
        # Desired control; not final control
        uf = self._ocp_interface.get_desired_input()
        state_cost_matrix = np.diag([0, 0, 0.1, 0] * self._ocp_interface.n_vehs)
        input_cost_matrix = np.diag([0.1] * self._ocp_interface.n_inputs)
        self.running_cost = opt.quadratic_cost(
            self.dynamic_system, state_cost_matrix, input_cost_matrix,
            self._ocp_desired_state, uf)
        terminal_cost_matrix = np.diag([0, 1000, 1000, 0]
                                       * self._ocp_interface.n_vehs)
        self.terminal_cost = opt.quadratic_cost(
            self.dynamic_system, terminal_cost_matrix, 0,
            x0=self._ocp_desired_state)

    def _set_constraints(self):
        self.terminal_constraints = None

        # Input constraints
        input_lower_bounds, input_upper_bounds = (
            self._ocp_interface.get_input_limits())
        self.constraints = [opt.input_range_constraint(
            self.dynamic_system, input_lower_bounds,
            input_upper_bounds)]

        # Safety constraints
        epsilon = 1e-10
        orig_lane_safety = NonlinearConstraint(
            self._safety_constraint_orig_lane_leader, -epsilon, epsilon)
        dest_lane_leader_safety = NonlinearConstraint(
            self._safety_constraint_dest_lane_leader, -epsilon, epsilon)
        dest_lane_follower_safety = NonlinearConstraint(
            self._safety_constraint_dest_lane_follower, -epsilon, epsilon)

        self.constraints.append(orig_lane_safety)
        self.constraints.append(dest_lane_leader_safety)
        self.constraints.append(dest_lane_follower_safety)

    def _safety_constraint_orig_lane_leader(self, states, inputs):
        return self._ocp_interface.lane_changing_safety_constraint(
            states, inputs, self.id, self.get_orig_lane_leader_id(),
            is_other_behind=False)

    def _safety_constraint_dest_lane_leader(self, states, inputs):
        return self._ocp_interface.lane_changing_safety_constraint(
            states, inputs, self.id, self.get_dest_lane_leader_id(),
            is_other_behind=False)

    def _safety_constraint_dest_lane_follower(self, states, inputs):
        return self._ocp_interface.lane_changing_safety_constraint(
            states, inputs, self.id, self.get_dest_lane_follower_id(),
            is_other_behind=True)

    def _solve_ocp(self):
        print("t={:.2f}, veh:{}. Calling ocp solver...".format(
            self.get_current_time(), self.id))

        u0 = self._ocp_interface.get_desired_input()
        n_ctrl_pts = min(round(self._ocp_horizon), 10)  # empirical
        time_pts = np.linspace(0, self._ocp_horizon, n_ctrl_pts, endpoint=True)
        result = opt.solve_ocp(
            self.dynamic_system, time_pts, self._ocp_initial_state,
            cost=self.running_cost,
            trajectory_constraints=self.constraints,
            terminal_cost=self.terminal_cost,
            terminal_constraints=self.terminal_constraints,
            initial_guess=u0,
            minimize_options={'maxiter': self._ocp_solver_max_iter},
            basis=flat.BezierFamily(5, T=self._ocp_horizon))
        # Note: the basis parameter above was set empirically - it might not
        # always work well
        self._ocp_inputs, self._ocp_times = result.inputs, result.time
        self._ocp_has_solution = result.success

        print("Solution{}found".format(
            " " if self._ocp_has_solution else " not "))

        # Warning for when we start working with platoons
        if result.inputs.shape[0] > self._n_optimal_inputs:
            warnings.warn("Too many inputs computed by the optimal control "
                          "solver. Maybe we're optimizing for multiple "
                          "vehicles?")
        elif result.inputs.shape[0] < self._n_optimal_inputs:
            warnings.warn("Too few inputs computed by the optimal control "
                          "solver.")

    def _set_up_lane_change_control(self):
        pass


class SafeAccelOptimalLCVehicle(OptimalControlVehicle):
    """ Safe acceleration (internally computed) and optimal control computed
    lane changes.
    States: [x, y, theta, v], external input: [phi], centered at the C.G.
    and accel is computed by a feedback law"""

    _n_optimal_inputs = 1

    def __init__(self):
        super().__init__()

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Sets the open loop control and phi (steering wheel angle) and
        computes the acceleration
        :param open_loop_controls: Dictionary containing 'phi' value
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = self._compute_acceleration(
            vehicles)
        delta_t = self.get_current_time() - self._lc_start_time
        if delta_t <= self._ocp_horizon:
            self._inputs[self._input_idx['phi']] = self._get_optimal_input()[0]
        else:
            slip_angle = self._compute_lane_keeping_slip_angle()
            self._inputs[self._input_idx['phi']] = (
                self._compute_steering_wheel_angle(slip_angle))

    def _update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        self._choose_min_accel_leader(vehicles)


class ClosedLoopVehicle(FourStateVehicle):
    """ Vehicle that computes all of its inputs by feedback laws.
     States: [x, y, theta, v], external input: None, centered at the C.G. """

    # delete after figuring out a better class organization
    _n_optimal_inputs = 0

    def __init__(self):
        super().__init__()
        self.set_mode(modes.CLLaneKeepingMode())

    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

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

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, BaseVehicle]):
        """
        Computes the acceleration and phi (steering wheel angle)
        :param open_loop_controls: irrelevant
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self._compute_acceleration(vehicles))
        slip_angle = self._compute_slip_angle()
        self._inputs[self._input_idx['phi']] = (
            self._compute_steering_wheel_angle(slip_angle))

    def _update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        self._choose_min_accel_leader(vehicles)

    def _compute_slip_angle(self):
        # if self.is_lane_change_complete():
        #     return 0.0

        delta_t = self.get_current_time() - self._lc_start_time
        if delta_t <= self._lane_change_duration:
            yr = sum([self._polynomial_lc_coeffs[i] * delta_t ** i
                      for i in range(len(self._polynomial_lc_coeffs))])
            vyr = sum([i * self._polynomial_lc_coeffs[i] * delta_t ** (i - 1)
                       for i in range(1, len(self._polynomial_lc_coeffs))])
            return self._compute_cbf_slip_angle(yr, vyr)
        else:
            return self._compute_lane_keeping_slip_angle()

    def _compute_polynomial_lc_trajectory(self):
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

    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, BaseVehicle]):
        pass

    def _set_up_lane_change_control(self):
        self.update_target_y()
        self._compute_polynomial_lc_trajectory()


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

    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        pass

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

    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, BaseVehicle]):
        pass

    def _set_up_lane_change_control(self):
        pass

    def _update_target_leader(self, vehicles: Dict[int, BaseVehicle]):
        """
        Does nothing, since this vehicle class does not have autonomous
        longitudinal control
        """
        pass


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
