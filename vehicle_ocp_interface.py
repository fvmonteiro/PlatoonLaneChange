from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd

import vehicle_models.base_vehicle as base
import vehicle_models.three_state_vehicles as tsv
import vehicle_models.four_state_vehicles as fsv


class BaseVehicleInterface(ABC):

    def __init__(self, vehicle: base.BaseVehicle):
        """

        """
        self.state_names, self.n_states = None, None
        self.input_names, self.n_inputs = None, None
        self.state_idx, self.input_idx = {}, {}

        self.free_flow_speed = vehicle.free_flow_speed
        self._orig_leader_id: int = vehicle.get_orig_lane_leader_id()
        self._destination_leader_id: int = vehicle.get_dest_lane_leader_id()
        self._destination_follower_id: int = vehicle.get_dest_lane_follower_id()
        self._leader_id = vehicle.get_current_leader_id()
        self.target_lane = vehicle.target_lane
        self.target_y = vehicle.target_y
        # The vehicle's current state is the starting point for the ocp
        self.initial_state = vehicle.get_states()

        # Some parameters
        self.id = vehicle.id
        self.name = vehicle.name
        self.lr = vehicle.lr  # dist from C.G. to rear wheel
        self.lf = vehicle.lf  # dist from C.G. to front wheel
        self.wheelbase = self.lr + self.lf
        self.phi_max = vehicle.phi_max  # max steering wheel angle
        self.safe_h = vehicle.safe_h
        self.c = vehicle.c  # standstill distance [m]

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

    def get_current_leader_id(self):
        """
        The 'current' leader is the vehicle being used to define this vehicle's
         acceleration
        :return:
        """
        return self._leader_id

    def has_leader(self):
        return self._leader_id >= 0

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


class FourStateVehicleInterface(BaseVehicleInterface, ABC):

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self, vehicle: fsv.FourStateVehicle):
        super().__init__(vehicle)
        self._set_model(self._state_names, self._input_names)
        self.brake_max = vehicle.brake_max
        self.accel_max = vehicle.accel_max

    def get_desired_input(self) -> List[float]:
        return [0] * self.n_inputs

    # def get_vel(self, values):
    #     return self.select_state_from_vector(values, 'v')

    def _set_speed(self, v0, state):
        state[self.state_idx['v']] = v0

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self._position_derivative_cg(vel, theta, phi, derivatives)
        derivatives[self.state_idx['v']] = accel


class OpenLoopVehicleInterfaceInterface(FourStateVehicleInterface):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G. """

    def __init__(self, vehicle: fsv.OpenLoopVehicle):
        super().__init__(vehicle)

    def compute_acceleration(self, ego_states, inputs, leader_states):
        return self.select_input_from_vector(inputs, 'a')

    def get_input_limits(self) -> (List[float], List[float]):
        return [self.brake_max, -self.phi_max], [self.accel_max, self.phi_max]


class SafeAccelVehicleInterface(FourStateVehicleInterface):
    """ States: [x, y, theta, v], inputs: [phi], centered at the C.G.
     and accel is computed by a feedback law"""

    _input_names = ['phi']

    def __init__(self,
                 vehicle: Union[fsv.FourStateVehicle]):
        super().__init__(vehicle)

        # Controller parameters
        self.h = vehicle.h  # time headway [s]
        # TODO: concentrate all accel computation functions in the control class
        self.long_controller = vehicle.long_controller

    def get_input_limits(self) -> (List[float], List[float]):
        return [-self.phi_max], [self.phi_max]

    def compute_acceleration(self, ego_states, inputs, leader_states) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.select_state_from_vector(ego_states, 'v')
        v_ff = self.free_flow_speed
        if leader_states is None or len(leader_states) == 0:
            return self.long_controller.compute_velocity_control(v_ff, v_ego)
        else:
            gap = (self.select_state_from_vector(leader_states, 'x')
                   - self.select_state_from_vector(ego_states, 'x'))
            v_leader = self.select_state_from_vector(leader_states, 'v')
            accel = self.long_controller.compute_gap_control(gap, v_ego,
                                                             v_leader)
            if v_ego >= v_ff and accel > 0:
                return self.long_controller.compute_velocity_control(
                    v_ff, v_ego)
            return accel


class ClosedLoopVehicleInterface(SafeAccelVehicleInterface):
    """ Vehicle that does not perform lane change and computes its own
    acceleration """

    _input_names = []

    def __init__(self, vehicle: fsv.ClosedLoopVehicle):
        super().__init__(vehicle)
        # self._set_model(self._state_names, self._input_names)

    def select_input_from_vector(self, inputs: List, input_name: str) -> float:
        return 0.0  # all inputs are computed internally

    def get_desired_input(self) -> List[float]:
        return []

    def get_input_limits(self) -> (List[float], List[float]):
        return [], []


# =========================== Three-State Vehicles =========================== #
# Three-state vehicles are used in initial tests with the optimization tool
# since they are simpler and were used in the tool's example.
class ThreeStateVehicleInterface(BaseVehicleInterface, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self, vehicle: tsv.ThreeStateVehicle):
        super().__init__(vehicle)
        self._set_model(self._state_names, self._input_names)

    def _set_speed(self, v0, state):
        # Does nothing because velocity is an input for this model
        pass

    def compute_acceleration(self, ego_states, inputs, leader_states):
        # Does nothing because velocity is an input for this model
        pass

    def get_desired_input(self) -> List[float]:
        return [self.free_flow_speed, 0]

    def get_input_limits(self) -> (List[float], List[float]):
        return [0, -self.phi_max], [self.free_flow_speed + 5, self.phi_max]


class ThreeStateVehicleRearWheelInterface(ThreeStateVehicleInterface):
    """ From the library's example.
    States: [x, y, theta], inputs: [v, phi], centered at the rear wheels """

    def __init__(self, vehicle: tsv.ThreeStateVehicleRearWheel):
        super().__init__(vehicle)

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self._position_derivative_rear_wheels(vel, theta, phi, derivatives)


class ThreeStateVehicleCGInterface(ThreeStateVehicleInterface):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self, vehicle: tsv.ThreeStateVehicleRearWheel):
        super().__init__(vehicle)

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self._position_derivative_cg(vel, theta, phi, derivatives)
