from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


class BaseVehicle(ABC):

    _counter = 0

    def __init__(self, free_flow_speed):
        """

        :param free_flow_speed: Desired speed if there's no vehicle ahead
        """
        self.state_names, self.n_states = None, None
        self.input_names, self.n_inputs = None, None
        self.state_idx, self.input_idx = {}, {}
        self.initial_state = None
        self.leader_id = -1

        # Some parameters
        self.id = BaseVehicle._counter
        BaseVehicle._counter += 1
        self.free_flow_speed = free_flow_speed
        self.lr = 2  # dist from C.G. to rear wheel
        self.lf = 1  # dist from C.G. to front wheel
        self.wheelbase = self.lr + self.lf
        self.phi_max = 0.1  # max steering wheel angle

    def __repr__(self):
        return self.__class__.__name__

    @staticmethod
    def reset_vehicle_counter():
        BaseVehicle._counter = 0

    # TODO: maybe most of these methods could be class methods since they don't
    #  depend on any 'internal' value of the instance
    def get_state(self, states: List, state_name: str) -> float:
        return states[self.state_idx[state_name]]

    def get_input(self, inputs: List, input_name: str) -> float:
        return inputs[self.input_idx[input_name]]

    def has_leader(self):
        return self.leader_id >= 0

    def compute_safe_gap(self, v_ego):
        return 0  # we ignore safety for vehicles without accel feedback

    def dynamics(self, ego_states, inputs, leader_states):
        theta = self.get_state(ego_states, 'theta')
        phi = self.get_input(inputs, 'phi')
        vel = self.get_vel(ego_states, inputs)

        accel = self.compute_acceleration(ego_states, inputs, leader_states)
        dxdt = np.zeros(self.n_states)
        self.compute_derivatives(vel, theta, phi, accel, dxdt)
        return dxdt
        # return self.compute_derivatives(ego_states, inputs, leader_states)

    def position_update_cg(self, vel: float, theta: float, phi: float,
                           derivatives: np.ndarray) -> None:

        beta = np.arctan(self.lr * np.tan(phi) / (self.lf + self.lr))
        derivatives[self.state_idx['x']] = vel * np.cos(theta + beta)
        derivatives[self.state_idx['y']] = vel * np.sin(theta + beta)
        derivatives[self.state_idx['theta']] = (vel * np.sin(beta)
                                                / self.lr)

    def position_update_rear_wheels(self, vel: float, theta: float, phi: float,
                                    derivatives: np.ndarray):
        derivatives[self.state_idx['x']] = vel * np.cos(theta)
        derivatives[self.state_idx['y']] = vel * np.sin(theta)
        derivatives[self.state_idx['theta']] = (vel * np.tan(phi)
                                                / self.wheelbase)

    def set_initial_state(self, x: float, y: float, theta: float,
                          v: float = None):
        self.initial_state = self.create_state_vector(x, y, theta, v)

    def create_state_vector(self, x: float, y: float, theta: float,
                            v: float = None):
        state_vector = np.zeros(self.n_states)
        state_vector[self.state_idx['x']] = x
        state_vector[self.state_idx['y']] = y
        state_vector[self.state_idx['theta']] = theta
        self.set_speed(v, state_vector)
        return state_vector

    def to_dataframe(self, time: np.ndarray,
                     states: np.ndarray, inputs: np.ndarray) -> pd.DataFrame:
        data = np.concatenate([time.reshape(1, -1), states, inputs])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        return pd.DataFrame(data=np.transpose(data), columns=columns)

    def get_vel(self, states, inputs):
        try:
            return states[self.state_idx['v']]
        except KeyError:
            return inputs[self.input_idx['v']]

    @abstractmethod
    def set_speed(self, v0, state):
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
    def compute_derivatives(self, vel, theta, phi, accel, derivatives):
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


class ThreeStateVehicle(BaseVehicle, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)
        self._set_model(self._state_names, self._input_names)

    def set_speed(self, v0, state):
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

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)

    def compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self.position_update_rear_wheels(vel, theta, phi, derivatives)


class ThreeStateVehicleCG(ThreeStateVehicle):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)

    def compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self.position_update_cg(vel, theta, phi, derivatives)


class FourStateVehicle(BaseVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G. """

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)
        self._set_model(self._state_names, self._input_names)
        self.brake_max = -4
        self.accel_max = 2

    def set_speed(self, v0, state):
        state[self.state_idx['v']] = v0

    def compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self.position_update_cg(vel, theta, phi, derivatives)
        derivatives[self.state_idx['v']] = accel
        return derivatives

    def compute_acceleration(self, ego_states, inputs, leader_states):
        return self.get_input(inputs, 'a')

    def get_desired_input(self) -> List[float]:
        return [0] * self.n_inputs

    def get_input_limits(self) -> (List[float], List[float]):
        return [self.brake_max, -self.phi_max], [self.accel_max, self.phi_max]


class FourStateVehicleAccelFB(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [phi], centered at the C.G.
     and accel is computed by a feedback law"""

    _input_names = ['phi']

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)

        # Controller parameters
        self.h = 1.0  # time headway [s]
        self.c = 1.0  # standstill distance [m]
        self.kg = 0.5
        self.kv = 0.5

    def compute_acceleration(self, ego_states, inputs, leader_states) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        v_ego = self.get_state(ego_states, 'v')
        if leader_states is None or len(leader_states) == 0:
            accel = self.kv * (self.free_flow_speed - v_ego)
        else:
            gap = (self.get_state(leader_states, 'x')
                   - self.get_state(ego_states, 'x'))
            v_leader = self.get_state(leader_states, 'v')
            accel = (self.kg * (gap - self.h * v_ego - self.c)
                     + self.kv * (v_leader - v_ego))
        return accel

    def get_input_limits(self) -> (List[float], List[float]):
        return [-self.phi_max], [self.phi_max]

    def compute_safe_gap(self, v_ego):
        return self.h * v_ego - self.c


class LongitudinalVehicle(FourStateVehicleAccelFB):
    """ Vehicle that does not perform lane change and computes its own
    acceleration """

    _input_names = []

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)
        self._set_model(self._state_names, self._input_names)

    def get_input(self, inputs: List, input_name: str) -> float:
        return 0.0  # all inputs are computed internally

    def get_desired_input(self) -> List[float]:
        return []

    def get_input_limits(self) -> (List[float], List[float]):
        return [], []
