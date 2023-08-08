from abc import ABC, abstractmethod
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_lane_width = 4


class BaseVehicle(ABC):

    def __init__(self, state_names: List[str], input_names: List[str],
                 free_flow_speed: float):
        self.n_states = len(state_names)
        self.state_names = state_names
        self.n_inputs = len(input_names)
        self.input_names = input_names
        self.state_idx = {state_names[i]: i for i in range(self.n_states)}
        self.input_idx = {input_names[i]: i for i in range(self.n_inputs)}

        # Some parameters
        self.lr = 2  # dist from C.G. to rear wheel
        self.lf = 1  # dist from C.G. to front wheel
        self.wheelbase = self.lr + self.lf
        self.phi_max = 0.1  # max steering wheel angle
        self.free_flow_speed = free_flow_speed

        # Initial states
        # self.initial_state = None

    def get_state(self, states: List, state_name: str) -> float:
        return states[self.state_idx[state_name]]

    def get_input(self, inputs: List, input_name: str) -> float:
        return inputs[self.input_idx[input_name]]

    def dynamics(self, states, inputs, params):
        theta = self.get_state(states, 'theta')
        # phi = np.clip(self.get_input(inputs, 'phi'),
        #               -5, 5)
        phi = self.get_input(inputs, 'phi')
        vel = self.get_vel(states, inputs)

        dxdt = np.zeros(self.n_states)
        self.position_update(vel, theta, float(phi), dxdt)
        self.velocity_update(states, inputs, params, dxdt)

        return dxdt

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

    def create_state_vector(self, x: float, y: float, theta: float,
                            v0: float = None):
        state_vector = np.zeros(self.n_states)
        state_vector[self.state_idx['x']] = x
        state_vector[self.state_idx['y']] = y
        state_vector[self.state_idx['theta']] = theta
        self.set_speed(v0, state_vector)
        return state_vector

    def to_dataframe(self, time: np.ndarray,
                     states: np.ndarray, inputs: np.ndarray) -> pd.DataFrame:
        data = np.concatenate([time.reshape(1, -1), states, inputs])
        columns = (['t'] + [s for s in self.state_names]
                   + [i for i in self.input_names])
        return pd.DataFrame(data=np.transpose(data), columns=columns)

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
    def get_vel(self, states, inputs):
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
    def position_update(self, vel: float, theta: float, phi: float,
                        derivatives: np.ndarray) -> None:
        """ Computes the derivatives of x, y, and theta, and stores them in the
         derivatives array """
        pass

    @abstractmethod
    def velocity_update(self, states, inputs, params, derivatives):
        pass

    # def plot_variable_vs_time(
    #         self, time: Iterable[float], variable_name: str,
    #         states: np.ndarray, inputs: np.ndarray, ax: plt.axes = None):
    #     # self.plot_multi_states_vs_time(time, states, [state_name])
    #     will_show = ax is None
    #     if will_show:
    #         _, ax = plt.subplots()
    #
    #     if variable_name in self.state_names:
    #         variable = states[self.state_idx[variable_name]]
    #     elif variable_name in self.input_names:
    #         variable = inputs[self.input_idx[variable_name]]
    #     else:
    #         raise ValueError("Requested variable is neither a state nor an "
    #                          "input of this model")
    #
    #     line, = ax.plot(time, variable)
    #     ax.set_xlabel("t [sec]")
    #     ax.set_ylabel(variable_name + ' [' + self._units[variable_name] + ']')
    #     if will_show:
    #         plt.tight_layout()
    #         plt.show()
    #     return line
    #
    # def plot_multi_states_vs_time(
    #         self, time: Iterable[float], states: np.ndarray,
    #         inputs: np.ndarray, states_to_plot: List[str]):
    #     n_plots = len(states_to_plot)
    #     fig, ax = plt.subplots(n_plots)
    #     for i in range(n_plots):
    #         self.plot_variable_vs_time(time, states_to_plot[i],
    #                                    states, inputs, ax[i])
    #     fig.tight_layout()
    #     fig.show()
    #
    # def plot_lane_change_and_inputs(self, time, states, inputs, n_vehs):
    #
    #     plt.subplot(3, 1, 1)
    #     for i in range(n_vehs):
    #         x_idx_per_veh = i * self.n_states + self.state_idx['x']
    #         y_idx_per_veh = i * self.n_states + self.state_idx['y']
    #         plt.plot(states[x_idx_per_veh, :], states[y_idx_per_veh],
    #                  label=str(i))
    #     plt.grid(True)
    #     plt.xlabel("x [m]")
    #     plt.ylabel("y [m]")
    #
    #     plt.subplot(3, 1, 2)
    #     for i in range(n_vehs):
    #         v_idx_per_veh = i * self.n_states + self.state_idx['v']
    #         plt.plot(time, states[v_idx_per_veh], label=str(i))
    #     plt.grid(True)
    #     plt.xlabel("t [sec]")
    #     plt.ylabel("v [m/s]")
    #
    #     plt.subplot(3, 1, 3)
    #     for i in range(n_vehs):
    #         plt.plot(time, inputs[i], label=str(i))
    #     plt.grid(True)
    #     plt.xlabel("t [sec]")
    #     plt.ylabel("phi [rad]")
    #
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()


class ThreeStateVehicle(BaseVehicle, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self, free_flow_speed: float):
        super().__init__(self._state_names, self._input_names, free_flow_speed)

    def set_speed(self, v0, state):
        # Does nothing because velocity is an input for this model
        pass

    def velocity_update(self, states, inputs, params, derivatives):
        # Does nothing because velocity is an input for this model
        pass

    def get_vel(self, states, inputs):
        return self.get_input(inputs, 'v')

    def get_desired_input(self) -> List[float]:
        return [self.free_flow_speed, 0]

    def get_input_limits(self) -> (List[float], List[float]):
        return [0, -self.phi_max], [self.free_flow_speed + 5, self.phi_max]


class ThreeStateVehicleRearWheel(ThreeStateVehicle):
    """ From the library's example.
    States: [x, y, theta], inputs: [v, phi], centered at the rear wheels """

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)

    def position_update(self, vel: float, theta: float, phi: float,
                        derivatives: np.ndarray) -> None:
        return self.position_update_rear_wheels(vel, theta, phi, derivatives)


class ThreeStateVehicleCG(ThreeStateVehicle):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)

    def position_update(self, vel: float, theta: float, phi: float,
                        derivatives: np.ndarray) -> None:
        return self.position_update_cg(vel, theta, phi, derivatives)


class FourStateVehicle(BaseVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G. """
    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self, free_flow_speed: float):
        super().__init__(self._state_names, self._input_names, free_flow_speed)
        self.brake_max = -4
        self.accel_max = 2

    def set_speed(self, v0, state):
        state[self.state_idx['v']] = v0
        pass

    def position_update(self, vel: float, theta: float, phi: float,
                        derivatives: np.ndarray) -> None:
        return self.position_update_cg(vel, theta, phi, derivatives)

    def velocity_update(self, states, inputs, params, derivatives):
        derivatives[self.state_idx['v']] = self.get_input(inputs, 'a')

    def get_vel(self, states, inputs):
        return self.get_state(states, 'v')

    def get_desired_input(self) -> List[float]:
        return [0] * self.n_inputs

    def get_input_limits(self) -> (List[float], List[float]):
        return [self.brake_max, -self.phi_max], [self.accel_max, self.phi_max]


class FourStateVehicleAccelFB(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [phi], centered at the C.G.
     and accel is computed by a feedback law"""

    def __init__(self, free_flow_speed: float):
        super().__init__(free_flow_speed)

        # Controller parameters
        self.h = 1.0  # time headway [s]
        self.c = 1.0  # standstill distance [m]
        self.kg = 0.1
        self.kv = 0.5

    def velocity_update(self, states, inputs, params, derivatives):
        derivatives[self.state_idx['v']] = self.compute_accel(
            states, params['leader_states'])

    def compute_accel(self, ego_states: List[float],
                      leader_states: List[float]) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        :param ego_states: ego vehicle states
        :param leader_states: leading vehicle states
        :return: acceleration
        """
        if len(leader_states) == 0:
            accel = self.kv * (self.free_flow_speed - self.get_vel(ego_states,
                                                                   None))
        else:
            gap = (self.get_state(leader_states, 'x')
                   - self.get_state(ego_states, 'x'))
            v_ego = self.get_state(ego_states, 'v')
            v_leader = self.get_state(leader_states, 'v')
            accel = (self.kg * (gap - self.h * v_ego - self.c)
                     + self.kv * (v_leader - v_ego))
        return accel

    def get_input_limits(self) -> (List[float], List[float]):
        return [-self.phi_max], [self.phi_max]
