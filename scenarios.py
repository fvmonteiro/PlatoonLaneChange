from abc import ABC, abstractmethod
import pickle
from typing import List, Type

import control as ct
import control.optimal as opt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import LinearConstraint, NonlinearConstraint

import vehicle_handler
from constants import lane_width, units
import dynamics
from vehicle_array import VehicleArray
from vehicle_handler import BaseVehicle


def load_simulated_scenario(pickle_file_name: str):
    with open(pickle_file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_lane_change(data: pd.DataFrame):
    """
    Plots the lane change in a y vs x plot along with speed and steering wheel
     angle vs time
    :param data:
    :return:
    """
    sns.set_style('whitegrid')
    x_axes = ['x', 't', 't']
    y_axes = ['y', 'v', 'phi']
    plot_scenario_results(x_axes, y_axes, data)


def plot_vehicle_following(data: pd.DataFrame):
    """
    Plots the x and vel vs time
    :param data:
    :return:
    """
    sns.set_style('whitegrid')
    data['gap'] = 0
    data.loc[data['id'] == 1, 'gap'] = (data.loc[data['id'] == 0, 'x']
                                        - data.loc[data['id'] == 1, 'x'])
    x_axes = ['t', 't']
    y_axes = ['gap', 'v']
    plot_scenario_results(x_axes, y_axes, data)


def plot_scenario_results(x_axes: List[str], y_axes: List[str],
                          data: pd.DataFrame):
    """

    :param x_axes: Name of the variable on the x-axis for each plot
    :param y_axes: Name of the variable on the y-axis for each plot
    :param data:
    :return:
    """
    fig, ax = plt.subplots(len(y_axes))
    for i, (x, y) in enumerate(zip(x_axes, y_axes)):
        sns.lineplot(data, x=x, y=y, hue='id', ax=ax[i])
        low, high = ax[i].get_ylim()
        if y == 'v' and high - low < 1:
            low, high = np.floor(low - 0.5), np.ceil(high + 0.5)
        ax[i].set(xlabel=_get_variable_with_unit(x),
                  ylabel=_get_variable_with_unit(y),
                  ylim=(low, high))
    fig.tight_layout()
    fig.show()


class SimulationScenario(ABC):
    def __init__(self, n_per_lane: List[int], vehicle_class: Type[BaseVehicle],
                 free_flow_speed: float):
        self.n_per_lane = n_per_lane
        self.vehicle_array = VehicleArray()
        self.vehicle_array.create_uniform_array(n_per_lane, vehicle_class,
                                                free_flow_speed)

        vehicle_handler.BaseVehicle.reset_vehicle_counter()
        # Simulation parameters
        self.initial_state, self.tf = None, None
        # Simulation results
        self.time, self.states, self.inputs = None, None, None

    def save_response_data(self, file_name: str) -> None:
        """
        Pickles time, inputs and states as a dataframe
        :param file_name:
        :return:
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.response_to_dataframe(),
                        f, pickle.HIGHEST_PROTOCOL)

    def set_boundary_conditions(self, tf: float):
        """ Sets the initial state, final time and, depending on the scenario,
        desired final states """
        self.tf = tf
        self.create_initial_state()
        self.create_final_state()

    def create_final_state(self):
        """ Default behavior: no final state specification """
        pass

    def response_to_dataframe(self):
        return self.vehicle_array.to_dataframe(
            self.time, self.states, self.inputs)

    @abstractmethod
    def create_initial_state(self):
        pass

    @abstractmethod
    def run(self, parameters):
        """

        :param parameters: Parameters depend on the concrete implementation
        :return:
        """
        pass


class OptimalControlScenario(SimulationScenario, ABC):

    def __init__(self, n_per_lane: List[int], vehicle_class: Type[BaseVehicle],
                 free_flow_speed: float):
        super().__init__(n_per_lane, vehicle_class, free_flow_speed)

        # All must be set by the derived class
        self.dynamic_system = None
        self.xf = None, None
        self.running_cost, self.terminal_cost = None, None
        self.terminal_constraints, self.constraints = None, None
        # self.response = None

    def create_dynamic_system(self) -> None:
        params = {'vehicle_array': self.vehicle_array,
                  'test': True}

        # Define the vehicle steering dynamics as an input/output system
        input_names = self.vehicle_array.create_input_names()
        output_names = self.vehicle_array.create_output_names()

        n_states = self.vehicle_array.n_states
        self.dynamic_system = ct.NonlinearIOSystem(
            dynamics.vehicle_update, dynamics.vehicle_output,
            params=params, states=n_states, name='vehicle_array',
            inputs=input_names, outputs=output_names)

    def set_optimal_control_problem_functions(self, tf: float):
        self.set_boundary_conditions(tf)
        self.set_costs()
        self.set_constraints()

    def set_input_boundaries(self):
        input_lower_bounds, input_upper_bounds = (
            self.vehicle_array.get_input_limits())
        self.constraints = [opt.input_range_constraint(
            self.dynamic_system, input_lower_bounds,
            input_upper_bounds)]

    def solve(self, max_iter: int = 100):
        # Initial guess; not initial control
        u0 = self.vehicle_array.get_desired_input()
        timepts = np.linspace(0, self.tf, (self.tf + 1), endpoint=True)
        result = opt.solve_ocp(
            self.dynamic_system, timepts, self.initial_state, cost=self.running_cost,
            trajectory_constraints=self.constraints,
            terminal_cost=self.terminal_cost,
            terminal_constraints=self.terminal_constraints,
            initial_guess=u0, minimize_options={'maxiter': max_iter})
        return timepts, result

    def run(self, max_iter: int = 100):
        """

        :param max_iter: Maximum number of iterations of the optimal control
         problem solver.
        :return: Nothing. Results are stored internally
        """
        timepts, result = self.solve(max_iter)

        # Simulate the system dynamics (open loop)
        response = ct.input_output_response(
            self.dynamic_system, timepts, result.inputs, self.initial_state,
            t_eval=np.linspace(0, self.tf, 100))
        self.time = response.time
        self.states = response.states
        self.inputs = response.inputs

    @abstractmethod
    def set_costs(self):
        """ Sets the running and terminal state costs """
        pass

    @abstractmethod
    def set_constraints(self):
        """ Sets all the problem constraints """
        pass


class ExampleScenario(OptimalControlScenario):
    """
    Two-lane scenario where all vehicles want to perform a lane change starting
    at t=0 and ending at tf.
    The scenario is used for testing the different dynamical models and
    variations in parameters
    """

    def create_initial_state(self):
        v0 = [veh.free_flow_speed for veh in self.vehicle_array.vehicles]
        gap = v0[0] + 1
        x0 = []
        y0 = []
        for lane in range(len(self.n_per_lane)):
            lane_center = lane * lane_width
            n = self.n_per_lane[lane]
            y0.extend([lane_center] * n)
            for i in range(n):
                x0.append(gap * (n - i - 1))
        theta0 = [0] * self.vehicle_array.n_vehs
        self.initial_state = self.vehicle_array.create_full_state_vector(
            x0, y0, theta0, v0)
        print(self.initial_state)

    def create_final_state(self):
        vf = [veh.free_flow_speed for veh in self.vehicle_array.vehicles]
        xf = []
        yf = []
        for lane in range(len(self.n_per_lane)):
            lane_center = lane * lane_width + (-1) ** lane * lane_width
            n = self.n_per_lane[lane]
            yf.extend([lane_center] * n)
            for i in range(n):
                xf.append(x0 + vf * self.tf)
        thetaf = [0] * self.vehicle_array.n_vehs
        self.xf = self.vehicle_array.create_full_state_vector(
            xf, yf, thetaf, vf)
        print(self.xf)

        # y_idx = self.vehicle_array.get_state_indices('y')
        #
        # self.xf = self.vehicle_array.create_desired_final_state(self.tf)
        # print(self.xf)

    def set_costs(self):
        Q, R, P = self.create_simple_weight_matrices()
        # Desired control; not final control
        uf = self.vehicle_array.get_desired_input()
        self.running_cost = opt.quadratic_cost(self.dynamic_system,
                                               Q, R, x0=self.xf, u0=uf)
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.xf)

    def create_simple_weight_matrices(self):
        # don't turn too sharply: matrix Q
        state_cost_weights = [0.] * self.vehicle_array.n_states
        theta_idx = self.vehicle_array.get_state_indices('theta')
        for i in theta_idx:
            state_cost_weights[i] = 0.1
        state_cost = np.diag(state_cost_weights)

        # keep inputs small: matrix R
        input_cost_weights = [1] * self.vehicle_array.n_inputs
        input_cost = np.diag(input_cost_weights)

        # get close to final point: matrix P
        if ('v' in self.vehicle_array.vehicles[0].input_names
                or 'a' in self.vehicle_array.vehicles[0].input_names):
            final_state_weights = [1000] * self.vehicle_array.n_states
        else:
            # If we can't control speed or acceleration, we shouldn't care about
            # final position and velocity
            final_state_weights = [0, 1000, 1000, 0] * self.vehicle_array.n_vehs
        terminal_cost = np.diag(final_state_weights)
        return state_cost, input_cost, terminal_cost

    def set_constraints(self):
        self.set_input_boundaries()


class LaneChangeWithConstraints(OptimalControlScenario):
    """
    Used to figure out how to code constraints
    """

    def __init__(self, free_flow_speed: float):
        super().__init__([1], vehicle_handler.FourStateVehicleAccelFB,
                         free_flow_speed)
        self.min_lc_x = 20

    def create_initial_state(self):
        v0 = self.vehicle_array.vehicles[0].free_flow_speed
        gap = v0 + 1
        self.initial_state = self.vehicle_array.set_initial_state(gap, v0)
        print(self.initial_state)

    def create_final_state(self):
        self.xf = self.vehicle_array.create_desired_final_state(self.tf)
        print(self.xf)

    def set_costs(self):
        # Desired control; not final control
        uf = self.vehicle_array.get_desired_input()
        Q = np.diag([0, 0, 0.1, 0] * self.vehicle_array.n_vehs)
        R = np.diag([0.1] * self.vehicle_array.n_vehs)
        self.running_cost = opt.quadratic_cost(self.dynamic_system, Q, R,
                                               self.xf, uf)
        P = np.diag([0, 1000, 1000, 0])
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.xf)

    def set_constraints(self):
        self.set_input_boundaries()
        # Linear constraint: lb <= A*[x; u] <= ub
        # vel_con = LinearConstraint([0, 0, 0, 1, 0, 0], 0, 15)
        # self.constraints.append(vel_con)
        # TODO: not getting the expected result. Could try a 'less' nonlinear
        #  constraint?

        epsilon = 1e-6
        nlc = NonlinearConstraint(self.safe_constraint, -epsilon, epsilon)

        self.constraints.append(nlc)

    def safe_constraint(self, states, inputs):
        x = states[0]
        phi = inputs[0]
        return min(x - self.min_lc_x, 0) * phi

    def smooth_safe_constraint(self, states, inputs):
        epsilon = 1e-5
        x = states[0] - self.min_lc_x
        phi = inputs[0]
        # Smooth min(x, 0)
        if x < -epsilon:
            min_x_0 = x
        elif x > epsilon:
            min_x_0 = 0
        else:
            min_x_0 = -(x - epsilon) ** 2 / 4 / epsilon
        return min_x_0 * phi


class VehicleFollowingScenario(SimulationScenario):
    """
    Scenario to test acceleration feedback laws. No lane changes.
    """

    def __init__(self, n_per_lane: List[int], free_flow_speed: float):
        super().__init__(n_per_lane, vehicle_handler.FourStateVehicleAccelFB,
                         free_flow_speed)

    def create_initial_state(self):
        v0 = self.vehicle_array.vehicles[0].free_flow_speed
        gap = 2
        self.initial_state = self.vehicle_array.set_initial_state(gap, v0)

    def run(self, parameters=None):
        """

        :param parameters: Parameters depend on the concrete implementation
        :return:
        """
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        states = np.zeros([len(self.initial_state), len(time)])
        states[:, 0] = self.initial_state
        steering_angle = np.zeros([self.vehicle_array.n_vehs, len(time)])
        for i in range(len(time) - 1):
            dxdt = self.vehicle_array.update(states[:, i], steering_angle[:, i],
                                             None)
            states[:, i + 1] = states[:, i] + dxdt * (time[i + 1] - time[i])
        self.time, self.states, self.inputs = time, states, steering_angle


def _get_variable_with_unit(variable: str):
    return variable + ' [' + units[variable] + ']'
