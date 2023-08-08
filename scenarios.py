from abc import ABC, abstractmethod
from typing import List

import control as ct
import control.optimal as opt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import dynamics
from vehicle_array import VehicleArray
from vehicle_handler import BaseVehicle


class SimulationScenario(ABC):
    _units = {'x': 'm', 'y': 'm', 'theta': 'rad', 'v': 'm/s', 'a': 'm/s^2',
              'phi': 'rad'}

    def __init__(self, n_per_lane: List[int]):
        self.n_per_lane = n_per_lane
        self.n_vehs = sum(n_per_lane)

        # All must be set by the derived class
        self.vehicle_class, self.vehicle_array = None, None
        self.dynamic_system = None
        self.x0, self.xf = None, None
        self.tf = None
        self.running_cost, self.terminal_cost = None, None
        self.constraints = None
        self.response = None

    def create_dynamic_system(self, vehicle_class: BaseVehicle) -> None:
        self.vehicle_class = vehicle_class
        self.vehicle_array = VehicleArray(self.n_per_lane, vehicle_class)
        params = {'vehicle_type': self.vehicle_class,
                  'n_per_lane': self.n_per_lane}

        # Define the vehicle steering dynamics as an input/output system
        input_names = self.vehicle_array.create_input_names()
        output_names = self.vehicle_array.create_output_names()

        n_states = self.vehicle_class.n_states
        self.dynamic_system = ct.NonlinearIOSystem(
            dynamics.vehicle_update, dynamics.vehicle_output,
            params=params, states=n_states * self.n_vehs, name='vehicle_array',
            inputs=input_names, outputs=output_names)

    def set_scenario(self):
        self.set_boundary_conditions()
        self.set_costs()
        self.set_constraints()

    def set_input_boundaries(self):
        input_lower_bounds, input_upper_bounds = (
            self.vehicle_class.get_input_limits())
        self.constraints = [opt.input_range_constraint(
            self.dynamic_system, input_lower_bounds * self.n_vehs,
            input_upper_bounds * self.n_vehs)]

    def plot_trajectory_and_inputs(self):
        data = self.vehicle_array.to_dataframe(
            self.response.time, self.response.states, self.response.inputs)
        fig, ax = plt.subplots(3)
        sns.set_style('whitegrid')  # TODO: ignored?
        x_axes = ['x', 't', 't']
        y_axes = ['y'] + [i for i in self.vehicle_class.input_names]
        sns.lineplot(data, x='x', y='y', hue='id', ax=ax[0])
        ax[0].set(xlabel='x [m]', ylabel='y [m]')
        sns.lineplot(data, x='t', y='v', hue='id', ax=ax[1])
        low, high = ax[1].get_ylim()
        if high - low < 1:
            low, high = np.floor(low - 1), np.ceil(high + 1)
        ax[1].set(xlabel='t [s]', ylabel='v [m/s]',
                  ylim=(low, high))
        sns.lineplot(data, x='t', y='phi', hue='id', ax=ax[2])
        ax[2].set(xlabel='t [s]', ylabel='phi [rad]')
        fig.tight_layout()
        fig.show()

    @abstractmethod
    def set_boundary_conditions(self):
        """ Sets the initial and desired final states """
        pass

    @abstractmethod
    def set_costs(self):
        """ Sets the running and terminal state costs """
        pass

    @abstractmethod
    def set_constraints(self):
        """ Sets all the problem constraints """
        pass


class ExampleScenario(SimulationScenario):
    """
    Two-lane scenario where all vehicles want to perform a lane change starting
    at t=0 and ending at tf.
    The scenario is used for testing the different dynamical models and
    variations in parameters
    """

    def solve(self, max_iter: int = 100):
        # Initial guess; not initial control
        u0 = self.vehicle_class.get_desired_input() * self.n_vehs
        timepts = np.linspace(0, self.tf, self.tf + 1, endpoint=True)
        result = opt.solve_ocp(
            self.dynamic_system, timepts, self.x0, self.running_cost,
            self.constraints, terminal_cost=self.terminal_cost,
            initial_guess=u0, minimize_options={'maxiter': max_iter})

        # Simulate the system dynamics (open loop)
        self.response = ct.input_output_response(
            self.dynamic_system, timepts, result.inputs, self.x0,
            t_eval=np.linspace(0, self.tf, 100))

    def set_boundary_conditions(self):
        v0 = self.vehicle_class.free_flow_speed
        gap = 10
        self.x0 = self.vehicle_array.create_initial_state(gap, v0)
        self.tf = 10
        self.xf = self.vehicle_array.create_desired_final_state(v0, self.tf)

    def set_costs(self):
        Q, R, P = self.create_simple_weight_matrices()
        # Desired control; not final control
        uf = self.vehicle_class.get_desired_input() * self.n_vehs
        self.running_cost = opt.quadratic_cost(self.dynamic_system,
                                               Q, R, x0=self.xf, u0=uf)
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.xf)

    def create_simple_weight_matrices(self):
        # don't turn too sharply: matrix Q
        state_cost_weights = [0] * self.vehicle_class.n_states
        theta_idx = self.vehicle_class.state_idx['theta']
        state_cost_weights[theta_idx] = 0.1
        state_cost = np.diag(state_cost_weights * self.n_vehs)

        # keep inputs small: matrix R
        input_cost_weights = [1] * self.vehicle_class.n_inputs
        input_cost = np.diag(input_cost_weights * self.n_vehs)

        # get close to final point: matrix P
        final_state_weights = [1000] * self.vehicle_class.n_states
        terminal_cost = np.diag(final_state_weights * self.n_vehs)
        return state_cost, input_cost, terminal_cost

    def set_constraints(self):
        self.set_input_boundaries()
