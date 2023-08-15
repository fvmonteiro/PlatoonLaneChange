from abc import ABC, abstractmethod
import pickle
from typing import List, Type

import control as ct
import control.optimal as opt
import numpy as np

from scipy.optimize import LinearConstraint, NonlinearConstraint

import vehicle_handler
from constants import lane_width
import dynamics
from vehicle_array import VehicleArray
from vehicle_handler import BaseVehicle


class SimulationScenario(ABC):
    def __init__(self):
        self.n_per_lane = []
        self.vehicle_array = VehicleArray()
        vehicle_handler.BaseVehicle.reset_vehicle_counter()
        # Simulation parameters
        self.initial_state, self.tf = None, None
        # Simulation results
        self.time, self.states, self.inputs = None, None, None

    def create_uniform_vehicles(
            self, n_per_lane: List[int], vehicle_class: Type[BaseVehicle],
            free_flow_speed: float):
        self.n_per_lane = n_per_lane
        self.vehicle_array.create_uniform_array(sum(n_per_lane),
                                                vehicle_class, free_flow_speed)

    def create_vehicles(self, vehicle_classes: List[List[Type[BaseVehicle]]],
                        free_flow_speeds: List[float]):
        for i in range(len(vehicle_classes)):
            self.n_per_lane.append(len(vehicle_classes[i]))
        flat_vehicle_list = [item for sublist in vehicle_classes for
                             item in sublist]
        self.vehicle_array.create_vehicle_array(flat_vehicle_list,
                                                free_flow_speeds)

    def save_response_data(self, file_name: str) -> None:
        """
        Pickles time, inputs and states as a dataframe
        :param file_name:
        :return:
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.response_to_dataframe(),
                        f, pickle.HIGHEST_PROTOCOL)

    def place_vehicles(self, gaps: List[List[float]], v0: List[float]):
        """
        Place vehicles at the center of their respective lanes and with zero
        orientation. Inter-vehicle distances and initial speeds for each
        vehicle must be passed as parameters.

        :param gaps: Inter-vehicle distance between consecutive vehicles on the
         same lane. Gaps must be of size: n_lanes x (n_vehicles_on_lane - 1),
         and the sequence is from first to last vehicle.
        :param v0: Initial speed for each vehicle
        :return:
        """

        x0 = []
        y0 = []
        for lane in range(len(self.n_per_lane)):
            lane_center = lane * lane_width
            n = self.n_per_lane[lane]
            y0.extend([lane_center] * n)
            veh_position = sum(gaps[lane])
            for i in range(n):
                x0.append(veh_position)
                veh_position -= gaps[lane][i]
        theta0 = [0] * self.vehicle_array.n_vehs
        self.initial_state = self.vehicle_array.create_full_state_vector(
            x0, y0, theta0, v0)

    def place_equally_spaced_vehicles(self, gap: float = None):
        """
        All vehicles start at the center of their respective lanes, with
        orientation angle zero, and at the same speed, which equals their
        desired free-flow speed. Vehicles on the same lane are 'gap' meters
        distant from each other.

        :param gap: Inter-vehicle distance for vehicles on the same lane. If
         the value is not given, it defaults to (v_ff + 1)
        :return:
        """
        v0 = [veh.free_flow_speed for veh in self.vehicle_array.vehicles]
        if gap is None:
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

    def __init__(self):
        super().__init__()

        # All must be set by the derived class
        self.dynamic_system = None
        self.final_state = None, None
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

    def set_free_flow_all_change_lanes_final_state(self):
        """
        Sets the final state where all vehicles change from their initial
        lanes while traveling at their free-flow speed.

        :return:
        """
        v_ff = [vehicle.free_flow_speed
                for vehicle in self.vehicle_array.vehicles]
        delta_x = []
        delta_y = []
        veh_counter = 0
        for lane in range(len(self.n_per_lane)):
            n = self.n_per_lane[lane]
            for i in range(n):
                delta_x.append(v_ff[veh_counter] * self.tf)
                delta_y.append((-1) ** lane * lane_width)
                veh_counter += 1
        delta_theta = [0.0] * self.vehicle_array.n_vehs
        delta_v = [0.0] * self.vehicle_array.n_vehs
        delta_state = self.vehicle_array.create_full_state_vector(
            delta_x, delta_y, delta_theta, delta_v)
        self.final_state = self.initial_state + delta_state

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
            self.dynamic_system, timepts, self.initial_state,
            cost=self.running_cost,
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
        self.place_equally_spaced_vehicles()
        self.vehicle_array.assign_leaders(self.initial_state)
        print(self.initial_state)

    def create_final_state(self):
        self.set_free_flow_all_change_lanes_final_state()
        print(self.final_state)

    def set_costs(self):
        Q, R, P = self.create_simple_weight_matrices()
        # Desired control; not final control
        uf = self.vehicle_array.get_desired_input()
        self.running_cost = opt.quadratic_cost(self.dynamic_system,
                                               Q, R, x0=self.final_state, u0=uf)
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.final_state)

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


class MinimumTimeControl(OptimalControlScenario):
    """
    Used to figure out how to get a minimum time controller
    """
    def __init__(self):
        super().__init__()
        self.create_uniform_vehicles(
            [1], vehicle_handler.FourStateVehicle, 10)

    def create_initial_state(self):
        self.place_equally_spaced_vehicles()
        self.vehicle_array.assign_leaders(self.initial_state)
        print(self.initial_state)

    def create_final_state(self):
        self.set_free_flow_all_change_lanes_final_state()
        print(self.final_state)

    def set_costs(self):
        Q, R, P = self.create_simple_weight_matrices()
        # Desired control; not final control
        uf = self.vehicle_array.get_desired_input()
        self.running_cost = opt.quadratic_cost(self.dynamic_system,
                                               Q, R, x0=self.final_state,
                                               u0=uf)
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.final_state)

    def set_constraints(self):
        self.set_input_boundaries()


class LaneChangeWithConstraints(OptimalControlScenario):
    """
    Used to test how to code safety constraints
    """

    def __init__(self, n_dest_lane_vehs: int, v_ff: float,
                 lane_changing_vehicle_id: int = 1):
        super().__init__()
        self.lc_id = lane_changing_vehicle_id
        veh_classes = [[]]
        n_orig_lane_vehs = 3
        for i in range(n_orig_lane_vehs):
            if i == lane_changing_vehicle_id:
                veh_classes[0].append(vehicle_handler.FourStateVehicleAccelFB)
            else:
                veh_classes[0].append(vehicle_handler.LongitudinalVehicle)
        if n_dest_lane_vehs > 0:
            veh_classes.append([])
        for i in range(n_dest_lane_vehs):
            veh_classes[1].append(vehicle_handler.LongitudinalVehicle)
        self.create_vehicles(veh_classes,
                             [v_ff] * (n_orig_lane_vehs + n_dest_lane_vehs))
        self.min_lc_x = 20  # only used for initial tests

    def create_initial_state(self):
        sample_vehicle = self.vehicle_array.vehicles[0]
        v_ff = sample_vehicle.free_flow_speed
        gap_steady_state = v_ff + 1
        gaps = [[]]
        # Orig lane
        for i in range(self.n_per_lane[0]):
            if i == self.lc_id:  # TODO: gap setting is wrong by one index
                gaps[0].append(gap_steady_state - 2)
            else:
                gaps[0].append(gap_steady_state)
        # Dest lane
        if len(self.n_per_lane) > 1:
            gaps.append([])
            for i in range(self.n_per_lane[1]):
                gaps[1].append(gap_steady_state)

        self.place_vehicles(gaps, [v_ff] * sum(self.n_per_lane))
        # self.place_equally_spaced_vehicles(gap_steady_state)
        self.vehicle_array.assign_leaders(self.initial_state)
        print(self.initial_state)

    def create_final_state(self):
        # xf and vf are irrelevant with the accel feedback model
        n_vehs = sum(self.n_per_lane)
        delta_x = [100.0] * n_vehs
        delta_y = [0.0] * n_vehs
        delta_y[self.lc_id] = 4.0
        delta_theta = [0.0] * n_vehs
        delta_v = [0.0] * n_vehs
        delta_state = self.vehicle_array.create_full_state_vector(
            delta_x, delta_y, delta_theta, delta_v)
        # self.set_free_flow_all_change_lanes_final_state()
        self.final_state = self.initial_state + delta_state
        print(self.final_state)

    def set_costs(self):
        # Desired control; not final control
        uf = self.vehicle_array.get_desired_input()
        Q = np.diag([0, 0, 0.1, 0] * self.vehicle_array.n_vehs)
        R = np.diag([0.01] * self.vehicle_array.n_inputs)
        self.running_cost = opt.quadratic_cost(self.dynamic_system, Q, R,
                                               self.final_state, uf)
        P = np.diag([0, 1000, 1000, 0] * self.vehicle_array.n_vehs)
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.final_state)

    def set_constraints(self):
        self.set_input_boundaries()
        # Linear constraint: lb <= A*[x; u] <= ub
        # vel_con = LinearConstraint([0, 0, 0, 1, 0, 0], 0, 15)
        # self.constraints.append(vel_con)

        epsilon = 1e-10
        nlc = NonlinearConstraint(self.safety_constraint,
                                  -epsilon, epsilon)
        self.constraints.append(nlc)

    def safety_constraint(self, states, inputs):
        gap = self.vehicle_array.compute_gap_to_leader(self.lc_id, states)
        lc_vehicle = self.vehicle_array.vehicles[self.lc_id]
        lc_veh_vel = self.vehicle_array.get_a_vehicle_state_by_id(
            self.lc_id, states, 'v')
        safe_gap = lc_vehicle.compute_safe_gap(lc_veh_vel)
        dist_to_safe_gap = gap - 10
        phi = self.vehicle_array.get_a_vehicle_input_by_id(
            self.lc_id, inputs, 'phi')
        return min(dist_to_safe_gap, 0) * phi

    def lane_change_starting_point_constraint(self, states, inputs):
        lc_veh_x = self.vehicle_array.get_a_vehicle_state_by_id(
            self.lc_id, states, 'x')
        phi = self.vehicle_array.get_a_vehicle_input_by_id(
            self.lc_id, inputs, 'phi')
        dist_to_point = lc_veh_x - self.min_lc_x
        return min(dist_to_point, 0) * phi

    def smooth_lane_change_starting_point_constraint(self, states, inputs):
        lc_veh_x = self.vehicle_array.get_a_vehicle_state_by_id(
            self.lc_id, states, 'x')
        phi = self.vehicle_array.get_a_vehicle_input_by_id(
            self.lc_id, inputs, 'phi')
        dist_to_point = lc_veh_x - self.min_lc_x

        # Smooth min(x, 0)
        epsilon = 1e-5
        if dist_to_point < -epsilon:
            min_x_0 = dist_to_point
        elif dist_to_point > epsilon:
            min_x_0 = 0
        else:
            min_x_0 = -(dist_to_point - epsilon) ** 2 / 4 / epsilon
        return min_x_0 * phi


class VehicleFollowingScenario(SimulationScenario):
    """
    Scenario to test acceleration feedback laws. No lane changes.
    """

    def __init__(self, n_per_lane: List[int], v_ff: float):
        super().__init__()
        self.create_uniform_vehicles(n_per_lane,
                                     vehicle_handler.LongitudinalVehicle,
                                     v_ff)

    def create_initial_state(self):
        gap = 2
        self.place_equally_spaced_vehicles(gap)
        self.vehicle_array.assign_leaders(self.initial_state)
        print(self.initial_state)

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
