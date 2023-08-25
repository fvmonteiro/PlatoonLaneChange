from abc import ABC, abstractmethod
import pickle
from typing import Dict, List, Type, Union

import control as ct
import control.optimal as opt
import numpy as np
import pandas as pd

from scipy.optimize import LinearConstraint, NonlinearConstraint

import vehicle_models
from constants import lane_width
from vehicle_group import VehicleGroup
from vehicle_models import BaseVehicle
import vehicle_group_ocp_interface as vgi


class SimulationScenario(ABC):
    def __init__(self):
        self.n_per_lane = []
        self.vehicle_group = VehicleGroup()
        vehicle_models.BaseVehicle.reset_vehicle_counter()
        # Simulation parameters
        self.initial_state, self.tf = None, None

    def create_uniform_vehicles(
            self, n_per_lane: List[int], vehicle_class: Type[BaseVehicle],
            free_flow_speed: float):
        self.n_per_lane = n_per_lane
        self.vehicle_group.create_vehicle_array(
            [vehicle_class] * sum(n_per_lane))
        self.vehicle_group.set_free_flow_speeds(free_flow_speed)

    def create_vehicles(self, vehicle_classes: List[List[Type[BaseVehicle]]]):
        for i in range(len(vehicle_classes)):
            self.n_per_lane.append(len(vehicle_classes[i]))
        flat_vehicle_list = [item for sublist in vehicle_classes for
                             item in sublist]
        self.vehicle_group.create_vehicle_array(flat_vehicle_list)

    def set_free_flow_speeds(self,
                             free_flow_speeds: Union[float, List, np.ndarray]):
        self.vehicle_group.set_free_flow_speeds(free_flow_speeds)

    def save_response_data(self, file_name: str) -> None:
        """
        Pickles time, inputs and states as a dataframe
        :param file_name:
        :return:
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.response_to_dataframe(),
                        f, pickle.HIGHEST_PROTOCOL)

    def place_equally_spaced_vehicles(self, gap: float = None):
        """
        All vehicles start at the center of their respective lanes, with
        orientation angle zero, and at the same speed, which equals their
        desired free-flow speed. Vehicles on the same lane are 'gap' meters
        distant from each other. Note: method always starts populating the
        scenario from the right-most lane, front-most vehicle.

        :param gap: Inter-vehicle distance for vehicles on the same lane. If
         the value is not given, it defaults to v_ff + 1
        :return:
        """
        if gap is None:
            gap = self.vehicle_group.vehicles[0].free_flow_speed + 1
        x0, y0, theta0, v0 = [], [], [], []
        for lane in range(len(self.n_per_lane)):
            lane_center = lane * lane_width
            n = self.n_per_lane[lane]
            for i in range(n):
                x0.append(gap * (n - i - 1))
                y0.append(lane_center)
                theta0.append(0.0)
        v0 = self.vehicle_group.get_free_flow_speeds()
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)
        self.initial_state = self.vehicle_group.get_full_initial_state_vector()

    def create_base_lane_change_initial_state(self, v_ego, v_lo, v_ld, v_fd,
                                              position_offset):
        """
        Creates an initial state for 5 vehicles: 3 at the origin lane, with
        the lane changing one between other two, and 2 at the destination lane.
        :param v_ego: lane changing vehicle free flow speed
        :param v_lo: origin lane leader free flow speed
        :param v_ld: destination lane leader free flow speed
        :param v_fd: destination lane follower free flow speed
        :param position_offset: determines by how much surrounding vehicles'
         positions deviate from the safe lane change values
        """
        # TODO: how to ensure proper vehicle 'sync' between creates vehicles
        #  and the initial states
        target_y = lane_width
        orig_lane_speed = 10
        dest_lane_speed = 10
        v_ff = [v_lo, v_ego, v_ego,
                v_ld, v_fd]
        self.set_free_flow_speeds(v_ff)

        # We assume all vehicles have the same parameters, so all safe gaps are
        # the same
        sample_veh = self.vehicle_group.vehicles[0]
        safe_gap = sample_veh.compute_safe_gap(v_lo)
        # Initial states
        x_ego = 0
        y_ego = 0
        x0 = [x_ego + safe_gap - position_offset, x_ego, x_ego - safe_gap,
              x_ego + safe_gap - 3 * position_offset,
              x_ego - safe_gap + position_offset]
        y0 = [y_ego, y_ego, y_ego,
              target_y, target_y]
        theta0 = [0., 0., 0., 0., 0.]
        v0 = [orig_lane_speed, orig_lane_speed, orig_lane_speed,
              dest_lane_speed, dest_lane_speed]
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)
        self.initial_state = self.vehicle_group.get_full_initial_state_vector()

    def create_final_state(self):
        """ Default behavior: no final state specification """
        pass

    def response_to_dataframe(self) -> pd.DataFrame:
        return self.vehicle_group.to_dataframe()

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


class VehicleFollowingScenario(SimulationScenario):
    """
    Scenario to test acceleration feedback laws. No lane changes.
    """

    def __init__(self, n_vehs: int):
        super().__init__()
        vehicle_classes = ([[vehicle_models.ClosedLoopVehicle]
                            * n_vehs])
        v_ff = [10] + [12] * n_vehs
        self.create_vehicles(vehicle_classes)
        self.set_free_flow_speeds(v_ff)

    def create_initial_state(self):
        gap = 2
        self.place_equally_spaced_vehicles(gap)
        print(self.initial_state)

    def run(self, final_time):
        """

        :param final_time: Total simulation time
        :return:
        """
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            self.vehicle_group.determine_inputs({})
            self.vehicle_group.compute_derivatives()
            self.vehicle_group.update_states(time[i + 1])
            self.vehicle_group.update_surrounding_vehicles()


class FeedbackLaneChangeScenario(SimulationScenario):

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        n_dest_lane_vehs = 2
        n_orig_lane_vehs = 3
        veh_classes = (
            [[vehicle_models.ClosedLoopVehicle]
             * (n_dest_lane_vehs + n_orig_lane_vehs)])
        self.create_vehicles(veh_classes)
        self.create_initial_state()

    def create_initial_state(self):

        v_ego = 12
        orig_lane_speed = 10
        dest_lane_speed = 10
        position_offset = 1
        self.create_base_lane_change_initial_state(
            v_ego, orig_lane_speed, dest_lane_speed, 0.8 * dest_lane_speed,
            position_offset)

    def run(self, final_time):
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            if i == 100:
                self.vehicle_group.set_lane_change_direction_by_id(
                    self.lc_veh_id, 1)
            self.vehicle_group.update_modes()
            self.vehicle_group.determine_inputs({})
            self.vehicle_group.compute_derivatives()
            self.vehicle_group.update_states(time[i + 1])
            self.vehicle_group.update_surrounding_vehicles()


class InternalOptimalControlScenario(SimulationScenario):

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        n_dest_lane_vehs = 2
        veh_classes = (
            [[vehicle_models.ClosedLoopVehicle,
              vehicle_models.SafeAccelOptimalLCVehicle,  # lane changing veh
              vehicle_models.ClosedLoopVehicle],
             [vehicle_models.ClosedLoopVehicle] * n_dest_lane_vehs])
        self.create_vehicles(veh_classes)
        self.create_initial_state()

    def create_initial_state(self):
        v_ego = 10
        orig_lane_speed = v_ego
        dest_lane_speed = 1.1 * v_ego
        v_fd = 0.9 * v_ego
        position_offset = 0
        self.create_base_lane_change_initial_state(
            v_ego, orig_lane_speed, dest_lane_speed, v_fd, position_offset)

    def run(self, final_time):
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            if i == 100:
                self.vehicle_group.set_lane_change_direction_by_id(
                    self.lc_veh_id, 1)
            self.vehicle_group.update_modes()
            self.vehicle_group.determine_inputs({})
            self.vehicle_group.compute_derivatives()
            self.vehicle_group.update_states(time[i + 1])
            self.vehicle_group.update_surrounding_vehicles()


class ExternalOptimalControlScenario(SimulationScenario, ABC):
    vehicles_ocp_interface: vgi.VehicleGroupInterface

    def __init__(self):
        super().__init__()

        # All must be set by the derived class
        self.dynamic_system = None
        self.final_state = None, None
        self.running_cost, self.terminal_cost = None, None
        self.terminal_constraints, self.constraints = None, None

        # self.response = None

    def set_boundary_conditions(self, tf: float):
        """ Sets the initial state, final time and desired final states """
        self.tf = tf
        self.create_initial_state()
        self.create_final_state()
        self.vehicle_group.update_surrounding_vehicles()

    def create_dynamic_system(self) -> None:
        self.vehicles_ocp_interface = vgi.VehicleGroupInterface(
            self.vehicle_group.vehicles)
        params = {'vehicle_group': self.vehicles_ocp_interface,
                  'test': True}

        # Define the vehicle steering dynamics as an input/output system
        input_names = self.vehicles_ocp_interface.create_input_names()
        output_names = self.vehicles_ocp_interface.create_output_names()

        n_states = self.vehicles_ocp_interface.n_states
        self.dynamic_system = ct.NonlinearIOSystem(
            vgi.vehicles_derivatives, vgi.vehicle_output,
            params=params, states=n_states, name='vehicle_group',
            inputs=input_names, outputs=output_names)

    def boundary_conditions_to_dataframe(self) -> pd.DataFrame:
        """
        Puts initial state and desired final conditions in a dataframe.
        """
        return self.vehicles_ocp_interface.to_dataframe(
            np.array([0, self.tf]),
            np.vstack((self.initial_state, self.final_state)).T,
            np.zeros([self.vehicles_ocp_interface.n_inputs, 2])
        )

    def set_optimal_control_problem_functions(self):
        self.set_costs()
        self.set_constraints()

    def set_input_boundaries(self):
        input_lower_bounds, input_upper_bounds = (
            self.vehicles_ocp_interface.get_input_limits())
        self.constraints = [opt.input_range_constraint(
            self.dynamic_system, input_lower_bounds,
            input_upper_bounds)]

    def solve(self, max_iter: int = 100):
        # Initial guess; not initial control
        u0 = self.vehicles_ocp_interface.get_desired_input()
        timepts = np.linspace(0, self.tf, 10, endpoint=True)
        result = opt.solve_ocp(
            self.dynamic_system, timepts, self.initial_state,
            cost=self.running_cost,
            trajectory_constraints=self.constraints,
            terminal_cost=self.terminal_cost,
            terminal_constraints=self.terminal_constraints,
            initial_guess=u0, minimize_options={'maxiter': max_iter})
        return result

    def run_ocp_solution(self, result: opt.OptimalControlResult):
        """
        Calls the control libraries function for running the dynamic system
        given the optimal control problem solution
        :param result: Dictionary ['time': x, 'result': x] containing the
         control time points and result obtained by the optimal control solver
        :return: Nothing. Results are stored internally
        """
        timepts, inputs = result.time, result.inputs

        # Simulate the system dynamics (open loop)
        response = ct.input_output_response(
            self.dynamic_system, timepts, inputs, self.initial_state,
            t_eval=np.arange(0, timepts[-1], 0.1))
        # TODO: from response to vehicle states and back
        # self.time = response.time
        # self.states = response.states
        # self.inputs = response.inputs

    def run(self, result: opt.OptimalControlResult):
        """
        Given the optimal control problem solution, runs the open loop system.
        Difference to method 'run' is that we directly (re)simulate the dynamics
        in this case. For debugging purposes
        """
        dt = 1e-2
        time = np.arange(0, result.time[-1], dt)
        inputs = np.zeros([result.inputs.shape[0], len(time)])
        for i in range(len(result.inputs)):
            inputs[i, :] = np.interp(time, result.time, result.inputs[i])
        inputs_per_vehicle = (
            self.vehicles_ocp_interface.map_ocp_solution_to_vehicle_inputs(
                inputs))
        self.vehicle_group.initialize_state_matrices(len(time))
        for i in range(len(time) - 1):
            current_inputs = {}
            for veh_id in inputs_per_vehicle.keys():
                ego_inputs = inputs_per_vehicle[veh_id][:, i]
                current_inputs[veh_id] = ego_inputs
            self.vehicle_group.update_modes()
            self.vehicle_group.determine_inputs(current_inputs)
            self.vehicle_group.compute_derivatives()
            self.vehicle_group.update_states(time[i + 1])
            self.vehicle_group.update_surrounding_vehicles()

    @abstractmethod
    def set_costs(self):
        """ Sets the running and terminal state costs """
        pass

    @abstractmethod
    def set_constraints(self):
        """ Sets all the problem constraints """
        pass


class ExampleScenarioExternal(ExternalOptimalControlScenario):
    """
    Two-lane scenario where all vehicles want to perform a lane change starting
    at t=0 and ending at tf.
    The scenario is used for testing the different dynamical models and
    variations in parameters
    """

    def create_initial_state(self):
        self.place_equally_spaced_vehicles()
        print(self.initial_state)

    def create_final_state(self):
        """
        Sets the final state where all vehicles change from their initial
        lanes while traveling at their free-flow speed.

        :return:
        """
        xf, yf, thetaf, vf = [], [], [], []
        for veh in self.vehicle_group.get_all_vehicles():
            v_ff = veh.free_flow_speed
            lane = veh.get_current_lane()
            xf.append(veh.get_a_state_by_name('x') + v_ff * self.tf)
            veh.set_lane_change_direction((-1) ** lane)
            veh.update_target_y()
            yf.append(veh.target_y)
            thetaf.append(0.0)
            vf.append(v_ff)
        self.final_state = self.vehicle_group.create_full_state_vector(
            xf, yf, thetaf, vf
        )
        print(self.final_state)

    def set_costs(self):
        Q, R, P = self.create_simple_weight_matrices()
        # Desired control; not final control
        uf = self.vehicles_ocp_interface.get_desired_input()
        self.running_cost = opt.quadratic_cost(self.dynamic_system,
                                               Q, R, x0=self.final_state, u0=uf)
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.final_state)

    def create_simple_weight_matrices(self):
        # don't turn too sharply: matrix Q
        state_cost_weights = [0.] * self.vehicles_ocp_interface.n_states
        theta_idx = self.vehicles_ocp_interface.get_state_indices('theta')
        for i in theta_idx:
            state_cost_weights[i] = 0.1
        state_cost = np.diag(state_cost_weights)

        # keep inputs small: matrix R
        input_cost_weights = [1] * self.vehicles_ocp_interface.n_inputs
        input_cost = np.diag(input_cost_weights)

        # get close to final point: matrix P
        if ('v' in self.vehicles_ocp_interface.vehicles[0].input_names
                or 'a' in self.vehicles_ocp_interface.vehicles[0].input_names):
            final_state_weights = [1000] * self.vehicles_ocp_interface.n_states
        else:
            # If we can't control speed or acceleration, we shouldn't care about
            # final position and velocity
            final_state_weights = [0, 1000, 1000, 0] * self.vehicle_group.n_vehs
        terminal_cost = np.diag(final_state_weights)
        return state_cost, input_cost, terminal_cost

    def set_constraints(self):
        self.set_input_boundaries()


class LaneChangeWithConstraints(ExternalOptimalControlScenario):
    """
    Used to test how to code safety constraints
    """

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        n_dest_lane_vehs = 2
        veh_classes = (
            [[vehicle_models.ClosedLoopVehicle,
              vehicle_models.SafeAccelOpenLoopLCVehicle,  # lane changing veh
              vehicle_models.ClosedLoopVehicle],
             [vehicle_models.ClosedLoopVehicle] * n_dest_lane_vehs])
        self.create_vehicles(veh_classes)

        self.min_lc_x = 20  # only used for initial tests

    def create_initial_state(self):
        v_ego = 10
        orig_lane_speed = v_ego
        dest_lane_speed = 1.1 * v_ego
        v_fd = 0.9 * v_ego
        position_offset = 0
        self.create_base_lane_change_initial_state(
            v_ego, orig_lane_speed, dest_lane_speed, v_fd, position_offset)

    def create_final_state(self):
        # Note: xf and vf are irrelevant with the accel feedback model
        lc_vehicle = self.vehicle_group.vehicles[self.lc_veh_id]
        lc_vehicle.set_lane_change_direction(1)
        lc_vehicle.update_target_y()
        xf, yf, thetaf, vf = [], [], [], []
        for veh in self.vehicle_group.get_all_vehicles():
            v_ff = veh.free_flow_speed
            xf.append(veh.get_a_state_by_name('x') + v_ff * self.tf)
            yf.append(veh.target_y)
            thetaf.append(0.0)
            vf.append(v_ff)
        self.final_state = self.vehicle_group.create_full_state_vector(
            xf, yf, thetaf, vf
        )
        print(self.final_state)

    def set_costs(self):
        # Desired control; not final control
        uf = self.vehicles_ocp_interface.get_desired_input()
        Q = np.diag([0, 0, 0.1, 0] * self.vehicles_ocp_interface.n_vehs)
        R = np.diag([0.1] * self.vehicles_ocp_interface.n_inputs)
        self.running_cost = opt.quadratic_cost(self.dynamic_system, Q, R,
                                               self.final_state, uf)
        P = np.diag([0, 1000, 1000, 0] * self.vehicles_ocp_interface.n_vehs)
        self.terminal_cost = opt.quadratic_cost(self.dynamic_system,
                                                P, 0, x0=self.final_state)

    def set_constraints(self):
        self.set_input_boundaries()
        # Linear constraint: lb <= A*[x; u] <= ub
        # vel_con = LinearConstraint([0, 0, 0, 1, 0, 0], 0, 15)
        # self.constraints.append(vel_con)

        epsilon = 1e-10
        orig_lane_safety = NonlinearConstraint(
            self.safety_constraint_orig_lane_leader, -epsilon, epsilon)
        dest_lane_leader_safety = NonlinearConstraint(
            self.safety_constraint_dest_lane_leader, -epsilon, epsilon)
        dest_lane_follower_safety = NonlinearConstraint(
            self.safety_constraint_dest_lane_follower, -epsilon, epsilon)

        self.constraints.append(orig_lane_safety)
        self.constraints.append(dest_lane_leader_safety)
        self.constraints.append(dest_lane_follower_safety)

    def safety_constraint_orig_lane_leader(self, states, inputs):
        lc_vehicle = self.vehicle_group.vehicles[self.lc_veh_id]
        return self.lane_changing_safety_constraint(
            states, inputs, lc_vehicle.id, lc_vehicle.get_orig_lane_leader_id())

    def safety_constraint_dest_lane_leader(self, states, inputs):
        lc_vehicle = self.vehicle_group.vehicles[self.lc_veh_id]
        return self.lane_changing_safety_constraint(
            states, inputs, lc_vehicle.id,
            lc_vehicle.get_dest_lane_leader_id())

    def safety_constraint_dest_lane_follower(self, states, inputs):
        lc_vehicle = self.vehicle_group.vehicles[self.lc_veh_id]
        return self.lane_changing_safety_constraint(
            states, inputs, lc_vehicle.get_dest_lane_follower_id(),
            lc_vehicle.id)

    def lane_changing_safety_constraint(self, states, inputs, follower_id,
                                        leader_id):
        if leader_id < 0:  # no leader
            return 0

        follower_veh = self.vehicles_ocp_interface.vehicles[follower_id]
        follower_states = (
            self.vehicles_ocp_interface.get_vehicle_state_vector_by_id(
                follower_id, states))
        leader_states = (
            self.vehicles_ocp_interface.get_vehicle_state_vector_by_id(
                leader_id, states))
        gap_error = follower_veh.compute_gap_error(follower_states,
                                                   leader_states)
        phi = self.vehicles_ocp_interface.get_a_vehicle_input_by_id(
            self.lc_veh_id, inputs, 'phi')
        margin = 1e-1
        # TODO: possible issue. When gap error becomes less than zero during
        #  the maneuver, then phi is forced to zero.
        return min(gap_error + margin, 0) * phi

    def lane_change_starting_point_constraint(self, states, inputs):
        lc_veh_x = self.vehicles_ocp_interface.get_a_vehicle_state_by_id(
            self.lc_veh_id, states, 'x')
        phi = self.vehicles_ocp_interface.get_a_vehicle_input_by_id(
            self.lc_veh_id, inputs, 'phi')
        dist_to_point = lc_veh_x - self.min_lc_x
        return min(dist_to_point, 0) * phi

    def smooth_lane_change_starting_point_constraint(self, states, inputs):
        lc_veh_x = self.vehicles_ocp_interface.get_a_vehicle_state_by_id(
            self.lc_veh_id, states, 'x')
        phi = self.vehicles_ocp_interface.get_a_vehicle_input_by_id(
            self.lc_veh_id, inputs, 'phi')
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
