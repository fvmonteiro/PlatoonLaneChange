from abc import ABC, abstractmethod
import pickle
from typing import List, Type, Union

import control as ct
import control.optimal as opt
import numpy as np
import pandas as pd

import controllers.optimal_controller as opt_ctrl
from constants import lane_width
from vehicle_group import VehicleGroup
from vehicle_models.base_vehicle import BaseVehicle
import vehicle_models.four_state_vehicles as fsv
import vehicle_group_ocp_interface as vgi


class SimulationScenario(ABC):
    def __init__(self):
        self.n_per_lane = []
        self.vehicle_group = VehicleGroup()
        BaseVehicle.reset_vehicle_counter()
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

    def create_base_lane_change_initial_state(self, v_ff, delta_x):
        """
        Creates an initial state for 5 vehicles: 3 at the origin lane, with
        the lane changing one between other two, and 2 at the destination lane.
        :param v_ff: Dictionary with free-flow speeds for ego, lo, ld and fd
        :param delta_x: Dictionary with deviation from minimum safe gap for lo,
         ld and fd. Positive values lead to smaller-than-safe initial gaps.
        """
        # TODO: how to ensure proper vehicle 'sync' between creates vehicles
        #  and the initial states
        self.vehicle_group.set_vehicle_names(['lo', 'ego', 'fo', 'ld', 'fd'])
        target_y = lane_width
        ordered_v_ff = [v_ff['lo'], v_ff['ego'], v_ff['ego'],
                        v_ff['ld'], v_ff['fd']]
        self.set_free_flow_speeds(ordered_v_ff)

        # We assume all vehicles have the same parameters, so all safe gaps are
        # the same
        sample_veh = self.vehicle_group.vehicles[0]
        safe_gap = sample_veh.compute_safe_gap(v_ff['lo'])
        # Initial states
        x_ego = 0
        y_ego = 0
        x0 = [x_ego + safe_gap - delta_x['lo'], x_ego, x_ego - safe_gap,
              x_ego + safe_gap - delta_x['ld'],
              x_ego - safe_gap + delta_x['fd']]
        y0 = [y_ego, y_ego, y_ego,
              target_y, target_y]
        theta0 = [0., 0., 0., 0., 0.]
        v0 = [v_ff['lo'], v_ff['lo'], v_ff['lo'],
              v_ff['lo'], v_ff['lo']]
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)
        self.initial_state = self.vehicle_group.get_full_initial_state_vector()

    def test_scenario(self):
        # Free-flow speeds
        v_ff = {'ego': 10, 'lo': 10}
        v_ff['ld'] = 1.1 * v_ff['lo']
        v_ff['fd'] = 0.9 * v_ff['lo']
        # Deviation from minimum safe gap
        delta_x = {'lo': -1.0, 'ld': -1.0, 'fd': -1.0}
        self.create_base_lane_change_initial_state(
            v_ff, delta_x)

    def response_to_dataframe(self) -> pd.DataFrame:
        return self.vehicle_group.to_dataframe()

    def simulate_one_time_step(self, new_time):
        self.vehicle_group.update_vehicle_modes()
        self.vehicle_group.determine_inputs({})
        self.vehicle_group.compute_derivatives()
        self.vehicle_group.update_states(new_time)
        self.vehicle_group.update_surrounding_vehicles()

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
        vehicle_classes = ([[fsv.ClosedLoopVehicle]
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
            self.simulate_one_time_step(time[i+1])


class FeedbackLaneChangeScenario(SimulationScenario):

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        n_dest_lane_vehs = 2
        n_orig_lane_vehs = 3
        veh_classes = (
            [[fsv.ClosedLoopVehicle]
             * (n_dest_lane_vehs + n_orig_lane_vehs)])
        self.create_vehicles(veh_classes)
        self.create_initial_state()

    def create_initial_state(self):
        self.test_scenario()
        self.vehicle_group.make_all_connected()

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
            self.simulate_one_time_step(time[i+1])


class InternalOptimalControlScenario(SimulationScenario):

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        n_dest_lane_vehs = 2
        veh_classes = (
            [[fsv.ClosedLoopVehicle,
              fsv.SafeAccelOptimalLCVehicle,  # lane changing veh
              fsv.ClosedLoopVehicle],
             [fsv.ClosedLoopVehicle] * n_dest_lane_vehs])
        self.create_vehicles(veh_classes)
        self.create_initial_state()

    def create_initial_state(self):
        self.test_scenario()
        self.vehicle_group.make_all_connected()

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
            self.simulate_one_time_step(time[i+1])


class ExternalOptimalControlScenario(SimulationScenario, ABC):
    vehicles_ocp_interface: vgi.VehicleGroupInterface
    ocp_response: ct.TimeResponseData

    def __init__(self):
        super().__init__()

        # All must be set by the derived class
        self.dynamic_system = None
        self.final_state = None
        # self.running_cost, self.terminal_cost = None, None
        # self.terminal_constraints, self.constraints = None, None

    def set_boundary_conditions(self, tf: float):
        """ Sets the initial state, final time and desired final states """
        self.tf = tf
        self.create_initial_state()
        self.create_final_state()
        self.vehicle_group.update_surrounding_vehicles()

    def boundary_conditions_to_dataframe(self) -> pd.DataFrame:
        """
        Puts initial state and desired final conditions in a dataframe.
        """
        return self.vehicles_ocp_interface.to_dataframe(
            np.array([0, self.tf]),
            np.vstack((self.initial_state, self.final_state)).T,
            np.zeros([self.vehicles_ocp_interface.n_inputs, 2])
        )

    def ocp_simulation_to_dataframe(self) -> pd.DataFrame:
        """
        Puts the states computed by the ocp solver tool (and saved) in a df
        """
        return self.vehicles_ocp_interface.to_dataframe(
            self.ocp_response.time,
            self.ocp_response.states,
            self.ocp_response.inputs)

    def solve(self, max_iter: int = 100):
        self.vehicles_ocp_interface = vgi.VehicleGroupInterface(
            self.vehicle_group.vehicles)
        self.controller = opt_ctrl.VehicleOptimalController(self.tf)
        self.controller.find_lane_change_trajectory(0.0, self.vehicle_group.vehicles,
                                               [])
        return self.controller.ocp_result

    def run_ocp_solution(self, result: opt.OptimalControlResult):
        """
        Calls the control libraries function for running the dynamic system
        given the optimal control problem solution
        :param result: Dictionary ['time': x, 'result': x] containing the
         control time points and result obtained by the optimal control solver
        :return: Nothing. Results are stored internally
        """
        time_pts, inputs = result.time, result.inputs

        # Simulate the system dynamics (open loop)
        response = ct.input_output_response(
            self.controller.dynamic_system, time_pts, inputs, self.initial_state,
            t_eval=np.arange(0, time_pts[-1], 0.1))
        self.ocp_response = response

    def run(self, result: opt.OptimalControlResult):
        """
        Given the optimal control problem solution, runs the open loop system.
        Difference to method 'run' is that we directly (re)simulate the dynamics
        in this case. For debugging purposes
        """
        # It is good to run our simulator with the ocp solution and to confirm
        # it yields the same response as the control library simulation
        self.run_ocp_solution(result)

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
            self.vehicle_group.update_vehicle_modes()
            self.vehicle_group.determine_inputs(current_inputs)
            self.vehicle_group.compute_derivatives()
            self.vehicle_group.update_states(time[i + 1])
            self.vehicle_group.update_surrounding_vehicles()

    @abstractmethod
    def create_final_state(self):
        """ Default behavior: no final state specification """
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

    def create_simple_weight_matrices(self):
        base = 0.01
        # don't turn too sharply: matrix Q
        state_cost_weights = [0.] * self.vehicles_ocp_interface.n_states
        theta_idx = self.vehicles_ocp_interface.get_state_indices('theta')
        for i in theta_idx:
            state_cost_weights[i] = base  # orig 0.1
        state_cost = np.diag(state_cost_weights)

        # keep inputs small: matrix R. Original value [1]
        input_cost_weights = [base] * self.vehicles_ocp_interface.n_inputs
        input_cost = np.diag(input_cost_weights)

        # get close to final point: matrix P
        p = base * 1e3  # orig 1000
        if ('v' in self.vehicles_ocp_interface.vehicles[0].input_names
                or 'a' in self.vehicles_ocp_interface.vehicles[0].input_names):
            final_state_weights = [p] * self.vehicles_ocp_interface.n_states
        else:
            # If we can't control speed or acceleration, we shouldn't care about
            # final position and velocity
            final_state_weights = [0, p, p, 0] * self.vehicle_group.n_vehs
        terminal_cost = np.diag(final_state_weights)
        return state_cost, input_cost, terminal_cost


class LaneChangeWithConstraints(ExternalOptimalControlScenario):
    """
    Used to test how to code safety constraints
    """

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        n_dest_lane_vehs = 2
        veh_classes = (
            [[fsv.ClosedLoopVehicle,
              fsv.SafeAccelOpenLoopLCVehicle,  # lane changing veh
              fsv.ClosedLoopVehicle],
             [fsv.ClosedLoopVehicle] * n_dest_lane_vehs])
        self.create_vehicles(veh_classes)

        self.min_lc_x = 20  # only used for initial tests

    def create_initial_state(self):
        # # Free-flow speeds
        # v_ff = {'ego': 10, 'lo': 10}
        # v_ff['ld'] = 1.1 * v_ff['ego']
        # v_ff['fd'] = 0.8 * v_ff['ego']
        # # Deviation from minimum safe gap
        # delta_x = {'lo': 0.0, 'ld': 0.0, 'fd': 0.0}
        # self.create_base_lane_change_initial_state(v_ff, delta_x)
        self.test_scenario()

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
