from __future__ import annotations

from collections import defaultdict
from functools import partial
import time
from typing import Callable, Dict, List, Union, Tuple

import control as ct
import control.optimal as opt
import control.flatsys as flat
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

import analysis  # only during tests
import controllers.optimal_control_costs as occ
import system_operating_mode as som
import vehicle_models.base_vehicle as base
import vehicle_group as vg
import vehicle_group_ocp_interface as vgi


def configure_optimal_controller(
        max_iter: int = None, solver_max_iter: int = None,
        discretization_step: float = None, time_horizon: float = None,
        has_terminal_constraints: bool = False,
        has_non_zero_initial_guess: bool = False,
        has_lateral_constraint: bool = False):
    """
    Sets the configurations of the optimal controller that do not vary during
    the simulation.
    :param max_iter: Maximum number of times the ocp will be solved until
     mode sequence convergence (different from max iteration of the solver)
    :param discretization_step: Fixed discretization step of the opc solver
    :param time_horizon: Final time of the optimal control problem
    :param solver_max_iter: Maximum number of iterations by the solver
    :param has_terminal_constraints: Whether to include terminal constraints. If
     true, then there are no terminal costs
    :param has_non_zero_initial_guess: Whether to set the initial control guess
     to something other than all zeros. If true, we assume max input for the
     first two time intervals and min input for the last two.
    :param has_lateral_constraint: Whether to include a constraint to keep the
     lane changing vehicles between y(t0)-1 and y(tf)+1. This can sometimes
     speed up simulations or prevent erratic behavior
    :return:
    """
    if max_iter:
        VehicleOptimalController.max_iter = max_iter
    if solver_max_iter:
        VehicleOptimalController.solver_max_iter = solver_max_iter
    if discretization_step:
        VehicleOptimalController.discretization_step = discretization_step
    if time_horizon:
        VehicleOptimalController.time_horizon = time_horizon

    VehicleOptimalController.has_terminal_constraints = (
        has_terminal_constraints)
    VehicleOptimalController.has_non_zero_initial_guess = (
        has_non_zero_initial_guess)
    VehicleOptimalController.has_lateral_constraint = has_lateral_constraint


class VehicleOptimalController:
    """
    The optimal controller follows the steps:
    1. Set an operating mode sequence
    2. Solve the OPC with given mode sequence
    3. Apply optimal input and simulate system
    4. Compare assumed operating mode sequence to obtained mode sequence
    5. Repeat steps 1-4 until convergence or iteration limit is reached

    We can leave the operating mode sequence empty and set the iteration limit
    to one to obtain a non-iterative optimal controller
    """

    _ocp_interface: vgi.VehicleGroupInterface
    _initial_state: np.ndarray
    _desired_state: np.ndarray

    solver_max_iter: int = 300
    max_iter: int = 3
    discretization_step: float = 1.0  # [s]
    time_horizon: float = 10.0  # [s]
    has_terminal_constraints: bool = False
    has_non_zero_initial_guess: bool = False
    has_lateral_constraint: bool = False

    def __init__(self):
        self._terminal_cost = None
        self._terminal_constraints = None
        self._constraints = []
        self._ocp_has_solution = False

        self._data_per_iteration = []
        self._results_summary: defaultdict[str, List] = defaultdict(list)
        self._cost_history: List[occ.OCPCostTracker] = []

    def get_time_horizon(self) -> float:
        return self.time_horizon

    def get_time_points(self, dt: float = None):
        if dt is None:
            dt = self.discretization_step  # [s]
        n = round(self.time_horizon / dt) + 1  # +1 to get 'round' times
        return np.linspace(0, self.time_horizon, n)

    def get_data_per_iteration(self):
        return self._data_per_iteration

    def get_desired_state(self):
        return self._desired_state

    def get_results_summary(self):
        return self._results_summary

    def has_solution(self) -> bool:
        return self._ocp_has_solution

    def get_running_cost_history(self):
        cost_history_per_iteration = []
        for c in self._cost_history:
            cost_history_per_iteration.append(c.get_running_cost())
        return cost_history_per_iteration

    def get_terminal_cost_history(self):
        cost_history_per_iteration = []
        for c in self._cost_history:
            cost_history_per_iteration.append(c.get_terminal_cost())
        return cost_history_per_iteration

    def get_ocp_response(self):
        """
        Gets the states as computed by the optimal control solver tool
        """
        time_pts, inputs = self.ocp_result.time, self.ocp_result.inputs
        return ct.input_output_response(
            self._dynamic_system, time_pts, inputs,
            self._initial_state,
            # t_eval=np.arange(0, time_pts[-1], 0.01)
        )

    def get_input(self, current_time: float, veh_ids: Union[int, List[int]]):
        """
        Gets the optimal input found by the solver at the given time by linear
        interpolation
        :param current_time: Time assuming t0 = 0. This means some callers must
         provide t-t0.
        :param veh_ids: Ids of vehicles for which we want the inputs
        :return: Dictionary with veh ids as keys and inputs as values if
         multiple veh ids passed, single input vector if single veh id
        """

        if np.isscalar(veh_ids):
            single_veh_ctrl = True
            veh_ids = [veh_ids]
        else:
            single_veh_ctrl = False

        current_inputs = {}
        for v_id in veh_ids:
            ego_inputs = self._ocp_inputs_per_vehicle[v_id]
            n_optimal_inputs = ego_inputs.shape[0]
            current_inputs[v_id] = []
            for i in range(n_optimal_inputs):
                current_inputs[v_id].append(np.interp(
                    current_time, self._ocp_time, ego_inputs[i]))
                # idx = np.searchsorted(self._ocp_time, delta_t)
                # current_inputs[v_id].append(ego_inputs[i][idx])

        if single_veh_ctrl:
            return current_inputs[veh_ids[0]]
        else:
            return current_inputs

    def set_time_horizon(self, value) -> None:
        self.time_horizon = value

    def find_single_vehicle_trajectory(
            self, vehicles: Dict[int, base.BaseVehicle],
            ego_id: int, mode_sequence: som.ModeSequence = None):
        """
        Solves the OPC assuming only the vehicle calling this function will
        perform a lane change
        :param vehicles: All relevant vehicles
        :param ego_id: ID of the vehicle calling the method
        :param mode_sequence: Predefined system mode sequence [optional].
        :return: nothing
        """
        self.find_multiple_vehicle_trajectory(vehicles,
                                              [ego_id], mode_sequence)

    def find_multiple_vehicle_trajectory(
            self, vehicles: Dict[int, base.BaseVehicle],
            controlled_veh_ids: List[int],
            mode_sequence: som.ModeSequence = None):
        """
        Solves the OPC for all listed controlled vehicles
        :param vehicles: All relevant vehicles
        :param controlled_veh_ids: IDs of controlled vehicles
        :param mode_sequence: Predefined system mode sequence [optional].
        :return:
        """

        if mode_sequence is None:
            input_sequence: som.ModeSequence = [(0.0, som.SystemMode(vehicles))]
        else:
            input_sequence = mode_sequence

        # We put the vehicle below at the origin (0, 0) when solving the opc
        # to make the solution independent of shifts in initial position
        center_veh_id = controlled_veh_ids[0]
        self._ocp_interface = vgi.VehicleGroupInterface(vehicles,
                                                        center_veh_id)
        self._set_ocp_configuration(controlled_veh_ids)
        running_cost = self._generate_running_cost()
        terminal_cost = self._generate_terminal_cost()
        time_points = self.get_time_points()
        converged = False
        last_input = None
        counter = 0
        while not converged and counter < self.max_iter:
            counter += 1

            self._ocp_interface.set_mode_sequence(input_sequence)
            # self._set_costs()
            ocp_cost_tracker = occ.OCPCostTracker(
                time_points, self._ocp_interface.n_states,
                running_cost, terminal_cost)
            self._cost_history.append(ocp_cost_tracker)
            start_time = time.time()
            self._solve_ocp(last_input)
            solve_time = time.time() - start_time

            # print(self._cost_history[-1].costs_to_string())

            # TODO: do we want the mode sequence seen by the ocp solver or the
            #  simulated one?
            # alt = self.get_ocp_solver_simulation(vehicles)
            # analysis.plot_constrained_lane_change(
            #     alt.to_dataframe(), vehicles[center_veh_id].get_id())

            simulated_vehicle_group = self.simulate_over_optimization_horizon(
                vehicles)
            self._data_per_iteration.append(
                simulated_vehicle_group.to_dataframe())
            analysis.plot_constrained_lane_change(
                self._data_per_iteration[-1],
                vehicles[center_veh_id].get_id())
            output_sequence = simulated_vehicle_group.get_mode_sequence()

            input_seq_str = som.mode_sequence_to_str(input_sequence)
            output_seq_str = som.mode_sequence_to_str(output_sequence)
            print("Input sequence:  {}\nOutput sequence: {}".format(
                input_seq_str, output_seq_str
            ))
            self._log_results(counter, input_seq_str, output_seq_str,
                              solve_time, self.ocp_result)

            converged = som.compare_mode_sequences(input_sequence,
                                                   output_sequence)
            print("Converged?", converged)
            input_sequence = output_sequence
            # if self.ocp_result.success:
            #     last_input = self.ocp_result.inputs
            # else:
            #     last_input = None

    def simulate_over_optimization_horizon(
            self, vehicles: Dict[int, base.BaseVehicle]):
        dt = 1e-2
        sim_time = self.get_time_points(dt)

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_vehicles(vehicles)
        vehicle_group.set_verbose(False)
        vehicle_group.prepare_to_start_simulation(len(sim_time))
        vehicle_group.update_surrounding_vehicles()
        for i in range(len(sim_time) - 1):
            vehicle_group.simulate_one_time_step(sim_time[i + 1])
        return vehicle_group

    def get_ocp_solver_simulation(self, vehicles):
        sim_time = self.ocp_result.time
        initial_state_per_vehicle = (
            self._ocp_interface.map_state_to_vehicle_ids(self._initial_state)
        )

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_vehicles(vehicles,
                                             initial_state_per_vehicle)
        vehicle_group.set_verbose(False)
        vehicle_group.prepare_to_start_simulation(len(sim_time))
        for i in range(1, len(sim_time)):
            vehicle_group.update_surrounding_vehicles()
            t = self.ocp_result.time[i]
            state = self.ocp_result.states[:, i]
            state_by_vehicle = (
                self._ocp_interface.map_state_to_vehicle_ids(state)
            )
            inputs = self.ocp_result.inputs[:, i]
            inputs_by_vehicle = (
                self._ocp_interface.map_input_to_vehicle_ids(inputs)
            )
            vehicle_group.write_vehicle_states(t, state_by_vehicle,
                                               inputs_by_vehicle)
        return vehicle_group

    def _set_ocp_configuration(self, controlled_veh_ids: List[int]):
        self._set_ocp_dynamics()
        self._initial_state = self._ocp_interface.get_initial_state()
        self._desired_state = self._ocp_interface.create_desired_state(
            self.time_horizon)
        # print("x0 vs xf:")
        # print(self._initial_state)
        # print(self._desired_state)
        self._constraints = []
        self._set_input_constraints()
        if self.has_terminal_constraints:
            self._set_terminal_constraints(controlled_veh_ids)
        self._set_safety_constraints(controlled_veh_ids)

    def _set_ocp_dynamics(self):
        params = {'vehicle_group_interface': self._ocp_interface}
        input_names = self._ocp_interface.create_input_names()
        output_names = self._ocp_interface.create_output_names()
        n_states = self._ocp_interface.n_states
        # Define the vehicle dynamics as an input/output system
        self._dynamic_system = ct.NonlinearIOSystem(
            vgi.vehicles_derivatives, vgi.vehicle_output,
            params=params, states=n_states, name='vehicle_group',
            inputs=input_names, outputs=output_names)

    def _generate_running_cost(self) -> Callable:
        # Note: running and terminal costs should depend on vehicle model
        # For three state: running costs y_cost=0, theta_cost=0.1 and

        u_ref = self._ocp_interface.get_desired_input()
        state_cost_matrix = self._ocp_interface.create_state_cost_matrix(
            y_cost=0.1, theta_cost=0., x_cost=0.)
        # input_cost_matrix = np.diag([0.01] * self._ocp_interface.n_inputs)
        running_cost = occ.quadratic_cost(
            self._ocp_interface.n_states, self._ocp_interface.n_inputs,
            state_cost_matrix, None, self._desired_state, u_ref
        )

        return running_cost

    def _generate_terminal_cost(self) -> Callable:
        # Note: running and terminal costs should depend on vehicle model
        # For three state: terminal costs 1000 for all states
        if self._terminal_constraints is None:
            terminal_state_cost_matrix = (
                self._ocp_interface.create_state_cost_matrix(y_cost=1,
                                                             theta_cost=10)
            )
            terminal_input_cost_matrix = np.diag(
                [0.] * self._ocp_interface.n_inputs)
            terminal_cost = occ.quadratic_cost(
                self._ocp_interface.n_states, self._ocp_interface.n_inputs,
                terminal_state_cost_matrix, terminal_input_cost_matrix,
                self._desired_state
            )
        else:
            terminal_cost = None

        return terminal_cost

    # def _set_costs(self):
    #     # Note: running and terminal costs should depend on vehicle model
    #     # For three state: running costs y_cost=0, theta_cost=0.1 and
    #     # terminal costs 1000 for all states
    #     time_points = self.get_time_points()
    #     u_ref = self._ocp_interface.get_desired_input()
    #     state_cost_matrix = self._ocp_interface.create_state_cost_matrix(
    #         y_cost=0.1, theta_cost=0., x_cost=0.)
    #     # input_cost_matrix = np.diag([0.01] * self._ocp_interface.n_inputs)
    #     running_cost = occ.quadratic_cost(
    #         self._ocp_interface.n_states, self._ocp_interface.n_inputs,
    #         state_cost_matrix, None, self._desired_state, u_ref
    #     )
    #
    #     if self._terminal_constraints is None:
    #         terminal_state_cost_matrix = (
    #             self._ocp_interface.create_state_cost_matrix(y_cost=1,
    #                                                          theta_cost=10)
    #         )
    #         terminal_input_cost_matrix = np.diag(
    #             [0.] * self._ocp_interface.n_inputs)
    #         terminal_cost = occ.quadratic_cost(
    #             self._ocp_interface.n_states, self._ocp_interface.n_inputs,
    #             terminal_state_cost_matrix, terminal_input_cost_matrix,
    #             self._desired_state
    #         )
    #     else:
    #         terminal_cost = None
    #
    #     ocp_cost_tracker = occ.OCPCostTracker(
    #         time_points, self._ocp_interface.n_states,
    #         running_cost, terminal_cost)
    #     self._cost_history.append(ocp_cost_tracker)

    def _set_terminal_constraints(self, controlled_veh_ids: List[int]):
        controlled_vehicles = [self._ocp_interface.vehicles[veh_id]
                               for veh_id in controlled_veh_ids]
        dim = [2 * len(controlled_vehicles),
               self._ocp_interface.n_states + self._ocp_interface.n_inputs]
        rows = np.zeros(dim)
        lower_boundaries = np.zeros(dim[0])
        upper_boundaries = np.zeros(dim[0])
        y_margin = 1e-1
        theta_margin = 1e-2
        for i in range(len(controlled_vehicles)):
            veh = controlled_vehicles[i]
            y_idx = 2 * i
            theta_idx = 2 * i + 1
            rows[y_idx, self._ocp_interface.get_a_vehicle_state_index(
                veh.get_id(), 'y')] = 1
            rows[theta_idx, self._ocp_interface.get_a_vehicle_state_index(
                veh.get_id(), 'theta')] = 1
            lower_boundaries[y_idx] = veh.get_target_y() - y_margin
            lower_boundaries[theta_idx] = -theta_margin
            upper_boundaries[y_idx] = veh.get_target_y() + y_margin
            upper_boundaries[theta_idx] = theta_margin

        self._terminal_constraints = LinearConstraint(
            rows, lb=lower_boundaries, ub=upper_boundaries)

    def _set_input_constraints(self):
        input_lower_bounds, input_upper_bounds = (
            self._ocp_interface.get_input_limits())
        m = np.hstack(
            [np.zeros((self._ocp_interface.n_inputs,
                       self._ocp_interface.n_states)),
             np.eye(self._ocp_interface.n_inputs)])
        input_constraint = LinearConstraint(
            m, input_lower_bounds, input_upper_bounds,
            keep_feasible=True)  # keep_feasible made no difference
        self._constraints.append(input_constraint)
        # self._constraints.extend([opt.input_range_constraint(
        #     self._dynamic_system, input_lower_bounds,
        #     input_upper_bounds)])

    def _set_safety_constraints(self, controlled_veh_ids: List[int]):
        # Safety constraints
        # TODO 1: play with keep_feasible param for constraints
        controlled_vehicles = [self._ocp_interface.vehicles[veh_id]
                               for veh_id in controlled_veh_ids]
        epsilon = 1e-10
        for veh in controlled_vehicles:
            d = np.zeros(self._ocp_interface.n_states
                         + self._ocp_interface.n_inputs)
            d[self._ocp_interface.get_a_vehicle_state_index(
                veh.get_id(), 'y')] = 1
            stay_in_lane = LinearConstraint(d, lb=veh.get_y0() - 1,
                                            ub=veh.get_target_y() + 1)
            if self.has_lateral_constraint:
                self._constraints.append(stay_in_lane)
            if veh.has_orig_lane_leader():
                orig_lane_safety = NonlinearConstraint(
                    partial(
                        self._ocp_interface.lane_changing_safety_constraint,
                        lc_veh_id=veh.get_id(),
                        other_id=veh.get_orig_lane_leader_id(),
                        is_other_behind=False),
                    -epsilon, epsilon)
                self._constraints.append(orig_lane_safety)
            if veh.has_dest_lane_leader():
                dest_lane_leader_safety = NonlinearConstraint(
                    partial(
                        self._ocp_interface.lane_changing_safety_constraint,
                        lc_veh_id=veh.get_id(),
                        other_id=veh.get_dest_lane_leader_id(),
                        is_other_behind=False),
                    -epsilon, epsilon)
                self._constraints.append(dest_lane_leader_safety)
            if veh.has_dest_lane_follower():
                dest_lane_follower_safety = NonlinearConstraint(
                    partial(
                        self._ocp_interface.lane_changing_safety_constraint,
                        lc_veh_id=veh.get_id(),
                        other_id=veh.get_dest_lane_follower_id(),
                        is_other_behind=True),
                    -epsilon, epsilon)
                self._constraints.append(dest_lane_follower_safety)

    def _solve_ocp(self, custom_initial_control: np.ndarray = None):
        time_pts = self.get_time_points()

        if custom_initial_control is not None:
            u0 = custom_initial_control
        elif self.has_non_zero_initial_guess:
            u0 = self._ocp_interface.get_initial_guess(time_pts)
        else:
            u0 = self._ocp_interface.get_desired_input()

        result = opt.solve_ocp(
            self._dynamic_system, time_pts, self._initial_state,
            # cost=self._running_cost,
            cost=self._cost_history[-1].running_cost,
            trajectory_constraints=self._constraints,
            # terminal_cost=self._terminal_cost,
            terminal_cost=self._cost_history[-1].terminal_cost,
            terminal_constraints=self._terminal_constraints,
            initial_guess=u0,
            minimize_options={
                'maxiter': self.solver_max_iter, 'disp': True,
                'callback': self._cost_history[-1].callback}
            # minimize_method='trust-constr',
            # log=True
            # basis=flat.BezierFamily(5, T=self._ocp_horizon)
        )
        # Note: the basis parameter above was set empirically - it might not
        # always work well
        self.ocp_result = result
        self._ocp_time = result.time
        self._ocp_inputs_per_vehicle = (
            self._ocp_interface.map_input_to_vehicle_ids(
                result.inputs))

        self._ocp_has_solution = result.success
        print("Solution{}found".format(
            " " if self._ocp_has_solution else " not "))
        # Threshold below based on terminal cost params
        # if result.success and result.cost > 4 * 1e3:
        #     print("but high cost indicates no LC.")
        #     self._ocp_has_solution = False

        # TODO: testing stuff here
        if not self._ocp_has_solution:
            new_result = self._cost_history[-1].get_best_iteration()
            print(new_result['cost'])
            self._ocp_inputs_per_vehicle = (
                self._ocp_interface.map_input_to_vehicle_ids(
                    new_result['inputs']))
        self._ocp_has_solution = True

    def _log_results(self, counter: int, input_sequence: str,
                     output_sequence: str, solve_time: float,
                     result: opt.OptimalControlResult):
        self._results_summary['iteration'].append(counter)
        self._results_summary['solution_found'].append(result.success)
        self._results_summary['message'].append(
            result.message if not result.success else '')
        self._results_summary['cost'].append(result.cost)
        self._results_summary['solver_iterations'].append(result.nit)
        self._results_summary['time'].append(solve_time)
        self._results_summary['input_sequence'].append(input_sequence)
        self._results_summary['output_sequence'].append(output_sequence)
