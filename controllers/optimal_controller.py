from __future__ import annotations

from collections import defaultdict
import time
from typing import Dict, List, Union, Tuple
import warnings

import control as ct
import control.optimal as opt
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

import analysis  # only during tests
import controllers.optimal_control_costs as occ
import system_operating_mode as som
import vehicle_models.base_vehicle as base
import vehicle_group as vg
import vehicle_group_ocp_interface as vgi


def set_solver_parameters(
        max_iter: int = None, discretization_step: float = None,
        ftol: float = None, estimate_gradient: bool = True
) -> None:
    """
    Sets the configurations of the underlying optimization tool.
    :param max_iter: Maximum number of iterations by the solver
    :param discretization_step: Fixed discretization step of the opc solver
    :param ftol: Scipy minimize parameter: "Precision goal for the value of f
     in the stopping criterion."
    :param estimate_gradient: Allow the optimizer to estimate the gradient
     or provide analytical cost gradient
    :return:
    """
    if max_iter:
        VehicleOptimalController.solver_max_iter = max_iter
    if discretization_step:
        VehicleOptimalController.discretization_step = discretization_step
    if ftol:
        VehicleOptimalController.ftol = ftol
    VehicleOptimalController.estimate_gradient = estimate_gradient


def set_controller_parameters(
        max_iter: int = None, time_horizon: float = None,
        has_terminal_lateral_constraints: bool = False,
        has_lateral_safety_constraint: bool = False,
        provide_initial_guess: bool = False,
        initial_acceleration_guess: Union[str, float] = 0.0,
        jumpstart_next_solver_call: bool = False
) -> None:
    """
    Sets the configurations of the optimal controller which iteratively
    calls the optimization tool.
    :param max_iter: Maximum number of times the ocp will be solved until
     mode sequence convergence (different from max iteration of the solver).
    :param time_horizon: Final time of the optimal control problem.
    :param has_terminal_lateral_constraints: Whether to include terminal
     lateral constraints, i.e., lane changing vehicles must finish with
     y_d - e <= y(tf) <= y_d + e. If true, then there are no terminal costs.
    :param has_lateral_safety_constraint: Whether to include a constraint to
     keep the lane changing vehicles between y(t0)-1 and y(tf)+1. This can
     sometimes speed up simulations or prevent erratic behavior.
    :param provide_initial_guess: If true, simulates the system given the
     initial input guess and passes an (inputs, states) tuple as initial guess
     to the solver. It only affects the first call to the solver if
     jumpstart_next_solver_call is True.
    :param initial_acceleration_guess: Initial guess of the optimal
     acceleration. We can provide the exact value or one of the strings 'zero',
     'max' (max acceleration), 'min' (max brake). The same value is used for
     the entire time horizon. It only affects the first call to the solver if
     jumpstart_next_solver_call is True.
    :param jumpstart_next_solver_call: Whether to use the solution of the
     previous call to the solver as starting point for the next call.
    :return:
    """
    if max_iter:
        VehicleOptimalController.max_iter = max_iter
    if time_horizon:
        VehicleOptimalController.time_horizon = time_horizon

    VehicleOptimalController.has_terminal_lateral_constraints = (
        has_terminal_lateral_constraints)
    VehicleOptimalController.has_safety_lateral_constraint = (
        has_lateral_safety_constraint)
    VehicleOptimalController.provide_initial_guess = (
        provide_initial_guess)
    VehicleOptimalController.initial_acceleration_guess = (
        initial_acceleration_guess)
    VehicleOptimalController.jumpstart_next_solver_call = (
        jumpstart_next_solver_call)


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
    _cost_with_tracker: occ.OCPCostTracker

    # Solver params
    solver_max_iter: int = 300
    discretization_step: float = 1.0  # [s]
    ftol: float = 1.0e-6  # [s]
    estimate_gradient: bool = True

    # Our controller's params
    max_iter: int = 3
    time_horizon: float = 10.0  # [s]
    has_terminal_lateral_constraints: bool = False
    has_safety_lateral_constraint: bool = False
    provide_initial_guess: bool = False,
    initial_acceleration_guess: Union[str, float] = 0.0
    jumpstart_next_solver_call: bool = False

    def __init__(self):
        self._terminal_cost = None
        self._terminal_constraints = []
        self._constraints = []
        self._ocp_has_solution = False
        self._controlled_veh_ids: List[int] = []
        self._platoon_vehicle_pairs: List[Tuple[int, int]] = []
        self._dest_lane_vehicles_ids: List[int] = []

        self._data_per_iteration = []
        self._results_summary: defaultdict[str, List] = defaultdict(list)

    def get_time_horizon(self) -> float:
        return self.time_horizon

    def get_time_points(self, dt: float = None):
        if dt is None:
            dt = self.discretization_step  # [s]
        n = round(self.time_horizon / dt) + 1  # +1 to get 'round' times
        # We use the rounding function to prevent minor numerical differences
        # between time steps.
        rounding_precision = int(np.floor(np.log10(dt)))
        return np.round(np.linspace(0, self.time_horizon, n),
                        -rounding_precision)

    def get_data_per_iteration(self):
        return self._data_per_iteration

    def get_desired_state(self):
        return self._desired_state

    def get_results_summary(self):
        return self._results_summary

    def has_solution(self) -> bool:
        return self._ocp_has_solution

    def get_running_cost_history(self):
        return self._cost_with_tracker.get_running_cost()

    def get_terminal_cost_history(self):
        return self._cost_with_tracker.get_terminal_cost()

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

        current_inputs: Dict[int, np.ndarray] = {}
        for v_id in veh_ids:
            ego_inputs = self._ocp_inputs_per_vehicle[v_id]
            n_optimal_inputs = ego_inputs.shape[0]
            current_inputs[v_id] = np.zeros(n_optimal_inputs)
            for i in range(n_optimal_inputs):
                current_inputs[v_id][i] = np.interp(
                    current_time, self._ocp_time, ego_inputs[i])
                # idx = np.searchsorted(self._ocp_time, delta_t)
                # current_inputs[v_id].append(ego_inputs[i][idx])

        if single_veh_ctrl:
            return current_inputs[veh_ids[0]]
        else:
            return current_inputs

    def set_time_horizon(self, value) -> None:
        self.time_horizon = value

    def set_controlled_vehicles_ids(
            self, controlled_veh_ids: Union[int, List[int]]):
        if np.isscalar(controlled_veh_ids):
            controlled_veh_ids = [controlled_veh_ids]
        self._controlled_veh_ids = controlled_veh_ids

    def add_controlled_vehicle_id(self, new_vehicle_id: int):
        if new_vehicle_id in self._controlled_veh_ids:
            warnings.warn(f'Trying to add vehicle {new_vehicle_id} to '
                          f'optimal controller twice')
        else:
            self._controlled_veh_ids.append(new_vehicle_id)

    def set_platoon_formation_constraint_parameters(
            self, platoon_vehicles_ids: List[int],
            dest_lane_vehicles_ids: List[int]):
        self._platoon_vehicle_pairs = []
        for i in range(len(platoon_vehicles_ids)):
            for j in range(i + 1, len(platoon_vehicles_ids)):
                self._platoon_vehicle_pairs.append(
                    (platoon_vehicles_ids[i], platoon_vehicles_ids[j]))
        self._dest_lane_vehicles_ids = dest_lane_vehicles_ids

        print('Platoon veh pairs:', self._platoon_vehicle_pairs)
        print('Dest lane vehs:', self._dest_lane_vehicles_ids)

    def find_trajectory(self, vehicles: Dict[int, base.BaseVehicle]):
        """
        Solves the OPC for all listed controlled vehicles
        :param vehicles: All relevant vehicles
        :return: Nothing. All relevant values are stored internally
        """

        input_sequence: som.ModeSequence = som.ModeSequence()
        input_sequence.add_mode(0.0, som.SystemMode(vehicles))
        # som.append_mode_to_sequence(input_sequence, 0.1,
        #                             {'fd1': {'lo': 'p1', 'leader': 'p1'}})

        # We put the vehicle below at the origin (0, 0) when solving the opc
        # to make the solution independent of shifts in initial position
        center_veh_id = self._controlled_veh_ids[0]
        self._ocp_interface = vgi.VehicleGroupInterface(
            vehicles, self._controlled_veh_ids, center_veh_id)
        self._set_ocp_dynamics()
        self._initial_state = self._ocp_interface.get_initial_state()
        self._desired_state = self._ocp_interface.create_desired_state(
            self.time_horizon)
        self._create_cost_with_tracker()

        initial_guess = (self.create_initial_guess(vehicles)
                         if self.provide_initial_guess else None)
        converged = False
        counter = 0
        while not converged and counter < self.max_iter:
            counter += 1

            # input_seq_str = som.mode_sequence_to_str(input_sequence)
            print("Setting mode sequence to:  {}".format(input_sequence))
            self._ocp_interface.set_mode_sequence(input_sequence)
            self._set_constraints()
            self._cost_with_tracker.start_recording()

            start_time = time.time()
            self._solve_ocp(initial_guess)
            solve_time = time.time() - start_time

            # Temp: just checking what the optimizer "sees"
            alt = self.get_ocp_solver_simulation(vehicles)
            if len(self._platoon_vehicle_pairs) < 1:
                analysis.plot_constrained_lane_change(
                    alt.to_dataframe(), vehicles[center_veh_id].get_id())
            else:
                analysis.plot_platoon_lane_change(alt.to_dataframe())

            simulated_vehicle_group = self.simulate_over_optimization_horizon(
                vehicles)
            self._data_per_iteration.append(
                simulated_vehicle_group.to_dataframe())
            if len(self._platoon_vehicle_pairs) < 1:
                analysis.plot_constrained_lane_change(
                    self._data_per_iteration[-1],
                    vehicles[center_veh_id].get_id())
            else:
                analysis.plot_platoon_lane_change(self._data_per_iteration[-1])
            output_sequence = simulated_vehicle_group.get_mode_sequence()

            # output_seq_str = som.mode_sequence_to_str(output_sequence)
            print("Input sequence:  {}\nOutput sequence: {}".format(
                input_sequence, output_sequence
            ))
            self._log_results(counter, str(input_sequence),
                              str(output_sequence), solve_time, self.ocp_result)

            converged = input_sequence.is_equal_to(output_sequence,
                                                   self.discretization_step)
            # converged = som.compare_mode_sequences(
            #     input_sequence, output_sequence, self.discretization_step
            # )

            print("Converged?", converged)
            input_sequence = output_sequence
            if self.jumpstart_next_solver_call and self.ocp_result.success:
                last_input = self.ocp_result.inputs
                last_states = self.ocp_result.states
                initial_guess = (last_states, last_input)
            else:
                initial_guess = None

    def _set_ocp_configuration(self):
        self._set_ocp_dynamics()
        self._initial_state = self._ocp_interface.get_initial_state()
        self._desired_state = self._ocp_interface.create_desired_state(
            self.time_horizon)
        # print("x0 vs xf:")
        # print(self._initial_state)
        # print(self._desired_state)

    def _create_cost_with_tracker(self):
        # Note: running and terminal costs should depend on vehicle model
        # For three state: running costs y_cost=0, theta_cost=0.1 and

        u_ref = self._ocp_interface.get_desired_input()
        Q = self._ocp_interface.create_state_cost_matrix(
            y_cost=0.1, theta_cost=0., v_cost=0.1)
        R = self._ocp_interface.create_input_cost_matrix(accel_cost=0.1,
                                                         phi_cost=0.0)
        running_cost = occ.quadratic_cost(
            self._ocp_interface.n_states, self._ocp_interface.n_inputs,
            Q, R, self._desired_state, u_ref
        )
        # print('===== TRYING NEW COST FUNCTION =====')
        # running_cost = self._ocp_interface.cost_function(controlled_veh_ids)

        if not self.has_terminal_lateral_constraints:
            Q_terminal = (
                self._ocp_interface.create_state_cost_matrix(y_cost=10.,
                                                             theta_cost=1.)
            )
            R_terminal = self._ocp_interface.create_input_cost_matrix(
                phi_cost=0.)
            terminal_cost = occ.quadratic_cost(
                self._ocp_interface.n_states, self._ocp_interface.n_inputs,
                Q_terminal, R_terminal, self._desired_state, u_ref
            )
        else:
            Q_terminal = None
            R_terminal = None
            terminal_cost = None

        time_points = self.get_time_points()
        if self.estimate_gradient:
            self._cost_gradient = occ.quadratic_cost_gradient(
                self._ocp_interface.n_states, self._ocp_interface.n_inputs,
                len(time_points), Q, R, Q_terminal, R_terminal,
                self._desired_state, u_ref)
        else:
            self._cost_gradient = None

        self._cost_with_tracker = occ.OCPCostTracker(
            time_points, self._ocp_interface.n_states,
            running_cost, terminal_cost, self.solver_max_iter
        )

    def _set_ocp_dynamics(self):
        params = {'vehicle_group_interface': self._ocp_interface}
        input_names = self._ocp_interface.create_input_names()
        output_names = self._ocp_interface.create_output_names()
        n_states = self._ocp_interface.n_states
        # Define the vehicle dynamics as an input/output system
        self._dynamic_system = ct.NonlinearIOSystem(
            vgi.update_vehicles, vgi.vehicle_output,
            params=params, states=n_states, name='vehicle_group',
            inputs=input_names, outputs=output_names)

    def _set_constraints(self):
        self._constraints = []
        self._set_input_constraints()
        if self.has_terminal_lateral_constraints:
            self._set_terminal_constraints()
        self._set_safety_constraints()
        self._set_platoon_formation_constraints()

        self._cost_with_tracker.set_constraints(self._constraints)

    def _set_terminal_constraints(self):
        controlled_vehicles = [self._ocp_interface.vehicles[veh_id]
                               for veh_id in self._controlled_veh_ids]
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

        self._terminal_constraints.append(LinearConstraint(
            rows, lb=lower_boundaries, ub=upper_boundaries))

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

    def _set_safety_constraints(self):
        # Safety constraints
        # TODO: play with keep_feasible param for constraints

        epsilon = 1e-16
        for veh_id in self._controlled_veh_ids:
            veh = self._ocp_interface.vehicles[veh_id]

            lc_safety_constraint = NonlinearConstraint(
                self._ocp_interface.lane_changing_safety_constraint(veh_id),
                -epsilon, epsilon
            )
            self._constraints.append(lc_safety_constraint)

            if veh.is_long_control_optimal():
                veh_foll_constraint = NonlinearConstraint(
                    self._ocp_interface.vehicle_following_safety_constraint(
                        veh_id),
                    0., np.inf
                )
                self._constraints.append(veh_foll_constraint)

            if self.has_safety_lateral_constraint:
                d = np.zeros(self._ocp_interface.n_states
                             + self._ocp_interface.n_inputs)
                d[self._ocp_interface.get_a_vehicle_state_index(
                    veh.get_id(), 'y')] = 1
                stay_in_lane = LinearConstraint(d, lb=veh.get_y0() - 1,
                                                ub=veh.get_target_y() + 1)
                self._constraints.append(stay_in_lane)

    def _set_platoon_formation_constraints(self):
        for other_id in self._dest_lane_vehicles_ids:
            for id_pair in self._platoon_vehicle_pairs:
                print(f'Creating terminal constraint that prevents veh '
                      f'{other_id} from ending between vehs {id_pair}')
                self._terminal_constraints.append(NonlinearConstraint(
                    self._ocp_interface.platoon_constraint(id_pair, other_id),
                    0., np.inf
                ))

    def _solve_ocp(self, custom_initial_guess: np.ndarray = None):
        time_pts = self.get_time_points()

        # Initial guess can be a guess just of the input or of both input and
        # states
        if custom_initial_guess is not None:
            x0 = custom_initial_guess
        else:
            x0 = self._ocp_interface.get_initial_inputs_guess(
                self.initial_acceleration_guess, as_dict=False)

        # try:
        result = opt.solve_ocp(
            self._dynamic_system, time_pts, self._initial_state,
            cost=self._cost_with_tracker.get_running_cost_fun(),
            trajectory_constraints=self._constraints,
            terminal_cost=self._cost_with_tracker.get_terminal_cost_fun(),
            terminal_constraints=self._terminal_constraints,
            initial_guess=x0,
            minimize_options={
                'maxiter': self.solver_max_iter, 'disp': True,
                'ftol': self.ftol,
                'callback': self._cost_with_tracker.callback,
                'jac': self._cost_gradient
            }
            # minimize_method='trust-constr',
            # log=True
            # basis=flat.BezierFamily(5, T=self._ocp_horizon)
        )
        # except RuntimeError:
        #     pass
        # Note: the basis parameter above was set empirically - it might not
        # always work well

        # TODO: testing stuff here
        self._solver_result = result
        if result.success:
            self.ocp_result = (
                occ.OptimalControlIterationResult.copy_original_result(result)
            )
        else:
            # self.ocp_result = (
            #     self._cost_with_tracker.get_best_iteration_result()
            # )
            # print('Using input from the minimum cost feasible iteration.\n'
            #       'Iteration: {}. Cost: {}'.format(self.ocp_result.iteration,
            #                                        self.ocp_result.cost))
            self.ocp_result = (
                self._cost_with_tracker.get_last_iteration_result()
            )
            print('Using input from the last iteration.')
        # self.ocp_result = result
        self._ocp_time = self.ocp_result.time
        self._ocp_inputs_per_vehicle = (
            self._ocp_interface.map_input_to_vehicle_ids(
                self.ocp_result.inputs))

        self._ocp_has_solution = True  # TODO: self.ocp_result.success
        print("Solution{}found".format(
            " " if self._ocp_has_solution else " not "))
        # Threshold below based on terminal cost params
        # if result.success and result.cost > 4 * 1e3:
        #     print("but high cost indicates no LC.")
        #     self._ocp_has_solution = False
        # self._ocp_has_solution = True

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

    def simulate_over_optimization_horizon(
            self, vehicles: Dict[int, base.BaseVehicle]):
        dt = 1e-2
        sim_time = self.get_time_points(dt)

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_vehicles(vehicles)
        veh_ids = vehicle_group.sorted_vehicle_ids
        vehicle_group.set_verbose(False)
        vehicle_group.prepare_to_start_simulation(len(sim_time))
        for i in range(len(sim_time) - 1):
            optimal_inputs = self.get_input(sim_time[i], veh_ids)
            vehicle_group.simulate_one_time_step(sim_time[i + 1],
                                                 optimal_inputs)
        return vehicle_group

    def create_initial_guess(self, vehicles: Dict[int, base.BaseVehicle]):
        sim_time = self.get_time_points()
        initial_state_per_vehicle = (
            self._ocp_interface.map_state_to_vehicle_ids(self._initial_state)
        )

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_vehicles(vehicles,
                                             initial_state_per_vehicle)
        vehicle_group.set_verbose(False)
        vehicle_group.prepare_to_start_simulation(len(sim_time))
        input_guess = self._ocp_interface.get_initial_inputs_guess(
            self.initial_acceleration_guess, as_dict=True)
        # time is the last state
        states = np.zeros([self._ocp_interface.n_states, len(sim_time)])
        states[:, 0] = np.hstack((vehicle_group.get_current_state(),
                                  0.0))
        for i in range(1, len(sim_time)):
            vehicle_group.simulate_one_time_step(sim_time[i], input_guess)
            states[:, i] = np.hstack((vehicle_group.get_current_state(),
                                      sim_time[i]))
        input_array = []
        [input_array.extend(a) for a in list(input_guess.values())]
        inputs = np.repeat(input_array, len(sim_time)
                           ).reshape(-1, len(sim_time))
        return states, inputs

    def _log_results(self, counter: int, input_sequence: str,
                     output_sequence: str, solve_time: float,
                     result: occ.OptimalControlIterationResult):
        self._results_summary['iteration'].append(counter)
        self._results_summary['solution_found'].append(result.success)
        self._results_summary['message'].append(
            result.message if not result.success else '')
        self._results_summary['cost'].append(result.cost)
        self._results_summary['solver_iterations'].append(result.nit)
        self._results_summary['time'].append(solve_time)
        self._results_summary['input_sequence'].append(input_sequence)
        self._results_summary['output_sequence'].append(output_sequence)
