from __future__ import annotations

from collections import defaultdict
import time
from typing import Mapping, Sequence, Union
import warnings

import control as ct
import control.optimal as opt
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

import analysis  # only during tests
import constants
import controllers.optimal_control_costs as occ
import operating_modes.system_operating_mode as som
import vehicle_models.base_vehicle as base
import vehicle_group as vg
import vehicle_group_ocp_interface as vgi

config = constants.Configuration


class VehicleOptimalController:
    """
    The optimal controller follows the steps:
    1. set an operating mode sequence
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

    _internal_sim_dt = 1.0e-2  # to simulate the system (not to solve the ocp)

    def __init__(self):
        self.time_horizon = 0.
        self._set_config()
        self._terminal_cost = None
        self._terminal_constraints = []
        self._constraints = []
        self._ocp_has_solution = False
        self._controlled_veh_ids: set[int] = set()
        # We center the system around some vehicle when solving the opc
        # to make the solution independent of shifts in initial position
        self._center_veh_id: int = 0
        self._platoon_vehicle_pairs: list[tuple[int, int]] = []
        self._dest_lane_vehicles_ids: list[int] = []
        self._activation_time = np.inf

        self._data_per_iteration = []
        self._results_summary: defaultdict[str, list] = defaultdict(list)

    def _set_config(self):
        # Solver params
        self.solver_max_iter: int = config.solver_max_iter
        self.discretization_step: float = config.discretization_step
        self.ftol: float = config.ftol
        self.estimate_gradient: bool = config.estimate_gradient

        # Our controller's params
        self.max_iter: int = config.max_iter
        self.time_horizon: float = config.time_horizon
        self.has_terminal_lateral_constraints: bool = (
            config.has_terminal_lateral_constraints)
        self.has_safety_lateral_constraint: bool = (
            config.has_safety_lateral_constraint)
        self.provide_initial_guess: bool = config.has_initial_state_guess
        self.initial_acceleration_guess: Union[str, float] = (
            config.initial_acceleration_guess)
        self.jumpstart_next_solver_call: bool = (
            config.jumpstart_next_solver_call)
        self.has_initial_mode_guess: bool = config.has_initial_mode_guess

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

    def get_activation_time(self):
        return self._activation_time

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

    def get_input(self, current_time: float, veh_ids: Union[int, Sequence[int]],
                  t0: float = None):
        """
        Gets the optimal input found by the solver at the given time by linear
        interpolation
        :param current_time:
        :param veh_ids: Ids of vehicles for which we want the inputs
        :param t0: If None, we assume t0 = activation_time
        :return: dictionary with veh ids as keys and inputs as values if
         multiple veh ids passed, single input vector if single veh id
        """

        if np.isscalar(veh_ids):
            single_veh_ctrl = True
            veh_ids = [veh_ids]
        else:
            single_veh_ctrl = False
        if t0 is None:
            t0 = self._activation_time

        delta_t = current_time - t0
        current_inputs: dict[int, np.ndarray] = {}
        for v_id in veh_ids:
            ego_inputs = self._ocp_inputs_per_vehicle[v_id]
            n_optimal_inputs = ego_inputs.shape[0]
            current_inputs[v_id] = np.zeros(n_optimal_inputs)
            for i in range(n_optimal_inputs):
                current_inputs[v_id][i] = np.interp(
                    delta_t, self._ocp_time, ego_inputs[i])
                # idx = np.searchsorted(self._ocp_time, delta_t)
                # current_inputs[v_id].append(ego_inputs[i][idx])

        if single_veh_ctrl:
            return current_inputs[veh_ids[0]]
        else:
            return current_inputs

    def is_active(self, current_time: float):
        return (self._activation_time <= current_time
                <= self._activation_time + self.time_horizon)

    def set_time_horizon(self, value) -> None:
        self.time_horizon = value

    def set_controlled_vehicles_ids(
            self, controlled_veh_ids: Union[int, Sequence[int]]):
        if np.isscalar(controlled_veh_ids):
            controlled_veh_ids = [controlled_veh_ids]
        self._center_veh_id = controlled_veh_ids[0]
        self._controlled_veh_ids = set(controlled_veh_ids)

    def add_controlled_vehicle_id(self, new_vehicle_id: int):
        if new_vehicle_id in self._controlled_veh_ids:
            warnings.warn(f'Trying to add vehicle {new_vehicle_id} to '
                          f'optimal controller twice')
        else:
            self._controlled_veh_ids.add(new_vehicle_id)

    def set_platoon_formation_constraint_parameters(
            self, platoon_vehicles_ids: Sequence[int],
            dest_lane_vehicles_ids: list[int]):
        self._platoon_vehicle_pairs = []
        for i in range(len(platoon_vehicles_ids)):
            for j in range(i + 1, len(platoon_vehicles_ids)):
                self._platoon_vehicle_pairs.append(
                    (platoon_vehicles_ids[i], platoon_vehicles_ids[j]))
        self._dest_lane_vehicles_ids = dest_lane_vehicles_ids

        print('Platoon veh pairs:', self._platoon_vehicle_pairs)
        print('Dest lane vehs:', self._dest_lane_vehicles_ids)

    def find_trajectory(self, vehicles: Mapping[int, base.BaseVehicle]):
        """
        Solves the OPC for all listed controlled vehicles
        :param vehicles: All relevant vehicles
        :return: Nothing. All relevant values are stored internally
        """

        self._activation_time = vehicles[self._center_veh_id].get_current_time()

        self._ocp_interface = vgi.VehicleGroupInterface(
            vehicles, self._center_veh_id)
        self._set_ocp_dynamics()
        self._initial_state = self._ocp_interface.get_initial_state()
        self._desired_state = self._ocp_interface.create_desired_state(
            self.time_horizon)
        self._create_cost_with_tracker()

        # state_guess, input_guess = (self.create_initial_guess(vehicles)
        #                             if self.provide_initial_guess else None)
        mode_sequence_guess, trajectory_guess = (
            self._create_initial_mode_guess(vehicles, plot_result=True))
        converged = False
        counter = 0
        while not converged and counter < self.max_iter:
            counter += 1

            print("Setting mode sequence to:\n"
                  f"{mode_sequence_guess.to_string(skip_lines=True)}")
            self._ocp_interface.set_mode_sequence(mode_sequence_guess)
            self._set_constraints()
            self._cost_with_tracker.start_recording()

            print("Calling ocp solver...")
            start_time = time.time()
            self._solve_ocp(trajectory_guess)
            solve_time = time.time() - start_time

            # Temp: just checking what the optimizer "sees"
            self.check_ocp_solver_simulation(vehicles)

            simulated_vehicle_group = self.simulate_over_optimization_horizon(
                vehicles, plot_result=True)
            self._data_per_iteration.append(
                simulated_vehicle_group.to_dataframe())
            output_sequence = simulated_vehicle_group.get_mode_sequence()

            print("Input sequence:  {}\nOutput sequence: {}".format(
                mode_sequence_guess, output_sequence
            ))
            self._log_results(counter, str(mode_sequence_guess),
                              str(output_sequence), solve_time, self.ocp_result)

            converged = mode_sequence_guess.is_equal_to(
                output_sequence, self.discretization_step)

            print("Converged?", converged)
            mode_sequence_guess = output_sequence
            if self.jumpstart_next_solver_call and self.ocp_result.success:
                trajectory_guess = (self.ocp_result.states,
                                    self.ocp_result.inputs)

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

    def _create_initial_mode_guess(
            self, vehicles: Mapping[int, base.BaseVehicle],
            plot_result: bool = False
    ) -> tuple[som.ModeSequence, tuple[np.ndarray, np.ndarray]]:
        if self.has_initial_mode_guess:
            # TODO: in this case we could also use the states and inputs
            #  as initial guess
            cl_simulation = self.get_strategy_mode_sequence(vehicles,
                                                            plot_result)
            mode_sequence = cl_simulation.get_mode_sequence()
            optimal_veh_ids = [
                veh_id for veh_id in self._controlled_veh_ids
                if vehicles[veh_id].get_has_open_loop_acceleration()]
            mode_sequence.remove_leader_from_modes(optimal_veh_ids)

            states = cl_simulation.get_all_states()
            vehicle_input_map = {
                veh_id: vehicles[veh_id].get_external_input_idx()
                for veh_id in self._controlled_veh_ids}
            inputs = cl_simulation.get_selected_inputs(vehicle_input_map)
            sampling = int(self.discretization_step // self._internal_sim_dt)
            states = states[:, [i for i in range(0, states.shape[1], sampling)]]
            inputs = inputs[:, [i for i in range(0, inputs.shape[1], sampling)]]
            trajectory_guess = states, inputs
        else:
            mode_sequence = som.ModeSequence()
            mode_sequence.add_mode(0.0, som.SystemMode(vehicles))
            trajectory_guess = (self.create_initial_guess(vehicles)
                                if self.provide_initial_guess else None)

        return mode_sequence, trajectory_guess

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
        epsilon = 1e-16
        for veh_id in self._controlled_veh_ids:
            veh = self._ocp_interface.vehicles[veh_id]
            if veh.has_lane_change_intention():
                # lc_safety_constraint = NonlinearConstraint(
                #     self._ocp_interface.lane_changing_safety_constraint(veh_id),
                #     -epsilon, epsilon
                # )
                # self._constraints.append(lc_safety_constraint)
                lc_safety_constraints = [NonlinearConstraint(
                    fun, -epsilon, epsilon) for fun in
                    self._ocp_interface.lane_changing_safety_constraints(
                        veh_id)]
                self._constraints.extend(lc_safety_constraints)
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

    def _solve_ocp(
            self, custom_initial_guess: tuple[np.ndarray, np.ndarray] = None):
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
            # log=True
        )
        # except RuntimeError:
        #     pass

        # TODO: testing stuff here
        self._solver_result = result
        if result.success:
            self.ocp_result = (
                occ.OptimalControlIterationResult.copy_original_result(result)
            )
        else:
            if self.has_initial_mode_guess or self.provide_initial_guess:
                self.ocp_result = (
                    self._cost_with_tracker.get_best_iteration_result()
                )
                print('Using input from the minimum cost feasible iteration.\n'
                      f'Iteration: {self.ocp_result.iteration}. '
                      f'Cost: {self.ocp_result.cost}')
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

    def get_strategy_mode_sequence(
            self, vehicles: Mapping[int, base.BaseVehicle],
            plot_result: bool = True):
        dt = self._internal_sim_dt
        sim_time = self.get_time_points(dt)
        initial_state_per_vehicle = (
            self._ocp_interface.map_state_to_vehicle_ids(self._initial_state)
        )
        chosen_strategy = -1
        best_cost = np.inf
        vehicle_groups = []
        for lc_strategy in config.platoon_strategies:
            # Simulate
            veh_group = vg.VehicleGroup()
            veh_group.set_platoon_lane_change_strategy(lc_strategy)
            veh_group.populate_with_closed_loop_copies(
                vehicles, self._controlled_veh_ids, initial_state_per_vehicle)
            veh_group.set_verbose(False)
            veh_group.prepare_to_start_simulation(len(sim_time))
            for i in range(len(sim_time) - 1):
                veh_group.simulate_one_time_step(sim_time[i + 1])
            # Check results
            success = veh_group.check_lane_change_success()
            vehicle_input_map = {
                veh_id: vehicles[veh_id].get_external_input_idx()
                for veh_id in self._controlled_veh_ids}
            relevant_inputs = veh_group.get_selected_inputs(vehicle_input_map)
            r_cost, t_cost = self._cost_with_tracker.compute_simulation_cost(
                veh_group.get_all_states(),
                relevant_inputs,
                sim_time)
            print(f'Strategy {lc_strategy} successful? {success}. '
                  f'Cost: {r_cost:.2f}(running) + {t_cost:.2f}(terminal) = '
                  f'{r_cost + t_cost:.2f}')
            # if plot_result:
            #     data = veh_group.to_dataframe()
            #     if len(self._controlled_veh_ids) <= 1:
            #         analysis.plot_constrained_lane_change(
            #             data, vehicles[self._center_veh_id].get_id())
            #     else:
            #         analysis.plot_platoon_lane_change(data)
            #     analysis.plot_trajectory(data)
            # Store
            if success and r_cost + t_cost < best_cost:
                best_cost = r_cost + t_cost
                chosen_strategy = lc_strategy
            vehicle_groups.append(veh_group)

        print(f'Strategy {chosen_strategy} chosen.')
        if plot_result:
            data = vehicle_groups[chosen_strategy].to_dataframe()
            if len(self._controlled_veh_ids) <= 1:
                analysis.plot_constrained_lane_change(
                    data, vehicles[self._center_veh_id].get_id())
            else:
                analysis.plot_platoon_lane_change(data)
            analysis.plot_trajectory(data)
        return vehicle_groups[chosen_strategy]

    def check_ocp_solver_simulation(
            self, vehicles: Mapping[int, base.BaseVehicle],
            plot_result: bool = True) -> vg.VehicleGroup:
        """
        Reads and plots the results of the OCP solution as the solver sees it,
        that is, the expected open loop solution with the given mode sequence.
        :param vehicles: All the simulation vehicles
        :param plot_result: If True, plots the results
        :return: The simulated vehicle group is returned in case further
         manipulation of the results is needed.
        """
        sim_time = self.ocp_result.time
        initial_state_per_vehicle = (
            self._ocp_interface.map_state_to_vehicle_ids(self._initial_state)
        )

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_open_loop_copies(
            vehicles, self._controlled_veh_ids, initial_state_per_vehicle)
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
        if plot_result:
            if len(self._platoon_vehicle_pairs) < 1:
                analysis.plot_constrained_lane_change(
                    vehicle_group.to_dataframe(),
                    vehicles[self._center_veh_id].get_id())
            else:
                analysis.plot_platoon_lane_change(vehicle_group.to_dataframe())
        return vehicle_group

    def simulate_over_optimization_horizon(
            self, vehicles: Mapping[int, base.BaseVehicle],
            plot_result: bool = False) -> vg.VehicleGroup:
        """
        Runs the system with the open loop solution.
        :param vehicles: All the simulation vehicles
        :param plot_result: If True, plots the results
        :return: The simulated vehicle group.
        """
        dt = self._internal_sim_dt
        sim_time = self.get_time_points(dt)

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_open_loop_copies(
            vehicles, self._controlled_veh_ids)
        veh_ids = vehicle_group.sorted_vehicle_ids
        vehicle_group.set_verbose(False)
        vehicle_group.prepare_to_start_simulation(len(sim_time))
        for i in range(len(sim_time) - 1):
            optimal_inputs = self.get_input(sim_time[i], veh_ids, t0=0.)
            vehicle_group.simulate_one_time_step(sim_time[i + 1],
                                                 optimal_inputs)
        if plot_result:
            if len(self._platoon_vehicle_pairs) < 1:
                analysis.plot_constrained_lane_change(
                    vehicle_group.to_dataframe(),
                    vehicles[self._center_veh_id].get_id())
            else:
                analysis.plot_platoon_lane_change(vehicle_group.to_dataframe())
        return vehicle_group

    def create_initial_guess(self, vehicles: Mapping[int, base.BaseVehicle]
                             ) -> (np.ndarray, np.ndarray):
        sim_time = self.get_time_points()
        initial_state_per_vehicle = (
            self._ocp_interface.map_state_to_vehicle_ids(self._initial_state)
        )

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_open_loop_copies(
            vehicles, self._controlled_veh_ids, initial_state_per_vehicle)
        vehicle_group.set_verbose(False)
        vehicle_group.prepare_to_start_simulation(len(sim_time))
        input_guess = self._ocp_interface.get_initial_inputs_guess(
            self.initial_acceleration_guess, as_dict=True)
        for i in range(1, len(sim_time)):
            vehicle_group.simulate_one_time_step(sim_time[i], input_guess)
        vehicle_input_map = {
            veh_id: vehicles[veh_id].get_external_input_idx()
            for veh_id in self._controlled_veh_ids}
        optimal_inputs = vehicle_group.get_selected_inputs(vehicle_input_map)
        return vehicle_group.get_all_states(), optimal_inputs

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
