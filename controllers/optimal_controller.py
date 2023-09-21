from __future__ import annotations

from functools import partial
from typing import Dict, List, Union

import control as ct
import control.optimal as opt
import control.flatsys as flat
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

import analysis  # only during tests
import system_operating_mode as som
import vehicle_models.base_vehicle as base
import vehicle_group as vg
import vehicle_group_ocp_interface as vgi


def configure_optimal_controller(
        max_iter: int = None, solver_max_iter: int = None,
        discretization_step: float = None):
    """
    Sets the configurations of the optimal controller that do not vary during
    the simulation.
    :param max_iter: Maximum number of times the ocp will be solved until
     mode sequence convergence (different from max iteration of the solver)
    :param discretization_step: Fixed discretization step of the opc solver
    :param solver_max_iter: Maximum number of iterations by the solver
    :return:
    """
    if max_iter:
        VehicleOptimalController.max_iter = max_iter
    if solver_max_iter:
        VehicleOptimalController.solver_max_iter = solver_max_iter
    if discretization_step:
        VehicleOptimalController.discretization_step = discretization_step


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

    solver_max_iter = 300
    max_iter = 3
    discretization_step = 1.0  # [s]

    # TODO: make max_iter mandatory?
    def __init__(self, ocp_horizon: float,):
        """

        :param ocp_horizon: Fixed time horizon when solving the optimal
         control problem
        """
        self._ocp_horizon: float = ocp_horizon
        self._terminal_cost = None
        self._terminal_constraints = None
        self._constraints = []
        self._ocp_has_solution = False

        self._data_per_iteration = []

    def get_horizon(self) -> float:
        return self._ocp_horizon

    def get_data_per_iteration(self):
        return self._data_per_iteration

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
        # We put the vehicle below at the origin (0, 0) when solving the opc
        # to make the solution independent of shifts in initial position
        center_veh_id = controlled_veh_ids[0]
        self._ocp_interface = vgi.VehicleGroupInterface(vehicles, center_veh_id)
        if mode_sequence is None:
            # TODO: guess a mode change?
            given_sequence: som.ModeSequence = [(0.0, som.SystemMode(vehicles))]
        else:
            given_sequence = mode_sequence

        converged = False
        counter = 0
        while not converged and counter < self.max_iter:
            self._ocp_interface.set_mode_sequence(given_sequence)
            self._set_ocp_configuration(controlled_veh_ids)
            self._solve_ocp()
            simulated_vehicle_group = self.simulate_over_optimization_horizon(
                vehicles)
            self._data_per_iteration.append(
                simulated_vehicle_group.to_dataframe())
            analysis.plot_constrained_lane_change(
                self._data_per_iteration[-1], vehicles[center_veh_id].get_id())
            simulated_sequence = simulated_vehicle_group.get_mode_sequence()
            print("Input sequence:  {}\nOutput sequence: {}".format(
                som.mode_sequence_to_str(given_sequence),
                som.mode_sequence_to_str(simulated_sequence)
            ))
            converged = som.compare_mode_sequences(given_sequence,
                                                   simulated_sequence)
            print("Converged?", converged)
            given_sequence = simulated_sequence
            counter += 1

    # TODO move somewhere else in the class
    def simulate_over_optimization_horizon(
            self, vehicles: Dict[int, base.BaseVehicle]):
        dt = 1e-2
        time = np.arange(0, self.get_horizon(), dt)

        vehicle_group = vg.VehicleGroup()
        vehicle_group.populate_with_vehicles(vehicles)
        vehicle_group.set_verbose(False)
        vehicle_group.prepare_to_start_simulation(len(time))
        vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            vehicle_group.simulate_one_time_step(time[i + 1])
        return vehicle_group

    def _set_ocp_configuration(self, controlled_veh_ids: List[int]):
        self._set_ocp_dynamics()
        self._set_input_constraints()
        self._initial_state = self._ocp_interface.get_initial_state()
        self._desired_state = self._ocp_interface.create_desired_state(
            self._ocp_horizon)
        print("x0 vs xf:")
        print(self._initial_state)
        print(self._desired_state)
        # self._set_terminal_constraints(controlled_veh_ids)
        self._set_safety_constraints(controlled_veh_ids)
        self._set_costs()

    def has_solution(self) -> bool:
        return self._ocp_has_solution

    def get_desired_state(self):
        return self._desired_state

    def get_input(self, time: float, veh_ids: Union[int, List[int]]):
        """
        Gets the optimal input found by the solver at the given time by linear
        interpolation
        :param time: Current time assuming t0 = 0. This means some callers must
         provide t-t0.
        :param veh_ids: Ids of vehicles for which we want the inputs
        :return: Dictionary with veh ids as keys and inputs as values if
         multiple veh ids passed, single input vector if single veh id
        """
        # delta_t = time - self._start_time

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
                    time, self._ocp_time, ego_inputs[i]))
                # idx = np.searchsorted(self._ocp_time, delta_t)
                # current_inputs[v_id].append(ego_inputs[i][idx])

        if single_veh_ctrl:
            return current_inputs[veh_ids[0]]
        else:
            return current_inputs

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

    def _set_ocp_dynamics(self):
        params = {'vehicle_group': self._ocp_interface}
        input_names = self._ocp_interface.create_input_names()
        output_names = self._ocp_interface.create_output_names()
        n_states = self._ocp_interface.n_states
        # Define the vehicle dynamics as an input/output system
        self._dynamic_system = ct.NonlinearIOSystem(
            vgi.vehicles_derivatives, vgi.vehicle_output,
            params=params, states=n_states, name='vehicle_group',
            inputs=input_names, outputs=output_names)

    def _set_costs(self):
        # TODO: should depend on the vehicle model
        #  Three state: y_cost=0, theta_cost=0.1
        #  Safe vehicles: y_cost=0.1, theta_cost=10
        # Desired control; not final control
        uf = self._ocp_interface.get_desired_input()

        state_cost_matrix = self._ocp_interface.create_state_cost_matrix(
            y_cost=0.01, theta_cost=0, x_cost=0)

        input_cost_matrix = np.diag([0.01] * self._ocp_interface.n_inputs)
        self._running_cost = opt.quadratic_cost(
            self._dynamic_system, state_cost_matrix, input_cost_matrix,
            self._desired_state, uf)

        if self._terminal_constraints is None:
            p = 1000
            terminal_cost_matrix = (
                self._ocp_interface.create_terminal_cost_matrix(p))
            self._terminal_cost = opt.quadratic_cost(
                self._dynamic_system, terminal_cost_matrix, None,
                x0=self._desired_state)

    def _set_terminal_constraints(self, controlled_veh_ids: List[int]):
        # TODO: set terminal constraints for all vehicles?
        controlled_vehicles = self._ocp_interface.vehicles
        # controlled_vehicles = [self._ocp_interface.vehicles[veh_id]
        #                        for veh_id in controlled_veh_ids]
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
        self._constraints.extend([opt.input_range_constraint(
            self._dynamic_system, input_lower_bounds,
            input_upper_bounds)])

    def _set_safety_constraints(self, controlled_veh_ids: List[int]):
        # Safety constraints
        # TODO 1: play with keep_feasible param for constraints
        # TODO 2: keep stay_in_lane constraint?
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
            # self._constraints.append(stay_in_lane)
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

    def _solve_ocp(self):
        # TODO: try providing 'relevant' time points, such as when we know the
        #  input can become non-zero
        # TODO: initial control guess all zeros or some value?

        dt = VehicleOptimalController.discretization_step  # [s]
        time_pts = np.arange(0, self._ocp_horizon, dt)
        # n_ctrl_pts = 10  # min(round(self._ocp_horizon), 10)  # empirical
        # time_pts = np.linspace(0, self._ocp_horizon, n_ctrl_pts, endpoint=True)
        u0 = self._ocp_interface.get_desired_input()
        # u0 = self._ocp_interface.get_initial_guess(time_pts)
        result = opt.solve_ocp(
            self._dynamic_system, time_pts, self._initial_state,
            cost=self._running_cost,
            trajectory_constraints=self._constraints,
            terminal_cost=self._terminal_cost,
            terminal_constraints=self._terminal_constraints,
            initial_guess=u0,
            minimize_options={'maxiter': self.solver_max_iter,
                              'disp': True},
            # log=False
            # basis=flat.BezierFamily(5, T=self._ocp_horizon)
        )
        # Note: the basis parameter above was set empirically - it might not
        # always work well
        self.ocp_result = result
        self._ocp_time = result.time
        self._ocp_inputs_per_vehicle = (
            self._ocp_interface.map_ocp_solution_to_vehicle_inputs(
                result.inputs))

        self._ocp_has_solution = result.success
        print("Solution{}found".format(
            " " if self._ocp_has_solution else " not "))
        # Threshold below based on terminal cost params
        if result.success and result.cost > 4 * 1e3:
            print("but high cost indicates no LC.")
            self._ocp_has_solution = False
        # TODO: tests only [Sept 20]
        self._ocp_has_solution = True

    def detail_costs(self):
        """
        Attempt to figure out where cost differences between apparently equal
        configurations come from. However, the opc solver costs and the costs
        computed here always differ.
        :return:
        """
        ocp_states = self.ocp_result.states
        ocp_inputs = self.ocp_result.inputs
        running_cost = sum(
            self._running_cost(ocp_states[:, i], ocp_inputs[:, i])
            for i in range(ocp_states.shape[1]))
        terminal_cost = self._terminal_cost(ocp_states[:, -1], 0)
        print("Computed costs:")
        print("\tL={}, T={}, total={}".format(running_cost, terminal_cost,
                                              running_cost + terminal_cost))

