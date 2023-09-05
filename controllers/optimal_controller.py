from __future__ import annotations

from functools import partial
from typing import Dict, List, Union

import control as ct
import control.optimal as opt
import control.flatsys as flat
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

import vehicle_models.base_vehicle as base
import vehicle_group_ocp_interface as vgi


class VehicleOptimalController:
    _ocp_interface: vgi.VehicleGroupInterface
    _ocp_initial_state: np.ndarray
    _ocp_desired_state: np.ndarray

    _constraints = []
    _ocp_solver_max_iter = 200

    def __init__(self, ocp_horizon: float):
        self._ocp_horizon: float = ocp_horizon
        self._terminal_constraints = None
        self._start_time = -np.inf

    def find_lane_change_trajectory(self, time: float,
                                    vehicles: Dict[int, base.BaseVehicle],
                                    lc_veh_ids: List[int]):
        self._start_time = time
        self._ocp_interface = vgi.VehicleGroupInterface(vehicles)
        self._set_ocp_dynamics()
        self._ocp_initial_state = self._ocp_interface.get_initial_state()
        self._ocp_desired_state = self._ocp_interface.create_desired_state(
            self._ocp_horizon)
        self._set_ocp_costs()
        self._set_input_constraints()
        self._set_safety_constraints([vehicles[veh_id]
                                      for veh_id in lc_veh_ids])
        self._solve_ocp()

    def has_solution(self) -> bool:
        return self._ocp_has_solution

    def get_desired_state(self):
        return self._ocp_desired_state

    def get_input(self, time: float, veh_ids: Union[int, List[int]]):
        """

        :param time: Current simulation time
        :param veh_ids: Ids of vehicles for which we want the inputs
        :return: Dictionary with veh ids as keys and inputs as values if
         multiple veh ids passed, single input vector is single veh id
        """
        delta_t = time - self._start_time

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
                    delta_t, self._ocp_time, ego_inputs[i]))

        if single_veh_ctrl:
            return current_inputs[veh_ids[0]]
        else:
            return current_inputs

    def _set_ocp_dynamics(self):
        params = {'vehicle_group': self._ocp_interface,
                  'mode_sequence': []}
        # TODO list of tuples: mode end time and mode. Maybe reverse order of
        #  time, so we can just pop elements as the time passes
        input_names = self._ocp_interface.create_input_names()
        output_names = self._ocp_interface.create_output_names()
        n_states = self._ocp_interface.n_states
        # Define the vehicle dynamics as an input/output system
        self._dynamic_system = ct.NonlinearIOSystem(
            vgi.vehicles_derivatives, vgi.vehicle_output,
            params=params, states=n_states, name='vehicle_group',
            inputs=input_names, outputs=output_names)

    def _set_ocp_costs(self):
        # Desired control; not final control
        uf = self._ocp_interface.get_desired_input()

        state_cost_matrix = self._ocp_interface.create_state_cost_matrix(
            y_cost=0.1, theta_cost=10)

        input_cost_matrix = np.diag([0.01] * self._ocp_interface.n_inputs)
        self._running_cost = opt.quadratic_cost(
            self._dynamic_system, state_cost_matrix, input_cost_matrix,
            self._ocp_desired_state, uf)

        p = 1000
        terminal_cost_matrix = (
            self._ocp_interface.create_terminal_cost_matrix(p))
        self._terminal_cost = opt.quadratic_cost(
            self._dynamic_system, terminal_cost_matrix, 0,
            x0=self._ocp_desired_state)

    def _set_input_constraints(self):
        input_lower_bounds, input_upper_bounds = (
            self._ocp_interface.get_input_limits())
        self._constraints.extend([opt.input_range_constraint(
            self._dynamic_system, input_lower_bounds,
            input_upper_bounds)])

    def _set_safety_constraints(self, lc_vehicles):
        # Safety constraints
        # TODO: play with keep_feasible param for constraints
        epsilon = 1e-10
        for veh in lc_vehicles:
            d = np.zeros(self._ocp_interface.n_states
                         + self._ocp_interface.n_inputs)
            d[self._ocp_interface.get_a_vehicle_state_index(veh.id, 'y')] = 1
            stay_in_lane = LinearConstraint(d, lb=-1, ub=5)
            self._constraints.append(stay_in_lane)
            if veh.has_orig_lane_leader():
                orig_lane_safety = NonlinearConstraint(
                    partial(
                        self._ocp_interface.lane_changing_safety_constraint,
                        lc_veh_id=veh.id,
                        other_id=veh.get_orig_lane_leader_id(),
                        is_other_behind=False),
                    -epsilon, epsilon)
                self._constraints.append(orig_lane_safety)
            if veh.has_dest_lane_leader():
                dest_lane_leader_safety = NonlinearConstraint(
                    partial(
                        self._ocp_interface.lane_changing_safety_constraint,
                        lc_veh_id=veh.id,
                        other_id=veh.get_dest_lane_leader_id(),
                        is_other_behind=False),
                    -epsilon, epsilon)
                self._constraints.append(dest_lane_leader_safety)
            if veh.has_dest_lane_follower():
                dest_lane_follower_safety = NonlinearConstraint(
                    partial(
                        self._ocp_interface.lane_changing_safety_constraint,
                        lc_veh_id=veh.id,
                        other_id=veh.get_dest_lane_follower_id(),
                        is_other_behind=True),
                    -epsilon, epsilon)
                self._constraints.append(dest_lane_follower_safety)

    def _solve_ocp(self):

        u0 = self._ocp_interface.get_desired_input()
        n_ctrl_pts = min(round(self._ocp_horizon), 10)  # empirical
        time_pts = np.linspace(0, self._ocp_horizon,
                               int(round(self._ocp_horizon)) + 1, endpoint=True)
        # time_pts = np.array([0, 1.5, 2.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0])
        # TODO: try providing 'relevant' time points, such as when we know the
        #  input can become non-zero
        result = opt.solve_ocp(
            self._dynamic_system, time_pts, self._ocp_initial_state,
            cost=self._running_cost,
            trajectory_constraints=self._constraints,
            terminal_cost=self._terminal_cost,
            terminal_constraints=self._terminal_constraints,
            initial_guess=u0,
            minimize_options={'maxiter': self._ocp_solver_max_iter,
                              'disp': True},
            log=True
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
