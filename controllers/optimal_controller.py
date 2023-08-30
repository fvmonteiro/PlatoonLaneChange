from __future__ import annotations

from functools import partial
from typing import Dict, List

import control as ct
import control.optimal as opt
import control.flatsys as flat
import numpy as np
from scipy.optimize import NonlinearConstraint

import vehicle_models.base_vehicle as base
import vehicle_group_ocp_interface as vgi


class VehicleOptimalController:
    _ocp_interface: vgi.VehicleGroupInterface
    _ocp_initial_state: np.ndarray
    _ocp_desired_state: np.ndarray

    _constraints = []
    _ocp_solver_max_iter = 200

    def __init__(self, ocp_horizon):
        self._ocp_horizon = ocp_horizon
        self.terminal_constraints = None
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

    def get_input(self, time: float, veh_ids: List[int]):
        delta_t = time - self._start_time

        current_inputs = []
        for v_id in veh_ids:
            ego_inputs = self._ocp_inputs_per_vehicle[v_id]
            n_optimal_inputs = ego_inputs.shape[0]
            for i in range(n_optimal_inputs):
                current_inputs.append(np.interp(
                    delta_t, self._ocp_time, ego_inputs[i]))

        # ego_inputs = self._ocp_interface.get_vehicle_inputs_vector_by_id(
        #     self.vehicle.id, self._ocp_inputs)
        # n_optimal_inputs = ego_inputs.shape[0]
        # current_inputs = np.zeros(n_optimal_inputs)
        # for i in range(n_optimal_inputs):
        #     current_inputs[i] = np.interp(
        #         delta_t, self._ocp_time, ego_inputs[i])
        return current_inputs

    def _set_ocp_dynamics(self):
        params = {'vehicle_group': self._ocp_interface}
        input_names = self._ocp_interface.create_input_names()
        output_names = self._ocp_interface.create_output_names()
        n_states = self._ocp_interface.n_states
        # Define the vehicle dynamics as an input/output system
        self.dynamic_system = ct.NonlinearIOSystem(
            vgi.vehicles_derivatives, vgi.vehicle_output,
            params=params, states=n_states, name='vehicle_group',
            inputs=input_names, outputs=output_names)

    def _set_ocp_costs(self):
        # Desired control; not final control
        uf = self._ocp_interface.get_desired_input()
        state_cost_matrix = np.diag([0, 0, 0.1, 0] * self._ocp_interface.n_vehs)
        input_cost_matrix = np.diag([0.1] * self._ocp_interface.n_inputs)
        self.running_cost = opt.quadratic_cost(
            self.dynamic_system, state_cost_matrix, input_cost_matrix,
            self._ocp_desired_state, uf)
        terminal_cost_matrix = np.diag([0, 1000, 1000, 0]
                                       * self._ocp_interface.n_vehs)
        self.terminal_cost = opt.quadratic_cost(
            self.dynamic_system, terminal_cost_matrix, 0,
            x0=self._ocp_desired_state)

    def _set_input_constraints(self):
        input_lower_bounds, input_upper_bounds = (
            self._ocp_interface.get_input_limits())
        self._constraints.extend([opt.input_range_constraint(
            self.dynamic_system, input_lower_bounds,
            input_upper_bounds)])

    def _set_safety_constraints(self, lc_vehicles):
        # Safety constraints
        epsilon = 1e-10
        for veh in lc_vehicles:
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

        # orig_lane_safety = NonlinearConstraint(
        #     self._safety_constraint_orig_lane_leader, -epsilon, epsilon)
        # dest_lane_leader_safety = NonlinearConstraint(
        #     self._safety_constraint_dest_lane_leader, -epsilon, epsilon)
        # dest_lane_follower_safety = NonlinearConstraint(
        #     self._safety_constraint_dest_lane_follower, -epsilon, epsilon)
        #
        # self._constraints.append(orig_lane_safety)
        # self._constraints.append(dest_lane_leader_safety)
        # self._constraints.append(dest_lane_follower_safety)

    # def _safety_constraint_orig_lane_leader(self, states, inputs):
    #     return self._ocp_interface.lane_changing_safety_constraint(
    #         states, inputs, self.vehicle.id,
    #         self.vehicle.get_orig_lane_leader_id(),
    #         is_other_behind=False)
    #
    # def _safety_constraint_dest_lane_leader(self, states, inputs):
    #     return self._ocp_interface.lane_changing_safety_constraint(
    #         states, inputs, self.vehicle.id,
    #         self.vehicle.get_dest_lane_leader_id(),
    #         is_other_behind=False)
    #
    # def _safety_constraint_dest_lane_follower(self, states, inputs):
    #     return self._ocp_interface.lane_changing_safety_constraint(
    #         states, inputs, self.vehicle.id,
    #         self.vehicle.get_dest_lane_follower_id(),
    #         is_other_behind=True)

    def _solve_ocp(self):

        u0 = self._ocp_interface.get_desired_input()
        n_ctrl_pts = min(round(self._ocp_horizon), 10)  # empirical
        time_pts = np.linspace(0, self._ocp_horizon, n_ctrl_pts, endpoint=True)
        result = opt.solve_ocp(
            self.dynamic_system, time_pts, self._ocp_initial_state,
            cost=self.running_cost,
            trajectory_constraints=self._constraints,
            terminal_cost=self.terminal_cost,
            terminal_constraints=self.terminal_constraints,
            initial_guess=u0,
            minimize_options={'maxiter': self._ocp_solver_max_iter},
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
