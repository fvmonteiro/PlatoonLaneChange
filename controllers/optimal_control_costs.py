from __future__ import annotations

from typing import Callable, Union

import control.optimal as opt
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, NonlinearConstraint


class OCPCostTracker:
    """
    Makes it possible to keep track of the computed costs during the optimal
    control problem solver iterations.
    """

    _relative_tolerance = 1.0e-2  # used when checking solution feasibility

    def __init__(
            self, time_points: np.ndarray, n_states: int,
            running_cost: Callable, terminal_cost: Callable = None,
            max_iterations: int = None
    ):
        self._time_points = time_points
        self._n_states = n_states
        self._running_cost_fun = running_cost
        self._terminal_cost_fun = terminal_cost
        self._max_iterations = max_iterations

        self._n_time_points: int = len(time_points)
        self._linear_constraints = []
        self._lc_absolute_tolerance = []
        self._non_linear_constraints = []
        self._current_call_costs: np.ndarray = np.zeros(self._n_time_points)
        self._call_counter: int = 0
        # One list of costs/states/inputs for each time we call the ocp solver
        self._running_cost_per_call: list[list[float]] = []
        self._terminal_cost_per_call: list[list[float]] = []
        self._running_cost_per_iteration: list[list[float]] = []
        self._terminal_cost_per_iteration: list[list[float]] = []
        self._states_per_iteration: list[list[np.ndarray]] = []
        self._inputs_per_iteration: list[list[np.ndarray]] = []

        # Best feasible iteration (for each call to the ocp solver)
        # self._best_states: list[np.array] = []
        # self._best_inputs: list[np.array] = []
        self._best_cost: float = np.inf
        self._best_iteration: list[int] = []

    def get_running_cost(self) -> list[np.array]:
        return [np.array(costs) for costs in self._running_cost_per_iteration]

    def get_terminal_cost(self) -> list[np.array]:
        return [np.array(costs) for costs in self._terminal_cost_per_iteration]

    def get_time_points(self) -> np.ndarray:
        return self._time_points

    def get_states_per_iteration(self):
        return self._states_per_iteration

    def get_inputs_per_iteration(self):
        return self._inputs_per_iteration

    def has_terminal_cost(self):
        return self._terminal_cost_fun is not None

    def start_recording(self):
        self._current_call_costs = np.zeros(self._n_time_points)
        self._call_counter = 0
        self._running_cost_per_call.append([])
        self._terminal_cost_per_call.append([])
        self._running_cost_per_iteration.append([])
        self._terminal_cost_per_iteration.append([])
        self._states_per_iteration.append([])
        self._inputs_per_iteration.append([])
        self._best_cost = np.inf
        self._best_iteration.append(-1)  # by default, the last run iteration
        # self._best_states.append(None)
        # self._best_inputs.append(None)

    def get_running_cost_fun(self) -> Callable:
        def support(states, inputs):
            cost = self._running_cost_fun(states, inputs)
            self._save_running_cost_per_call(cost)
            return cost

        return support

    def get_terminal_cost_fun(self):
        def support(states, inputs):
            cost = self._terminal_cost_fun(states, inputs)
            self._terminal_cost_per_call[-1].append(cost)
            return cost

        if self.has_terminal_cost():
            return support
        return None

    def print_cost_all_solutions(self):
        for i in range(len(self._running_cost_per_iteration)):
            print("==== OCP solution {} ====".format(i))
            self.print_cost_from_solution_i(i)

    def print_cost_from_solution_i(self, ocp_solution_number: int):
        rc = self._running_cost_per_iteration[ocp_solution_number]
        tc = self._terminal_cost_per_iteration[ocp_solution_number]
        output = '{:10s}\t{:10s}\t{:10s}\t{:10s}\n'.format(
            'Iteration', 'Running', 'Terminal', 'Total')
        for i in range(len(rc)):
            output += '{:<10d}\t{:<10.6g}\t{:<10.6g}\t{:<10.6g}\n'.format(
                i, rc[i], tc[i], rc[i] + tc[i])
        print(output)

    def get_total_cost(self, ocp_solution_number: int, iteration: int):
        return (
                self._running_cost_per_iteration[ocp_solution_number][iteration]
                + self._terminal_cost_per_iteration[ocp_solution_number][
                    iteration])

    def get_best_iteration_result(self, ocp_solution_number: int = -1):
        """
        Gets the states and inputs from the iteration that had the minimum cost
        :return:
        """
        # rc = np.array(self._running_cost_per_iteration[ocp_solution_number])
        # tc = np.array(self._terminal_cost_per_iteration[ocp_solution_number])
        # best_idx = np.argmin(rc + tc)
        # return OptimalControlIterationResult.get_result_from_iteration(
        #     self, best_idx)
        best_idx = self._best_iteration[ocp_solution_number]
        return OptimalControlIterationResult.get_result_from_iteration(
            self, ocp_solution_number, best_idx
        )

    def get_last_iteration_result(self, ocp_solution_number: int = -1):
        return OptimalControlIterationResult.get_result_from_iteration(
            self, ocp_solution_number, -1
        )

    # def get_worst_iteration_result(self, ocp_solution_number: int = -1):
    #     rc = np.array(self._running_cost_per_iteration[ocp_solution_number])
    #     tc = np.array(self._terminal_cost_per_iteration[ocp_solution_number])
    #     worst_idx = np.argmax(rc + tc)
    #     return OptimalControlIterationResult.get_result_from_iteration(
    #         self, worst_idx)

    def callback(self, x) -> None:
        states, inputs = self._compute_states_inputs(x)

        # Note: we're recomputing costs here, but given how simple they are
        # this should not affect performance

        # Cost over time
        costs = [self._running_cost_fun(states[:, i], inputs[:, i]) for
                 i in range(self._n_time_points)]
        # Compute the time intervals
        dt = np.diff(self._time_points)
        # Integrate the cost
        running_cost = 0
        for i in range(self._n_time_points - 1):
            # Approximate the integral using trapezoidal rule
            running_cost += 0.5 * (costs[i] + costs[i + 1]) * dt[i]

        terminal_cost = 0.
        if self.has_terminal_cost():
            terminal_cost = self._terminal_cost_fun(states[:, -1],
                                                    inputs[:, -1])

        # Store everyone
        self._running_cost_per_iteration[-1].append(running_cost)
        self._terminal_cost_per_iteration[-1].append(terminal_cost)
        self._states_per_iteration[-1].append(states)
        self._inputs_per_iteration[-1].append(inputs)

        # Store min cost values
        iteration_cost = running_cost + terminal_cost
        if (iteration_cost < self._best_cost
                and self.check_feasibility(states, inputs)):
            self._best_cost = iteration_cost
            iteration = len(self._running_cost_per_iteration[-1]) - 1
            self._best_iteration[-1] = iteration
            # self._best_states[-1] = states
            # self._best_inputs[-1] = inputs

        if self._max_iterations is not None:
            iterations = len(self._running_cost_per_iteration[-1])
            percentage = iterations * 100 / self._max_iterations
            if percentage % 5 == 0:
                print(f'{iterations}/{self._max_iterations} iterations. '
                      f'Last cost: {iteration_cost:.5g}.')

        # Compare iterations:
        # delta_cost = (np.diff(self._running_cost_per_iteration[-1][-2:])
        #               + np.diff(self._terminal_cost_per_iteration[-1][-2:]))
        # if np.abs(delta_cost) < 1.0:
        #     raise RuntimeError(
        #         "Terminating optimization: small variation in cost"
        #         " between iterations: {}".format(delta_cost))

    # def cost_gradient(self, x):
    # States are appended to end of (input) the optimization variables
    # states = x[-self._n_states * self._n_time_points:].reshape(
    #     self._n_states, -1)
    # x = x[:-self._n_states * self._n_time_points]
    #
    # # Note: this only works if we are not using any basis functions for the
    # # control input.
    # inputs = x.reshape((-1, self._n_time_points))

    def check_feasibility(self, states, inputs):
        # margin = 1.0e-3
        for i in range(self._n_time_points):
            for j in range(len(self._linear_constraints)):
                lc = self._linear_constraints[j]
                margin_l, margin_u = self._lc_absolute_tolerance[j]
                rl, ru = lc.residual(np.hstack([states[:, i], inputs[:, i]]))
                if np.any(rl < -margin_l) or np.any(ru < -margin_u):
                    return False
            for nlc in self._non_linear_constraints:
                if not ((nlc.lb * (1.0 + self._relative_tolerance))
                        <= nlc.fun(states[:, i], inputs[:, i]) <=
                        (nlc.ub * (1.0 + self._relative_tolerance))):
                    return False
        return True

    def _save_running_cost_per_call(self, current_cost):
        self._current_call_costs[self._call_counter] = current_cost
        self._call_counter += 1

        # The running cost is called n_time_points for every call the optimizer
        # makes to the cost function.
        if self._call_counter >= self._n_time_points:
            # Compute the time intervals
            dt = np.diff(self._time_points)
            total_cost = 0
            for i in range(self._n_time_points - 1):
                # Approximate the integral using trapezoidal rule
                total_cost += 0.5 * (self._current_call_costs[i]
                                     + self._current_call_costs[i + 1]) * dt[i]
            self._running_cost_per_call[-1].append(total_cost)
            self._current_call_costs = np.zeros(self._n_time_points)
            self._call_counter = 0

    def _compute_states_inputs(self, x):
        """
        Extracts state and input matrices with shape n_states (or n_inputs)
         by n_times from the vector of optimization variables.
        Follows the implementation of the Optimal Control Solver library
        :param x: Optimization variables
        :return:
        """
        # States are appended to end of (input) the optimization variables
        states = x[-self._n_states * self._n_time_points:].reshape(
            self._n_states, -1)
        x = x[:-self._n_states * self._n_time_points]

        # Note: this only works if we are not using any basis functions for the
        # control input.
        inputs = x.reshape((-1, self._n_time_points))

        return states, inputs

    def set_constraints(self, constraints):
        self._linear_constraints = []
        self._lc_absolute_tolerance = []
        self._non_linear_constraints = []
        if constraints:
            for c in constraints:
                if isinstance(c, LinearConstraint):
                    self._linear_constraints.append(c)
                    self._lc_absolute_tolerance.append(
                        (np.abs(c.lb) * self._relative_tolerance,
                         np.abs(c.ub) * self._relative_tolerance))
                elif isinstance(c, NonlinearConstraint):
                    self._non_linear_constraints.append(c)
                else:
                    raise ValueError('Unknown constraint type: {}'.format(
                        type(c)
                    ))


class OptimalControlIterationResult:
    # Members from opt.OptimalControlResult
    success: bool
    nit: int
    cost: float
    message: str
    time: np.ndarray
    inputs: np.ndarray
    states: np.ndarray
    # New members
    iteration: int

    def __init__(self, iteration: int):
        self.iteration = iteration

    @classmethod
    def copy_original_result(cls, result: opt.OptimalControlResult
                             ) -> OptimalControlIterationResult:
        obj = OptimalControlIterationResult(result.nit - 1)
        for key, val in result.items():
            setattr(obj, key, val)
        return obj

    @classmethod
    def get_result_from_iteration(
            cls, cost_tracker: OCPCostTracker, ocp_solution_number: int,
            iteration: int
    ) -> OptimalControlIterationResult:
        obj = OptimalControlIterationResult(iteration)
        obj.success = True  # TODO: How to check?
        obj.time = cost_tracker.get_time_points()
        obj.states = cost_tracker.get_states_per_iteration()[
            ocp_solution_number][iteration]
        obj.inputs = cost_tracker.get_inputs_per_iteration()[
            ocp_solution_number][iteration]
        # feasibility check may be redundant sometimes
        obj.success = cost_tracker.check_feasibility(obj.states, obj.inputs)
        obj.message = (' ' if obj.success
                       else 'OCP solution {}, iteration {} is not '
                            'feasible'.format(ocp_solution_number, iteration))
        obj.nit = len(cost_tracker.get_states_per_iteration()[
                          ocp_solution_number])
        rc = cost_tracker.get_running_cost()[ocp_solution_number][iteration]
        tc = cost_tracker.get_terminal_cost()[ocp_solution_number][iteration]
        obj.cost = rc + tc
        return obj


def quadratic_cost(n_states: int, n_inputs: int, Q, R,
                   x0: Union[np.ndarray, float] = 0,
                   u0: Union[np.ndarray, float] = 0) -> Callable:
    """
    Create quadratic cost function
    Returns a quadratic cost function that can be used for an optimal
    control problem. The cost function is of the form
      cost = (x - x0)^T Q (x - x0) + (u - u0)^T R (u - u0)
    (Inspired from the Control library with modifications to keep track of
    costs over iterations.)

    :param n_states: number of states in the system
    :param n_inputs: number of inputs in the system
    :param Q: 2D array_like
        Weighting matrix for state cost. Dimensions must match system state.
    :param R: 2D array_like
        Weighting matrix for input cost. Dimensions must match system input.
    :param x0: 1D array
        Nominal value of the system state (for which cost should be zero).
    :param u0: 1D array
        Nominal value of the system input (for which cost should be zero).

    :return: callable
        Function that can be used to evaluate the cost at a given state and
        input.  The call signature of the function is cost_fun(x, u).

    """
    # Process the input arguments
    Q, R = _process_matrix_sizes(n_states, n_inputs, Q, R, make_zero=False)

    if Q is None:
        return lambda x, u: ((u - u0) @ R @ (u - u0)).item()
    if R is None:
        return lambda x, u: ((x - x0) @ Q @ (x - x0)).item()
    # Received both Q and R matrices
    return lambda x, u: (
            (x - x0) @ Q @ (x - x0) + (u - u0) @ R @ (u - u0)).item()


def quadratic_cost_gradient(n_states: int, n_inputs: int, n_times: int, Q, R,
                            Q_terminal=None, R_terminal=None,
                            x0: Union[np.ndarray, float] = 0,
                            u0: Union[np.ndarray, float] = 0) -> Callable:
    # Process the input arguments
    Q, R = _process_matrix_sizes(n_states, n_inputs, Q, R, make_zero=True)
    Q_terminal, R_terminal = _process_matrix_sizes(n_states, n_inputs,
                                                   Q_terminal, R_terminal,
                                                   make_zero=True)
    # TODO: it is faster to reorganize the 'big' matrices (only once) than to
    #  reshape vector x at every iteration
    q_list = [Q] * (n_times - 1) + [Q + Q_terminal]
    r_list = [R] * (n_times - 1) + [R + R_terminal]
    big_Q = block_diag(*q_list)
    big_R = block_diag(*r_list)
    big_x0 = np.tile(x0, n_times)
    big_u0 = np.tile(u0, n_times)

    def support(x):
        # Note 1: this only works for simple quadratic costs
        inputs = x[:n_inputs * n_times].reshape(n_inputs, -1
                                                ).transpose().flatten()
        states = x[n_inputs * n_times:].reshape(n_states, -1
                                                ).transpose().flatten()
        return 2 * np.hstack(
            ((big_R @ (inputs - big_u0)).reshape(-1, n_inputs
                                                 ).transpose().flatten(),
             (big_Q @ (states - big_x0)).reshape(-1, n_states
                                                 ).transpose().flatten())
        )

    return support

    # return lambda x, u: 2 * np.vstack((big_R @ (u-np.repeat(u0, n_times)),
    #                                    big_Q @ (x-np.repeat(x0, n_times))))


def _process_matrix_sizes(n_states: int, n_inputs: int, Q, R,
                          make_zero: bool = False):
    """
    If make_zero True and the matrix is None, creates a matrix of zeros.
    """
    if Q is not None:
        Q = np.atleast_2d(Q)
        if Q.size == 1:  # allow scalar weights
            Q = np.eye(n_states) * Q.item()
        elif Q.shape != (n_states, n_states):
            raise ValueError("Q matrix is the wrong shape")
    elif make_zero:
        Q = np.zeros(n_states)

    if R is not None:
        R = np.atleast_2d(R)
        if R.size == 1:  # allow scalar weights
            R = np.eye(n_inputs) * R.item()
        elif R.shape != (n_inputs, n_inputs):
            raise ValueError("R matrix is the wrong shape")
    elif make_zero:
        R = np.zeros(n_inputs)

    return Q, R

# def _input_only_cost(x, u, R, u0):
#     cost = ((u - u0) @ R @ (u - u0)).item()
#     return cost
#
#
# def _state_only_cost(x, u, Q, x0):
#     cost = ((x - x0) @ Q @ (x - x0)).item()
#     return cost
#
#
# def _full_cost(x, u, Q, R, x0, u0):
#     cost = ((x-x0) @ Q @ (x-x0) + (u-u0) @ R @ (u-u0)).item()
#     return cost
