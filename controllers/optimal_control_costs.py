from functools import partial
from typing import Callable, List, Union

import numpy as np


class OCPCostTracker:
    """
    Makes it possible to keep track of the computes costs during the optimal
    control problem solver iterations.
    """
    def __init__(self, time_points: np.ndarray, n_states: int,
                 running_cost: Callable, terminal_cost: Callable = None):
        self._time_points = time_points
        self._n_states = n_states
        self._running_cost_fun = running_cost
        self._terminal_cost_fun = terminal_cost

        self._n_time_points: int = len(time_points)
        self._current_call_costs: np.ndarray = np.zeros(self._n_time_points)
        self._call_counter: int = 0
        self._running_cost_per_call: List[float] = []
        self._terminal_cost_per_call: List[float] = []
        self._running_cost_per_iteration: List[float] = []
        self._terminal_cost_per_iteration: List[float] = []
        self._states_per_iteration: List[np.ndarray] = []
        self._inputs_per_iteration: List[np.ndarray] = []

    def get_running_cost(self):
        return self._running_cost_per_iteration

    def get_terminal_cost(self):
        return self._terminal_cost_per_iteration

    def running_cost(self, states, inputs):
        cost = self._running_cost_fun(states, inputs)
        self._save_running_cost_per_call(cost)
        return cost

    def terminal_cost(self, states, inputs):
        cost = self._terminal_cost_fun(states, inputs)
        self._terminal_cost_per_call.append(cost)
        return cost

    def costs_to_string(self):
        output = '{:10s}\t{:10s}\t{:10s}\t{:10s}\n'.format(
            'Iteration', 'Running', 'Terminal', 'Total')
        for i in range(len(self._running_cost_per_iteration)):
            output += '{:<10d}\t{:<10.6g}\t{:<10.6g}\t{:<10.6g}\n'.format(
                i, self._running_cost_per_iteration[i],
                self._terminal_cost_per_iteration[i],
                self._running_cost_per_iteration[i] +
                self._terminal_cost_per_iteration[i])
        return output

    def get_best_iteration(self):
        """
        Gets the states and inputs from the iteration that had the minimum cost
        :return:
        """
        best_idx = np.argmin(np.array(self._running_cost_per_iteration)
                             + np.array(self._terminal_cost_per_iteration))
        min_cost = (self._running_cost_per_iteration[best_idx]
                    + self._terminal_cost_per_iteration[best_idx])
        print('Best iteration idx:', best_idx)
        return {'cost': min_cost,
                'states': self._states_per_iteration[best_idx],
                'inputs': self._inputs_per_iteration[best_idx]}
    # TODO: return object with members: success, time, states (n x time),
    #  inputs (n x time), message, nit, cost

    def callback(self, x):
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

        terminal_cost = 0
        if self._terminal_cost_fun is not None:
            terminal_cost = self._terminal_cost_fun(states[:, -1],
                                                    inputs[:, -1])

        # Store everyone
        self._running_cost_per_iteration.append(running_cost)
        self._terminal_cost_per_iteration.append(terminal_cost)
        self._states_per_iteration.append(states)
        self._inputs_per_iteration.append(inputs)

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
            self._running_cost_per_call.append(total_cost)
            self._current_call_costs = np.zeros(self._n_time_points)
            self._call_counter = 0

    def _compute_states_inputs(self, x):
        """
        Extracts state and input matrices with shape n_states (n_inputs) x n_times
        from the vector of optimization variables.
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
    if Q is not None:
        Q = np.atleast_2d(Q)
        if Q.size == 1:  # allow scalar weights
            Q = np.eye(n_states) * Q.item()
        elif Q.shape != (n_states, n_states):
            raise ValueError("Q matrix is the wrong shape")

    if R is not None:
        R = np.atleast_2d(R)
        if R.size == 1:  # allow scalar weights
            R = np.eye(n_inputs) * R.item()
        elif R.shape != (n_inputs, n_inputs):
            raise ValueError("R matrix is the wrong shape")

    if Q is None:
        return partial(_input_only_cost, R=R, u0=u0)

    if R is None:
        return partial(_state_only_cost, Q=Q, x0=x0)

    # Received both Q and R matrices
    return partial(_full_cost, Q=Q, R=R, x0=x0, u0=u0)


def _input_only_cost(x, u, R, u0):
    cost = ((u - u0) @ R @ (u - u0)).item()
    return cost


def _state_only_cost(x, u, Q, x0):
    cost = ((x - x0) @ Q @ (x - x0)).item()
    return cost


def _full_cost(x, u, Q, R, x0, u0):
    cost = ((x-x0) @ Q @ (x-x0) + (u-u0) @ R @ (u-u0)).item()
    return cost