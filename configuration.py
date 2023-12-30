from __future__ import annotations

import os
from typing import Mapping, Union

import numpy as np

# =================================== Types ================================== #
LCOrder = list[set[int]]
CoopOrder = list[int]
QuantizedState = tuple[int]
Strategy = tuple[LCOrder, CoopOrder]
StrategyMap = dict[QuantizedState,
                   dict[frozenset[int], tuple[Strategy, float]]]

# ================================ Constants ================================= #
LANE_WIDTH = 4  # [m]
TIME_HEADWAY = 1.0  # [s]
STANDSTILL_DISTANCE = 1.0  # [m]

UNIT_MAP = {'t': 's', 'x': 'm', 'y': 'm', 'theta': 'rad', 'v': 'm/s',
            'a': 'm/s^2', 'phi': 'rad', 'gap': 'm'}

COLORS = {'gray': (0.5, 0.5, 0.5), 'red': (1.0, 0.0, 0.0),
          'green': (0.0, 1.0, 0.0), 'blue': (0.0, 0.0, 1.0),
          'purple': (0.5, 0.0, 0.5), 'orange': (1.0, 0.5, 0.0),
          'dark_blue': (0, 0.0, 0.5)}

ACCEPTED_GRAPH_COSTS = {'time', 'accel'}

# TODO: make dependent on current machine
# DATA_FOLDER_PATH = 'C:\\Users\\fvall\\Documents\\Research\\data'
DATA_FOLDER_PATH = os.path.join('c:', os.sep, 'Users', 'fvall', 'Documents',
                                'Research', 'data')
SHARED_IMAGES_PATH = os.path.join('g:', os.sep, 'My Drive', 'PhD Research',
                                  'Lane Change', 'images_for_platoon_paper',
                                  'results')


# ====================== Simulation Parameters =============================== #
class Configuration:
    # Solver parameters
    solver_max_iter: int = 100
    discretization_step: float = 1.0  # [s]
    ftol: float = 1.0e-6  # [s]
    estimate_gradient: bool = True

    # Our optimal controller's parameters
    max_iter: int = 3
    time_horizon: float = 10.0  # [s]
    has_terminal_lateral_constraints: bool = False
    has_safety_lateral_constraint: bool = False
    initial_input_guess: Union[None, str, float] = None
    jumpstart_next_solver_call: bool = False
    has_initial_mode_guess: bool = False

    # Scenario parameters
    sim_time: float = 10.0  # [s]
    platoon_strategies = [0]
    increase_lc_time_headway: bool = False

    @staticmethod
    def set_solver_parameters(
            max_iter: int = None, discretization_step: float = None,
            ftol: float = None, estimate_gradient: bool = True
    ) -> None:
        """
        Sets the configurations of the underlying optimization tool.
        :param max_iter: Maximum number of iterations by the solver
        :param discretization_step: Fixed discretization step of the opc solver
        :param ftol: Scipy minimize parameter: "Precision goal for the value of
         f in the stopping criterion."
        :param estimate_gradient: Allow the optimizer to estimate the gradient
         or provide analytical cost gradient
        :return:
        """
        if max_iter:
            Configuration.solver_max_iter = max_iter
        if discretization_step:
            Configuration.discretization_step = discretization_step
        if ftol:
            Configuration.ftol = ftol
        Configuration.estimate_gradient = estimate_gradient

    @staticmethod
    def set_optimal_controller_parameters(
            max_iter: int = None, time_horizon: float = None,
            has_terminal_lateral_constraints: bool = False,
            has_lateral_safety_constraint: bool = False,
            initial_input_guess: Union[str, float, None] = None,
            jumpstart_next_solver_call: bool = False,
            has_initial_mode_guess: bool = False
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
        :param initial_input_guess: How to create the initial guess (a
         state/input tuple) for the solver. It only affects the first call to
         the solver if jumpstart_next_solver_call is True.
         None: provides no initial guess to the solver, i.e., the
         initial input/state tuple is all zeros.
         Numerical value: Assumes all controlled vehicles apply the given
         acceleration value during the entire time horizon
         'max'/'min'/'zero': Similar to a numerical value but uses the
         maximum/minimum/zero acceleration
         'random': Randomly samples between min and max accel for each time
         step
         'mode': Uses the state/input tuple that generates the initial mode
         guess (see has_initial_mode_guess)
        :param jumpstart_next_solver_call: Whether to use the solution of the
         previous call to the solver as starting point for the next call.
        :param has_initial_mode_guess: If True, runs the system with closed
         loop feedback controllers and uses the resulting mode sequence as
         initial guess for the optimal controller
        :return:
        """
        if not has_initial_mode_guess and initial_input_guess == 'mode':
            raise ValueError("has_initial_mode_guess must be set to True " 
                             "if initial_input_guess is set to 'mode'")
        if max_iter:
            Configuration.max_iter = max_iter
        if time_horizon:
            if time_horizon > Configuration.sim_time:
                raise ValueError("Opt controller's horizon time is larger than "
                                 "the simulation time")
            Configuration.time_horizon = time_horizon
        Configuration.has_terminal_lateral_constraints = (
            has_terminal_lateral_constraints)
        Configuration.has_safety_lateral_constraint = (
            has_lateral_safety_constraint)
        Configuration.initial_input_guess = initial_input_guess
        Configuration.jumpstart_next_solver_call = (
            jumpstart_next_solver_call)
        Configuration.has_initial_mode_guess = has_initial_mode_guess

    @staticmethod
    def set_scenario_parameters(
            sim_time: float = None,
            platoon_strategies: Union[list[int], int] = None,
            increase_lc_time_headway: bool = None):
        """
        Sets scenario parameters. If a certain parameter is not set, we
        keep its current value.
        :param sim_time: Simulation run time
        :param platoon_strategies: Defines the strategies that may be used by
         the optimal controller to generate the initial mode sequence guess.
         Accepted numerical values are 0: no platoon strategy, 1: synch,
         2: leader first, 3: last first, 4: leader first reverse. The only
         accepted string is 'all'.
        :param increase_lc_time_headway: If True, the safe time headway for
         lane changing is greater than the safe time headway for lane keeping.
         approach. Accepted values are 'time' and 'accel'
        :return:
        """
        if sim_time:
            Configuration.sim_time = sim_time
        if platoon_strategies is not None:
            if np.isscalar(platoon_strategies):
                platoon_strategies = [platoon_strategies]
            Configuration.platoon_strategies = platoon_strategies
        if increase_lc_time_headway is not None:
            Configuration.increase_lc_time_headway = increase_lc_time_headway


def get_lane_changing_time_headway() -> float:
    return TIME_HEADWAY + (0.2 if Configuration.increase_lc_time_headway
                           else 0.)
