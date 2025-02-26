from __future__ import annotations

import os
import warnings
from typing import Mapping, Union

import numpy as np

# =================================== Types ================================== #
LCOrder = list[Union[set[int], frozenset[int]]]
CoopOrder = list[int]
QuantizedState = tuple[int, ...]
Query = tuple[QuantizedState, set[int]]
Strategy = tuple[LCOrder, CoopOrder]
StrategyMap = dict[QuantizedState,
                   dict[frozenset[int], tuple[Strategy, float]]]

# ================================ Constants ================================= #
LANE_WIDTH = 4  # [m]
KMH_TO_MS = 1/3.6
SAFE_TIME_HEADWAY = 2.0  # [s]
SAFE_CONNECTED_TIME_HEADWAY = 1.0  # [s]
STANDSTILL_DISTANCE = 1.0  # [m]
TIME_HEADWAY_MARGIN = 0.1  # [s]
MAX_DISTANCE = 300.  # [m] for when we need to create virtual far away vehicles
FREE_FLOW_SPEED = 110  # km/h
# TODO: read these from a file
DELTA_X = 9  # quantization interval [m]
DELTA_V = 2  # quantization interval [m]
DELTA_Y = LANE_WIDTH  # quantization interval [m]
MAX_COMPUTATION_TIME = 1.e10  # when exploring the graph

UNIT_MAP = {'t': 's', 'x': 'm', 'y': 'm', 'theta': 'rad', 'v': 'm/s',
            'a': 'm/s^2', 'phi': 'rad', 'gap': 'm'}

COLORS = {'gray': (0.5, 0.5, 0.5), 'red': (1.0, 0.0, 0.0),
          'green': (0.0, 1.0, 0.0), 'blue': (0.0, 0.0, 1.0),
          'purple': (0.5, 0.0, 0.5), 'orange': (1.0, 0.5, 0.0),
          'dark_blue': (0, 0.0, 0.5)}

ACCEPTED_GRAPH_COSTS = {'time', 'accel'}

DATA_FOLDER_PATH = os.path.join('c:', os.sep, 'Users', 'fvall', 'Documents',
                                'Research', 'data')
SHARED_IMAGES_PATH = os.path.join('g:', os.sep, 'My Drive', 'PhD Research',
                                  'Lane Change', 'images_for_platoon_paper',
                                  'results')
SHARED_DATA_PATH = os.path.join('g:', os.sep, 'My Drive', 'PhD Research',
                                'Lane Change', 'data_exchange')


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
    time_step: float = 1.0e-2  # [s]
    platoon_strategies = [0]
    increase_lc_time_headway: bool = False
    is_warm_up: bool = False

    # Graph exploration parameters
    should_use_bfs = False
    epsilon: float = 1.0
    max_computation_time: float = 1e5

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
            sim_time: float = None, time_step: float = None,
            platoon_strategies: Union[list[int], int] = None,
            increase_lc_time_headway: bool = None,
            is_warm_up: bool = None):
        """
        Sets scenario parameters. If a certain parameter is not set, we
        keep its current value.
        :param sim_time: Simulation run time
        :param time_step: Time discretization
        :param platoon_strategies: Defines the strategies that may be used by
         the optimal controller to generate the initial mode sequence guess.
         Accepted numerical values are 0: no platoon strategy, 1: synch,
         2: leader first, 3: last first, 4: leader first reverse. The only
         accepted string is 'all'.
        :param increase_lc_time_headway: If True, the safe time headway for
         lane changing is greater than the safe time headway for lane keeping.
         approach. Accepted values are 'time' and 'accel'
        :param is_warm_up: when true, simulations stop as soon as the
         platoon finds a suitable lc gap, and stores the query
        :return:
        """
        if sim_time:
            Configuration.sim_time = sim_time
        if time_step:
            Configuration.time_step = time_step
        if platoon_strategies is not None:
            if np.isscalar(platoon_strategies):
                platoon_strategies = [platoon_strategies]
            Configuration.platoon_strategies = platoon_strategies
        if increase_lc_time_headway is not None:
            Configuration.increase_lc_time_headway = increase_lc_time_headway
        if is_warm_up is not None:
            Configuration.is_warm_up = is_warm_up

    @staticmethod
    def set_graph_exploration_parameters(
            should_use_bfs: bool = None, epsilon: float = None,
            max_computation_time: float = None):
        # TODO: there could be more checks
        if should_use_bfs is not None and should_use_bfs:
            Configuration.should_use_bfs = should_use_bfs
            if epsilon is not None:
                warnings.warn("Parameter epsilon only makes sense if "
                              "should_use_bfs is False")
            if max_computation_time is not None:
                warnings.warn("Parameter max_computation_time only makes sense "
                              "if should_use_bfs is False")
        elif epsilon is not None:
            Configuration.should_use_bfs = False
            Configuration.epsilon = epsilon

        if max_computation_time is not None:
            Configuration.max_computation_time = max_computation_time


# TODO: remove default value. Each caller should know. Or maybe remove this
#  function from here and figure out better design
def get_lane_changing_time_headway(are_connected: bool = False) -> float:
    h = SAFE_CONNECTED_TIME_HEADWAY if are_connected else SAFE_TIME_HEADWAY
    return h + (0.2 if Configuration.increase_lc_time_headway else 0.)
