from typing import Union, Dict

INCREASE_LC_TIME_HEADWAY = False
LANE_WIDTH = 4  # [m]
# Lane Keeping [s]
LK_TIME_HEADWAY = 1.0
# Lane Changing [s]
LC_TIME_HEADWAY = LK_TIME_HEADWAY + (0.2 if INCREASE_LC_TIME_HEADWAY else 0)
STANDSTILL_DISTANCE = 1.0  # [m]

UNIT_MAP = {'t': 's', 'x': 'm', 'y': 'm', 'theta': 'rad', 'v': 'm/s',
            'a': 'm/s^2', 'phi': 'rad', 'gap': 'm'}

COLORS = {'gray': (0.5, 0.5, 0.5), 'red': (1.0, 0.0, 0.0),
          'green': (0.0, 1.0, 0.0), 'blue': (0.0, 0.0, 1.0),
          'purple': (0.5, 0.0, 0.5), 'orange': (1.0, 0.5, 0.0),
          'dark_blue': (0, 0.0, 0.5)}


class Configuration:
    # Solver parameters
    solver_max_iter: int = 100
    discretization_step: float = 1.0  # [s]
    ftol: float = 1.0e-6  # [s]
    estimate_gradient: bool = True

    # Our controller's parameters
    max_iter: int = 3
    time_horizon: float = 10.0  # [s]
    has_terminal_lateral_constraints: bool = False
    has_safety_lateral_constraint: bool = False
    provide_initial_guess: bool = False,
    initial_acceleration_guess: Union[str, float] = 0.0
    jumpstart_next_solver_call: bool = False

    # Scenario parameters
    v_ref = {'lo': 10., 'ld': 10., 'p': 10.,
             'fo': 10., 'fd': 10.}
    delta_x = {'lo': 0.0, 'ld': 0.0, 'p': 0.0, 'fo': 0.0, 'fd': 0.0}
    platoon_strategy = 1  # 1: synch, 2: leader first, 3: last first,
    # 4: leader first reverse

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
         initial input guess and passes an (inputs, states) tuple as initial
         guess to the solver. It only affects the first call to the solver if
         jumpstart_next_solver_call is True.
        :param initial_acceleration_guess: Initial guess of the optimal
         acceleration. We can provide the exact value or one of the strings
         'zero', 'max' (max acceleration), 'min' (max brake). The same value
         is used for the entire time horizon. It only affects the first call to
         the solver if jumpstart_next_solver_call is True.
        :param jumpstart_next_solver_call: Whether to use the solution of the
         previous call to the solver as starting point for the next call.
        :return:
        """
        if max_iter:
            Configuration.max_iter = max_iter
        if time_horizon:
            Configuration.time_horizon = time_horizon

        Configuration.has_terminal_lateral_constraints = (
            has_terminal_lateral_constraints)
        Configuration.has_safety_lateral_constraint = (
            has_lateral_safety_constraint)
        Configuration.provide_initial_guess = (
            provide_initial_guess)
        Configuration.initial_acceleration_guess = (
            initial_acceleration_guess)
        Configuration.jumpstart_next_solver_call = (
            jumpstart_next_solver_call)

    @staticmethod
    def set_scenario_parameters(
            v_ref: Dict[str, float] = None, delta_x: Dict[str, float] = None,
            platoon_strategy: int = 1):
        """

        :param v_ref: Free-flow speed for all vehicles. The accepted keys are:
         lo (origin leaders), ld (destination leaders), p (platoon or ego
         vehicles), fo (origin followers), fd (destination followers)
        :param delta_x: Deviation from equilibrium position. The accepted keys
         are: lo (origin leader), ld (destination leader), p (intra platoon),
         fo (origin follower), fd (destination follower)
        :param platoon_strategy: 1: synch, 2: leader first, 3: last first,
         4: leader first reverse
        :return:
        """
        if v_ref:
            for key, value in v_ref.items():
                Configuration.v_ref[key] = value
        if delta_x:
            for key, value in delta_x.items():
                Configuration.delta_x[key] = value
        Configuration.platoon_strategy = platoon_strategy
