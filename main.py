from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping, Sequence
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np

import analysis
import configuration
import graph_tools
import post_processing
import scenarios
import vehicle_models

trajectory_file_name = 'trajectory_data.pickle'  # temp
cost_file_name = 'cost_data.pickle'


def run_no_lc_scenario():
    tf = 10
    scenario = scenarios.VehicleFollowingScenario(2)
    scenario.create_initial_state()
    scenario.run(tf)
    analysis.plot_vehicle_following(scenario.response_to_dataframe())


def run_fast_lane_change():
    tf = 5
    scenario = scenarios.FastLaneChange()
    scenario.run(tf)
    data = scenario.response_to_dataframe()
    analysis.plot_lane_change(data)


def run_base_ocp_scenario():
    v_ff = 10
    tf = 10

    scenario = scenarios.ExampleScenarioExternal()
    vehicles = [
        [vehicle_models.three_state_vehicles.ThreeStateVehicleRearWheel()],
        [vehicle_models.three_state_vehicles.ThreeStateVehicleRearWheel()]
    ]
    scenario.create_vehicle_group(vehicles)
    scenario.set_free_flow_speeds(v_ff)
    scenario.create_initial_state()
    scenario.set_desired_final_states(tf)
    scenario.solve()
    scenario.run(tf)
    scenario.save_response_data(trajectory_file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_lane_change(data)


def run_with_external_controller(
        n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
        n_dest_ahead: int, n_dest_behind: int, is_acceleration_optimal: bool,
        are_vehicles_cooperative: bool, v_ref: Mapping[str, float],
        delta_x: Mapping[str, float]):
    # Set-up
    tf = configuration.Configuration.time_horizon
    scenario = scenarios.LaneChangeWithExternalController(
        n_platoon, are_vehicles_cooperative)
    scenario.create_initial_state(
        n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind, v_ref,
        delta_x, is_acceleration_optimal)
    scenario.set_desired_final_states(configuration.Configuration.time_horizon)
    # Solve
    print("Calling OCP solver")
    scenario.solve()
    run_save_and_plot(scenario, tf)
    # analysis.plot_constrained_lane_change(
    #     scenario.ocp_simulation_to_dataframe(), 'p1')
    # analysis.compare_desired_and_actual_final_states(
    #     scenario.boundary_conditions_to_dataframe(), data)


def run_brute_force_strategy_test(
        n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
        n_dest_ahead: int, n_dest_behind: int, are_vehicles_cooperative: bool,
        v_ref: Mapping[str, float], delta_x: Mapping[str, float]):

    tf = configuration.Configuration.sim_time
    scenario = scenarios.AllLaneChangeStrategies(
        n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind,
        v_ref, delta_x, are_vehicles_cooperative)
    run_save_and_plot(scenario, tf, 'Brute Force: Best Order')
    analysis.plot_cost_vs_ordering(scenario.costs,
                                   scenario.completion_times,
                                   scenario.named_strategies_positions)


def run_save_and_plot(scenario: scenarios.SimulationScenario, tf: float,
                      scenario_name: str = None):
    scenario.run(tf)

    scenario.save_response_data(trajectory_file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_trajectory(data, scenario_name)
    if scenario.get_n_platoon() < 1:
        analysis.plot_constrained_lane_change(data, 'p1')  # TODO: ego or p1
    else:
        analysis.plot_platoon_lane_change(data)

    try:
        scenario.save_cost_data(cost_file_name)
        running_cost, terminal_cost = scenario.get_opc_cost_history()
        analysis.plot_costs_vs_iteration(running_cost, terminal_cost)
    except AttributeError:
        pass


def run_closed_loop_test(
        n_platoon: int, are_vehicles_cooperative: bool,
        v_orig: float, v_ff_platoon: float, v_dest: float,
        delta_x: Mapping[str, float],
        strategies: Iterable[int], plot_results: bool = True):
    scenario_manager = scenarios.LaneChangeScenarioManager()
    v_ref = {'orig': v_orig, 'platoon': v_ff_platoon, 'dest': v_dest}
    scenario_manager.set_parameters(
        n_platoon, are_vehicles_cooperative, v_ref, delta_x)
    n_orig_ahead = 1
    n_orig_behind = 1
    n_dest_ahead = 1
    n_dest_behind = 1
    scenario_manager.set_plotting(plot_results)
    scenario_manager.run_strategy_comparison_on_test_scenario(
        n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind, strategies
    )


def run_all_scenarios_for_comparison(
        n_platoon: int, v_orig: float, v_ff_platoon: float,
        are_vehicles_cooperative: bool):
    run_scenarios_for_comparison(
        n_platoon, v_orig, v_ff_platoon, are_vehicles_cooperative,
        [5, 6, 12, 13], [0, -5, 5], has_plots=False, save=True)


def run_scenarios_for_comparison(
        n_platoon: int, v_orig: float, v_ff_platoon: float,
        are_vehicles_cooperative: bool, strategies: Iterable[int],
        v_dest: Sequence[float], gap_positions: Iterable[int] = None,
        has_plots: bool = True, save: bool = False):

    scenario_manager = scenarios.LaneChangeScenarioManager()
    scenario_manager.set_plotting(has_plots)
    v_ref = {'orig': v_orig, 'platoon': v_ff_platoon}
    print('Starting multiple runs')
    for vd in v_dest:
        print(f'  v_dest={vd * 3.6}')
        v_ref['dest'] = vd
        scenario_manager.set_parameters(n_platoon, are_vehicles_cooperative,
                                        v_ref)
        for s in strategies:
            print(f'    strategy number={s}')
            if gap_positions is None:
                scenario_manager.run_all_single_gap_cases(s)
            else:
                for gp in gap_positions:
                    scenario_manager.run_single_gap_scenario(s, gp)
    result = scenario_manager.get_results()
    print(result.groupby('strategy')[
              ['success', 'completion_time', 'accel_cost', 'decision_time']
          ].mean())
    result.to_csv('results_temp_name.csv', index=False)
    if save:
        scenario_manager.append_results_to_csv()


# def run_optimal_platoon_test(
#         n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
#         n_dest_ahead: int, n_dest_behind: int, is_acceleration_optimal: bool,
#         are_vehicles_cooperative: bool):
#     v_ref = dict()  # TODO: make param
#     delta_x = dict()  # TODO: make param
#     scenario = scenarios.LaneChangeScenario(n_platoon,
#     are_vehicles_cooperative)
#     scenario.set_control_type('optimal', is_acceleration_optimal)
#     scenario.create_test_scenario(n_dest_ahead, n_dest_behind, n_orig_ahead,
#                                   n_orig_behind, v_ref, delta_x)
#     # scenario.create_optimal_control_test_scenario(
#     #     n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind)
#     tf = configuration.Configuration.time_horizon + 2
#     run_save_and_plot(scenario, tf)


def load_and_plot_latest_scenario():
    trajectory_data = analysis.load_latest_simulated_scenario(
        trajectory_file_name)
    analysis.plot_trajectory(trajectory_data)
    analysis.plot_constrained_lane_change(trajectory_data, 'p1')
    analysis.plot_platoon_lane_change(trajectory_data)

    try:
        cost_data = analysis.load_latest_simulated_scenario(cost_file_name)
        analysis.plot_costs_vs_iteration(cost_data[0], cost_data[1],
                                         plot_separately=False)
    except EOFError:
        # no cost data
        pass


def create_graph(n_platoon: int, has_fd: bool, vel_orig_lane: Sequence[float],
                 vel_ff_platoon: float, v_dest_lane: Sequence[float],
                 max_dist: float, mode: str = "as"):
    graph_creator = graph_tools.GraphCreator(n_platoon, has_fd)
    graph_creator.create_graph(vel_orig_lane, vel_ff_platoon, v_dest_lane,
                               max_dist, mode)
    graph_creator.save_vehicle_state_graph_to_file()
    graph_creator.save_quantization_parameters_to_file()
    create_and_save_strategy_maps(graph_creator)


def create_and_save_strategy_maps(vsg: graph_tools.GraphCreator):
    for cost_name in ['time', 'accel']:
        vsg.save_minimum_cost_strategies_to_json(cost_name)


def filter_test():
    import controllers.longitudinal_controller as long_ctrl
    tf = 40
    dt = configuration.Configuration.time_step
    v0 = 0.
    max_vel = 20.
    vel_ctrl = long_ctrl.VelocityController(-4, 2)
    vel_ctrl.reset(v0)
    sim_time = []
    accel = []
    v_ego = []
    v_ref = []
    v_ff = []
    for i in range(int(tf/dt)):
        if i == 0:
            sim_time.append(0)
            v_ego.append(v0)
        else:
            sim_time.append(sim_time[-1] + dt)
            v_ego.append(v_ego[-1] + accel[-1] * dt)

        if np.isclose(sim_time[-1], 5):
            vel_ctrl.reset(v_ego[-1])
            v_ref.append(25)
        else:
            v_ref.append(max_vel)
        v_ff.append(vel_ctrl.get_v_ff())
        accel.append(vel_ctrl.compute_input(v_ego[-1], v_ref[-1]))

    fig, ax = plt.subplots(2)
    ax[0].grid(visible=True)
    ax[0].plot(sim_time, v_ref, label='v_ref')
    ax[0].plot(sim_time, v_ff, label='v_ff')
    ax[0].plot(sim_time, v_ego, label='v_ego')
    ax[0].legend()
    ax[1].grid(visible=True)
    ax[1].plot(sim_time, accel, label='accel')
    ax[1].legend()
    fig.show()


def test_1():
    """
    Loads initial nodes which raised problems and rebuilds trees starting
    from them
    :return:
    """
    n_platoon = 2
    with open("unsolved_root_nodes.pickle", "rb") as f:
        unsolved_initial_states = pickle.load(f)

    graph_creator = graph_tools.GraphCreator(2, False)
    graph_creator.load_a_graph()
    # graph_creator._initial_states |= set(unsolved_initial_states)
    visited_nodes = set()

    free_flow_speeds = np.array([70.] + [110.]*n_platoon + [0.]) / 3.6
    possible_vel_dest = np.array([50, 70, 90]) / 3.6
    n = len(unsolved_initial_states)
    for i in range(n):
        x0 = unsolved_initial_states[i]
        free_flow_speeds[-1] = (
            graph_creator.state_quantizer.dequantize_free_flow_velocities(
                x0[-1], possible_vel_dest))
        print(f"{i+1}/{n}")
        graph_creator.add_all_from_initial_state_to_graph(
            x0, visited_nodes, free_flow_speeds, "ao")
        # tracker = graph_tools.PlatoonLCTracker(n_platoon)
        # tracker.move_vehicles([0], -1)
        # root = (tracker, x0)
        # vsg._explore_until_maneuver_completion(root, visited_nodes,
        #                                        free_flow_speeds)
    if n > 0:
        graph_creator.save_vehicle_state_graph_to_file()
        graph_creator.save_quantization_parameters_to_file()
        create_and_save_strategy_maps(graph_creator)


def main():

    n_platoon = 2
    n_orig_ahead, n_orig_behind = 1, 1
    n_dest_ahead, n_dest_behind = 1, 1

    sim_time = 15.0 * n_platoon
    # time_horizon = sim_time - 2
    # configuration.Configuration.set_solver_parameters(
    #     max_iter=100, discretization_step=0.2,
    #     ftol=1.0e-3, estimate_gradient=True
    # )

    # configuration.Configuration.set_optimal_controller_parameters(
    #     max_iter=3, time_horizon=time_horizon,
    #     has_terminal_lateral_constraints=False,
    #     has_lateral_safety_constraint=False,
    #     initial_input_guess='mode',
    #     jumpstart_next_solver_call=True, has_initial_mode_guess=True
    # )

    configuration.Configuration.set_scenario_parameters(
        sim_time=sim_time, increase_lc_time_headway=False
    )

    # TODO: test values only
    v_orig = [70/3.6]
    v_ff_platoon = 110/3.6  # make 110
    v_dest = np.array([50, 70, 90])/3.6  # make 50, 70, 90
    max_dist = v_ff_platoon * configuration.SAFE_TIME_HEADWAY * 1.5

    is_acceleration_optimal = True
    are_vehicles_cooperative = False
    graph_includes_fd = False

    start_time = time.time()

    print("===== CREATING SMALL-SCALE GRAPHS ONLY ======")
    graph_t0 = time.time()
    for vd in v_dest:
        create_graph(n_platoon, graph_includes_fd, v_orig, v_ff_platoon,
                     [vd], max_dist, "ao")
    print(f"Time to create graph: {time.time() - graph_t0}")

    # delta_x = {'ld': -200., 'lo': 0., 'fd': 0.}
    # run_closed_loop_test(n_platoon, are_vehicles_cooperative,
    #                      v_orig[0], v_ff_platoon, v_dest[0], delta_x,
    #                      [6], plot_results=True)

    # run_scenarios_for_comparison(
    #     n_platoon, v_orig[0], v_ff_platoon, are_vehicles_cooperative,
    #     [5], v_dest, gap_positions=None, has_plots=False, save=False
    # )

    # for n_platoon in [2, 3, 4]:
    #     print(f'############ N={n_platoon} ############')
    #     graph_t0 = time.time()
    #     create_graph(n_platoon, graph_includes_fd, v_orig, v_ff_platoon,
    #                  v_dest - v_orig[0])
    #     print(f'Time to create graph: {time.time() - graph_t0}')
    #     configuration.Configuration.set_scenario_parameters(
    #         sim_time=20.0 * n_platoon
    #     )
    #     sim_t0 = time.time()
    #     run_all_scenarios_for_comparison(n_platoon, v_orig, v_ff_platoon,
    #                                      are_vehicles_cooperative)
    #     print(f'Time simulated: {time.time() - sim_t0}')
    # analysis.compare_approaches(save_fig=False)
    # analysis.compare_graph_to_best_heuristic(save_fig=True)

    # post_processing.export_strategy_maps_to_cloud()

    end_time = time.time()

    exec_time = datetime.timedelta(seconds=end_time - start_time)
    print("Execution time:", str(exec_time).split(".")[0])


if __name__ == "__main__":
    main()
