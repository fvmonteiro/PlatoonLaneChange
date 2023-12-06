from __future__ import annotations

from collections.abc import Iterable, Mapping
import datetime
import time

import analysis
import configuration
import graph_tools
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
        [vehicle_models.three_state_vehicles.ThreeStateVehicleRearWheel],
        [vehicle_models.three_state_vehicles.ThreeStateVehicleRearWheel]
    ]
    # vehicles = [
    #     [vehicle_models.four_state_vehicles.SafeAccelOpenLoopLCVehicle,
    #      vehicle_models.four_state_vehicles.SafeAccelOpenLoopLCVehicle]
    # ]
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

    tf = configuration.Configuration.time_horizon
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


def run_all_scenarios_for_comparison(
        n_platoon: int, are_vehicles_cooperative: bool,
        graph_includes_fd: bool):
    run_scenarios_for_comparison(n_platoon, are_vehicles_cooperative,
                                 [4, 12, 13], [0, -5, 5], graph_includes_fd,
                                 False, save=True)


def run_scenarios_for_comparison(
        n_platoon: int, are_vehicles_cooperative: bool,
        strategies: Iterable[int], delta_v: Iterable[float],
        graph_includes_fd: bool, has_plots: bool, save: bool = False):
    try:
        vsg = graph_tools.VehicleStatesGraph.load_from_file(n_platoon,
                                                            graph_includes_fd)
    except FileNotFoundError:
        vsg = graph_tools.VehicleStatesGraph(n_platoon, graph_includes_fd)
        vsg.create_graph()
        vsg.save_to_file()

    scenario_manager = scenarios.LaneChangeScenarioManager()
    scenario_manager.set_plotting(has_plots)
    scenario_manager.set_lane_change_graph(vsg)
    v_base = 20.
    p_speed = v_base * 1.25
    v_ref = {'lo': v_base, 'p': p_speed,
             'fo': v_base, 'fd': v_base}
    print('Starting multiple runs')
    for dv in delta_v:
        print(f'  delta_v={dv}')
        v_ref['ld'] = v_base + dv
        scenario_manager.set_parameters(n_platoon, are_vehicles_cooperative,
                                        v_ref)
        for s in strategies:
            print(f'    strategy number={s}')
            scenario_manager.run_all_single_gap_cases(s)
    result = scenario_manager.get_results()
    print(result.groupby('strategy')[
              ['success', 'completion_time', 'accel_cost']].mean())
    result.to_csv('results_temp_name.csv', index=False)
    if save:
        scenario_manager.append_results_to_csv()


# def run_optimal_platoon_test(
#         n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
#         n_dest_ahead: int, n_dest_behind: int, is_acceleration_optimal: bool,
#         are_vehicles_cooperative: bool):
#     v_ref = dict()  # TODO: make param
#     delta_x = dict()  # TODO: make param
#     scenario = scenarios.LaneChangeScenario(n_platoon, are_vehicles_cooperative)
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


def create_graph(n_platoon: int, has_fd: bool):
    vsg = graph_tools.VehicleStatesGraph(n_platoon, has_fd)
    vsg.create_graph()
    print(f'#nodes={vsg.states_graph.number_of_nodes()}')
    vsg.save_to_file()


def main():

    n_platoon = 3
    n_orig_ahead, n_orig_behind = 1, 1
    n_dest_ahead, n_dest_behind = 1, 1

    configuration.Configuration.set_solver_parameters(
        max_iter=100, discretization_step=0.2,
        ftol=1.0e-3, estimate_gradient=True
    )
    configuration.Configuration.set_optimal_controller_parameters(
        max_iter=3, time_horizon=20.0,
        has_terminal_lateral_constraints=False,
        has_lateral_safety_constraint=False,
        initial_input_guess='mode',
        jumpstart_next_solver_call=True, has_initial_mode_guess=True
    )
    configuration.Configuration.set_scenario_parameters(
        increase_lc_time_headway=False
    )

    v_base = 20
    p_speed = v_base * 1.25
    v_ref = {'lo': v_base, 'ld': v_base, 'p': p_speed,
             'fo': v_base, 'fd': v_base}
    delta_x = {'lo': 0., 'ld': 0., 'p': 0., 'fd': 0.}

    is_acceleration_optimal = True
    are_vehicles_cooperative = False
    graph_includes_fd = False

    start_time = time.time()

    # create_graph(n_platoon, graph_includes_fd)
    # run_scenarios_for_comparison(n_platoon, are_vehicles_cooperative,
    #                              [4], [5], graph_includes_fd, True)
    # analysis.compare_to_approach()
    # for n in [2, 3]:
    #     run_all_scenarios_for_comparison(n, are_vehicles_cooperative,
    #                                      graph_includes_fd)

    vsg = graph_tools.VehicleStatesGraph.load_from_file(n_platoon,
                                                        graph_includes_fd)
    lcsm = scenarios.LaneChangeScenarioManager()
    lcsm.set_lane_change_graph(vsg)
    lcsm.set_parameters(n_platoon, are_vehicles_cooperative, v_ref, delta_x)
    lcsm.run_strategy_comparison_on_test_scenario(
        n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind,
        [4])

    # run_brute_force_strategy_test(
    #     n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead,
    #     n_dest_behind, are_vehicles_cooperative)

    # run_with_external_controller(
    #     n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind,
    #     is_acceleration_optimal, are_vehicles_cooperative
    # )

    # load_and_plot_latest_scenario()

    end_time = time.time()

    exec_time = datetime.timedelta(seconds=end_time - start_time)
    print("Execution time:", str(exec_time).split(".")[0])


if __name__ == "__main__":
    main()
