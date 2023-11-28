from __future__ import annotations

import datetime
import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx

import analysis
import configuration
import graph_tools
import platoon_lane_change_strategies as lc_strategy
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


def run_base_opc_scenario():
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
        are_vehicles_cooperative: bool):
    # Set-up
    tf = configuration.Configuration.time_horizon
    scenario = scenarios.LaneChangeWithExternalController(
        n_platoon, are_vehicles_cooperative)
    scenario.create_initial_state(n_orig_ahead, n_orig_behind, n_dest_ahead,
                                  n_dest_behind, is_acceleration_optimal)
    scenario.set_desired_final_states(configuration.Configuration.time_horizon)
    # Solve
    print("Calling OCP solver")
    scenario.solve()
    run_save_and_plot(scenario, tf)
    # analysis.plot_constrained_lane_change(
    #     scenario.ocp_simulation_to_dataframe(), 'p1')
    # analysis.compare_desired_and_actual_final_states(
    #     scenario.boundary_conditions_to_dataframe(), data)


def run_cbf_lc_scenario(n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
                        n_dest_ahead: int, n_dest_behind: int,
                        are_vehicles_cooperative: bool):
    for strategy_number in configuration.Configuration.platoon_strategies:
        scenario = scenarios.LaneChangeScenario(n_platoon,
                                                are_vehicles_cooperative)
        scenario.set_control_type('closed_loop', is_acceleration_optimal=False)
        scenario.create_test_scenario(
            n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind)
        scenario.vehicle_group.set_verbose(False)
        scenario.set_lane_change_strategy(strategy_number)
        tf = configuration.Configuration.time_horizon
        run_save_and_plot(scenario, tf)


def test_full_lane_scenario(n_platoon: int, are_vehicles_cooperative: bool):
    tf = configuration.Configuration.time_horizon
    for strategy_number in configuration.Configuration.platoon_strategies:
        for i in range(0, n_platoon + 2):
            scenario = scenarios.LaneChangeScenario(n_platoon,
                                                    are_vehicles_cooperative)
            scenario.set_control_type('closed_loop',
                                      is_acceleration_optimal=False)
            scenario.vehicle_group.set_verbose(False)
            scenario.create_single_gap_initial_state(i)
            scenario.set_lane_change_strategy(strategy_number)
            # run_save_and_plot(scenario, tf)


def run_brute_force_strategy_test(
        n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
        n_dest_ahead: int, n_dest_behind: int, are_vehicles_cooperative: bool):

    tf = configuration.Configuration.time_horizon
    scenario = scenarios.AllLaneChangeStrategies(
        n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind,
        are_vehicles_cooperative)
    run_save_and_plot(scenario, tf, 'Brute Force: Best Order')
    analysis.plot_cost_vs_ordering(scenario.costs,
                                   scenario.completion_times,
                                   scenario.named_strategies_positions)


def run_graph_based_scenario(
        n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
        n_dest_ahead: int, n_dest_behind: int, are_vehicles_cooperative: bool,
        load_from_file: bool = False):
    tf = configuration.Configuration.time_horizon

    if load_from_file:
        with open(f'graph_{n_platoon}_vehicles', 'rb') as f:
            vsg = pickle.load(f)
    else:
        vsg = graph_tools.VehicleStatesGraph(n_platoon)
        vsg.create_graph()
    for i in range(2, n_platoon + 2):
        scenario = scenarios.LaneChangeScenario(n_platoon,
                                                are_vehicles_cooperative)
        scenario.set_control_type('closed_loop', is_acceleration_optimal=False)
        scenario.create_single_gap_initial_state(i)
        scenario.set_lane_change_strategy(lc_strategy.GraphStrategy.get_id())
        scenario.set_lane_change_states_graph(vsg)
        scenario.vehicle_group.set_verbose(False)
        run_save_and_plot(scenario, tf)


def run_optimal_platoon_test(
        n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
        n_dest_ahead: int, n_dest_behind: int, is_acceleration_optimal: bool,
        are_vehicles_cooperative: bool):
    scenario = scenarios.LaneChangeScenario(n_platoon, are_vehicles_cooperative)
    scenario.set_control_type('optimal', is_acceleration_optimal)
    scenario.create_test_scenario(n_orig_ahead, n_orig_behind, n_dest_ahead,
                                  n_dest_behind)
    # scenario.create_optimal_control_test_scenario(
    #     n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind)
    tf = configuration.Configuration.time_horizon + 2
    run_save_and_plot(scenario, tf)


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


def create_graph(n_platoon: int):
    vsg = graph_tools.VehicleStatesGraph(n_platoon)
    vsg.create_graph()
    nx.draw_circular(vsg.states_graph)
    plt.show()
    # vsg.find_minimum_time_maneuver_order()

    vsg.save_to_file(f'graph_{n_platoon}_vehicles')
    # data = vsg.vehicle_group.to_dataframe()
    # analysis.plot_trajectory(data)
    # analysis.plot_platoon_lane_change(data)


def main():

    n_platoon = 2
    n_orig_ahead, n_orig_behind = 1, 0
    n_dest_ahead, n_dest_behind = 1, 1

    configuration.Configuration.set_solver_parameters(
        max_iter=100, discretization_step=0.2,
        ftol=1.0e-3, estimate_gradient=True
    )
    configuration.Configuration.set_controller_parameters(
        max_iter=3, time_horizon=8.0,
        has_terminal_lateral_constraints=False,
        has_lateral_safety_constraint=False,
        initial_input_guess='mode',
        jumpstart_next_solver_call=True, has_initial_mode_guess=True
    )
    base_speed = 20.
    p_speed = base_speed * (1.25 if n_orig_ahead > 0 else 1.0)
    configuration.Configuration.set_scenario_parameters(
        v_ref={'lo': base_speed, 'ld': base_speed / 1.25, 'p': p_speed,
               'fo': base_speed, 'fd': base_speed},
        delta_x={'lo': 0., 'ld': 0., 'p': 0., 'fd': 0.},
        platoon_strategies=[13], increase_lc_time_headway=False
    )
    is_acceleration_optimal = True
    are_vehicles_cooperative = False

    start_time = time.time()
    # run_cbf_lc_scenario(n_platoon, n_orig_ahead, n_orig_behind,
    #                     n_dest_ahead, n_dest_behind,
    #                     are_vehicles_cooperative)
    test_full_lane_scenario(n_platoon, are_vehicles_cooperative)
    # run_brute_force_strategy_test(
    #     n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead,
    #     n_dest_behind, are_vehicles_cooperative)

    # run_with_external_controller(
    #     n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind,
    #     is_acceleration_optimal, are_vehicles_cooperative
    # )
    # run_platoon_test(n_platoon, n_orig_ahead, n_orig_behind,
    #                  n_dest_ahead, n_dest_behind,
    #                  is_acceleration_optimal,
    #                  are_vehicles_cooperative
    #                  )
    # load_and_plot_latest_scenario()

    # run_graph_based_scenario(
    #     n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind,
    #     are_vehicles_cooperative, load_from_file=True)

    end_time = time.time()

    exec_time = datetime.timedelta(seconds=end_time - start_time)
    print("Execution time:", str(exec_time).split(".")[0])


if __name__ == "__main__":
    main()
