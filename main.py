from __future__ import annotations

import random
from collections.abc import Sequence
import datetime
import time

import configuration
import analysis
from platoon_functionalities import graph_tools, platoon_lane_change_strategies
from platoon_functionalities import traffic_state_graph
import post_processing
import scenarios
from vissim_handler import vissim_interface
from vissim_handler import file_handling


def run_optimal_control_scenario():
    configuration.Configuration.set_solver_parameters(
        max_iter=100, discretization_step=0.2,
        ftol=1.0e-3, estimate_gradient=True
    )
    configuration.Configuration.set_optimal_controller_parameters(
        max_iter=3, time_horizon=configuration.Configuration.sim_time - 2,
        has_terminal_lateral_constraints=False,
        has_lateral_safety_constraint=False,
        initial_input_guess='mode',
        jumpstart_next_solver_call=True, has_initial_mode_guess=True
    )
    scenarios.run_base_ocp_scenario()


def load_and_plot_latest_scenario():
    trajectory_file_name = 'data/trajectory_data.pickle'  # temp
    cost_file_name = 'data/cost_data.pickle'
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
    graph_creator.save_minimum_cost_strategies_to_json()


def run_q_learning_unit_tests():
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover('platoon_functionalities',
                            pattern='q_learning_test.py')
    runner = unittest.TextTestRunner()
    runner.run(suite)


def q_learning_tests():
    from platoon_functionalities import q_learning
    grid_size = (5, 5)
    # obstacles = {(i, i) for i in range(1, min(grid_size) - 1)}
    # obstacles.add((9, 8))
    obstacles = {(1, i) for i in range(1, grid_size[1])}
    obstacles.update({(3, i) for i in range(grid_size[1] - 1)})
    goal_nodes = [(grid_size[0] - 1, 0),
                  (0, grid_size[1] - 1)]
    q_learning_agent = q_learning.QLearningAgent(
        grid_size, goal_nodes, obstacles, alpha=1.)
    q_learning_agent.train((2, 2), 1000)


def test_map_2d_example():
    from platoon_functionalities import map_2d_example
    grid_size = (11, 11)
    # obstacles = {(i, i) for i in range(1, min(grid_size) - 1)}
    # obstacles.add((9, 8))
    start_state = (grid_size[0] // 2, grid_size[1] // 2)
    obstacles = {(start_state[0]-1, i) for i in range(1, grid_size[1])}
    obstacles.update({(start_state[0]+1, i) for i in range(grid_size[1] - 1)})
    goal_states = [(grid_size[0] - 1, 0),
                   (0, grid_size[1] - 1)]
    my_map = map_2d_example.ProblemMap(grid_size, obstacles, start_state,
                                       goal_states)
    print(my_map.to_string())
    map_2d_example.train(start_state, 100, my_map, epsilon=0.8,
                         verbose_level=1)


def run_traffic_graph_explorer():
    platoon_sizes = [i for i in range(5, 6)]
    epsilon_range = [i for i in range(1, 11, 3)]
    all_cost_types = ["time", "accel"]
    # configuration.Configuration.set_graph_exploration_parameters(should_use_bfs=True)
    for n_platoon in platoon_sizes:
        for epsilon_idx in epsilon_range:
            for cost_type in all_cost_types:
                random.seed(1)
                epsilon = epsilon_idx/10
                print("=" * 79 + f"\nCost type: {cost_type}, eps={epsilon}")
                try:
                    traffic_state_graph.solve_queries_from_simulations(
                        "python", "", n_platoon, cost_type=cost_type,
                        epsilon=epsilon, verbose_level=0)
                except MemoryError:
                    print("MEMORY ERROR! but following on to next")
                    continue


def test_analysis():
    # analyzer = analysis.ResultAnalyzer(save_figs=False)
    # analyzer.get_python_results_for_paper()
    # analyzer.print_average_number_of_maneuver_steps()
    # analyzer.compare_approaches()
    # analyzer.compare_to_approach("time")

    n_platoon = 5

    # analysis.compare_bfs_and_dfs(n_platoon, "time")
    ra = analysis.ResultAnalyzer(is_bfs=False)
    ra.plot_cost_vs_max_computation_time()

    # for i in range(1, 11):
    #     analysis.plot_average_cost_vs_computation_time(5, "time", i/10)

    # scenarios.run_with_varying_max_computation_time(1.0)
    # analysis.plot_several_cost_vs_computation_time(
    #     [5], ["time", "accel"], [i/10 for i in range(1, 11, 3)])


def main():
    start_time = time.time()

    # test_map_2d_example()
    # run_traffic_graph_explorer()

    # n_orig_ahead, n_orig_behind = 1, 1
    # n_dest_ahead, n_dest_behind = 1, 1
    n_platoon = 2
    v_orig = 70 / 3.6
    v_ff_platoon = 110 / 3.6
    v_dest = 50 / 3.6
    # is_acceleration_optimal = True
    are_vehicles_cooperative = False

    test_analysis()
    # configuration.Configuration.set_scenario_parameters(
    #     sim_time=10 * n_platoon
    # )

    # scenarios.run_with_varying_max_computation_time(epsilon=0.4)

    # configuration.Configuration.set_graph_exploration_parameters(
    #     should_use_bfs=False, epsilon=0.1, max_computation_time=0.1)
    #
    # scenarios.run_scenarios_for_comparison(
    #     n_platoon, v_orig, v_dest, v_ff_platoon, are_vehicles_cooperative,
    #     [platoon_lane_change_strategies.StrategyMap.graph_min_time],
    #     # gap_positions=[0]
    # )

    # scenarios.run_all_scenarios_for_comparison()

    # platoon_sizes = [3]
    # vissim_interface.run_platoon_simulations(is_warm_up=True,
    #                                          platoon_size=platoon_sizes)
    # post_processing.import_strategy_maps_from_cloud(platoon_sizes)
    # for n_platoon in platoon_sizes:
    #     print(f"===== Exploring scenario with platoon size {n_platoon} =====")
    #     graph_creator = graph_tools.GraphCreator(n_platoon)
    #     # graph_creator.solve_queries_from_simulations("python", "as")
    #     graph_creator.solve_queries_from_simulations("vissim", "as")
    #     try:
    #         post_processing.export_strategy_maps_to_cloud([n_platoon])
    #     except FileNotFoundError:
    #         print("Couldn't share strategy maps")
    #         continue
    # try:
    #     vissim_interface.run_platoon_simulations(platoon_size=platoon_sizes)
    # except:
    #     print("Some error when running vissim after computing graphs")

    # scenarios.run_all_scenarios_for_comparison(

    # vissim_interface.run_platoon_simulations()

    exec_time = datetime.timedelta(seconds=time.time() - start_time)
    print("Main execution time:", str(exec_time).split(".")[0])


if __name__ == "__main__":
    main()
