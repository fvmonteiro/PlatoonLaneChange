from __future__ import annotations

import random
from collections.abc import Sequence
import datetime
import time

import configuration
import analysis
import vissim_handler.scenario_handling
from platoon_functionalities import graph_tools, platoon_lane_change_strategies
from platoon_functionalities import traffic_state_graph
import post_processing
import scenarios
from vissim_handler import vissim_interface, vissim_vehicle
from vissim_handler import file_handling


def create_graph(n_platoon: int, has_fd: bool, vel_orig_lane: Sequence[float],
                 vel_ff_platoon: float, v_dest_lane: Sequence[float],
                 max_dist: float, mode: str = "as"):
    graph_creator = graph_tools.GraphCreator(n_platoon, has_fd)
    graph_creator.create_graph(vel_orig_lane, vel_ff_platoon, v_dest_lane,
                               max_dist, mode)
    graph_creator.save_vehicle_state_graph_to_file()
    graph_creator.save_quantization_parameters_to_file()
    graph_creator.save_minimum_cost_strategies_to_json()


def run_traffic_graph_explorer(platoon_sizes: list[int], simulator: str,
                               mode: str):
    epsilon_range = [4]  # [i for i in range(1, 11, 3)]
    all_cost_types = ["time"]
    # configuration.Configuration.set_graph_exploration_parameters(should_use_bfs=True)
    for n_platoon in platoon_sizes:
        for epsilon_idx in epsilon_range:
            for cost_type in all_cost_types:
                random.seed(1)
                epsilon = epsilon_idx/10
                print("=" * 79 + f"\nCost type: {cost_type}, eps={epsilon}")
                traffic_state_graph.solve_queries_from_simulations(
                    simulator, mode, n_platoon, cost_type=cost_type,
                    epsilon=epsilon, verbose_level=0)
                # for scenario in range(63):
                #     try:
                #         print(f"scenario: {scenario}")
                #         traffic_state_graph.solve_queries_from_simulations(
                #             simulator, mode, n_platoon, cost_type=cost_type,
                #             epsilon=epsilon, verbose_level=0,
                #             scenario_number=scenario)
                #     except MemoryError:
                #         print("MEMORY ERROR! but following on to next")
                #         continue


def test_analysis(n_platoon: int):
    # analyzer = analysis.ResultAnalyzer(save_figs=False)
    # analyzer.get_python_results_for_paper()
    # analyzer.print_average_number_of_maneuver_steps()
    # analyzer.compare_approaches()
    # analyzer.compare_to_approach("time")

    # analysis.compare_bfs_and_dfs(n_platoon, "time")
    ra = analysis.ResultAnalyzer(is_bfs=False)
    # ra.plot_cost_vs_max_computation_time(n_platoon)

    epsilon = [i/10 for i in range(1, 11, 3)]
    analysis.plot_several_cost_vs_computation_time(
        [n_platoon], ["time", "accel"], epsilon,
        simulator="python", save_fig=False)
    # analysis.plot_several_cost_vs_computation_time(
    #     [n_platoon], ["time", "accel"], epsilon,
    #     simulator="vissim")
    # analysis.plot_several_cost_vs_computation_time(
    #     [n_platoon], ["time", "accel"], epsilon,
    #     simulator="python")


def failed_lc_case():
    scenario_name = "platoon_discretionary_lane_change"
    epsilon = 0.4
    strategy = vissim_vehicle.PlatoonLaneChangeStrategy.graph_min_time
    other_vehicles = {vissim_vehicle.VehicleType.HDV: 100}
    vehicles_per_lane = 500
    orig_and_dest_lane_speeds = ("70", "50")
    comp_time = 20
    scenario = vissim_handler.scenario_handling.ScenarioInfo(
        other_vehicles, vehicles_per_lane, strategy,
        orig_and_dest_lane_speeds, 5, comp_time
    )
    vi = vissim_interface.VissimInterface()
    vi.load_simulation(scenario_name)


def main():
    start_time = time.time()
    n_platoon = 5

    # test_map_2d_example()

    # n_orig_ahead, n_orig_behind = 1, 1
    # n_dest_ahead, n_dest_behind = 1, 1
    v_orig = 70 / 3.6
    v_ff_platoon = 110 / 3.6
    v_dest = 50 / 3.6
    # is_acceleration_optimal = True
    are_vehicles_cooperative = False

    # pympler_test()

    # run_traffic_graph_explorer([n_platoon], "python", "as")
    test_analysis(n_platoon)

    # configuration.Configuration.set_scenario_parameters(
    #   sim_time=10 * n_platoon
    # )

    # scenarios.run_with_varying_max_computation_time(epsilon=0.4)
    # vissim_interface.run_platoon_simulations_with_varying_computation_time(
    #     n_platoon
    # )

    # configuration.Configuration.set_graph_exploration_parameters(
    #     should_use_bfs=False, epsilon=0.1, max_computation_time=0.1)
    #
    # scenarios.run_scenarios_for_comparison(
    #     n_platoon, v_orig, v_dest, v_ff_platoon, are_vehicles_cooperative,
    #     [platoon_lane_change_strategies.StrategyMap.graph_min_time],
    #     # gap_positions=[0]
    # )

    # scenarios.run_all_scenarios_for_comparison()

    # platoon_sizes = [4]
    # vissim_interface.run_a_platoon_simulation(
    #     vehicles_per_lane=1000, platoon_size=3)
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

    exec_time = datetime.timedelta(seconds=time.time() - start_time)
    print("Main execution time:", str(exec_time).split(".")[0])


if __name__ == "__main__":
    main()
