from __future__ import annotations

from collections.abc import Sequence
import datetime
import time

import configuration
import analysis
from platoon_functionalities import graph_tools, platoon_lane_change_strategies
import post_processing
import scenarios
from vissim_handler import vissim_interface
from vissim_handler import file_handling


def load_and_plot_latest_scenario():
    trajectory_file_name = 'trajectory_data.pickle'  # temp
    cost_file_name = 'cost_data.pickle'
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


def main():
    start_time = time.time()

    n_platoon = 2
    n_orig_ahead, n_orig_behind = 1, 1
    n_dest_ahead, n_dest_behind = 1, 1
    v_orig = 70 / 3.6
    v_ff_platoon = 110 / 3.6
    v_dest = 50 / 3.6
    is_acceleration_optimal = True
    are_vehicles_cooperative = False

    configuration.Configuration.set_scenario_parameters(
        sim_time=10 * n_platoon
    )
    # configuration.Configuration.set_solver_parameters(
    #     max_iter=100, discretization_step=0.2,
    #     ftol=1.0e-3, estimate_gradient=True
    # )
    # configuration.Configuration.set_optimal_controller_parameters(
    #     max_iter=3, time_horizon=configuration.Configuration.sim_time - 2,
    #     has_terminal_lateral_constraints=False,
    #     has_lateral_safety_constraint=False,
    #     initial_input_guess='mode',
    #     jumpstart_next_solver_call=True, has_initial_mode_guess=True
    # )
    scenarios.run_scenarios_for_comparison(
        n_platoon, v_orig, v_dest, v_ff_platoon, are_vehicles_cooperative,
        [platoon_lane_change_strategies.StrategyMap.graph_min_time],
        gap_positions=[1]
    )
    # scenarios.run_base_ocp_scenario()

    # scenarios.run_all_scenarios_for_comparison(warmup=True)

    # for n_platoon in [5]:
    #     print(f"===== Exploring scenario with platoon size {n_platoon} =====")
    #     graph_creator = graph_tools.GraphCreator(n_platoon)
    #     graph_creator.solve_queries_from_simulations("python", "as")
    # #     graph_creator.solve_queries_from_simulations("vissim", "as")
    #     try:
    #         post_processing.export_strategy_maps_to_cloud([n_platoon])
    #     except FileNotFoundError:
    #         print("Couldn't share strategy maps")
    #         continue

    # scenarios.run_all_scenarios_for_comparison()
    # post_processing.export_results_to_cloud()

    analyzer = analysis.ResultAnalyzer(save_figs=False)
    # analyzer.get_python_results_for_paper()
    # analyzer.print_average_number_of_maneuver_steps()
    # analyzer.compare_approaches()
    # analyzer.compare_to_approach("time")

    # vissim_interface.run_platoon_simulations()

    exec_time = datetime.timedelta(seconds=time.time() - start_time)
    print("Main execution time:", str(exec_time).split(".")[0])


if __name__ == "__main__":
    main()
