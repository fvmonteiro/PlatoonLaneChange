import warnings
from collections import defaultdict
import datetime
import pickle
import time

import pandas as pd

import analysis
from controllers.optimal_controller import configure_optimal_controller
import vehicle_models
import scenarios

file_name = 'data.pickle'  # temp


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


def run_base_opc_scenario(max_iter=100):
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
    scenario.set_boundary_conditions(tf)
    scenario.solve()
    scenario.run()
    scenario.save_response_data(file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_lane_change(data)


def run_constraints_scenario(has_lo: bool, has_fo: bool, has_ld: bool,
                             has_fd: bool):
    tf = 10

    # Set-up
    scenario = scenarios.LaneChangeWithConstraints(has_lo, has_fo,
                                                   has_ld, has_fd)
    scenario.set_boundary_conditions(tf)
    # Solve
    print("Calling OCP solver")
    scenario.solve()
    scenario.run()
    # Check results
    scenario.save_response_data(file_name)
    data = scenario.response_to_dataframe()
    # lc_veh_id = scenario.lc_veh_id
    analysis.plot_constrained_lane_change(data, 'ego')
    analysis.plot_constrained_lane_change(
        scenario.ocp_simulation_to_dataframe(), 'ego')
    analysis.compare_desired_and_actual_final_states(
        scenario.boundary_conditions_to_dataframe(), data)


def load_and_plot_latest_scenario():
    data = analysis.load_simulated_scenario(file_name)
    # analysis.plot_initial_and_final_states(data, custom_colors=True)
    analysis.plot_constrained_lane_change(data, 'ego')


def run_save_and_plot(scenario: scenarios.LaneChangeScenario, tf: float = 10.):
    scenario.create_test_initial_state()
    scenario.run(tf)
    scenario.save_response_data(file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_initial_and_final_states(data, custom_colors=True)
    analysis.plot_constrained_lane_change(data, 'p1')  # TODO: ego or p1


def run_cbf_lc_scenario(n_orig_ahead: int, n_orig_behind: int,
                        n_dest_ahead: int, n_dest_behind: int):
    scenario = scenarios.LaneChangeScenario.single_vehicle_feedback_lane_change(
        n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind
    )
    tf = 10.
    run_save_and_plot(scenario, tf)


def run_internal_optimal_controller(n_orig_ahead: int, n_orig_behind: int,
                                    n_dest_ahead: int, n_dest_behind: int):
    scenario = scenarios.LaneChangeScenario.single_vehicle_optimal_lane_change(
        n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind
    )
    scenario.create_test_initial_state()
    tf = 10.
    run_save_and_plot(scenario, tf)


def run_platoon_test(n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
                     n_dest_ahead: int, n_dest_behind: int):
    tf = 7.0
    scenario = scenarios.LaneChangeScenario.platoon_lane_change(
        n_platoon, n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind)
    run_save_and_plot(scenario, tf)


def mode_convergence_base_tests():
    configure_optimal_controller(
        max_iter=5, solver_max_iter=1000, discretization_step=0.1,
        time_horizon=5.0, has_terminal_constraints=False,
        has_non_zero_initial_guess=False, has_lateral_constraint=False
    )

    # Eventually we'll move to descriptive names, but for now let's just avoid
    # overwriting data
    result_file_name = 'result_summary_' + "{:%Y_%m_%d}".format(
        datetime.datetime.now()) + '.pickle'

    n_platoon = 1
    tf = 7
    results = defaultdict(list)
    for i in range(1, 16):
        try:
            # All combinations
            n_ld, n_fd, n_lo, n_fo = (int(b) for b in "{0:04b}".format(i))
            print("============ Running scenario {} ============\t\n"
                  "n_ld={}, n_fd={}, n_lo={}, n_fo={}".format(
                      i, n_ld, n_fd, n_lo, n_fo))
            scenario = scenarios.LaneChangeScenario.platoon_lane_change(
                n_platoon, n_lo, n_fo, n_ld, n_fd)
            scenario.create_safe_uniform_speed_initial_state()
            scenario.run(tf)
            n_iter = len(scenario.get_opc_results_summary()['iteration'])
            results['scenario'].extend([i] * n_iter)
            results['n_ld'].extend([n_ld] * n_iter)
            results['n_fd'].extend([n_fd] * n_iter)
            results['n_lo'].extend([n_lo] * n_iter)
            results['n_fo'].extend([n_fo] * n_iter)
            for key, values in scenario.get_opc_results_summary().items():
                results[key].extend(values)
        except Exception as e:  # Just to make sure we don't waste simulations
            print(e)
            warnings.warn("... will try to run next scenario ...")
    with open(result_file_name, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    data = pd.DataFrame(data=results)
    data.to_csv('./../data/optimal_control_results/'
                'safe_scenarios_summary.csv',
                index=False)


def main():
    n_platoon = 1
    n_orig_ahead, n_orig_behind = 1, 1
    n_dest_ahead, n_dest_behind = 0, 0

    configure_optimal_controller(max_iter=5, solver_max_iter=500,
                                 discretization_step=0.1, time_horizon=5.0,
                                 has_terminal_constraints=False,
                                 has_non_zero_initial_guess=False,
                                 has_lateral_constraint=False)

    start_time = time.time()

    # run_no_lc_scenario()
    # run_fast_lane_change()
    # run_base_opc_scenario(max_iter=400)
    # run_constraints_scenario(n_orig_ahead > 0, n_orig_behind > 0,
    #                          n_dest_ahead > 0, n_dest_behind > 0)
    # run_cbf_lc_scenario(n_orig_ahead, n_orig_behind, n_dest_ahead,
    #                     n_dest_behind)
    # run_internal_optimal_controller(n_orig_ahead, n_orig_behind, n_dest_ahead,
    #                                 n_dest_behind)
    run_platoon_test(n_platoon, n_orig_ahead, n_orig_behind,
                     n_dest_ahead, n_dest_behind)
    # load_and_plot_latest_scenario()
    # mode_convergence_base_tests()

    end_time = time.time()

    print("Execution time: ", end_time - start_time)


if __name__ == "__main__":
    main()
