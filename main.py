import time

import analysis
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


def run_base_opc_scenario(n_per_lane, max_iter=100):
    v_ff = 10
    tf = 10

    scenario = scenarios.ExampleScenarioExternal()
    # vehicles = [
    #     [vehicle_models.three_state_vehicles.ThreeStateVehicleRearWheel],
    #     [vehicle_models.three_state_vehicles.ThreeStateVehicleRearWheel]
    # ]
    vehicles = [
        [vehicle_models.four_state_vehicles.SafeAccelOpenLoopLCVehicle,
         vehicle_models.four_state_vehicles.SafeAccelOpenLoopLCVehicle]
    ]
    scenario.create_vehicle_group(vehicles)
    scenario.set_free_flow_speeds(v_ff)
    scenario.set_boundary_conditions(tf)
    result = scenario.solve(max_iter)
    scenario.run(result)
    scenario.save_response_data(file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_lane_change(data)


def run_constraints_scenario(has_lo: bool, has_fo: bool,
                             has_ld: bool, has_fd: bool):
    tf = 10

    # Set-up
    scenario = scenarios.LaneChangeWithConstraints(has_lo, has_fo, has_ld,
                                                   has_fd)
    scenario.set_boundary_conditions(tf)
    # Solve
    print("Calling OCP solver")
    result = scenario.solve(300)
    scenario.run(result)
    # Check results
    scenario.save_response_data(file_name)
    data = scenario.response_to_dataframe()
    lc_veh_id = scenario.lc_veh_id
    analysis.plot_constrained_lane_change(data, lc_veh_id)
    analysis.plot_constrained_lane_change(
        scenario.ocp_simulation_to_dataframe(), lc_veh_id)
    analysis.compare_desired_and_actual_final_states(
        scenario.boundary_conditions_to_dataframe(), data)


def load_and_plot_latest_scenario():
    data = analysis.load_simulated_scenario(file_name)
    analysis.plot_constrained_lane_change(data, 1)
    # analysis.plot_initial_and_final_states(data)
    # analysis.check_constraint_satisfaction(data, 1)


def run_lane_change_scenario(scenario: scenarios.LaneChangeScenario):
    tf = 15
    scenario.run(tf)
    data = scenario.response_to_dataframe()
    scenario.save_response_data(file_name)
    analysis.plot_initial_and_final_states(data)
    analysis.plot_constrained_lane_change(data, 'ego')


def run_cbf_lc_scenario(has_lo: bool, has_fo: bool,
                        has_ld: bool, has_fd: bool):
    scenario = scenarios.LaneChangeScenario.closed_loop(has_lo, has_fo,
                                                        has_ld, has_fd)
    run_lane_change_scenario(scenario)


def run_internal_optimal_controller(has_lo: bool, has_fo: bool,
                                    has_ld: bool, has_fd: bool):
    scenario = scenarios.LaneChangeScenario.optimal_control(has_lo, has_fo,
                                                            has_ld, has_fd)
    run_lane_change_scenario(scenario)


def main():
    has_lo, has_fo = True, True
    has_ld, has_fd = True, True

    start_time = time.time()

    # run_no_lc_scenario()
    # run_base_scenario([1], max_iter=400)
    run_constraints_scenario(has_lo, has_fo, has_ld, has_fd)
    # run_cbf_lc_scenario(has_lo, has_fo, has_ld, has_fd)
    # run_internal_optimal_controller(has_lo, has_fo, has_ld, has_fd)
    # load_and_plot_latest_scenario()

    end_time = time.time()
    print("Execution time: ", end_time - start_time)


if __name__ == "__main__":
    main()
