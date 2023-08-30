import time

import analysis
import vehicle_models
import scenarios

file_name = 'data.pickle'  # temp


def run_base_scenario(n_per_lane, max_iter=100):
    v_ff = 10
    tf = 10

    base_scenario = scenarios.ExampleScenarioExternal()
    # vehicles = [[vehicle_models.ThreeStateVehicleRearWheel],
    #             [vehicle_models.SafeAccelOptimalLCVehicle]]
    vehicles = [[vehicle_models.four_state_vehicles.SafeAccelOpenLoopLCVehicle,
                vehicle_models.four_state_vehicles.SafeAccelOpenLoopLCVehicle]]
    # base_scenario.set_uniform_vehicles(n_per_lane, vehicle_type, v_ff)
    base_scenario.create_vehicles(vehicles)
    base_scenario.set_free_flow_speeds(v_ff)
    base_scenario.set_boundary_conditions(tf)
    base_scenario.create_dynamic_system()
    base_scenario.set_optimal_control_problem_functions()
    result = base_scenario.solve(max_iter)
    base_scenario.run(result)
    base_scenario.save_response_data(file_name)
    data = base_scenario.response_to_dataframe()
    analysis.plot_lane_change(data)


def run_no_lc_scenario():
    tf = 10
    scenario = scenarios.VehicleFollowingScenario(2)
    scenario.create_initial_state()
    scenario.run(tf)
    analysis.plot_vehicle_following(scenario.response_to_dataframe())


def run_constraints_scenario():
    tf = 10

    # Set-up
    scenario = scenarios.LaneChangeWithConstraints()
    scenario.set_boundary_conditions(tf)
    scenario.create_dynamic_system()
    scenario.set_optimal_control_problem_functions()
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


def run_cbf_lc_scenario():
    tf = 15
    scenario = scenarios.FeedbackLaneChangeScenario()
    scenario.run(tf)
    data = scenario.response_to_dataframe()
    analysis.plot_initial_and_final_states(data)
    analysis.plot_constrained_lane_change(data, scenario.lc_veh_id)


def run_internal_optimal_controller():
    tf = 15
    scenario = scenarios.InternalOptimalControlScenario()
    scenario.run(tf)
    data = scenario.response_to_dataframe()
    analysis.plot_initial_and_final_states(data)
    analysis.plot_constrained_lane_change(data, scenario.lc_veh_id)


def main():
    start_time = time.time()

    # run_no_lc_scenario()
    # run_base_scenario([1], max_iter=200)
    run_constraints_scenario()
    # run_cbf_lc_scenario()
    # run_internal_optimal_controller()
    # load_and_plot_latest_scenario()

    end_time = time.time()
    print("Execution time: ", end_time - start_time)


if __name__ == "__main__":
    main()
