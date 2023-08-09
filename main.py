import pickle

import vehicle_handler
import scenarios

file_name = 'data.pickle'  # temp


def run_base_scenario(n_per_lane, max_iter=100):
    v_ff = 10
    tf = 10
    vehicle_type = vehicle_handler.FourStateVehicleAccelFB(v_ff)

    base_scenario = scenarios.ExampleScenario(n_per_lane, vehicle_type)
    base_scenario.create_dynamic_system()
    base_scenario.set_optimal_control_problem_functions(tf)
    base_scenario.run(max_iter)
    base_scenario.save_response_data(file_name)
    data = base_scenario.response_to_dataframe()
    scenarios.plot_lane_change(data)


def load_and_plot_base_scenario():
    data = scenarios.load_simulated_scenario(file_name)
    scenarios.plot_lane_change(data)


def run_no_lc_scenario():
    v_ff = 10
    tf = 10
    scenario = scenarios.VehicleFollowingScenario([2], v_ff)
    scenario.set_boundary_conditions(tf)
    scenario.run()
    scenarios.plot_vehicle_following(scenario.response_to_dataframe())


def run_constraints_scenario():
    v_ff = 10
    tf = 10

    base_scenario = scenarios.LaneChangeWithConstraints(v_ff)
    base_scenario.create_dynamic_system()
    base_scenario.set_optimal_control_problem_functions(tf)
    base_scenario.run(200)
    base_scenario.save_response_data(file_name)
    data = base_scenario.response_to_dataframe()
    scenarios.plot_scenario_results(['x', 't', 't'],
                                    ['y', 'v', 'phi'], data)


def main():
    # run_base_scenario([2], max_iter=200)
    # load_and_plot_base_scenario()
    # run_no_lc_scenario()
    run_constraints_scenario()


if __name__ == "__main__":
    main()
