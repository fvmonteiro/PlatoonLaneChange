import plot_helpers
import vehicle_array
import vehicle_handler
import scenarios

file_name = 'data.pickle'  # temp


def run_base_scenario(n_per_lane, max_iter=100):
    v_ff = 10
    tf = 10
    vehicle_type = vehicle_handler.FourStateVehicle

    base_scenario = scenarios.ExampleScenario()
    vehicles = [[vehicle_handler.FourStateVehicleAccelFB],
                [vehicle_handler.FourStateVehicleAccelFB]]
    # base_scenario.set_uniform_vehicles(n_per_lane, vehicle_type, v_ff)
    base_scenario.create_vehicles(vehicles, [10, 10])
    base_scenario.create_dynamic_system()
    base_scenario.set_optimal_control_problem_functions(tf)
    base_scenario.run(max_iter)
    base_scenario.save_response_data(file_name)
    data = base_scenario.response_to_dataframe()
    plot_helpers.plot_lane_change(data)


def load_and_plot_latest_scenario():
    data = plot_helpers.load_simulated_scenario(file_name)
    plot_helpers.plot_constrained_lane_change(data, 1, 0)


def run_no_lc_scenario():
    v_ff = 10
    tf = 10
    scenario = scenarios.VehicleFollowingScenario([2], v_ff)
    scenario.set_boundary_conditions(tf)
    scenario.run()
    plot_helpers.plot_vehicle_following(scenario.response_to_dataframe())


def run_constraints_scenario():
    v_ff = 10
    tf = 10

    lc_veh_id = 1
    scenario = scenarios.LaneChangeWithConstraints(1, v_ff, lc_veh_id)
    scenario.create_dynamic_system()
    scenario.set_optimal_control_problem_functions(tf)

    scenario.run(300)
    scenario.save_response_data(file_name)
    data = scenario.response_to_dataframe()
    plot_helpers.plot_constrained_lane_change(data, lc_veh_id, lc_veh_id-1)


def main():
    # run_no_lc_scenario()
    # run_base_scenario([1, 1], max_iter=200)
    # run_constraints_scenario()
    load_and_plot_latest_scenario()


if __name__ == "__main__":
    main()
