import vehicle_handler
import scenarios


def simple_model_tests(n_per_lane, max_iter=100):

    v_ff = 10
    vehicle_type = vehicle_handler.ThreeStateVehicleRearWheel(v_ff)

    base_scenario = scenarios.ExampleScenario(n_per_lane)
    base_scenario.create_dynamic_system(vehicle_type)
    base_scenario.set_scenario()
    base_scenario.solve(max_iter)
    time, states, inputs = (base_scenario.response.time,
                            base_scenario.response.outputs,
                            base_scenario.response.inputs)
    base_scenario.plot_trajectory_and_inputs()


def main():
    simple_model_tests([1, 1], max_iter=200)


if __name__ == "__main__":
    main()
