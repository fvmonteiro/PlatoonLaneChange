from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import datetime
import os
import pickle
import time
import warnings
from typing import Union

import control as ct
import networkx as nx
import numpy as np
import pandas as pd

import analysis
import controllers.optimal_controller as opt_ctrl
import controllers.optimal_control_costs as occ
import configuration
import platoon_functionalities.graph_tools as graph_tools
import platoon_functionalities.platoon_lane_change_strategies as lc_strategy
import post_processing as pp
import vehicle_group as vg
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv
import vehicle_models.three_state_vehicles as tsv


config = configuration.Configuration


all_platoon_simulation_configurations = {
    "strategies": [
        lc_strategy.StrategyMap.single_body_platoon,
        lc_strategy.StrategyMap.last_vehicle_first,
        lc_strategy.StrategyMap.leader_first_and_reverse,
        lc_strategy.StrategyMap.graph_min_accel,
        lc_strategy.StrategyMap.graph_min_time
    ],
    "orig_and_dest_lane_speeds": [(70, 50), (70, 70), (70, 90)],
    "platoon_size": [2, 3, 4, 5],
}


# ================== Functions to quickly run simulations ==================== #
def run_no_lc_scenario():
    time_start = time.time()
    tf = 10
    scenario = VehicleFollowingScenario(2)
    scenario.create_initial_state()
    scenario.run(tf)
    analysis.plot_vehicle_following(scenario.response_to_dataframe())
    exec_time = datetime.timedelta(seconds=time.time() - time_start)
    print("run_no_lc_scenario time:", str(exec_time).split(".")[0])


def run_fast_lane_change():
    time_start = time.time()
    tf = 5
    scenario = FastLaneChange()
    scenario.run(tf)
    data = scenario.response_to_dataframe()
    analysis.plot_lane_change(data)
    exec_time = datetime.timedelta(seconds=time.time() - time_start)
    print("run_fast_lane_change time:", str(exec_time).split(".")[0])


def run_base_ocp_scenario():
    trajectory_file_name = 'trajectory_data.pickle'  # temp
    v_ff = 10
    tf = 10

    scenario = ExampleScenarioExternal()
    vehicles = [
        [tsv.ThreeStateVehicleRearWheel()],
        [tsv.ThreeStateVehicleRearWheel()]
    ]
    scenario.create_vehicle_group(vehicles)
    scenario.set_free_flow_speeds(v_ff)
    scenario.create_initial_state()
    scenario.set_desired_final_states(tf)
    scenario.solve()
    scenario.run(tf)
    scenario.save_response_data(trajectory_file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_lane_change(data)


def run_optimal_platoon_test(
        n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
        n_dest_ahead: int, n_dest_behind: int, is_acceleration_optimal: bool,
        are_vehicles_cooperative: bool):
    trajectory_file_name = 'trajectory_data.pickle'  # temp
    cost_file_name = 'cost_data.pickle'
    v_ref = dict()  # TODO: make param
    delta_x = dict()  # TODO: make param
    scenario = LaneChangeScenario(n_platoon,
    are_vehicles_cooperative)
    scenario.set_control_type('optimal', is_acceleration_optimal)
    scenario.create_test_scenario(n_dest_ahead, n_dest_behind, n_orig_ahead,
                                  n_orig_behind, v_ref, delta_x)
    # scenario.create_optimal_control_test_scenario(
    #     n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind)
    tf = configuration.Configuration.time_horizon + 2
    # run_save_and_plot(scenario, tf)
    scenario.run(tf)
    scenario.save_response_data(trajectory_file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_trajectory(data)
    if scenario.get_n_platoon() < 1:
        analysis.plot_constrained_lane_change(data, 'p1')  # TODO: ego or p1
    else:
        analysis.plot_platoon_lane_change(data)
    scenario.save_cost_data(cost_file_name)
    running_cost, terminal_cost = scenario.get_opc_cost_history()
    analysis.plot_costs_vs_iteration(running_cost, terminal_cost)


def run_with_external_controller(
        n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
        n_dest_ahead: int, n_dest_behind: int, is_acceleration_optimal: bool,
        are_vehicles_cooperative: bool, v_ref: Mapping[str, float],
        delta_x: Mapping[str, float]):
    trajectory_file_name = 'trajectory_data.pickle'  # temp
    cost_file_name = 'cost_data.pickle'
    # Set-up
    tf = configuration.Configuration.time_horizon
    scenario = LaneChangeWithExternalController(
        n_platoon, are_vehicles_cooperative)
    scenario.create_initial_state(
        n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind, v_ref,
        delta_x, is_acceleration_optimal)
    scenario.set_desired_final_states(configuration.Configuration.time_horizon)
    # Solve
    print("Calling OCP solver")
    scenario.solve()
    # run_save_and_plot(scenario, tf)
    scenario.run(tf)
    scenario.save_response_data(trajectory_file_name)
    data = scenario.response_to_dataframe()
    analysis.plot_trajectory(data)
    if scenario.get_n_platoon() < 1:
        analysis.plot_constrained_lane_change(data, 'p1')  # TODO: ego or p1
    else:
        analysis.plot_platoon_lane_change(data)
    scenario.save_cost_data(cost_file_name)
    running_cost, terminal_cost = scenario.get_opc_cost_history()
    analysis.plot_costs_vs_iteration(running_cost, terminal_cost)

    # analysis.plot_constrained_lane_change(
    #     scenario.ocp_simulation_to_dataframe(), 'p1')
    # analysis.compare_desired_and_actual_final_states(
    #     scenario.boundary_conditions_to_dataframe(), data)


def run_all_scenarios_for_comparison(warmup: bool = False):
    if warmup:
        strategies = [lc_strategy.StrategyMap.graph_min_time]
        save_results = False
    else:
        strategies = all_platoon_simulation_configurations[
            "strategies"]
        save_results = True

    orig_and_dest_lane_speeds = (
        all_platoon_simulation_configurations[
            "orig_and_dest_lane_speeds"])
    v_ff_platoon = configuration.FREE_FLOW_SPEED * configuration.KMH_TO_MS
    # n_platoon = [4, 5]
    n_platoon = all_platoon_simulation_configurations["platoon_size"]
    are_vehicles_cooperative = False

    start_time = time.time()
    for n in n_platoon:
        sim_time = 20.0 * n
        configuration.Configuration.set_scenario_parameters(
            sim_time=sim_time, increase_lc_time_headway=False
        )
        for v_orig, v_dest in orig_and_dest_lane_speeds:
            run_scenarios_for_comparison(
                n, v_orig * configuration.KMH_TO_MS,
                v_dest * configuration.KMH_TO_MS, v_ff_platoon,
                are_vehicles_cooperative, strategies,
                has_plots=False, save=save_results
            )

    warmup_time = datetime.timedelta(seconds=time.time() - start_time)
    print("Time to run all platoon scenarios:",
          str(warmup_time).split(".")[0])


def run_scenarios_for_comparison(
        n_platoon: int, v_orig: float, v_dest: float, v_ff_platoon: float,
        are_vehicles_cooperative: bool,
        strategies: Iterable[lc_strategy.StrategyMap],
        gap_positions: Iterable[int] = None,
        has_plots: bool = True, save: bool = False):
    v_ref = {'orig': v_orig, 'platoon': v_ff_platoon, 'dest': v_dest}
    scenario_manager = LaneChangeScenarioManager(
        n_platoon, are_vehicles_cooperative, v_ref)
    scenario_manager.set_plotting(has_plots)

    print(f'Starting multiple runs with n_platoon={n_platoon}, '
          f'vo={v_orig:.1f}, vd={v_dest:.1f}')
    for s in strategies:
        print(f' strategy number={s}')
        if gap_positions is None:
            scenario_manager.run_all_single_gap_cases(s)
        else:
            for gp in gap_positions:
                scenario_manager.run_single_gap_scenario(s, gp)
                # result = scenario_manager.get_results()
                # print(result.groupby('strategy')[
                #           ['success', 'completion_time', 'accel_cost',
                #            'decision_time']
                #       ].mean())
                # result.to_csv('results_temp_name.csv', index=False)
    if save:
        scenario_manager.append_results_to_csv()


def run_closed_loop_test(
        n_platoon: int, are_vehicles_cooperative: bool,
        v_orig: float, v_ff_platoon: float, v_dest: float,
        delta_x: Mapping[str, float],
        strategies: Iterable[lc_strategy.StrategyMap],
        plot_results: bool = True):
    v_ref = {'orig': v_orig, 'platoon': v_ff_platoon, 'dest': v_dest}
    scenario_manager = LaneChangeScenarioManager(
        n_platoon, are_vehicles_cooperative, v_ref, delta_x)
    scenario_manager.set_plotting(plot_results)
    n_orig_ahead = 1
    n_orig_behind = 1
    n_dest_ahead = 1
    n_dest_behind = 1
    scenario_manager.run_strategy_comparison_on_test_scenario(
        n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind, strategies
    )


class LaneChangeScenarioManager:

    # scenario: LaneChangeScenario
    # v_ref: Mapping[str, float]
    # delta_x: Mapping[str, float]
    # n_platoon: int
    # are_vehicles_cooperative: bool
    # _lane_change_graph: graph_tools.VehicleStatesGraph
    # _strategy_map: graph_tools.StrategyMap

    def __init__(self, n_platoon: int, are_vehicles_cooperative: bool,
                 v_ref: Mapping[str, float],
                 delta_x: Mapping[str, float] = None):
        # TODO: names should depend on the scenario
        self.trajectory_file_name = "trajectory_data.pickle"
        self.cost_file_name = "cost_data.pickle"
        self.results: dict[str, list] = defaultdict(list)
        self._has_plots = True

        self.n_platoon = n_platoon
        self.are_vehicles_cooperative = are_vehicles_cooperative
        self.v_ref = v_ref
        if delta_x is not None:
            self.delta_x = delta_x
        else:
            self.delta_x: dict[str, float] = defaultdict(float)

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    # def set_parameters(self, n_platoon: int, are_vehicles_cooperative: bool,
    #                    v_ref: Mapping[str, float],
    #                    delta_x: Mapping[str, float] = None):
    #     self.n_platoon = n_platoon
    #     self.are_vehicles_cooperative = are_vehicles_cooperative
    #     self.v_ref = v_ref
    #     if delta_x is not None:
    #         self.delta_x = delta_x

    def set_plotting(self, value: bool) -> None:
        self._has_plots = value

    def run_scenarios_from_file(self, simulator_name: str,
                                scenario_number: int = None):
        """
        Runs initial states found during runs in VISSIM
        :param simulator_name: vissim or python
        :param scenario_number: For debugging, in case we want to run a single
         scenario
        """
        df, _ = (
            graph_tools.GraphCreator.load_initial_states_seen_in_simulations(
                self.n_platoon, simulator_name)
        )
        if scenario_number is not None:
            df = df.iloc[scenario_number: scenario_number + 1]

        states_idx = [i for i in range(len(df.columns))
                      if df.columns[i].startswith("x")]
        tf = config.sim_time
        strategy = lc_strategy.StrategyMap.graph_min_time

        lc_intention_time = 0.0
        quantizer = graph_tools.StateQuantizer()  # assuming default params
        print(f"Running {len(df.index)} scenarios.")
        for index, row in df.iterrows():
            print(f"Scenario {index}")
            scenario = self.initialize_closed_loop_scenario(strategy)
            scenario.set_lc_intention_time(lc_intention_time)

            free_flow_speeds = {"lo": row["orig"], "ld": row["dest"]}
            self.v_ref = {"orig": row["orig"], "dest": row["dest"]}
            for i in range(self.n_platoon):
                free_flow_speeds["p" + str(i+1)] = row["platoon"]
            quantized_initial_state = tuple(row.iloc[states_idx].to_numpy(int))
            initial_state = quantizer.dequantize_state(quantized_initial_state)
            initial_state_per_vehicle = (
                graph_tools.VehicleStateGraph.split_state_vector(initial_state)
            )
            # Helps ensure that
            initial_state_per_vehicle["p1"][0] = 0
            initial_state_per_vehicle["ld"][0] += (
                    (free_flow_speeds["p1"] - free_flow_speeds["ld"])
                    * lc_intention_time
            )
            scenario.create_vehicle_group_from_initial_state(
                initial_state_per_vehicle, free_flow_speeds,
                fill_destination_lane_leaders=True)
            scenario.vehicle_group.set_verbose(False)
            try:
                scenario.run(tf)
            except nx.NodeNotFound:
                print("Simulation interrupted because some nodes were not "
                      "found in the graph.")
                analysis.plot_state_vector(
                    scenario.vehicle_group.get_full_initial_state_vector(),
                    "initial state")
                analysis.plot_state_vector(
                    scenario.vehicle_group.get_state(),
                    "last state")
                continue
            except ValueError:
                print("Simulation interrupted because a query was not "
                      "found in the strategies map.")
                continue
            except vg.CollisionException:
                warnings.warn(f"Collision at simulation {index}.")

            # self.store_results(scenario, gap_position)
            # scenario.save_response_data(self.trajectory_file_name)

            if self._has_plots:
                data = scenario.response_to_dataframe()
                analysis.plot_trajectory(data, "Scenario " + str(index))
                analysis.plot_platoon_lane_change(data)

    def run_strategy_comparison_on_test_scenario(
            self, n_orig_ahead: int, n_orig_behind: int, n_dest_ahead: int,
            n_dest_behind: int, strategies: Iterable[lc_strategy.StrategyMap]):
        for sn in strategies:
            self.run_test_scenario(n_orig_ahead, n_orig_behind, n_dest_ahead,
                                   n_dest_behind, sn)

    def run_strategy_comparison_on_single_gap_scenario(
            self, strategies: Iterable[lc_strategy.StrategyMap]):
        for sn in strategies:
            self.run_all_single_gap_cases(sn)

    def run_test_scenario(
            self, n_orig_ahead: int, n_orig_behind: int, n_dest_ahead: int,
            n_dest_behind: int, strategy: lc_strategy.StrategyMap):
        tf = config.sim_time
        scenario_name = self._create_scenario_name(strategy)
        scenario = self.initialize_closed_loop_scenario(strategy)
        scenario.set_test_initial_state(
            n_dest_ahead, n_dest_behind, n_orig_ahead, n_orig_behind,
            self.v_ref, self.delta_x)
        gap_position = -1
        self.run_save_and_plot(scenario, tf, gap_position, scenario_name)

    def run_all_single_gap_cases(self, strategy: lc_strategy.StrategyMap):
        first_gap = 1
        last_gap = self.n_platoon
        if self.v_ref["orig"] > self.v_ref["dest"]:
            first_gap = 0
        elif self.v_ref["orig"] < self.v_ref["dest"]:
            last_gap = self.n_platoon + 1
        for gap_position in range(first_gap, last_gap + 1):
            self.run_single_gap_scenario(strategy, gap_position)

    def run_single_gap_scenario(self, strategy: lc_strategy.StrategyMap,
                                gap_position: int):
        tf = 20 * self.n_platoon
        scenario_name = self._create_scenario_name(strategy)
        print(f"      gap position={gap_position}")
        scenario = self.initialize_closed_loop_scenario(strategy)
        scenario.set_allow_early_termination(True)
        scenario.set_single_gap_initial_state(gap_position, self.v_ref,
                                              self.delta_x["lo"])
        self.run_save_and_plot(scenario, tf, gap_position,
                               scenario_name + f" gap at {gap_position}")

    def initialize_closed_loop_scenario(self, strategy: lc_strategy.StrategyMap
                                        ) -> LaneChangeScenario:
        scenario = LaneChangeScenario(self.n_platoon,
                                      self.are_vehicles_cooperative)
        scenario.set_control_type("closed_loop",
                                  is_acceleration_optimal=False)
        scenario.set_lane_change_strategy(strategy)
        # if strategy_number in lc_strategy.graphStrategyIds:
        #     scenario.set_lane_change_states_graph(self._lane_change_graph)
        return scenario

    def initialize_optimal_control_scenario(self, is_acceleration_optimal: bool
                                            ) -> LaneChangeScenario:
        scenario = LaneChangeScenario(self.n_platoon,
                                      self.are_vehicles_cooperative)
        scenario.set_control_type("optimal", is_acceleration_optimal)
        return scenario

    # def set_test_scenario_initial_state(
    #         self, scenario: LaneChangeScenario, n_orig_ahead: int,
    #         n_orig_behind: int, n_dest_ahead: int, n_dest_behind: int):
    #     scenario.set_test_initial_state(
    #         n_dest_ahead, n_dest_behind, n_orig_ahead, n_orig_behind,
    #         self.v_ref, self.delta_x)

    # def set_single_gap_initial_state(
    #         self, scenario: LaneChangeScenario, gap_position: int):
    #     scenario.set_single_gap_initial_state(gap_position, self.v_ref,
    #                                           self.delta_x["lo"])

    def create_closed_loop_test_scenario(
            self, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int,
            strategy: lc_strategy.StrategyMap
    ) -> LaneChangeScenario:
        scenario = self.initialize_closed_loop_scenario(strategy)
        scenario.set_test_initial_state(
            n_dest_ahead, n_dest_behind, n_orig_ahead, n_orig_behind,
            self.v_ref, self.delta_x)
        return scenario

    def create_closed_loop_single_gap_scenario(
            self, strategy: lc_strategy.StrategyMap, gap_position: int
    ) -> LaneChangeScenario:
        scenario = self.initialize_closed_loop_scenario(strategy)
        scenario.set_single_gap_initial_state(gap_position, self.v_ref,
                                              self.delta_x["lo"])
        return scenario

    def create_optimal_control_test_scenario(
            self, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int,
            is_acceleration_optimal: bool):
        # tf = config.time_horizon + 2
        scenario = self.initialize_optimal_control_scenario(
            is_acceleration_optimal)
        scenario.set_test_initial_state(
            n_dest_ahead, n_dest_behind, n_orig_ahead, n_orig_behind,
            self.v_ref, self.delta_x)
        return scenario

    def run_save_and_plot(self, scenario: LaneChangeScenario, tf: float,
                          gap_position: int, scenario_name: str = None):

        scenario.vehicle_group.set_verbose(False)
        try:
            scenario.run(tf)
        except nx.NodeNotFound:
            print("Simulation interrupted because some nodes were not "
                  "found in the graph.")
            return
        except KeyError:
            print("Simulation interrupted because a query was not "
                  "found in the strategies map.")
            return

        self.store_results(scenario, gap_position)
        # scenario.save_response_data(self.trajectory_file_name)

        if self._has_plots:
            data = scenario.response_to_dataframe()
            analysis.plot_trajectory(data, scenario_name)
            if scenario.get_n_platoon() < 1:
                analysis.plot_constrained_lane_change(data, "p1")
            else:
                analysis.plot_platoon_lane_change(data)

            try:
                scenario.save_cost_data(self.cost_file_name)
                running_cost, terminal_cost = scenario.get_opc_cost_history()
                analysis.plot_costs_vs_iteration(running_cost, terminal_cost)
            except AttributeError:
                pass

    def store_results(self, scenario: LaneChangeScenario, gap_position: int):
        veh_group = scenario.vehicle_group
        strategy = veh_group.get_platoon_lane_change_strategy()
        if strategy is not None:
            strategy_name = strategy.get_name()
            lane_change_order = strategy.get_lane_change_order()
            cooperation_order = strategy.get_cooperation_order()
        else:
            print("### Dealing with a scenario without strategy ###\n"
                  "### Might have to rethink this part ###")
            strategy_name = "none"
            lane_change_order = []
            cooperation_order = []
        success = veh_group.check_lane_change_success()
        completion_time = np.max(veh_group.get_lc_end_times())
        accel_cost = veh_group.compute_acceleration_cost()
        decision_time = veh_group.get_decision_time()
        if success:
            print(f"      Jt={completion_time:.1f}, Ju={accel_cost:.0f}")
        else:
            print(f"      Lane change failure")

        result = {
            "n_platoon": scenario.get_n_platoon(),
            "vo": self.v_ref["orig"], "vd": self.v_ref["dest"],
            "gap_position": gap_position,
            "strategy": strategy_name,
            "lane_change_order": lane_change_order,
            "cooperation_order": cooperation_order,
            "success": veh_group.check_lane_change_success(),
            "completion_time": completion_time, "accel_cost": accel_cost,
            "decision_time": decision_time
        }

        for key, value in result.items():
            self.results[key].append(value)

    def append_results_to_csv(self):
        file_name = "result_summary.csv"
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "platoon_strategy_results", file_name)
        try:
            results_history = pd.read_csv(file_path)
            experiment_counter = results_history["experiment_counter"].max() + 1
            write_header = False
        except FileNotFoundError:
            experiment_counter = 0
            write_header = True
        current_results = self.get_results()
        current_results["experiment_counter"] = experiment_counter
        current_results.to_csv(file_path, mode="a", index=False,
                               header=write_header)
        print(f"File {file_name} saved")

    # def append_initial_state_to_csv(
    #         self, initial_state: configuration.QuantizedState) -> None:
    #     file_name = "_".join(["unsolved_x0", str(self.n_platoon),
    #                           "vehicles.csv"])
    #     file_path = os.path.join(configuration.DATA_FOLDER_PATH,
    #                              "vehicle_state_graphs", file_name)
    #     new_line = (",".join([str(self.v_ref[k])
    #                           for k in sorted(self.v_ref.keys())]
    #                          + [str(i) for i in initial_state]) + "\n")
    #     if not os.path.isfile(file_path):
    #         header = (",".join(
    #             [str(k) for k in sorted(self.v_ref.keys())]
    #             + ["x" + str(i) for i in range(len(initial_state))]) + "\n")
    #         new_line = header + new_line
    #     with open(file_path, "a") as file:
    #         file.write(new_line)

    def _create_scenario_name(self, strategy_id: lc_strategy.StrategyMap):
        scenario_name = strategy_id.get_implementation().get_name()
        if self.v_ref["orig"] > self.v_ref["dest"]:
            scenario_name += " vo > vd"
        elif self.v_ref["orig"] < self.v_ref["dest"]:
            scenario_name += " vo < vd"
        else:
            scenario_name += " vo = vd"
        return scenario_name


class SimulationScenario(ABC):

    _str_to_vehicle_type = {
        "optimal": fsv.OptimalControlVehicle,
        "closed_loop": fsv.ClosedLoopVehicle,
        "open_loop": fsv.OpenLoopVehicle
    }

    _is_acceleration_optimal: bool
    _lc_veh_type: type[fsv.FourStateVehicle]
    _n_platoon: int

    def __init__(self):
        self.n_per_lane: list[int] = []
        base.BaseVehicle.reset_vehicle_counter()
        self.vehicle_group: vg.VehicleGroup = vg.VehicleGroup()
        self.result_summary: dict = {}
        self._are_vehicles_cooperative = False
        self.lc_vehicle_names = []
        self.final_time: float = 0.
        self.dt = config.time_step

    def get_n_platoon(self):
        return self._n_platoon

    @abstractmethod
    def get_opc_results_summary(self):
        pass

    @abstractmethod
    def get_opc_cost_history(self):
        pass

    def get_lc_vehicle_ids(self):
        return [veh.get_id() for veh in self.vehicle_group.get_all_vehicles()
                if veh.get_name() in self.lc_vehicle_names]

    def set_control_type(self, control_type: str,
                         is_acceleration_optimal: bool = None):
        self._lc_veh_type = SimulationScenario._str_to_vehicle_type[
            control_type]
        if isinstance(self._lc_veh_type, fsv.ClosedLoopVehicle):
            if is_acceleration_optimal is None:
                is_acceleration_optimal = False
            elif is_acceleration_optimal:
                warnings.warn("Cannot set optimal acceleration True when the "
                              "vehicles only have feedback controllers")
        elif is_acceleration_optimal is None:
            is_acceleration_optimal = True
        self._is_acceleration_optimal = is_acceleration_optimal

    def save_cost_data(self, file_name: str) -> None:
        """
        Pickles running and terminal costs 2D lists
        :param file_name:
        :return:
        """
        with open(file_name, "wb") as f:
            pickle.dump(self.get_opc_cost_history(),
                        f, pickle.HIGHEST_PROTOCOL)

    def create_vehicle_group(
            self, vehicles: Sequence[Sequence[base.BaseVehicle]]):
        for i in range(len(vehicles)):
            self.n_per_lane.append(len(vehicles[i]))
        flat_vehicle_list = [item for sublist in vehicles for
                             item in sublist]
        self.vehicle_group.fill_vehicle_array(flat_vehicle_list)

    def create_vehicle_group_from_initial_state(
            self, initial_state: Mapping[str, np.ndarray],
            free_flow_speed: Mapping[str, float],
            is_acceleration_optimal: bool = False,
            fill_destination_lane_leaders: bool = False):
        """

        :param initial_state: Initial state per vehicle
        :param free_flow_speed: Free flow speed per vehicle
        :param is_acceleration_optimal: Determines whether the optimal
         controller is in charge of the acceleration (if there is an optimal
         controller at all)
        :param fill_destination_lane_leaders: If true, creates vehicles in
         front of the destination lane leader. This is useful to force the
         platoon to merge behind the destination lane leader
        :return:
        """
        y_idx = fsv.FourStateVehicle.get_idx_of_state("y")
        for name, x0 in initial_state.items():
            if name.startswith("p"):
                veh = self._lc_veh_type(True, is_acceleration_optimal, True)
            else:
                veh = fsv.ClosedLoopVehicle(False, is_acceleration_optimal,
                                            self._are_vehicles_cooperative)
            veh.set_name(name)
            veh.set_free_flow_speed(free_flow_speed[name])
            x0[y_idx] = configuration.LANE_WIDTH * (x0[y_idx] > 0)
            veh.set_initial_state(full_state=x0)
            self.vehicle_group.add_vehicle(veh)
        if fill_destination_lane_leaders:
            x_idx = fsv.FourStateVehicle.get_idx_of_state("x")
            ld = self.vehicle_group.get_vehicle_by_name("ld")
            p1 = self.vehicle_group.get_vehicle_by_name("p1")
            i = 1
            previous_veh = ld
            while previous_veh.get_x() < p1.get_x():
                dest_lane_veh = fsv.ClosedLoopVehicle(
                    False, is_acceleration_optimal,
                    self._are_vehicles_cooperative)
                dest_lane_veh.set_name("d" + str(i))
                i += 1
                dest_lane_veh.set_free_flow_speed(free_flow_speed["ld"])
                safe_gap = previous_veh.compute_non_connected_reference_gap()
                x0 = previous_veh.get_initial_state()
                x0[x_idx] += (safe_gap * 1.1)  # we want the vehicles to be
                # close to each other, but not so close that they'll leave
                # velocity control mode
                dest_lane_veh.set_initial_state(full_state=x0)
                previous_veh = dest_lane_veh
                self.vehicle_group.add_vehicle(previous_veh)

    def place_origin_lane_vehicles(
            self, v_ff_lo: float, v_ff_platoon: float,
            is_acceleration_optimal: bool, n_orig_ahead: int = 1,
            n_orig_behind: int = 1, delta_x_lo: float = 0,
            ):
        """
        Creates n_platoon vehicles plus an origin lane leader and inserts them
        in self"s vehicle group.
        :param v_ff_lo: Origin lane leader"s free-flow speed
        :param v_ff_platoon: Platoon"s free-flow speed
        :param is_acceleration_optimal: Determines if acceleration is an optimal
         input (for optimal control vehicles only)
        :param n_orig_behind: Number of vehicles in front of the platoon
        :param n_orig_ahead: Number of vehicles behind the platoon
        :param delta_x_lo: Distance from the reference gap between the platoon
        leader and its leader. Positive values mean farther away.
        :return:
        """
        y_orig = 0.

        # Platoon vehicles
        x0 = 0
        v0_platoon = v_ff_lo
        for i in range(self._n_platoon):
            p_i = self._lc_veh_type(True, is_acceleration_optimal, True)
            p_i.set_name("p" + str(i + 1))
            p_i.set_free_flow_speed(v_ff_platoon)
            p_i.set_initial_state(x0, y_orig, 0., v0_platoon)
            self.vehicle_group.add_vehicle(p_i)
            x0 -= p_i.compute_initial_reference_gap_to(p_i)  # since platoon
            # vehicles have the same parameters, the gap reference gap from a
            # vehicle to itself is the same as between any two platoon vehicles

        # Origin lane ahead
        p1 = self.vehicle_group.get_vehicle_by_name("p1")
        x0 = p1.get_x() + p1.compute_non_connected_reference_gap() - delta_x_lo
        v0_lo = v_ff_lo
        for i in range(n_orig_ahead):
            lo = fsv.ClosedLoopVehicle(False, False,
                                       self._are_vehicles_cooperative)
            lo.set_name("lo" + str(i))
            lo.set_free_flow_speed(v_ff_lo)
            lo.set_initial_state(x0, y_orig, 0., v0_lo)
            self.vehicle_group.add_vehicle(lo)
            x0 += lo.compute_non_connected_reference_gap(v0_lo)

        # Origin lane behind
        x0 = self.vehicle_group.get_vehicle_by_name(
            "p" + str(self._n_platoon)).get_x()
        for i in range(n_orig_behind):
            veh = fsv.ClosedLoopVehicle(False, False,
                                        self._are_vehicles_cooperative)
            veh.set_name("fo" + str(i))
            veh.set_free_flow_speed(v_ff_lo)
            x0 -= veh.compute_non_connected_reference_gap(v0_lo)
            veh.set_initial_state(x0, y_orig, 0., v0_lo)
            self.vehicle_group.add_vehicle(veh)

        self.n_per_lane = [self._n_platoon + n_orig_behind + n_orig_behind]

    def place_dest_lane_vehicles_around_platoon(
            self, v_ff: float, delta_x: Mapping[str, float],
            n_ahead: int, n_behind: int):
        y_dest = configuration.LANE_WIDTH
        p1 = self.vehicle_group.get_vehicle_by_name("p1")
        x0 = (p1.get_x() + p1.compute_non_connected_reference_gap()
              - delta_x["ld"])
        for i in range(n_ahead):
            veh = fsv.ClosedLoopVehicle(False, False,
                                        self._are_vehicles_cooperative)
            veh.set_name("ld" + str(i))
            veh.set_free_flow_speed(v_ff)
            veh.set_initial_state(x0, y_dest, 0., v_ff)
            self.vehicle_group.add_vehicle(veh)
            x0 += veh.compute_non_connected_reference_gap(v_ff)

        x0 = self.vehicle_group.get_vehicle_by_name(
            "p" + str(self._n_platoon)).get_x() + delta_x["fd"]
        for i in range(n_behind):
            veh = fsv.ClosedLoopVehicle(False, False,
                                        self._are_vehicles_cooperative)
            veh.set_name("fd" + str(i))
            veh.set_free_flow_speed(v_ff)
            x0 -= veh.compute_non_connected_reference_gap(v_ff)
            veh.set_initial_state(x0, y_dest, 0., v_ff)
            self.vehicle_group.add_vehicle(veh)

        self.n_per_lane.append(n_ahead + n_behind)

    def set_test_initial_state(
            self, n_dest_ahead: int, n_dest_behind: int, n_orig_ahead: int,
            n_orig_behind: int, v_ref: Mapping[str, float],
            delta_x: Mapping[str, float]):
        """
        Scenario where all vehicles are at steady state distances. Vehicles
        at the destination lane are placed either in front of or behind the
        whole platoon. To have a destination lane vehicle longitudinally
        between platoon vehicles, use delta_x from Configurations
        :return:
        """

        v_ff_lo = v_ref["orig"]
        v_ff_platoon = v_ref["platoon"]
        v_ff_d = v_ref["dest"]
        self.place_origin_lane_vehicles(
            v_ff_lo, v_ff_platoon, self._is_acceleration_optimal,
            n_orig_ahead, n_orig_behind, delta_x["lo"])
        self.place_dest_lane_vehicles_around_platoon(
            v_ff_d, delta_x, n_dest_ahead, n_dest_behind)
        # analysis.plot_initial_state_vector(
        #     self.vehicle_group.get_full_initial_state_vector())

    def set_free_flow_speeds(self,
                             free_flow_speeds: Union[float, list, np.ndarray]):
        self.vehicle_group.set_free_flow_speeds(free_flow_speeds)

    def save_response_data(self, file_name: str) -> None:
        """
        Pickles time, inputs and states as a dataframe
        :param file_name:
        :return:
        """
        with open(file_name, "wb") as f:
            pickle.dump(self.response_to_dataframe(),
                        f, pickle.HIGHEST_PROTOCOL)

    def place_equally_spaced_vehicles(self, gap: float = None):
        """
        All vehicles start at the center of their respective lanes, with
        orientation angle zero, and at the same speed, which equals their
        desired free-flow speed. Vehicles on the same lane are "gap" meters
        distant from each other. Note: method always starts populating the
        scenario from the right-most lane, front-most vehicle.

        :param gap: Inter-vehicle distance for vehicles on the same lane. If
         the value is not given, it defaults to v_ff + 1
        :return:
        """
        if gap is None:
            gap = self.vehicle_group.vehicles[0].get_free_flow_speed() + 1
        x0, y0, theta0, v0 = [], [], [], []
        for lane in range(len(self.n_per_lane)):
            lane_center = lane * configuration.LANE_WIDTH
            n = self.n_per_lane[lane]
            for i in range(n):
                x0.append(gap * (n - i - 1))
                y0.append(lane_center)
                theta0.append(0.0)
        v0 = self.vehicle_group.get_free_flow_speeds()
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)
        # self.initial_state =
        # self.vehicle_group.get_full_initial_state_vector()

    def response_to_dataframe(self) -> pd.DataFrame:
        return self.vehicle_group.to_dataframe()

    def create_simulation_time_steps(self, final_time):
        self.final_time = final_time
        return np.arange(0, self.final_time + self.dt, self.dt)

    @abstractmethod
    def run(self, final_time):
        """

        :param final_time:
        :return:
        """
        pass


class LaneChangeScenario(SimulationScenario):

    def __init__(self, n_platoon: int,
                 are_vehicles_cooperative: bool = False):
        if n_platoon < 1:
            raise ValueError("Scenario must have at least one platoon vehicle")

        super().__init__()
        self._lc_intention_time = 1.0
        self._n_platoon = n_platoon
        self._are_vehicles_cooperative = are_vehicles_cooperative
        self._allow_early_termination = False

    def get_opc_results_summary(self):
        # We assume there's either a single optimally controlled vehicles
        # or that they all share the same controller
        try:
            opc_vehicle = self.vehicle_group.get_optimal_control_vehicles()[0]
        except IndexError:
            raise AttributeError  # no optimal control vehicles in this group
        return (opc_vehicle.get_opt_controller().get_running_cost_history(),
                opc_vehicle.get_opt_controller().get_terminal_cost_history())

    def get_opc_cost_history(self):
        try:
            opc_vehicle = self.vehicle_group.get_optimal_control_vehicles()[0]
        except IndexError:
            raise AttributeError  # no optimal control vehicles in this group
        return (opc_vehicle.get_opt_controller().get_running_cost_history(),
                opc_vehicle.get_opt_controller().get_terminal_cost_history())

    def set_allow_early_termination(self, value: bool) -> None:
        self._allow_early_termination = value

    def set_lc_intention_time(self, value):
        self._lc_intention_time = value

    def set_single_gap_initial_state(
            self, gap_position: int, v_ref: Mapping[str, float],
            delta_x_lo: float):
        base.BaseVehicle.reset_vehicle_counter()
        v_orig_leader = v_ref["orig"]
        v_dest_leader = v_ref["dest"]
        v_platoon = v_ref["platoon"]
        delta_x_ld = (v_orig_leader - v_dest_leader) * self._lc_intention_time
        self.place_origin_lane_vehicles(
            v_orig_leader, v_platoon, self._is_acceleration_optimal,
            delta_x_lo=delta_x_lo)
        self.place_dest_lane_vehicles_with_single_gap(
            v_dest_leader, gap_position, self._is_acceleration_optimal,
            delta_x_ld)
        # analysis.plot_state_vector(
        #     self.vehicle_group.get_full_initial_state_vector())

    def place_dest_lane_vehicles_with_single_gap(
            self, v_dest: float, gap_position: int,
            is_acceleration_optimal: bool, delta_x: float = None):
        """
        Creates destination lanes on position spanning from ahead the origin
        lane leader to behind the origin lane follower. One vehicle is removed
        to create a suitable lane change gap
        :param v_dest: Destination lane free-flow speed
        :param gap_position: Where, in relation to the platoon vehicles, is
         the gap. If i=0: aligned with lo; i=platoon size + 1: aligned with fo;
         otherwise: aligned with platoon veh i
        :param is_acceleration_optimal: If optimal control vehicles"
         acceleration is also an optimal control input
        :param delta_x: Deviation of the gap"s position (and by consequence
         all vehicles) from being perfectly aligned with vehicle i. If None,
         delta_x is set such that, at lane change time, the gap is aligned
         with vehicle i.
        :return:
        """
        # Number of vehs ahead and behind the platoon in the dest lane. This
        # choice ensures at least one dest lane vehicle ahead of lo and another
        # behind fd
        n_ahead = 1 + gap_position
        n_behind = gap_position + self._n_platoon + 1

        # We make the gap align with one of the origin lane vehicles
        if gap_position == 0:
            center_vehicle = self.vehicle_group.get_vehicle_by_name("lo0")
        elif gap_position > self._n_platoon:
            center_vehicle = self.vehicle_group.get_vehicle_by_name("fo0")
        else:
            center_vehicle = self.vehicle_group.get_vehicle_by_name(
                "p" + str(gap_position))

        if delta_x is None:
            delta_x = ((center_vehicle.get_vel() - v_dest)
                       * self._lc_intention_time)

        # assuming uniform vehicles:
        reference_gap = center_vehicle.compute_non_connected_reference_gap(
            v_dest)

        # min_gap_to_decelerate = (
        #         (center_vehicle.get_vel() ** 2 - v_dest ** 2)
        #         / 2 / np.abs(center_vehicle.brake_comfort_max)
        #         )
        # gap_to_leader = max(reference_gap, min_gap_to_decelerate)
        safe_gap_leader = center_vehicle.compute_safe_lane_change_gap(
            configuration.SAFE_TIME_HEADWAY, v_dest,
            center_vehicle.brake_max, is_other_ahead=True
        )
        gap_to_leader = safe_gap_leader
        safe_gap_follower = center_vehicle.compute_safe_lane_change_gap(
            configuration.SAFE_TIME_HEADWAY, v_dest,
            center_vehicle.brake_max, is_other_ahead=False
        )
        gap_to_follower = safe_gap_follower
        x_gap = center_vehicle.get_x() + delta_x
        # x0 = x_gap + n_ahead * reference_gap

        # Destination lane vehicles
        ld_counter = 0
        fd_counter = 0
        pN = self.vehicle_group.get_vehicle_by_name("p" + str(self._n_platoon))
        # print(f"x_gap={x_gap:.1f}")
        x0 = x_gap + gap_to_leader + 0.1  # good to have some margin
        for n in range(n_ahead):
            veh = fsv.ClosedLoopVehicle(False, is_acceleration_optimal,
                                        self._are_vehicles_cooperative)
            veh_name = "ld" + str(n)
            veh.set_name(veh_name)
            veh.set_free_flow_speed(v_dest)
            veh.set_initial_state(x0, configuration.LANE_WIDTH, 0., v_dest)
            self.vehicle_group.add_vehicle(veh)
            x0 += reference_gap

        x0 = x_gap - gap_to_follower - 0.1
        for n in range(n_behind):
            veh = fsv.ClosedLoopVehicle(False, is_acceleration_optimal,
                                        self._are_vehicles_cooperative)
            veh_name = "fd" + str(n)
            veh.set_name(veh_name)
            veh.set_free_flow_speed(v_dest)
            veh.set_initial_state(x0, configuration.LANE_WIDTH, 0., v_dest)
            self.vehicle_group.add_vehicle(veh)
            x0 -= reference_gap
        self.n_per_lane.append(n_ahead + n_behind)

    def set_lane_change_strategy(self,
                                 platoon_strategy: lc_strategy.StrategyMap):
        self._platoon_strategy = platoon_strategy
        # self.vehicle_group.set_platoon_lane_change_strategy(
        #     platoon_strategy)

    def make_control_centralized(self):
        self.vehicle_group.centralize_control()

    def reset_platoons(self):
        self.vehicle_group.reset_platoons()

    def run(self, final_time: float,
            expected_states_at_lc_time: Mapping[str, np.ndarray] = None
            ) -> None:
        """
        :param final_time: Simulation max time. The simulation might end
         earlier if the platoon "misses" the intended lane change gap
        :param expected_states_at_lc_time: For debugging purposes. We force
         the vehicle states to equal this value at the lane change intention
         time.
        """
        sim_time = self.create_simulation_time_steps(final_time)
        for i in range(len(sim_time) - 1):
            if np.abs(sim_time[i] - self._lc_intention_time) < self.dt / 10:
                self.vehicle_group.set_vehicles_lane_change_direction(
                    1, self.lc_vehicle_names)
            # There's a one dt delay between setting the intention and starting
            # the procedure for the maneuver
            elif (expected_states_at_lc_time is not None
                  and (np.abs(sim_time[i] - (self._lc_intention_time + self.dt))
                       < self.dt / 10)):
                self.vehicle_group.force_state(expected_states_at_lc_time)
            self.vehicle_group.simulate_one_time_step(sim_time[i + 1],
                                                      detect_collision=True)
            # Early termination conditions
            if self.vehicle_group.is_platoon_out_of_range():
                print(f"Platoon out of simulation range at {sim_time[i]}")
                self.vehicle_group.truncate_simulation_history()
                break
            if (self._allow_early_termination
                    and sim_time[i] > self._lc_intention_time + self.dt
                    and self.vehicle_group.are_all_at_target_lane_center()):
                print(f"Lane change finished at {sim_time[i]}")
                self.vehicle_group.truncate_simulation_history()
                break


class LaneChangeWithClosedLoopControl(LaneChangeScenario):
    _platoon_lc_strategy: lc_strategy.StrategyMap

    def run(self, final_time: float,
            expected_states_at_lc_time: Mapping[str, np.ndarray] = None
            ) -> None:
        pass
        # self.vehicle_group.prepare_to_start_simulation(
        #     len(time), self._platoon_lc_strategy)


class LaneChangeWithOptimalControl(LaneChangeScenario):
    pass

class AllLaneChangeStrategies(LaneChangeScenario):
    """
    Class to run all possible one-by-one strategies and pick the best
    """

    def __init__(self, n_platoon: int, n_orig_ahead: int, n_orig_behind: int,
                 n_dest_ahead: int, n_dest_behind: int,
                 v_ref: Mapping[str, float], delta_x: Mapping[str, float],
                 are_vehicles_cooperative: bool = False):
        super().__init__(n_platoon, are_vehicles_cooperative)
        self._noa = n_orig_ahead
        self._nob = n_orig_behind
        self._nda = n_dest_ahead
        self._ndb = n_dest_behind
        self.v_ref = v_ref
        self.delta_x = delta_x
        self.set_control_type("closed_loop", is_acceleration_optimal=False)

        self.named_strategies_positions = {"LdF": -1, "LVF": -1, "LdFR": -1}
        self.best_strategy = {"merging_order": [], "coop_order": []}
        self.costs = []
        self.completion_times = []
        self.accel_costs = []

    def run(self, final_time,
            expected_states_at_lc_time: Mapping[str, np.ndarray] = None
            ) -> None:
        sg = lc_strategy.StrategyGenerator()
        all_positions = [i for i in range(self._n_platoon)]

        ldf_lc_order = all_positions
        ldf_coop_order = [-1] * self._n_platoon
        lvf_lc_order = all_positions[::-1]
        lvf_coop_order = [-1] + lvf_lc_order[:1]
        ldfr_lc_order = all_positions
        ldfr_coop_order = [-1] + ldfr_lc_order[:-1]

        remaining_vehicles = set(all_positions)
        success = []
        best_cost = np.inf
        best_result = self.vehicle_group
        counter = 0
        for i in range(self._n_platoon):
            print("Starting with veh", i)
            for merging_order, coop_order in sg.generate_order_all(
                    i, [], [], remaining_vehicles.copy()):
                if (merging_order == ldf_lc_order
                        and coop_order == ldf_coop_order):
                    self.named_strategies_positions["LdF"] = counter
                elif (merging_order == lvf_lc_order
                        and coop_order == lvf_coop_order):
                    self.named_strategies_positions["LVF"] = counter
                elif (merging_order == ldfr_lc_order
                        and coop_order == ldfr_coop_order):
                    self.named_strategies_positions["LdFR"] = counter

                counter += 1
                base.BaseVehicle.reset_vehicle_counter()
                self.vehicle_group = vg.VehicleGroup()
                self.set_test_initial_state(self._nda, self._ndb, self._noa,
                                            self._nob, self.v_ref, self.delta_x)
                strategy = lc_strategy.TemplateStrategy()
                strategy.set_maneuver_order(merging_order, coop_order)
                self.vehicle_group.set_platoon_lane_change_strategy(
                    lc_strategy.StrategyMap.template)
                self.vehicle_group.set_predefined_lane_change_order(
                    merging_order, coop_order)
                self.vehicle_group.set_verbose(False)
                LaneChangeScenario.run(self, final_time)

                # data = self.vehicle_group.to_dataframe()
                # analysis.plot_trajectory(data, "#" + str(counter))
                # analysis.plot_platoon_lane_change(data)

                # ============ Computing cost ============== #
                # TODO: a mess for now

                n_states = self.vehicle_group.get_n_states()
                n_inputs = self.vehicle_group.get_n_inputs()
                desired_input = np.zeros(n_inputs)  # TODO: hard codded

                all_vehicles = self.vehicle_group.get_all_vehicles_in_order()
                controlled_veh_ids = set(self.get_lc_vehicle_ids())
                desired_state = occ.create_desired_state(all_vehicles,
                                                         final_time)
                q_matrix = occ.create_state_cost_matrix(
                    all_vehicles, controlled_veh_ids,
                    x_cost=0.0, y_cost=0.2, theta_cost=0.0, v_cost=0.1)
                r_matrix = occ.create_input_cost_matrix(
                    all_vehicles, controlled_veh_ids,
                    accel_cost=0.1, phi_cost=0.1)
                running_cost = occ.quadratic_cost(
                    n_states, n_inputs, q_matrix, r_matrix,
                    desired_state, desired_input
                )
                q_terminal = occ.create_state_cost_matrix(
                    all_vehicles, controlled_veh_ids, y_cost=100.,
                    theta_cost=1.)
                r_terminal = occ.create_input_cost_matrix(
                    all_vehicles, controlled_veh_ids, phi_cost=0.)
                terminal_cost = occ.quadratic_cost(
                    n_states, n_inputs, q_terminal, r_terminal,
                    desired_state, desired_input
                )
                cost_with_tracker = occ.OCPCostTracker(
                    np.array([0]), n_states,
                    running_cost, terminal_cost,
                    config.solver_max_iter
                )

                success.append(self.vehicle_group.check_lane_change_success())
                r_cost, t_cost = cost_with_tracker.compute_simulation_cost(
                    self.vehicle_group.get_all_states(),
                    self.vehicle_group.get_all_inputs(),
                    self.vehicle_group.get_simulated_time())
                self.costs.append(r_cost + t_cost)
                if self.costs[-1] < best_cost:
                    best_cost = self.costs[-1]
                    self.best_strategy["merging_order"] = merging_order[:]
                    self.best_strategy["coop_order"] = coop_order[:]
                    best_result = self.vehicle_group

                # print(
                #     f"Strategy #{counter}. "
                #     f"Order and coop: {merging_order}, {coop_order}"
                #     f"\nSuccessful? {success[-1]}. "
                #     f"Cost: {r_cost:.2f}(running) + {t_cost:.2f}(terminal) = "
                #     f"{r_cost + t_cost:.2f}"
                #     )
                # ========================================= #

                # ============= Other costs =============== #
                self.completion_times.append(
                    np.max(self.vehicle_group.get_lc_end_times()))
                self.accel_costs.append(
                    sum(pp.compute_acceleration_costs(
                        self.vehicle_group.to_dataframe()).values())
                )

        self.vehicle_group = best_result
        print(f"{sg.counter} strategies tested.\n"
              f"Success rate: {sum(success) / sg.counter * 100}%\n"
              f"Best strategy: cost={best_cost:.2f}, "
              f"merging order={self.best_strategy['merging_order']}, "
              f"coop order={self.best_strategy['coop_order']}")


class ExternalOptimalControlScenario(SimulationScenario, ABC):
    controller: opt_ctrl.VehicleOptimalController
    ocp_response: ct.TimeResponseData

    def __init__(self):
        super().__init__()
        self.tf: float = 0.0

    def get_opc_results_summary(self):
        return (self.controller.get_running_cost_history(),
                self.controller.get_terminal_cost_history())

    def get_opc_cost_history(self):
        return (self.controller.get_running_cost_history(),
                self.controller.get_terminal_cost_history())

    def set_desired_final_states(self, tf: float):
        """ Sets the final time and desired final states """
        self.tf = tf
        self.vehicle_group.prepare_to_start_simulation(1)
        self.set_desired_lane_changes()

    def solve(self):
        time_start = time.time()
        self.vehicle_group.update_surrounding_vehicles()
        self.controller = opt_ctrl.VehicleOptimalController()
        self.controller.set_time_horizon(self.tf)
        self.controller.set_controlled_vehicles_ids(
            [self.vehicle_group.get_vehicle_id_by_name(veh_name) for veh_name
             in self.lc_vehicle_names])
        self.controller.find_trajectory(self.vehicle_group.vehicles)
        solve_time = datetime.timedelta(seconds=time.time() - time_start)
        print("solve time:", str(solve_time).split(".")[0])
        # return self.controller.ocp_result


    def run_ocp_solution(self) -> None:
        """
        Calls the control libraries function for running the dynamic system
        given the optimal control problem solution
        :return: Nothing. Results are stored internally
        """
        self.ocp_response = self.controller.get_ocp_response()

    def run(self, final_time: float):
        """
        Given the optimal control problem solution, runs the open loop system.
        Difference to method "run" is that we directly (re)simulate the dynamics
        in this case. For debugging purposes
        """
        # It is good to run our simulator with the ocp solution and to confirm
        # it yields the same response as the control library simulation
        self.run_ocp_solution()
        result = self.controller.ocp_result

        time = self.create_simulation_time_steps(final_time)
        veh_ids = self.vehicle_group.sorted_vehicle_ids
        self.vehicle_group.prepare_to_start_simulation(len(time))
        for i in range(len(time) - 1):
            current_inputs = self.controller.get_input(time[i], veh_ids)
            self.vehicle_group.simulate_one_time_step(time[i + 1],
                                                      current_inputs)

    @abstractmethod
    def set_desired_lane_changes(self):
        pass


class LaneChangeWithExternalController(ExternalOptimalControlScenario):
    """
    Used to test how to code safety constraints
    """

    def __init__(self, n_platoon: int,
                 are_vehicles_cooperative: bool = False):
        super().__init__()
        self._n_platoon = n_platoon
        self._are_vehicles_cooperative = are_vehicles_cooperative

    def create_initial_state(
            self, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int,
            v_ref: Mapping[str, float], delta_x: Mapping[str, float],
            is_acceleration_optimal: bool = True
    ):
        self.set_control_type("open_loop", is_acceleration_optimal)
        self.set_test_initial_state(n_dest_ahead, n_dest_behind, n_orig_ahead,
                                    n_orig_behind, v_ref, delta_x)
        # self.create_base_lane_change_initial_state(self._has_lo, self._has_fo,
        #                                            self._has_ld, self._has_fd)

    def set_desired_lane_changes(self):
        self.vehicle_group.set_vehicles_lane_change_direction(
            1, self.lc_vehicle_names)


# ================================ OLD TESTS ================================= #

class VehicleFollowingScenario(SimulationScenario):
    """
    Scenario to test acceleration feedback laws. No lane changes.
    """

    def __init__(self, n_vehs: int):
        super().__init__()
        vehicles = [fsv.ClosedLoopVehicle(can_change_lanes=False)]
        # vehicle_classes = [fsv.SafeLongitudinalVehicle] * n_vehs
        v_ff = [10] + [12] * n_vehs
        self.n_per_lane = [n_vehs]
        self.vehicle_group.fill_vehicle_array(vehicles)
        # self.vehicle_group.create_vehicle_array_from_classes(vehicle_classes)
        self.vehicle_group.set_free_flow_speeds(v_ff)

    def get_opc_results_summary(self):
        raise AttributeError("Scenario does not have optimal control")

    def get_opc_cost_history(self):
        raise AttributeError("Scenario does not have optimal control")

    def create_initial_state(self):
        gap = 2
        self.place_equally_spaced_vehicles(gap)
        print(self.vehicle_group.get_full_initial_state_vector())

    def run(self, final_time):
        """

        :param final_time: Total simulation time
        :return:
        """
        time = self.create_simulation_time_steps(final_time)
        self.vehicle_group.prepare_to_start_simulation(len(time))
        # self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            self.vehicle_group.simulate_one_time_step(time[i + 1])


class FastLaneChange(SimulationScenario):

    def __init__(self):
        super().__init__()
        # vehicle_classes = [fsv.SafeAccelOpenLoopLCVehicle]
        # self.vehicle_group.create_vehicle_array_from_classes(vehicle_classes)
        vehicles = [fsv.OpenLoopVehicle(can_change_lanes=True,
                                        has_open_loop_acceleration=False)]
        self.n_per_lane = [len(vehicles)]
        self.vehicle_group.fill_vehicle_array(vehicles)
        self.create_initial_state()

    def get_opc_results_summary(self):
        raise AttributeError("Scenario does not have optimal control")

    def get_opc_cost_history(self):
        raise AttributeError("Scenario does not have optimal control")

    def create_initial_state(self):
        v_ff = 10
        self.vehicle_group.set_free_flow_speeds(v_ff)

        # Initial states
        x0 = [0.]
        y0 = [0.]
        theta0 = [0.]
        v0 = [v_ff]
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)

    def run(self, final_time):
        """

        :param final_time: Total simulation time
        :return:
        """
        time = self.create_simulation_time_steps(final_time)
        ego = self.vehicle_group.vehicles[0]
        inputs = {}
        self.vehicle_group.prepare_to_start_simulation(len(time))
        # self.vehicle_group.update_surrounding_vehicles()
        t_c = 1.08
        for i in range(len(time) - 1):
            if time[i] <= t_c:  # ego.get_y() <= 2 * LANE_WIDTH / 3:
                phi = ego.phi_max
            elif time[i] <= 2 * t_c:  # np.abs(ego.get_theta()) >= 1e-4:
                phi = -ego.phi_max
            else:
                phi = 0
            inputs[ego.get_id()] = np.array([phi])
            self.vehicle_group.simulate_one_time_step(time[i + 1], inputs)


class ExampleScenarioExternal(ExternalOptimalControlScenario):
    """
    Two-lane scenario where all vehicles want to perform a lane change starting
    at t=0 and ending at tf.
    The scenario is used for testing the different dynamical models and
    variations in parameters
    """

    def create_initial_state(self):
        self.place_equally_spaced_vehicles()
        print(self.vehicle_group.get_full_initial_state_vector())

    def set_desired_lane_changes(self):
        """
        In this scenario, all vehicles try to perform a lane change

        :return:
        """
        for veh in self.vehicle_group.get_all_vehicles():
            lane = veh.get_current_lane()
            veh.set_lane_change_direction((-1) ** lane)
