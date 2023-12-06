from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
import os
import pickle
from typing import Union

import control as ct
import numpy as np
import pandas as pd

import analysis
import controllers.optimal_controller as opt_ctrl
import controllers.optimal_control_costs as occ
import configuration
import graph_tools
import platoon_lane_change_strategies as lc_strategy
import post_processing as pp
from vehicle_group import VehicleGroup
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv

config = configuration.Configuration


class LaneChangeScenarioManager:

    # scenario: LaneChangeScenario
    v_ref: Mapping[str, float]
    delta_x: Mapping[str, float]
    n_platoon: int
    are_vehicles_cooperative: bool
    _lane_change_graph: graph_tools.VehicleStatesGraph

    def __init__(self):
        # TODO: names should depend on the scenario
        self.trajectory_file_name = 'trajectory_data.pickle'
        self.cost_file_name = 'cost_data.pickle'
        self.results: dict[str, list] = {
            'n_platoon': [], 'vo': [], 'vd': [], 'strategy': [],
            'lane_change_order': [], 'cooperation_order': [],
            'success': [], 'completion_time': [], 'accel_cost': []
        }
        self._has_plots = True

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def set_parameters(self, n_platoon: int, are_vehicles_cooperative: bool,
                       v_ref: Mapping[str, float],
                       delta_x: Mapping[str, float] = None):
        self.n_platoon = n_platoon
        self.are_vehicles_cooperative = are_vehicles_cooperative
        self.v_ref = v_ref
        if delta_x is None:
            delta_x = {'lo': 0., 'fo': 0., 'ld': 0., 'fd': 0.}
        self.delta_x = delta_x

    def set_lane_change_graph(
            self, lane_change_graph: graph_tools.VehicleStatesGraph):
        self._lane_change_graph = lane_change_graph

    def set_plotting(self, value: bool):
        self._has_plots = value

    def run_strategy_comparison_on_test_scenario(
            self, n_orig_ahead: int, n_orig_behind: int, n_dest_ahead: int,
            n_dest_behind: int, strategy_numbers: Iterable[int]):
        tf = configuration.Configuration.time_horizon
        for sn in strategy_numbers:
            scenario_name = self._create_scenario_name(sn)
            scenario = self.create_closed_loop_test_scenario(
                n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind, sn)
            self.run_save_and_plot(scenario, tf, scenario_name)

    def run_strategy_comparison_on_single_gap_scenario(
            self, strategy_numbers: Iterable[int]):
        for sn in strategy_numbers:
            self.run_all_single_gap_cases(sn)

    def run_all_single_gap_cases(self, strategy_number: int):
        tf = configuration.Configuration.time_horizon
        first_gap = 1
        last_gap = self.n_platoon
        if self.v_ref['lo'] > self.v_ref['ld']:
            first_gap = 0
        elif self.v_ref['lo'] < self.v_ref['ld']:
            last_gap = self.n_platoon + 1
        scenario_name = self._create_scenario_name(strategy_number)
        for gap_position in range(first_gap, last_gap + 1):
            print(f'      gap position={gap_position}')
            scenario = self.initialize_closed_loop_scenario(strategy_number)
            self.set_single_gap_initial_state(scenario, gap_position)
            self.run_save_and_plot(scenario, tf,
                                   scenario_name + f' gap at {gap_position}')

    def initialize_closed_loop_scenario(self, strategy_number: int
                                        ) -> LaneChangeScenario:
        scenario = LaneChangeScenario(self.n_platoon,
                                      self.are_vehicles_cooperative)
        scenario.set_control_type('closed_loop',
                                  is_acceleration_optimal=False)
        scenario.set_lane_change_strategy(strategy_number)
        if strategy_number == lc_strategy.GraphStrategy.get_id():
            scenario.set_lane_change_states_graph(self._lane_change_graph)
        return scenario

    def initialize_optimal_control_scenario(self, is_acceleration_optimal: bool
                                            ) -> LaneChangeScenario:
        scenario = LaneChangeScenario(self.n_platoon,
                                      self.are_vehicles_cooperative)
        scenario.set_control_type('optimal', is_acceleration_optimal)
        return scenario

    def set_test_scenario_initial_state(
            self, scenario: LaneChangeScenario, n_orig_ahead: int,
            n_orig_behind: int, n_dest_ahead: int, n_dest_behind: int):
        scenario.set_test_initial_state(
            n_dest_ahead, n_dest_behind, n_orig_ahead, n_orig_behind,
            self.v_ref, self.delta_x)

    def set_single_gap_initial_state(
            self, scenario: LaneChangeScenario, gap_position: int):
        scenario.set_single_gap_initial_state(gap_position, self.v_ref,
                                              self.delta_x['lo'])

    def create_closed_loop_test_scenario(
            self, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int, strategy_number: int
    ) -> LaneChangeScenario:
        scenario = self.initialize_closed_loop_scenario(strategy_number)
        self.set_test_scenario_initial_state(
            scenario, n_orig_ahead, n_orig_behind, n_dest_ahead, n_dest_behind)
        return scenario

    def create_closed_loop_single_gap_scenario(
            self, strategy_number: int, gap_position: int
    ) -> LaneChangeScenario:
        scenario = self.initialize_closed_loop_scenario(strategy_number)
        self.set_single_gap_initial_state(scenario, gap_position)
        return scenario

    def create_optimal_control_test_scenario(
            self, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int,
            is_acceleration_optimal: bool):
        tf = configuration.Configuration.time_horizon + 2
        scenario = self.initialize_optimal_control_scenario(
            is_acceleration_optimal)
        self.set_test_scenario_initial_state(
            scenario, n_dest_ahead, n_dest_behind, n_orig_ahead, n_orig_behind)
        return scenario

    def run_save_and_plot(self, scenario: LaneChangeScenario, tf: float,
                          scenario_name: str = None):

        scenario.vehicle_group.set_verbose(False)
        scenario.run(tf)

        self.store_results(scenario)
        scenario.save_response_data(self.trajectory_file_name)

        if self._has_plots:
            data = scenario.response_to_dataframe()
            analysis.plot_trajectory(data, scenario_name)
            if scenario.get_n_platoon() < 1:
                analysis.plot_constrained_lane_change(data, 'p1')
            else:
                analysis.plot_platoon_lane_change(data)

            try:
                scenario.save_cost_data(self.cost_file_name)
                running_cost, terminal_cost = scenario.get_opc_cost_history()
                analysis.plot_costs_vs_iteration(running_cost, terminal_cost)
            except AttributeError:
                pass

    def store_results(self, scenario: LaneChangeScenario):
        strategy = scenario.vehicle_group.get_platoon_lane_change_strategy()
        if strategy is not None:
            strategy_name = strategy.get_name()
            lane_change_order = strategy.get_lane_change_order()
            cooperation_order = strategy.get_cooperation_order()
        else:
            print('### Dealing with a scenario without strategy ###\n'
                  '### Might have to rethink this part ###')
            strategy_name = 'none'
            lane_change_order = []
            cooperation_order = []
        completion_time = np.max(scenario.vehicle_group.get_lc_end_times())
        accel_cost = scenario.vehicle_group.compute_acceleration_cost()

        result = {
            'n_platoon': scenario.get_n_platoon(),
            'vo': self.v_ref['lo'], 'vd': self.v_ref['ld'],
            'strategy': strategy_name,
            'lane_change_order': lane_change_order,
            'cooperation_order': cooperation_order,
            'success': scenario.vehicle_group.check_lane_change_success(),
            'completion_time': completion_time, 'accel_cost': accel_cost
        }
        for key, value in result.items():
            self.results[key].append(value)

    def append_results_to_csv(self):
        file_name = 'result_summary.csv'
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 'platoon_strategy_results', file_name)
        try:
            results_history = pd.read_csv(file_path)
            experiment_counter = results_history['experiment_counter'].max() + 1
            write_header = False
        except FileNotFoundError:
            experiment_counter = 0
            write_header = True
        current_results = self.get_results()
        current_results['experiment_counter'] = experiment_counter
        current_results.to_csv(file_path, mode='a', index=False,
                               header=write_header)
        print(f'File {file_name} saved')

    def _create_scenario_name(self, strategy_number: int):
        scenario_name = lc_strategy.strategy_map[strategy_number].get_name()
        if self.v_ref['lo'] > self.v_ref['ld']:
            scenario_name += ' vo > vd'
        elif self.v_ref['lo'] < self.v_ref['ld']:
            scenario_name += ' vo < vd'
        else:
            scenario_name += ' vo = vd'
        return scenario_name


class SimulationScenario(ABC):

    _str_to_vehicle_type = {
        'optimal': fsv.OptimalControlVehicle,
        'closed_loop': fsv.ClosedLoopVehicle,
        'open_loop': fsv.OpenLoopVehicle
    }

    _is_acceleration_optimal: bool
    _lc_veh_type: type[fsv.FourStateVehicle]
    _n_platoon: int

    def __init__(self):
        self.n_per_lane: list[int] = []
        base.BaseVehicle.reset_vehicle_counter()
        self.vehicle_group: VehicleGroup = VehicleGroup()
        self.result_summary: dict = {}
        self._are_vehicles_cooperative = False
        self.lc_vehicle_names = []

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
                warnings.warn('Cannot set optimal acceleration True when the '
                              'vehicles only have feedback controllers')
        elif is_acceleration_optimal is None:
            is_acceleration_optimal = True
        self._is_acceleration_optimal = is_acceleration_optimal

    def save_cost_data(self, file_name: str) -> None:
        """
        Pickles running and terminal costs 2D lists
        :param file_name:
        :return:
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.get_opc_cost_history(),
                        f, pickle.HIGHEST_PROTOCOL)

    def create_uniform_vehicles(
            self, n_per_lane: Iterable[int],
            vehicle_class: type[base.BaseVehicle],
            free_flow_speed: float):
        array_2d = [[vehicle_class] * n for n in n_per_lane]
        self.create_vehicle_group(array_2d)
        self.vehicle_group.set_free_flow_speeds(free_flow_speed)

    def create_vehicle_group(
            self, vehicle_classes: Sequence[Sequence[type[base.BaseVehicle]]]):
        for i in range(len(vehicle_classes)):
            self.n_per_lane.append(len(vehicle_classes[i]))
        flat_vehicle_list = [item for sublist in vehicle_classes for
                             item in sublist]
        self.vehicle_group.create_vehicle_array_from_classes(flat_vehicle_list)

    def place_origin_lane_vehicles(
            self, v_ff_lo: float, v_ff_platoon: float,
            is_acceleration_optimal: bool, n_orig_ahead: int = 1,
            n_orig_behind: int = 1, delta_x_lo: float = 0,
            ):
        """
        Creates n_platoon vehicles plus an origin lane leader and inserts them
        in self's vehicle group.
        :param v_ff_lo: Origin lane leader's free-flow speed
        :param v_ff_platoon: Platoon's free-flow speed
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
            p_i.set_name('p' + str(i + 1))
            p_i.set_free_flow_speed(v_ff_platoon)
            p_i.set_initial_state(x0, y_orig, 0., v0_platoon)
            self.vehicle_group.add_vehicle(p_i)
            x0 -= p_i.compute_lane_keeping_desired_gap(v0_platoon)

        # Origin lane ahead
        p1 = self.vehicle_group.get_vehicle_by_name('p1')
        x0 = p1.get_x() + p1.compute_lane_keeping_desired_gap() - delta_x_lo
        v0_lo = v_ff_lo
        for i in range(n_orig_ahead):
            lo = fsv.ClosedLoopVehicle(False, False,
                                       self._are_vehicles_cooperative)
            lo.set_name('lo' + str(i))
            lo.set_free_flow_speed(v_ff_lo)
            lo.set_initial_state(x0, y_orig, 0., v0_lo)
            self.vehicle_group.add_vehicle(lo)
            x0 += lo.compute_lane_keeping_desired_gap(v0_lo)

        # Origin lane behind
        x0 = self.vehicle_group.get_vehicle_by_name(
            'p' + str(self._n_platoon)).get_x()
        for i in range(n_orig_behind):
            veh = fsv.ClosedLoopVehicle(False, False,
                                        self._are_vehicles_cooperative)
            veh.set_name('fo' + str(i))
            veh.set_free_flow_speed(v_ff_lo)
            x0 -= veh.compute_lane_keeping_desired_gap(v0_lo)
            veh.set_initial_state(x0, y_orig, 0., v0_lo)
            self.vehicle_group.add_vehicle(veh)

        self.n_per_lane = [self._n_platoon + n_orig_behind + n_orig_behind]

    def place_dest_lane_vehicles_around_platoon(
            self, v_ff: float, delta_x: Mapping[str, float],
            n_ahead: int, n_behind: int):
        y_dest = configuration.LANE_WIDTH
        p1 = self.vehicle_group.get_vehicle_by_name('p1')
        x0 = p1.get_x() + p1.compute_lane_keeping_desired_gap() - delta_x['ld']
        for i in range(n_ahead):
            veh = fsv.ClosedLoopVehicle(False, False,
                                        self._are_vehicles_cooperative)
            veh.set_name('ld' + str(i))
            veh.set_free_flow_speed(v_ff)
            veh.set_initial_state(x0, y_dest, 0., v_ff)
            self.vehicle_group.add_vehicle(veh)
            x0 += veh.compute_lane_keeping_desired_gap(v_ff)

        x0 = self.vehicle_group.get_vehicle_by_name(
            'p' + str(self._n_platoon)).get_x() + delta_x['fd']
        for i in range(n_behind):
            veh = fsv.ClosedLoopVehicle(False, False,
                                        self._are_vehicles_cooperative)
            veh.set_name('fd' + str(i))
            veh.set_free_flow_speed(v_ff)
            x0 -= veh.compute_lane_keeping_desired_gap(v_ff)
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

        v_ff_lo = v_ref['lo']
        v_ff_platoon = v_ref['p']
        v_ff_d = v_ref['ld']
        self.place_origin_lane_vehicles(
            v_ff_lo, v_ff_platoon, self._is_acceleration_optimal,
            n_orig_ahead, n_orig_behind, delta_x['lo'])
        self.place_dest_lane_vehicles_around_platoon(
            v_ff_d, delta_x, n_dest_ahead, n_dest_behind)
        # analysis.plot_initial_state_vector(
        #     self.vehicle_group.get_full_initial_state_vector())

    # def _create_vehicles_for_test_scenario(
    #         self, n_orig_ahead: int, n_orig_behind: int, n_dest_ahead: int,
    #         n_dest_behind: int):
    #
    #     coop = self._are_vehicles_cooperative
    #     vehicles_ahead = [
    #         fsv.ClosedLoopVehicle(can_change_lanes=False,
    #                               is_connected=coop) for _
    #         in range(n_orig_ahead)]
    #     lc_vehs = [self._lc_veh_type(
    #         can_change_lanes=True,
    #         has_open_loop_acceleration=self._is_acceleration_optimal,
    #         is_connected=True
    #     ) for _ in range(self._n_platoon)]
    #     vehicles_behind = [
    #         fsv.ClosedLoopVehicle(can_change_lanes=False,
    #                               is_connected=coop) for _
    #         in range(n_orig_behind)]
    #     orig_lane_vehs = vehicles_ahead + lc_vehs + vehicles_behind
    #     dest_lane_vehs = [
    #         fsv.ClosedLoopVehicle(can_change_lanes=False,
    #                               is_connected=coop) for _
    #         in range(n_dest_ahead + n_dest_behind)]
    #     # dest_lane_vehs = [
    #     #     fsv.OptimalControlVehicle(can_change_lanes=False,
    #     #                               is_connected=are_vehicles_cooperative)
    #     #     for _ in range(n_dest_ahead + n_dest_behind)]
    #     self.n_per_lane = [len(orig_lane_vehs), len(dest_lane_vehs)]
    #     self.vehicle_group.fill_vehicle_array(orig_lane_vehs + dest_lane_vehs)
    #     self.lc_vehicle_names = ['p' + str(i) for i
    #                              in range(1, self._n_platoon + 1)]
    #     vehicle_names = (
    #             ['lo' + str(i) for i in range(n_orig_ahead, 0, -1)]
    #             + self.lc_vehicle_names
    #             + ['fo' + str(i) for i in range(1, n_orig_behind + 1)]
    #             + ['ld' + str(i) for i in range(n_dest_ahead, 0, -1)]
    #             + ['fd' + str(i) for i in range(1, n_dest_behind + 1)]
    #     )
    #     self.vehicle_group.set_vehicle_names(vehicle_names)

    # def _create_test_initial_state(self, n_orig_ahead: int,
    #                                n_orig_behind: int, n_dest_ahead: int):
    #     """
    #     For ongoing tests.
    #     :return:
    #     """
    #
    #     v_orig_leader = config.v_ref['lo']
    #     v_dest_leader = config.v_ref['ld']
    #     v_platoon = config.v_ref['p']
    #     v_orig_foll = config.v_ref['fo']
    #     v_dest_foll = config.v_ref['fd']
    #     v_ff_array = ([v_orig_leader] * n_orig_ahead
    #                   + [v_platoon] * self._n_platoon
    #                   + [v_orig_foll] * n_orig_behind
    #                   + [v_dest_leader]
    #                   + [v_dest_foll] * (self.n_per_lane[1] - 1))
    #     self.vehicle_group.set_free_flow_speeds(v_ff_array)
    #     y_orig = 0
    #     y_dest = configuration.LANE_WIDTH
    #     y0_array = ([y_orig] * self.n_per_lane[0]
    #                 + [y_dest] * self.n_per_lane[1])
    #     theta0_array = [0.] * sum(self.n_per_lane)
    #     v0 = ([v_orig_leader] * self.n_per_lane[0]
    #           + self.n_per_lane[1] * [v_dest_leader])
    #     x0 = self.create_x0_with_deltas(v0, config.delta_x,
    #                                     n_orig_ahead, n_dest_ahead)
    #     self.vehicle_group.set_vehicles_initial_states(x0, y0_array,
    #                                                    theta0_array, v0)

    # def create_x0_with_deltas(self, v0: Sequence[float],
    #                           delta_x: Mapping[str, float],
    #                           n_orig_ahead: int, n_dest_ahead: int
    #                           ) -> np.ndarray:
    #     """
    #     Computes the initial position of all vehicles. Puts all vehicles at
    #     steady state distances. Vehicles at the destination lane are placed
    #     either in front of or behind the whole platoon. To have a destination
    #     lane vehicle longitudinally between platoon vehicles, use delta_x.
    #     :param v0: Initial velocities of all vehicles
    #     :param delta_x: Deviation from safe distance.
    #     :param n_orig_ahead:
    #     :param n_dest_ahead:
    #     :return:
    #     """
    #
    #     ref_gaps = self.vehicle_group.get_initial_desired_gaps(v0)
    #     idx_p1 = n_orig_ahead  # platoon leader idx
    #     idx_p_last = idx_p1 + self._n_platoon - 1
    #     x0_array = np.zeros(sum(self.n_per_lane))
    #     x0_p1 = 0.
    #     # Ahead of the platoon in origin lane
    #     leader_x0 = x0_p1 + ref_gaps[idx_p1] - delta_x['lo']
    #     for i in range(idx_p1 - 1, -1, -1):  # lo_0 to lo_N
    #         x0_array[i] = leader_x0
    #         leader_x0 += ref_gaps[i]
    #     # The platoon (note that p1's position is already set at zero)
    #     # Loop goes from p_1 to p_N and then continues to fo_0 till fo_N
    #     for i in range(idx_p1 + 1, self.n_per_lane[0]):
    #         x0_array[i] = x0_array[i - 1] - ref_gaps[i]
    #     # Ahead of the platoon in dest lane
    #     leader_x0 = x0_array[idx_p1] + ref_gaps[idx_p1] - delta_x['ld']
    #     for i in range(self.n_per_lane[0] + n_dest_ahead - 1,
    #                    self.n_per_lane[0] - 1, -1):  # ld_0 to ld_N
    #         x0_array[i] = leader_x0
    #         leader_x0 += ref_gaps[i]
    #     # Behind the platoon in dest lane
    #     follower_x0 = x0_array[idx_p_last] + delta_x['fd']
    #     for i in range(self.n_per_lane[0] + n_dest_ahead,
    #                    sum(self.n_per_lane)):
    #         follower_x0 -= ref_gaps[i]
    #         x0_array[i] = follower_x0
    #
    #     return x0_array

    def set_free_flow_speeds(self,
                             free_flow_speeds: Union[float, list, np.ndarray]):
        self.vehicle_group.set_free_flow_speeds(free_flow_speeds)

    def save_response_data(self, file_name: str) -> None:
        """
        Pickles time, inputs and states as a dataframe
        :param file_name:
        :return:
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.response_to_dataframe(),
                        f, pickle.HIGHEST_PROTOCOL)

    def place_equally_spaced_vehicles(self, gap: float = None):
        """
        All vehicles start at the center of their respective lanes, with
        orientation angle zero, and at the same speed, which equals their
        desired free-flow speed. Vehicles on the same lane are 'gap' meters
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

    @abstractmethod
    def run(self, parameters):
        """

        :param parameters: Parameters depend on the concrete implementation
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

    def set_single_gap_initial_state(
            self, gap_position: int, v_ref: Mapping[str, float],
            delta_x_lo: float):
        base.BaseVehicle.reset_vehicle_counter()
        v_orig_leader = v_ref['lo']
        v_dest_leader = v_ref['ld']
        v_platoon = v_ref['p']
        delta_x_ld = (v_orig_leader - v_dest_leader) / self._lc_intention_time
        self.place_origin_lane_vehicles(
            v_orig_leader, v_platoon, self._is_acceleration_optimal,
            delta_x_lo=delta_x_lo)
        self.place_dest_lane_vehicles_with_single_gap(
            v_dest_leader, gap_position, self._is_acceleration_optimal,
            delta_x_ld)
        # analysis.plot_state_vector(
        #     self.vehicle_group.get_full_initial_state_vector())

    def place_dest_lane_vehicles_with_single_gap(
            self, v_ff: float, gap_position: int,
            is_acceleration_optimal: bool, delta_x: float = None):
        """
        Creates destination lanes on position spanning from ahead the origin
        lane leader to behind the origin lane follower. One vehicle is removed
        to create a suitable lane change gap
        :param v_ff: Destination lane free-flow speed
        :param gap_position: Where, in relation to the platoon vehicles, is
         the gap. If i=0: aligned with lo; i=platoon size + 1: aligned with fo;
         otherwise: aligned with platoon veh i
        :param is_acceleration_optimal: If optimal control vehicles'
         acceleration is also an optimal control input
        :param delta_x: Deviation of the gap's position (and by consequence
         all vehicles) from being perfectly aligned with vehicle i. If None,
         delta_x is set such that, at lane change time, the gap is aligned
         with vehicle i.
        :return:
        """
        # Number of vehs ahead and behind the platoon in the dest lane. This
        # choice ensures at least one dest lane vehicle ahead of lo and another
        # behind fd
        n_ahead = 1 + gap_position
        n_behind = 3

        # We make the gap align with one of the origin lane vehicles
        if gap_position == 0:
            center_vehicle = self.vehicle_group.get_vehicle_by_name('lo0')
        elif gap_position > self._n_platoon:
            center_vehicle = self.vehicle_group.get_vehicle_by_name('fo0')
        else:
            center_vehicle = self.vehicle_group.get_vehicle_by_name(
                'p' + str(gap_position))

        if delta_x is None:
            delta_x = ((center_vehicle.get_vel() - v_ff)
                       / self._lc_intention_time)

        # assuming uniform vehicles:
        reference_gap = center_vehicle.compute_lane_keeping_desired_gap(v_ff)
        x_gap = center_vehicle.get_x() + delta_x
        x0 = x_gap + n_ahead * reference_gap

        ld_counter = 0
        fd_counter = 0
        pN = self.vehicle_group.get_vehicle_by_name('p' + str(self._n_platoon))
        # print(f'x_gap={x_gap:.1f}')
        while x0 > pN.get_x() - n_behind * reference_gap:
            if np.abs(x0 - x_gap) > 0.1:  # skip the vehicle at x_gap
                veh = fsv.ClosedLoopVehicle(False, is_acceleration_optimal,
                                            self._are_vehicles_cooperative)
                if x0 > x_gap:
                    veh_name = 'ld' + str(ld_counter)
                    ld_counter += 1
                else:
                    veh_name = 'fd' + str(fd_counter)
                    fd_counter += 1
                veh.set_name(veh_name)
                veh.set_free_flow_speed(v_ff)
                veh.set_initial_state(x0, configuration.LANE_WIDTH, 0., v_ff)
                self.vehicle_group.add_vehicle(veh)
            # print(f'x0={x0:.1f}, #ld={ld_counter}, #fd={fd_counter}')
            x0 -= reference_gap
        self.n_per_lane.append(fd_counter + ld_counter + 1)
        # print(f'Final: #ld={ld_counter}, #fd={fd_counter} ')

    def set_lane_change_strategy(self, platoon_strategy_number: int):
        self.vehicle_group.set_platoon_lane_change_strategy(
            platoon_strategy_number)

    def set_lane_change_states_graph(
            self, states_graph: graph_tools.VehicleStatesGraph):
        self.vehicle_group.set_vehicle_states_graph(states_graph)

    def make_control_centralized(self):
        self.vehicle_group.centralize_control()

    def run(self, final_time):
        dt = 1.0e-2
        time = np.arange(0, final_time + dt, dt)
        self.vehicle_group.prepare_to_start_simulation(len(time))
        self.vehicle_group.initialize_platoons()
        # analysis.plot_initial_state(self.response_to_dataframe())
        # self.make_control_centralized()
        for i in range(len(time) - 1):
            if np.abs(time[i] - self._lc_intention_time) < dt / 10:
                self.vehicle_group.set_vehicles_lane_change_direction(
                    1, self.lc_vehicle_names)
                # analysis.plot_state_vector(
                #     self.vehicle_group.get_current_state())
            self.vehicle_group.simulate_one_time_step(time[i + 1])


class AllLaneChangeStrategies(LaneChangeScenario):

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
        self.set_control_type('closed_loop', is_acceleration_optimal=False)

        self.named_strategies_positions = {'LdF': -1, 'LVF': -1, 'LdFR': -1}
        self.best_strategy = {'merging_order': [], 'coop_order': []}
        self.costs = []
        self.completion_times = []
        self.accel_costs = []

    def run(self, final_time):
        tf = final_time
        sg = lc_strategy.StrategyGenerator()
        strategy_number = lc_strategy.TemplateStrategy.get_id()
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

            print('Starting with veh', i)

            # all_merging_orders, all_coop_orders = sg.get_all_orders(
            #     self._n_platoon, [i])
            # for merging_order, coop_order in zip(all_merging_orders[6:7],
            #                                      all_coop_orders[6:7]):
            for merging_order, coop_order in sg.generate_order_all(
                    i, [], [], remaining_vehicles.copy()):
                if (merging_order == ldf_lc_order
                        and coop_order == ldf_coop_order):
                    self.named_strategies_positions['LdF'] = counter
                elif (merging_order == lvf_lc_order
                        and coop_order == lvf_coop_order):
                    self.named_strategies_positions['LVF'] = counter
                elif (merging_order == ldfr_lc_order
                        and coop_order == ldfr_coop_order):
                    self.named_strategies_positions['LdFR'] = counter

                counter += 1
                base.BaseVehicle.reset_vehicle_counter()
                self.vehicle_group = VehicleGroup()
                self.set_test_initial_state(self._nda, self._ndb, self._noa,
                                            self._nob, self.v_ref, self.delta_x)
                self.vehicle_group.set_platoon_lane_change_strategy(
                    strategy_number)
                self.vehicle_group.set_predefined_lane_change_order(
                    merging_order, coop_order)
                self.vehicle_group.set_verbose(False)
                LaneChangeScenario.run(self, tf)

                # data = self.vehicle_group.to_dataframe()
                # analysis.plot_trajectory(data, '#' + str(counter))
                # analysis.plot_platoon_lane_change(data)

                # ============ Computing cost ============== #
                # TODO: a mess for now

                n_states = self.vehicle_group.get_n_states()
                n_inputs = self.vehicle_group.get_n_inputs()
                desired_input = [0.] * n_inputs  # TODO: hard codded

                all_vehicles = self.vehicle_group.get_all_vehicles_in_order()
                controlled_veh_ids = set(self.get_lc_vehicle_ids())
                desired_state = occ.create_desired_state(all_vehicles, tf)
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
                    configuration.Configuration.solver_max_iter
                )

                success.append(self.vehicle_group.check_lane_change_success())
                r_cost, t_cost = cost_with_tracker.compute_simulation_cost(
                    self.vehicle_group.get_all_states(),
                    self.vehicle_group.get_all_inputs(),
                    self.vehicle_group.get_simulated_time())
                self.costs.append(r_cost + t_cost)
                if self.costs[-1] < best_cost:
                    best_cost = self.costs[-1]
                    self.best_strategy['merging_order'] = merging_order[:]
                    self.best_strategy['coop_order'] = coop_order[:]
                    best_result = self.vehicle_group

                # print(
                #     f'Strategy #{counter}. '
                #     f'Order and coop: {merging_order}, {coop_order}'
                #     f'\nSuccessful? {success[-1]}. '
                #     f'Cost: {r_cost:.2f}(running) + {t_cost:.2f}(terminal) = '
                #     f'{r_cost + t_cost:.2f}'
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
        print(f'{sg.counter} strategies tested.\n'
              f'Success rate: {sum(success) / sg.counter * 100}%\n'
              f'Best strategy: cost={best_cost:.2f}, '
              f'merging order={self.best_strategy["merging_order"]}, '
              f'coop order={self.best_strategy["coop_order"]}')


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

    # @abstractmethod
    # def create_initial_state(self):
    #     pass

    # def boundary_conditions_to_dataframe(self) -> pd.DataFrame:
    #     """
    #     Puts initial state and desired final conditions in a dataframe.
    #     """
    #     return self.controller._ocp_interface.to_dataframe(
    #         np.array([0, self.tf]),
    #         np.vstack((self.vehicle_group.get_full_initial_state_vector(),
    #                    self.controller.get_desired_state())).T,
    #         np.zeros([self.controller._ocp_interface.n_inputs, 2])
    #     )

    # def ocp_simulation_to_dataframe(self) -> pd.DataFrame:
    #     """
    #     Puts the states computed by the ocp solver tool (and saved) in a df
    #     """
    #     return self.controller._ocp_interface.to_dataframe(
    #         # self.controller.ocp_result.time,
    #         # self.controller.ocp_result.states,
    #         # self.controller.ocp_result.inputs,
    #         self.ocp_response.time,
    #         self.ocp_response.states,
    #         self.ocp_response.inputs
    #     )

    def solve(self):
        self.vehicle_group.update_surrounding_vehicles()
        self.controller = opt_ctrl.VehicleOptimalController()
        self.controller.set_time_horizon(self.tf)
        self.controller.set_controlled_vehicles_ids(
            [self.vehicle_group.get_vehicle_id_by_name(veh_name) for veh_name
             in self.lc_vehicle_names])
        self.controller.find_trajectory(self.vehicle_group.vehicles)
        # return self.controller.ocp_result

    def run_ocp_solution(self) -> None:
        """
        Calls the control libraries function for running the dynamic system
        given the optimal control problem solution
        :return: Nothing. Results are stored internally
        """
        self.ocp_response = self.controller.get_ocp_response()

    def run(self, tf: float):
        """
        Given the optimal control problem solution, runs the open loop system.
        Difference to method 'run' is that we directly (re)simulate the dynamics
        in this case. For debugging purposes
        """
        # It is good to run our simulator with the ocp solution and to confirm
        # it yields the same response as the control library simulation
        self.run_ocp_solution()

        dt = 1e-2
        result = self.controller.ocp_result
        time = np.arange(0, tf + dt, dt)
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
        self.set_control_type('open_loop', is_acceleration_optimal)
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
        raise AttributeError('Scenario does not have optimal control')

    def get_opc_cost_history(self):
        raise AttributeError('Scenario does not have optimal control')

    def create_initial_state(self):
        gap = 2
        self.place_equally_spaced_vehicles(gap)
        print(self.vehicle_group.get_full_initial_state_vector())

    def run(self, final_time):
        """

        :param final_time: Total simulation time
        :return:
        """
        dt = 1e-2
        time = np.arange(0, final_time, dt)
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
        raise AttributeError('Scenario does not have optimal control')

    def get_opc_cost_history(self):
        raise AttributeError('Scenario does not have optimal control')

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
        dt = 1e-2
        time = np.arange(0, final_time, dt)
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
