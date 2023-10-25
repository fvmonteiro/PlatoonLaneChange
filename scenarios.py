from __future__ import annotations

from abc import ABC, abstractmethod
import pickle
from typing import Dict, List, Type, Union

import control as ct
import numpy as np
import pandas as pd

import analysis
import controllers.optimal_controller as opt_ctrl
import constants
from vehicle_group import VehicleGroup
from vehicle_models.base_vehicle import BaseVehicle
import vehicle_models.four_state_vehicles as fsv


config = constants.Configuration


class SimulationScenario(ABC):
    def __init__(self):
        self.n_per_lane: List[int] = []
        BaseVehicle.reset_vehicle_counter()
        self.vehicle_group = VehicleGroup()
        self.result_summary: Dict = {}

    def create_uniform_vehicles(
            self, n_per_lane: List[int], vehicle_class: Type[BaseVehicle],
            free_flow_speed: float):
        self.n_per_lane = n_per_lane
        self.vehicle_group.create_vehicle_array(
            [vehicle_class] * sum(n_per_lane))
        self.vehicle_group.set_free_flow_speeds(free_flow_speed)

    def create_vehicle_group(
            self, vehicle_classes: List[List[Type[BaseVehicle]]]):
        for i in range(len(vehicle_classes)):
            self.n_per_lane.append(len(vehicle_classes[i]))
        flat_vehicle_list = [item for sublist in vehicle_classes for
                             item in sublist]
        self.vehicle_group.create_vehicle_array(flat_vehicle_list)

    def create_base_lane_change_initial_state(self, has_lo: bool, has_fo: bool,
                                              has_ld: bool, has_fd: bool):
        """
        Creates an initial state for up to 5 vehicles: the lane changing one,
        leader and follower at the origin lane, and leader and follower at the
        destination lane
        """
        all_names = ['lo'] if has_lo else []
        all_names.append('ego')
        all_names.extend(['fo'] if has_fo else [])
        all_names.extend(['ld'] if has_ld else [])
        all_names.extend(['fd'] if has_fd else [])
        self.vehicle_group.set_vehicle_names(all_names)

        # Parameters:
        # 1. Deviation from reference gap: positive values lead to
        # smaller-than-desired initial gaps.
        delta_x = {'lo': 0.0, 'ld': 0.0, 'fd': 0.0}
        # 2. Free-flow speeds
        v_ff = {'ego': 10, 'lo': 10}
        v_ff['fo'] = v_ff['ego']
        v_ff['ld'] = 1.0 * v_ff['lo']
        v_ff['fd'] = 1.0 * v_ff['lo']

        ordered_v_ff = [v_ff['lo']] if has_lo else []
        ordered_v_ff.append(v_ff['ego'])
        ordered_v_ff.extend([v_ff['fo']] if has_fo else [])
        ordered_v_ff.extend([v_ff['ld']] if has_ld else [])
        ordered_v_ff.extend([v_ff['fd']] if has_fd else [])
        self.vehicle_group.set_free_flow_speeds(ordered_v_ff)

        # Initial states
        v_orig = ordered_v_ff[0]
        v_dest = ordered_v_ff[-1]
        n_orig, n_dest = self.n_per_lane[0], self.n_per_lane[1]
        v0 = [v_orig] * n_orig + [v_dest] * n_dest
        ref_gaps = self.vehicle_group.map_values_to_names(
            self.vehicle_group.get_initial_desired_gaps(v0)
        )
        x_ego = 0  # max(ref_gaps.get('fd', 0), ref_gaps.get('fo', 0))
        x0 = [x_ego + ref_gaps['ego'] - delta_x['lo']] if has_lo else []
        x0.append(x_ego)
        x0.extend([x_ego - ref_gaps['fo']] if has_fo else [])
        x0.extend([x_ego + ref_gaps['ego'] - delta_x['ld']] if has_ld else [])
        x0.extend([x_ego - ref_gaps['fd'] + delta_x['fd']] if has_fd else [])
        y_orig, y_dest = 0, constants.LANE_WIDTH
        y0 = [y_orig] * n_orig + [y_dest] * n_dest
        theta0 = [0.] * (n_orig + n_dest)
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)

    def set_free_flow_speeds(self,
                             free_flow_speeds: Union[float, List, np.ndarray]):
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
            gap = self.vehicle_group.vehicles[0].free_flow_speed + 1
        x0, y0, theta0, v0 = [], [], [], []
        for lane in range(len(self.n_per_lane)):
            lane_center = lane * constants.LANE_WIDTH
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


class VehicleFollowingScenario(SimulationScenario):
    """
    Scenario to test acceleration feedback laws. No lane changes.
    """

    def __init__(self, n_vehs: int):
        super().__init__()
        vehicle_classes = [fsv.SafeLongitudinalVehicle] * n_vehs
        v_ff = [10] + [12] * n_vehs
        self.n_per_lane = [n_vehs]
        self.vehicle_group.create_vehicle_array(vehicle_classes)
        # self.create_vehicle_group(vehicle_classes)
        self.vehicle_group.set_free_flow_speeds(v_ff)

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
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            self.vehicle_group.simulate_one_time_step(time[i + 1])


class FastLaneChange(SimulationScenario):

    def __init__(self):
        super().__init__()
        vehicle_classes = [fsv.SafeAccelOpenLoopLCVehicle]
        self.n_per_lane = [len(vehicle_classes)]
        self.vehicle_group.create_vehicle_array(vehicle_classes)
        # self.create_vehicle_group(vehicle_classes)
        self.create_initial_state()

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
        self.vehicle_group.update_surrounding_vehicles()
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


class LaneChangeScenario(SimulationScenario):

    def __init__(self, lc_veh_class: Type[BaseVehicle], n_platoon: int,
                 n_orig_ahead: int, n_orig_behind: int,
                 n_dest_ahead: int, n_dest_behind: int):
        if n_platoon < 1:
            raise ValueError("Scenario must have at least one platoon vehicle")

        super().__init__()

        self.lc_intention_time = 1.0
        self._n_orig_ahead, self._n_orig_behind = n_orig_ahead, n_orig_behind
        self._n_platoon = n_platoon
        self._n_dest_ahead, self._n_dest_behind = n_dest_ahead, n_dest_behind

        orig_veh_classes = (
                (fsv.SafeLongitudinalVehicle,) * n_orig_ahead
                + (lc_veh_class,) * n_platoon
                + (fsv.SafeLongitudinalVehicle,) * n_orig_behind)
        dest_veh_classes = (fsv.SafeLongitudinalVehicle,) * (n_dest_ahead
                                                             + n_dest_behind)
        # TODO: quick tests (Oct 10)
        # dest_veh_classes = (fsv.OptimalCoopNoLCVehicle,) * (n_dest_ahead
        #                                                     + n_dest_behind)
        self.n_per_lane = [len(orig_veh_classes), len(dest_veh_classes)]
        self.vehicle_group.create_vehicle_array(
            list(orig_veh_classes + dest_veh_classes))

        # TODO: name differently based on whether n_xxx > 1
        self.lc_vehicle_names = ['p' + str(i) for i in range(1, n_platoon + 1)]
        vehicle_names = (
                ['lo' + str(i) for i in range(n_orig_ahead, 0, -1)]
                + self.lc_vehicle_names
                + ['fo' + str(i) for i in range(1, n_orig_behind + 1)]
                + ['ld' + str(i) for i in range(n_dest_ahead, 0, -1)]
                + ['fd' + str(i) for i in range(1, n_dest_behind + 1)]
        )
        self.vehicle_group.set_vehicle_names(vehicle_names)

    @classmethod
    def platoon_lane_change(cls, n_platoon: int, n_orig_ahead: int,
                            n_orig_behind: int, n_dest_ahead: int,
                            n_dest_behind: int, is_acceleration_optimal: bool
                            ) -> LaneChangeScenario:
        scenario = cls(fsv.PlatoonVehicle, n_platoon, n_orig_ahead,
                       n_orig_behind, n_dest_ahead, n_dest_behind)
        for vehicle in scenario.vehicle_group.get_platoon_vehicles():
            vehicle.set_acceleration_controller_type(is_acceleration_optimal)
        return scenario

    @classmethod
    def single_vehicle_optimal_lane_change(
            cls, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int) -> LaneChangeScenario:
        return cls(fsv.SafeAccelOptimalLCVehicle, 1, n_orig_ahead,
                   n_orig_behind, n_dest_ahead, n_dest_behind)

    @classmethod
    def single_vehicle_feedback_lane_change(
            cls, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int) -> LaneChangeScenario:
        return cls(fsv.ClosedLoopVehicle, 1, n_orig_ahead,
                   n_orig_behind, n_dest_ahead, n_dest_behind)

    def get_n_platoon(self):
        return self._n_platoon

    def get_opc_results_summary(self):
        # We assume there's either a single optimally controlled vehicles
        # or that they all share the same controller
        try:
            opc_vehicle = self.vehicle_group.get_optimal_control_vehicles()[0]
        except IndexError:
            raise AttributeError  # no optimal control vehicles in this group
        return (opc_vehicle.opt_controller.get_running_cost_history(),
                opc_vehicle.opt_controller.get_terminal_cost_history())

    def get_opc_cost_history(self):
        try:
            opc_vehicle = self.vehicle_group.get_optimal_control_vehicles()[0]
        except IndexError:
            raise AttributeError  # no optimal control vehicles in this group
        return (opc_vehicle.opt_controller.get_running_cost_history(),
                opc_vehicle.opt_controller.get_terminal_cost_history())

    def save_cost_data(self, file_name: str) -> None:
        """
        Pickles running and terminal costs 2D lists
        :param file_name:
        :return:
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.get_opc_cost_history(),
                        f, pickle.HIGHEST_PROTOCOL)

    def name_vehicles(self):
        # NOT READY
        # compare to code in constructor
        if self._n_platoon > 1:
            self.lc_vehicle_names = ['p' + str(i) for i
                                     in range(1, self._n_platoon + 1)]
        else:
            self.lc_vehicle_names = ['ego']

        vehicle_counts = [
            self._n_orig_ahead, self._n_platoon, self._n_orig_behind,
            self._n_dest_ahead, self._n_dest_behind]
        vehicle_base_names = ['lo', 'p', 'fo', 'ld', 'fd']
        all_names = []
        for i in range(len(vehicle_counts)):
            n = vehicle_counts[i]
            base_name = vehicle_base_names[i]
            if n != 1:
                all_names.extend([base_name + str(i) for i
                                  in range(self._n_orig_ahead, 0, -1)])
            else:
                all_names.append(base_name)

        self.vehicle_group.set_vehicle_names(all_names)

    def create_safe_uniform_speed_initial_state(self):
        # Free-flow speeds
        v_orig = 10
        v_dest = 10
        v_ff = 10
        v_ff_array = ([v_orig] + [v_ff] * (self.n_per_lane[0] - 1)
                      + [v_dest] + [v_ff] * (self.n_per_lane[1] - 1))
        self.vehicle_group.set_free_flow_speeds(v_ff_array)
        # Deviation from equilibrium position:
        delta_x = {'lo': 0.0, 'ld': 0.0, 'fd': 0.0}
        self.create_initial_state(v_orig, v_dest, delta_x)

    def create_test_initial_state(self):
        """
        For ongoing tests. NOT TO BE USED IN DATA COLLECTION
        :return:
        """
        print("======= RUNNING AN EXPLORATORY TEST SCENARIO =======")
        v_orig_leader = config.v_ref['lo']
        v_dest_leader = config.v_ref['ld']
        v_others = config.v_ref['p']
        v_orig_foll = config.v_ref['fo']
        v_dest_foll = config.v_ref['fd']
        v_ff_array = ([v_orig_leader] * self._n_orig_ahead
                      + [v_others] * self._n_platoon
                      + [v_orig_foll] * self._n_orig_behind
                      + [v_dest_leader]
                      + [v_dest_foll] * (self.n_per_lane[1] - 1))
        self.vehicle_group.set_free_flow_speeds(v_ff_array)
        # Deviation from equilibrium position:
        delta_x = config.delta_x
        self.create_initial_state(v_orig_leader, v_dest_leader, delta_x)

    def create_initial_state(self, v_orig: float, v_dest: float,
                             delta_x: Dict[str, float]):

        # Initial states
        v0_array = ([v_orig] * self.n_per_lane[0]
                    + self.n_per_lane[1] * [v_dest])
        ref_gaps = self.vehicle_group.get_initial_desired_gaps(v0_array)
        idx_p0 = self._n_orig_ahead
        idx_p_last = idx_p0 + self._n_platoon - 1
        x0_array = np.zeros(sum(self.n_per_lane))
        x0_p1 = 0
        # Ahead of the platoon in origin lane
        leader_x0 = x0_p1 + ref_gaps[idx_p0] - delta_x['lo']
        for i in range(idx_p0 - 1, -1, -1):  # lo_0 to lo_N
            x0_array[i] = leader_x0
            leader_x0 += ref_gaps[i]
        # The platoon
        follower_x0 = x0_p1
        for i in range(idx_p0 + 1, idx_p_last + 1):  # p_1 to p_N
            follower_x0 -= ref_gaps[i]
            x0_array[i] = follower_x0
        # Behind the platoon in origin lane
        follower_x0 = x0_array[idx_p_last]
        for i in range(idx_p_last + 1, self.n_per_lane[0]):  # fo_0 to fo_N
            follower_x0 -= ref_gaps[i]
            x0_array[i] = follower_x0
        # Ahead of the platoon in dest lane
        leader_x0 = x0_p1 + ref_gaps[idx_p0] - delta_x['ld']
        for i in range(self.n_per_lane[0] + self._n_dest_ahead - 1,
                       self.n_per_lane[0] - 1, -1):  # ld_0 to ld_N
            x0_array[i] = leader_x0
            leader_x0 += ref_gaps[i]
        # Behind the platoon in origin lane
        follower_x0 = x0_array[idx_p_last] + delta_x['fd']
        for i in range(self.n_per_lane[0] + self._n_dest_ahead,
                       sum(self.n_per_lane)):
            follower_x0 -= ref_gaps[i]
            x0_array[i] = follower_x0

        y_orig = 0
        y_dest = constants.LANE_WIDTH
        y0_array = ([y_orig] * self.n_per_lane[0]
                    + [y_dest] * self.n_per_lane[1])
        theta0_array = [0.] * sum(self.n_per_lane)
        self.vehicle_group.set_vehicles_initial_states(x0_array, y0_array,
                                                       theta0_array, v0_array)

    def make_control_centralized(self):
        self.vehicle_group.centralize_control()

    def run(self, final_time):
        dt = 1.0e-2
        time = np.arange(0, final_time + dt, dt)
        self.vehicle_group.prepare_to_start_simulation(len(time))
        analysis.plot_initial_state(self.response_to_dataframe())
        # self.make_control_centralized()
        for i in range(len(time) - 1):
            if np.isclose(time[i], self.lc_intention_time, atol=dt / 10):
                self.vehicle_group.set_vehicles_lane_change_direction(
                    self.lc_vehicle_names, 1
                )
            self.vehicle_group.simulate_one_time_step(time[i + 1])


class ExternalOptimalControlScenario(SimulationScenario, ABC):
    controller: opt_ctrl.VehicleOptimalController
    ocp_response: ct.TimeResponseData

    def __init__(self):
        super().__init__()
        self.tf: float = 0.0

    def set_boundary_conditions(self, tf: float):
        """ Sets the initial state, final time and desired final states """
        self.tf = tf
        self.create_initial_state()
        self.vehicle_group.prepare_to_start_simulation(1)
        self.set_desired_lane_changes()
        self.vehicle_group.update_surrounding_vehicles()

    @abstractmethod
    def create_initial_state(self):
        pass

    def boundary_conditions_to_dataframe(self) -> pd.DataFrame:
        """
        Puts initial state and desired final conditions in a dataframe.
        """
        return self.controller._ocp_interface.to_dataframe(
            np.array([0, self.tf]),
            np.vstack((self.vehicle_group.get_full_initial_state_vector(),
                       self.controller.get_desired_state())).T,
            np.zeros([self.controller._ocp_interface.n_inputs, 2])
        )

    def ocp_simulation_to_dataframe(self) -> pd.DataFrame:
        """
        Puts the states computed by the ocp solver tool (and saved) in a df
        """
        return self.controller._ocp_interface.to_dataframe(
            # self.controller.ocp_result.time,
            # self.controller.ocp_result.states,
            # self.controller.ocp_result.inputs,
            self.ocp_response.time,
            self.ocp_response.states,
            self.ocp_response.inputs
        )

    def solve(self):
        self.controller = opt_ctrl.VehicleOptimalController()
        self.controller.set_time_horizon(self.tf)
        self.controller.set_controlled_vehicles_ids(
            self.vehicle_group.get_vehicle_id_by_name('ego'))
        self.controller.find_trajectory(self.vehicle_group.vehicles)
        # return self.controller.ocp_result

    def run_ocp_solution(self) -> None:
        """
        Calls the control libraries function for running the dynamic system
        given the optimal control problem solution
        :return: Nothing. Results are stored internally
        """
        self.ocp_response = self.controller.get_ocp_response()

    def run(self, params=None):
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
        time = np.arange(0, result.time[-1], dt)
        # inputs = np.zeros([result.inputs.shape[0], len(time)])
        veh_ids = self.vehicle_group.sorted_vehicle_ids
        # for i in range(len(result.inputs)):
        #     inputs[i, :] = np.interp(time, result.time, result.inputs[i])
        self.vehicle_group.prepare_to_start_simulation(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            current_inputs = self.controller.get_input(time[i], veh_ids)
            self.vehicle_group.simulate_one_time_step(time[i + 1],
                                                      current_inputs)

    @abstractmethod
    def set_desired_lane_changes(self):
        pass


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
            # veh.update_target_y()


class LaneChangeWithConstraints(ExternalOptimalControlScenario):
    """
    Used to test how to code safety constraints
    """

    def __init__(self, has_lo: bool, has_fo: bool, has_ld: bool, has_fd: bool):
        super().__init__()

        self._has_lo, self._has_fo, self._has_ld, self._has_fd = (
            has_lo, has_fo, has_ld, has_fd
        )
        lc_veh_class = fsv.SafeAccelOpenLoopLCVehicle
        orig_veh_classes = (lc_veh_class,)
        if has_lo:
            orig_veh_classes = (fsv.ClosedLoopVehicle,) + orig_veh_classes
        if has_fo:
            orig_veh_classes = orig_veh_classes + (fsv.ClosedLoopVehicle,)

        n_orig = 1 + has_lo + has_fo
        n_dest = has_ld + has_fd
        self.n_per_lane = [n_orig, n_dest]
        veh_classes = orig_veh_classes + (fsv.ClosedLoopVehicle,) * n_dest
        self.vehicle_group.create_vehicle_array(list(veh_classes))

    def create_initial_state(self):
        self.create_base_lane_change_initial_state(self._has_lo, self._has_fo,
                                                   self._has_ld, self._has_fd)

    def set_desired_lane_changes(self):
        lc_vehicle = self.vehicle_group.get_vehicle_by_name('ego')
        lc_vehicle.set_lane_change_direction(1)
