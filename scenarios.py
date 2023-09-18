from __future__ import annotations
from abc import ABC, abstractmethod
import pickle
from typing import List, Tuple, Type, Union

import control as ct
import numpy as np
import pandas as pd

import controllers.optimal_controller as opt_ctrl
from constants import LANE_WIDTH
from vehicle_group import VehicleGroup
from vehicle_models.base_vehicle import BaseVehicle
import vehicle_models.four_state_vehicles as fsv
import system_operating_mode as som


class SimulationScenario(ABC):
    def __init__(self):
        self.n_per_lane: List[int] = []
        self.vehicle_group = VehicleGroup()
        BaseVehicle.reset_vehicle_counter()
        self.tf = None

    @abstractmethod
    def get_restart_copy(self) -> SimulationScenario:
        pass

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
        y_orig, y_dest = 0, LANE_WIDTH
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
            lane_center = lane * LANE_WIDTH
            n = self.n_per_lane[lane]
            for i in range(n):
                x0.append(gap * (n - i - 1))
                y0.append(lane_center)
                theta0.append(0.0)
        v0 = self.vehicle_group.get_free_flow_speeds()
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)
        # self.initial_state =
        # self.vehicle_group.get_full_initial_state_vector()

    # def lane_change_scenario_vehicle_filter(self, values: List):
    #     # Used together with lane change scenarios. We define values for the
    #     # 4 relevant surrounding vehicles, but they might not all be in the
    #     # current scenario
    #     # TODO: poor organization
    #     if self.n_per_lane[0] < 3:
    #         values.pop(2)
    #         if self.n_per_lane[0] < 2:
    #             values.pop(0)
    #     if self.n_per_lane[1] < 2:
    #         values.pop()
    #         if self.n_per_lane[1] < 1:
    #             values.pop()
    #     return values

    def response_to_dataframe(self) -> pd.DataFrame:
        return self.vehicle_group.to_dataframe()

    def simulate_one_time_step(self, new_time, open_loop_controls=None):
        if open_loop_controls is None:
            open_loop_controls = {}
        self.vehicle_group.update_platoons()
        self.vehicle_group.update_vehicle_modes()
        self.vehicle_group.determine_inputs(open_loop_controls)
        self.vehicle_group.compute_derivatives()
        self.vehicle_group.update_states(new_time)
        self.vehicle_group.update_surrounding_vehicles()

    @abstractmethod
    def create_initial_state(self):
        pass

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
        vehicle_classes = [fsv.ClosedLoopVehicle] * n_vehs
        v_ff = [10] + [12] * n_vehs
        self.n_per_lane = [n_vehs]
        self.vehicle_group.create_vehicle_array(vehicle_classes)
        # self.create_vehicle_group(vehicle_classes)
        self.set_free_flow_speeds(v_ff)

    def get_restart_copy(self) -> VehicleFollowingScenario:
        return VehicleFollowingScenario(self.n_per_lane[0])

    def create_initial_state(self):
        gap = 2
        self.place_equally_spaced_vehicles(gap)
        print(self.vehicle_group.get_full_initial_state_vector())

    def run(self, final_time):
        """

        :param final_time: Total simulation time
        :return:
        """
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.prepare_to_start_simulation(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            self.simulate_one_time_step(time[i + 1])


class FastLaneChange(SimulationScenario):

    def __init__(self):
        super().__init__()
        vehicle_classes = [fsv.SafeAccelOpenLoopLCVehicle]
        self.n_per_lane = [len(vehicle_classes)]
        self.vehicle_group.create_vehicle_array(vehicle_classes)
        # self.create_vehicle_group(vehicle_classes)
        self.create_initial_state()

    def get_restart_copy(self) -> FastLaneChange:
        return FastLaneChange()

    def create_initial_state(self):
        v_ff = 10
        self.set_free_flow_speeds(v_ff)

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
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
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
            self.simulate_one_time_step(time[i + 1], inputs)


class LaneChangeScenario(SimulationScenario):

    def __init__(self, lc_veh_class: Type[BaseVehicle], has_lo: bool,
                 has_fo: bool, has_ld: bool, has_fd: bool):
        """

        :param lc_veh_class:
        :param has_lo: if True, includes the origin lane leader
        :param has_fo: if True, includes the origin lane follower
        :param has_ld: if True, includes the destination lane leader
        :param has_fd: if True, includes the destination lane follower
        """
        super().__init__()

        self.lc_intention_time = 1.0
        self.leader_sequence = dict()  # used when iterating over the OCP
        self._has_lo, self._has_fo, self._has_ld, self._has_fd = (
            has_lo, has_fo, has_ld, has_fd
        )

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
        self.create_initial_state()

    @classmethod
    def closed_loop(cls, has_lo: bool, has_fo: bool,
                    has_ld: bool, has_fd: bool):
        """
        Creates a scenario instance where all vehicles have longitudinal and
        lateral feedback controllers.
        :param has_lo: if True, includes the origin lane leader
        :param has_fo: if True, includes the origin lane follower
        :param has_ld: if True, includes the destination lane leader
        :param has_fd: if True, includes the destination lane follower
        """
        return cls(fsv.ClosedLoopVehicle, has_lo, has_fo, has_ld, has_fd)

    @classmethod
    def optimal_control(cls, has_lo: bool, has_fo: bool,
                        has_ld: bool, has_fd: bool):
        """

        :param has_lo: if True, includes the origin lane leader
        :param has_fo: if True, includes the origin lane follower
        :param has_ld: if True, includes the destination lane leader
        :param has_fd: if True, includes the destination lane follower
        :return:
        """
        return cls(fsv.SafeAccelOptimalLCVehicle,
                   has_lo, has_fo, has_ld, has_fd)

    @classmethod
    def platoon_lane_change(cls, has_lo: bool, has_fo: bool,
                            has_ld: bool, has_fd: bool):
        """
        Only tests platoons with a single vehicle. Used for initial tests.
        :param has_lo: if True, includes the origin lane leader
        :param has_fo: if True, includes the origin lane follower
        :param has_ld: if True, includes the destination lane leader
        :param has_fd: if True, includes the destination lane follower
        """
        scenario = cls(fsv.PlatoonVehicle, has_lo, has_fo, has_ld, has_fd)
        return scenario

    def get_restart_copy(self) -> LaneChangeScenario:
        lc_veh_type = self.vehicle_group.get_vehicle_by_name('ego')
        return LaneChangeScenario(type(lc_veh_type), self._has_lo, self._has_fo,
                                  self._has_ld, self._has_fd)

    def get_initial_system_mode(self):
        self.vehicle_group.prepare_to_start_simulation(1)
        self.vehicle_group.update_surrounding_vehicles()
        return self.vehicle_group.mode_sequence[:]

    def create_initial_state(self):
        self.create_base_lane_change_initial_state(self._has_lo, self._has_fo,
                                                   self._has_ld, self._has_fd)
        # self.vehicle_group.make_all_connected()

    def run(self, final_time):
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.prepare_to_start_simulation(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        self.vehicle_group.set_ocp_leader_sequence(self.leader_sequence)
        for i in range(len(time) - 1):
            if np.isclose(time[i], self.lc_intention_time, atol=dt / 10):
                self.vehicle_group.set_single_vehicle_lane_change_direction(
                    'ego', 1)
            self.simulate_one_time_step(time[i + 1])


class PlatoonLaneChange(SimulationScenario):

    def __init__(self, n_platoon: int, n_orig_ahead: int,
                 n_orig_behind: int, n_dest_ahead: int, n_dest_behind: int):
        if n_platoon < 1:
            raise ValueError("Scenario must have at least one platoon vehicle")

        super().__init__()

        self.lc_intention_time = 1.0
        self.leader_sequence = dict()  # used when iterating over the OCP
        self._n_orig_ahead = n_orig_ahead
        self._n_platoon = n_platoon
        self._n_dest_ahead = n_dest_ahead

        orig_veh_classes = (
                (fsv.ClosedLoopVehicle,) * n_orig_ahead
                + (fsv.PlatoonVehicle,) * n_platoon  # ignore type checker here
                + (fsv.ClosedLoopVehicle,) * n_orig_behind)
        dest_veh_classes = (fsv.ClosedLoopVehicle,) * (n_dest_ahead
                                                       + n_dest_behind)
        self.n_per_lane = [len(orig_veh_classes), len(dest_veh_classes)]
        self.vehicle_group.create_vehicle_array(
            list(orig_veh_classes + dest_veh_classes))
        # veh_classes = ([orig_veh_classes, dest_veh_classes])
        # self.create_vehicle_group(veh_classes)

        vehicle_names = (['lo' + str(i) for i in range(n_orig_ahead, 0, -1)]
                         + ['p' + str(i) for i in range(1, n_platoon + 1)]
                         + ['fo' + str(i) for i in range(1, n_orig_behind + 1)]
                         + ['ld' + str(i) for i in range(n_dest_ahead, 0, -1)]
                         + ['fd' + str(i) for i in range(1, n_dest_behind + 1)])
        self.vehicle_group.set_vehicle_names(vehicle_names)
        self.create_initial_state()

    def get_restart_copy(self) -> PlatoonLaneChange:
        pass  # TODO

    def create_initial_state(self):
        # Free-flow speeds
        v_orig = 10
        v_dest = 10
        v_ff = 10
        v_ff_array = ([v_orig] + [v_ff] * (self.n_per_lane[0] - 1)
                      + [v_dest] + [v_ff] * (self.n_per_lane[1] - 1))
        self.set_free_flow_speeds(v_ff_array)

        # Initial states
        delta_x = {'lo': 0.0, 'ld': 0.0,
                   'fd': 0.0}  # deviation from equilibrium
        v0_array = ([v_orig] * self.n_per_lane[0]
                    + [v_dest] * self.n_per_lane[1])
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
        y_dest = LANE_WIDTH
        y0_array = ([y_orig] * self.n_per_lane[0]
                    + [y_dest] * self.n_per_lane[1])
        theta0_array = [0.] * sum(self.n_per_lane)
        self.vehicle_group.set_vehicles_initial_states(x0_array, y0_array,
                                                       theta0_array, v0_array)

    def run(self, final_time):
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.prepare_to_start_simulation(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        self.vehicle_group.set_ocp_leader_sequence(self.leader_sequence)
        for i in range(len(time) - 1):
            if np.isclose(time[i], self.lc_intention_time, atol=dt / 10):
                self.vehicle_group.set_single_vehicle_lane_change_direction(
                    'p1', 1)
            self.simulate_one_time_step(time[i + 1])


class ModeSwitchTests:

    def __init__(self, scenario: Union[LaneChangeScenario]):
        self.scenario = scenario
        self.data: List[pd.DataFrame] = []

    @classmethod
    def single_vehicle_lane_change(cls, has_lo: bool, has_fo: bool,
                                   has_ld: bool, has_fd: bool):
        scenario = LaneChangeScenario.optimal_control(has_lo, has_fo,
                                                      has_ld, has_fd)
        return ModeSwitchTests(scenario)

    def run(self, final_time: float):
        scenario = self.scenario
        mode_sequence = scenario.get_initial_system_mode()
        solved = False
        counter = 0
        while not solved and counter < 3:
            counter += 1
            print("==== Attempt {} with mode sequence: {}====".format(
                counter, ModeSwitchTests.mode_sequence_to_str(mode_sequence)))

            scenario.leader_sequence = som.mode_sequence_to_leader_sequence(
                mode_sequence)
            scenario.run(final_time)
            self.data.append(scenario.response_to_dataframe())
            solved = ModeSwitchTests.compare_mode_sequences(
                mode_sequence, scenario.vehicle_group.mode_sequence)

            if not solved:
                # Preparing for the next iteration: we save the simulated
                # mode sequence and restart the scenario
                mode_sequence = scenario.vehicle_group.mode_sequence[:]
                scenario = scenario.get_restart_copy()

    @staticmethod
    def compare_mode_sequences(s1: List[Tuple[float, som.SystemMode]],
                               s2: List[Tuple[float, som.SystemMode]]) -> bool:

        if len(s1) != len(s2):
            return False
        for i in range(len(s1)):
            t1, mode1 = s1[i]
            t2, mode2 = s2[i]
            if not (np.abs(t1 - t2) <= 0.1 and mode1 == mode2):
                return False
        return True

    @staticmethod
    def mode_sequence_to_str(s: List[Tuple[float, som.SystemMode]]):
        ret = ""
        for t, m in s:
            ret += "(" + str(t) + ": " + str(m) + ") "
        return ret


class ExternalOptimalControlScenario(SimulationScenario, ABC):
    # vehicles_ocp_interface: vgi.VehicleGroupInterface
    controller: opt_ctrl.VehicleOptimalController
    ocp_response: ct.TimeResponseData

    def __init__(self):
        super().__init__()

    def set_boundary_conditions(self, tf: float):
        """ Sets the initial state, final time and desired final states """
        self.tf = tf
        self.create_initial_state()
        self.vehicle_group.prepare_to_start_simulation(1)
        self.set_desired_lane_changes()
        self.vehicle_group.update_surrounding_vehicles()

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

    def solve(self, max_iter: int = 100):
        self.controller = opt_ctrl.VehicleOptimalController(self.tf)
        self.controller.set_max_iter(max_iter)
        self.controller.find_multiple_vehicle_trajectory(
            0.0, self.vehicle_group.vehicles,
            [self.vehicle_group.get_vehicle_id_by_name('ego')])
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
        inputs = np.zeros([result.inputs.shape[0], len(time)])
        veh_ids = self.vehicle_group.sorted_vehicle_ids
        for i in range(len(result.inputs)):
            inputs[i, :] = np.interp(time, result.time, result.inputs[i])
        self.vehicle_group.prepare_to_start_simulation(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            current_inputs = self.controller.get_input(time[i], veh_ids)
            self.simulate_one_time_step(time[i + 1], current_inputs)
            # self.vehicle_group.update_vehicle_modes()
            # self.vehicle_group.determine_inputs(current_inputs)
            # self.vehicle_group.compute_derivatives()
            # self.vehicle_group.update_states(time[i + 1])
            # self.vehicle_group.update_surrounding_vehicles()

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

    def get_restart_copy(self) -> ExampleScenarioExternal:
        return ExampleScenarioExternal()

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

    def get_restart_copy(self) -> LaneChangeWithConstraints:
        return LaneChangeWithConstraints(self._has_lo, self._has_fo,
                                         self._has_ld, self._has_fd)

    def create_initial_state(self):
        self.create_base_lane_change_initial_state(self._has_lo, self._has_fo,
                                                   self._has_ld, self._has_fd)

    def set_desired_lane_changes(self):
        lc_vehicle = self.vehicle_group.get_vehicle_by_name('ego')
        lc_vehicle.set_lane_change_direction(1)
