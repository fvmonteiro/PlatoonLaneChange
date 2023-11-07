from __future__ import annotations

from abc import ABC, abstractmethod
import pickle
from typing import Iterable, Mapping, Sequence, Type, Union

import control as ct
import numpy as np
import pandas as pd

import analysis
import controllers.optimal_controller as opt_ctrl
import configuration
from vehicle_group import VehicleGroup
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv

config = configuration.Configuration


class SimulationScenario(ABC):

    _n_platoon: int

    def __init__(self):
        self.n_per_lane: list[int] = []
        base.BaseVehicle.reset_vehicle_counter()
        self.vehicle_group = VehicleGroup()
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
            vehicle_class: Type[base.BaseVehicle],
            free_flow_speed: float):
        array_2d = [[vehicle_class] * n for n in n_per_lane]
        self.create_vehicle_group(array_2d)
        self.vehicle_group.set_free_flow_speeds(free_flow_speed)

    def create_vehicle_group(
            self, vehicle_classes: Sequence[Sequence[Type[base.BaseVehicle]]]):
        for i in range(len(vehicle_classes)):
            self.n_per_lane.append(len(vehicle_classes[i]))
        flat_vehicle_list = [item for sublist in vehicle_classes for
                             item in sublist]
        self.vehicle_group.create_vehicle_array_from_classes(flat_vehicle_list)

    def create_test_scenario(
            self, lc_veh_class_name: str,
            n_orig_ahead: int, n_orig_behind: int, n_dest_ahead: int,
            n_dest_behind: int, is_acceleration_optimal: bool):
        """
        Scenario where all vehicles are at steady state distances. Vehicles
        at the destination lane are placed either in front of or behind the
        whole platoon. To have a destination lane vehicle longitudinally
        between platoon vehicles, use delta_x from Configurations
        :return:
        """

        if lc_veh_class_name == 'optimal':
            lc_veh_type = fsv.OptimalControlVehicle
        elif lc_veh_class_name == 'closed_loop':
            lc_veh_type = fsv.ClosedLoopVehicle
        elif lc_veh_class_name == 'open_loop':
            lc_veh_type = fsv.OpenLoopVehicle
        else:
            raise ValueError('Unknown vehicle class name for lane changing '
                             'vehicles.\n'
                             'Accepted values are: optimal, closed_loop, '
                             'open_loop')
        coop = self._are_vehicles_cooperative
        vehicles_ahead = [
            fsv.ClosedLoopVehicle(can_change_lanes=False,
                                  is_connected=coop) for _
            in range(n_orig_ahead)]
        lc_vehs = [lc_veh_type(
            can_change_lanes=True,
            has_open_loop_acceleration=is_acceleration_optimal,
            is_connected=True
        ) for _ in range(self._n_platoon)]
        vehicles_behind = [
            fsv.ClosedLoopVehicle(can_change_lanes=False,
                                  is_connected=coop) for _
            in range(n_orig_behind)]
        orig_lane_vehs = vehicles_ahead + lc_vehs + vehicles_behind
        dest_lane_vehs = [
            fsv.ClosedLoopVehicle(can_change_lanes=False,
                                  is_connected=coop) for _
            in range(n_dest_ahead + n_dest_behind)]
        # dest_lane_vehs = [
        #     fsv.OptimalControlVehicle(can_change_lanes=False,
        #                               is_connected=are_vehicles_cooperative)
        #     for _ in range(n_dest_ahead + n_dest_behind)]
        self.n_per_lane = [len(orig_lane_vehs), len(dest_lane_vehs)]
        self.vehicle_group.fill_vehicle_array(orig_lane_vehs + dest_lane_vehs)
        self.lc_vehicle_names = ['p' + str(i) for i
                                 in range(1, self._n_platoon + 1)]
        vehicle_names = (
                ['lo' + str(i) for i in range(n_orig_ahead, 0, -1)]
                + self.lc_vehicle_names
                + ['fo' + str(i) for i in range(1, n_orig_behind + 1)]
                + ['ld' + str(i) for i in range(n_dest_ahead, 0, -1)]
                + ['fd' + str(i) for i in range(1, n_dest_behind + 1)]
        )
        self.vehicle_group.set_vehicle_names(vehicle_names)
        self.create_test_initial_state(n_orig_ahead, n_orig_behind,
                                       n_dest_ahead)

    def create_test_initial_state(self, n_orig_ahead: int,
                                  n_orig_behind: int, n_dest_ahead: int):
        """
        For ongoing tests.
        :return:
        """

        v_orig_leader = config.v_ref['lo']
        v_dest_leader = config.v_ref['ld']
        v_platoon = config.v_ref['p']
        v_orig_foll = config.v_ref['fo']
        v_dest_foll = config.v_ref['fd']
        v_ff_array = ([v_orig_leader] * n_orig_ahead
                      + [v_platoon] * self._n_platoon
                      + [v_orig_foll] * n_orig_behind
                      + [v_dest_leader]
                      + [v_dest_foll] * (self.n_per_lane[1] - 1))
        self.vehicle_group.set_free_flow_speeds(v_ff_array)
        y_orig = 0
        y_dest = configuration.LANE_WIDTH
        y0_array = ([y_orig] * self.n_per_lane[0]
                    + [y_dest] * self.n_per_lane[1])
        theta0_array = [0.] * sum(self.n_per_lane)
        v0 = ([v_orig_leader] * self.n_per_lane[0]
              + self.n_per_lane[1] * [v_dest_leader])
        x0 = self.create_x0_with_deltas(v0, config.delta_x,
                                        n_orig_ahead, n_dest_ahead)
        self.vehicle_group.set_vehicles_initial_states(x0, y0_array,
                                                       theta0_array, v0)

    def create_x0_with_deltas(self, v0: Sequence[float],
                              delta_x: Mapping[str, float],
                              n_orig_ahead: int, n_dest_ahead: int
                              ) -> np.ndarray:
        """
        Computes the initial position of all vehicles. Puts all vehicles at
        steady state distances. Vehicles at the destination lane are placed
        either in front of or behind the whole platoon. To have a destination
        lane vehicle longitudinally between platoon vehicles, use delta_x.
        :param v0: Initial velocities of all vehicles
        :param delta_x: Deviation from safe distance.
        :param n_orig_ahead:
        :param n_dest_ahead:
        :return:
        """

        # Initial states
        ref_gaps = self.vehicle_group.get_initial_desired_gaps(v0)
        idx_p1 = n_orig_ahead  # platoon leader idx
        idx_p_last = idx_p1 + self._n_platoon - 1
        x0_array = np.zeros(sum(self.n_per_lane))
        x0_p1 = 0.
        # Ahead of the platoon in origin lane
        leader_x0 = x0_p1 + ref_gaps[idx_p1] - delta_x['lo']
        for i in range(idx_p1 - 1, -1, -1):  # lo_0 to lo_N
            x0_array[i] = leader_x0
            leader_x0 += ref_gaps[i]
        # The platoon (note that p1's position is already set at zero)
        # Loop goes from p_1 to p_N and then continues to fo_0 till fo_N
        for i in range(idx_p1 + 1, self.n_per_lane[0]):
            x0_array[i] = x0_array[i - 1] - ref_gaps[i]
        # Ahead of the platoon in dest lane
        leader_x0 = x0_array[idx_p1] + ref_gaps[idx_p1] - delta_x['ld']
        for i in range(self.n_per_lane[0] + n_dest_ahead - 1,
                       self.n_per_lane[0] - 1, -1):  # ld_0 to ld_N
            x0_array[i] = leader_x0
            leader_x0 += ref_gaps[i]
        # Behind the platoon in origin lane
        follower_x0 = x0_array[idx_p_last] + delta_x['fd']
        for i in range(self.n_per_lane[0] + n_dest_ahead,
                       sum(self.n_per_lane)):
            follower_x0 -= ref_gaps[i]
            x0_array[i] = follower_x0

        return x0_array

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
            gap = self.vehicle_group.vehicles[0].free_flow_speed + 1
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

    def optimal_platoon_lane_change(
            self, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int,
            is_acceleration_optimal: bool):
        self.create_test_scenario(
            'optimal', n_orig_ahead, n_orig_behind, n_dest_ahead,
            n_dest_behind, is_acceleration_optimal)

    def platoon_full_feedback_lane_change(
            self, n_orig_ahead: int, n_orig_behind: int,
            n_dest_ahead: int, n_dest_behind: int):
        self.create_test_scenario(
            'closed_loop', n_orig_ahead, n_orig_behind, n_dest_ahead,
            n_dest_behind, False)

    def create_full_lanes_initial_state(self):
        v_orig_leader = config.v_ref['lo']
        v_dest_leader = config.v_ref['ld']
        v_platoon = config.v_ref['p']
        v_ff_array = ([v_orig_leader]
                      + [v_platoon] * self._n_platoon
                      + [v_dest_leader] * (self.n_per_lane[1]))
        self.vehicle_group.set_free_flow_speeds(v_ff_array)
        y_orig = 0
        y_dest = configuration.LANE_WIDTH
        y0_array = ([y_orig] * self.n_per_lane[0]
                    + [y_dest] * self.n_per_lane[1])
        theta0_array = [0.] * sum(self.n_per_lane)
        v0 = ([v_orig_leader] * self.n_per_lane[0]
              + self.n_per_lane[1] * [v_dest_leader])
        x0 = self.create_full_lane_x0(v0)
        self.vehicle_group.set_vehicles_initial_states(x0, y0_array,
                                                       theta0_array, v0)

    def create_full_lane_x0(self, v0: Sequence[float]) -> np.ndarray:
        # TODO: we must change the parts populating the veh group too
        # The goal here is to make the platoon see a full destination lane,
        # that is, no large gaps anywhere, while including the minimum number
        # of vehicles in the simulation.

        self.n_per_lane = [self._n_platoon + 1, self._n_platoon + 1]

        ref_gaps = self.vehicle_group.get_initial_desired_gaps(v0)
        idx_p1 = 1  # platoon leader idx
        idx_p_last = idx_p1 + self._n_platoon - 1
        x0_array = np.zeros(sum(self.n_per_lane))
        # Ahead of the platoon in origin lane
        x0_array[0] = x0_array[idx_p1] + ref_gaps[idx_p1]
        # The platoon (note that p1's position is already set at zero)
        for i in range(idx_p1 + 1, idx_p_last + 1):  # p_1 to p_N
            x0_array[i] = x0_array[i - 1] - ref_gaps[i]
        # From front-most to last dest lane vehicle
        x0_array[self.n_per_lane[0]] = x0_array[idx_p1] + ref_gaps[idx_p1] / 2
        for i in range(self.n_per_lane[0] + 1, sum(self.n_per_lane)):
            x0_array[i] = x0_array[i - 1] - ref_gaps[i]
        return x0_array

    def make_control_centralized(self):
        self.vehicle_group.centralize_control()

    def run(self, final_time):
        dt = 1.0e-2
        time = np.arange(0, final_time + dt, dt)
        self.vehicle_group.prepare_to_start_simulation(len(time))
        analysis.plot_initial_state(self.response_to_dataframe())
        self.make_control_centralized()
        for i in range(len(time) - 1):
            if np.abs(time[i] - self._lc_intention_time) < dt / 10:
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

    @abstractmethod
    def create_initial_state(self):
        pass

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
        # inputs = np.zeros([result.inputs.shape[0], len(time)])
        veh_ids = self.vehicle_group.sorted_vehicle_ids
        # for i in range(len(result.inputs)):
        #     inputs[i, :] = np.interp(time, result.time, result.inputs[i])
        self.vehicle_group.prepare_to_start_simulation(len(time))
        # self.vehicle_group.update_surrounding_vehicles()
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
            self, n_orig_ahead: int = 0, n_orig_behind: int = 0,
            n_dest_ahead: int = 0, n_dest_behind: int = 0,
            is_acceleration_optimal: bool = True
    ):
        self.create_test_scenario(
            'open_loop', n_orig_ahead, n_orig_behind, n_dest_ahead,
            n_dest_behind, is_acceleration_optimal)
        # self.create_base_lane_change_initial_state(self._has_lo, self._has_fo,
        #                                            self._has_ld, self._has_fd)

    def set_desired_lane_changes(self):
        self.vehicle_group.set_vehicles_lane_change_direction(
            self.lc_vehicle_names, 1
        )


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
