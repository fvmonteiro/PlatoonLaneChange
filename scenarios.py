from abc import ABC, abstractmethod
import pickle
from typing import List, Type, Union

import control as ct
import control.optimal as opt
import numpy as np
import pandas as pd

import controllers.optimal_controller as opt_ctrl
from constants import LANE_WIDTH
import platoon
from vehicle_group import VehicleGroup
from vehicle_models.base_vehicle import BaseVehicle
import vehicle_models.four_state_vehicles as fsv


class SimulationScenario(ABC):
    def __init__(self):
        self.n_per_lane = []
        self.vehicle_group = VehicleGroup()
        BaseVehicle.reset_vehicle_counter()
        # Simulation parameters
        self.tf = None
        # self.initial_state = None

    def create_uniform_vehicles(
            self, n_per_lane: List[int], vehicle_class: Type[BaseVehicle],
            free_flow_speed: float):
        self.n_per_lane = n_per_lane
        self.vehicle_group.create_vehicle_array(
            [vehicle_class] * sum(n_per_lane))
        self.vehicle_group.set_free_flow_speeds(free_flow_speed)

    def create_vehicle_group(self,
                             vehicle_classes: List[List[Type[BaseVehicle]]]):
        for i in range(len(vehicle_classes)):
            self.n_per_lane.append(len(vehicle_classes[i]))
        flat_vehicle_list = [item for sublist in vehicle_classes for
                             item in sublist]
        self.vehicle_group.create_vehicle_array(flat_vehicle_list)

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

    def create_base_lane_change_initial_state(self, v_ff, delta_x):
        """
        Creates an initial state for 5 vehicles: 3 at the origin lane, with
        the lane changing one between other two, and 2 at the destination lane.
        :param v_ff: Dictionary with free-flow speeds for ego, lo, ld and fd
        :param delta_x: Dictionary with deviation from minimum safe gap for lo,
         ld and fd. Positive values lead to smaller-than-safe initial gaps.
        """
        # TODO: how to ensure proper vehicle 'sync' between creates vehicles
        #  and the initial states
        target_y = LANE_WIDTH
        ordered_v_ff = [v_ff['lo'], v_ff['ego'], v_ff['ego'],
                        v_ff['ld'], v_ff['fd']]
        self.set_free_flow_speeds(self.lane_change_scenario_vehicle_filter(
            ordered_v_ff
        ))

        # Initial states
        v0 = self.lane_change_scenario_vehicle_filter(
            [v_ff['lo'], v_ff['lo'], v_ff['lo'], v_ff['lo'], v_ff['lo']])
        ref_gaps = self.vehicle_group.map_values_to_names(
            self.vehicle_group.get_initial_desired_gaps(v0)
        )
        x_ego = max(ref_gaps.get('fd', 0), ref_gaps.get('fo', 0))
        y_ego = 0
        x0 = [x_ego + ref_gaps['ego'] - delta_x['lo'],
              x_ego,
              x_ego - ref_gaps.get('fo', 0),
              x_ego + ref_gaps['ego'] - delta_x['ld'],
              x_ego - ref_gaps.get('fd', 0) + delta_x['fd']]
        x0 = self.lane_change_scenario_vehicle_filter(x0)
        y0 = self.lane_change_scenario_vehicle_filter([y_ego, y_ego, y_ego,
                                                       target_y, target_y])
        theta0 = self.lane_change_scenario_vehicle_filter([0., 0., 0., 0., 0.])
        self.vehicle_group.set_vehicles_initial_states(x0, y0, theta0, v0)

    def test_scenario(self):
        all_names = ['lo', 'ego', 'fo', 'ld', 'fd']
        self.lane_change_scenario_vehicle_filter(all_names)
        self.vehicle_group.set_vehicle_names(all_names)
        # Free-flow speeds
        v_ff = {'ego': 10, 'lo': 10}
        v_ff['ld'] = 1.0 * v_ff['lo']
        v_ff['fd'] = 1.0 * v_ff['lo']
        # Deviation from minimum safe gap
        delta_x = {'lo': 0.0, 'ld': 4.0, 'fd': 0.0}
        self.create_base_lane_change_initial_state(
            v_ff, delta_x)
        # self.vehicle_group.make_all_connected()

    def lane_change_scenario_vehicle_filter(self, values: List):
        # Used together with lane change scenarios.
        # TODO: poor organization
        if self.n_per_lane[0] < 3:
            values.pop(2)
            if self.n_per_lane[0] < 2:
                values.pop(0)
        if self.n_per_lane[1] < 2:
            values.pop()
            if self.n_per_lane[1] < 1:
                values.pop()
        return values

    def response_to_dataframe(self) -> pd.DataFrame:
        return self.vehicle_group.to_dataframe()

    def simulate_one_time_step(self, new_time, open_loop_controls=None):
        if open_loop_controls is None:
            open_loop_controls = {}
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
        vehicle_classes = ([[fsv.ClosedLoopVehicle]
                            * n_vehs])
        v_ff = [10] + [12] * n_vehs
        self.create_vehicle_group(vehicle_classes)
        self.set_free_flow_speeds(v_ff)

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
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            self.simulate_one_time_step(time[i+1])


class FastLaneChange(SimulationScenario):

    def __init__(self):
        super().__init__()
        vehicle_classes = ([[fsv.SafeAccelOpenLoopLCVehicle]])
        self.create_vehicle_group(vehicle_classes)
        self.create_initial_state()

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
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        t_c = 1.08
        for i in range(len(time) - 1):
            if time[i] <= t_c:  # ego.get_y() <= 2 * LANE_WIDTH / 3:
                phi = ego.phi_max
            elif time[i] <= 2 * t_c:  # np.abs(ego.get_theta()) >= 1e-4:
                phi = -ego.phi_max
            else:
                phi = 0
            inputs[ego.id] = np.array([phi])
            self.simulate_one_time_step(time[i+1], inputs)


class LaneChangeScenario(SimulationScenario):

    def __init__(self, lc_veh_class: Type[BaseVehicle], n_orig: int = 3,
                 n_dest: int = 2):
        super().__init__()

        self.lc_intention_time = 0.0
        if n_orig == 3:
            orig_veh_classes = [fsv.ClosedLoopVehicle, lc_veh_class,
                                fsv.ClosedLoopVehicle]
        elif n_orig == 2:
            orig_veh_classes = [fsv.ClosedLoopVehicle, lc_veh_class]
        else:
            orig_veh_classes = [lc_veh_class]

        veh_classes = (
            [orig_veh_classes, [fsv.ClosedLoopVehicle] * n_dest])
        self.create_vehicle_group(veh_classes)
        self.create_initial_state()

    @classmethod
    def closed_loop(cls, n_orig: int = 3, n_dest: int = 2):
        """
        Creates a scenario instance where all vehicles have longitudinal and
        lateral feedback controllers.
        :param n_orig: number of vehicles on the origin lane. Options:
         1: only lane changing, 2: includes leader, 3: includes follower
        :param n_dest: number of vehicles on the destination lane. Options:
         1: leader, 2: includes followers
        """
        return cls(fsv.ClosedLoopVehicle, n_orig, n_dest)

    @classmethod
    def optimal_control(cls, n_orig: int = 3, n_dest: int = 2):
        return cls(fsv.SafeAccelOptimalLCVehicle, n_orig, n_dest)

    def create_initial_state(self):
        self.test_scenario()
        # self.vehicle_group.make_all_connected()

    def run(self, final_time):
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            if np.isclose(time[i], self.lc_intention_time):
                self.vehicle_group.set_single_vehicle_lane_change_direction(
                    'ego', 1)
            self.simulate_one_time_step(time[i+1])

    # def run_ocp_solution(self) -> None:
    #     """
    #     Calls the control libraries function for running the dynamic system
    #     given the optimal control problem solution
    #     :return: Nothing. Results are stored internally
    #     """
    #     self.ocp_response = self.controller.get_ocp_response()


class ModeSwitchTests(LaneChangeScenario):

    def run(self, final_time):
        self.tf = final_time
        ego_id = self.vehicle_group.get_vehicle_id_by_name('ego')
        lo_id = self.vehicle_group.get_vehicle_id_by_name('lo')
        ld_id = self.vehicle_group.get_vehicle_id_by_name('ld')
        ego_modes = {ego_id: [(0.0, lo_id), (3.0, ld_id)]}
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            if np.isclose(time[i], 2.0, atol=dt/10):
                self.vehicle_group.set_single_vehicle_lane_change_direction(
                    'ego', 1)
                self.vehicle_group.set_ocp_leader_sequence(ego_modes)
            self.simulate_one_time_step(time[i+1])


class PlatoonLaneChangeScenario(SimulationScenario):

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        lc_veh_class = fsv.PlatoonVehicle
        n_dest_lane_vehs = 2
        veh_classes = (
            [[fsv.ClosedLoopVehicle, lc_veh_class, fsv.ClosedLoopVehicle],
             [fsv.ClosedLoopVehicle] * n_dest_lane_vehs])
        self.create_vehicle_group(veh_classes)
        self.create_initial_state()
        PlatoonLaneChangeScenario.create_platoons([[]])

    def create_initial_state(self):
        self.test_scenario()
        self.vehicle_group.make_all_connected()

    @staticmethod
    def create_platoons(platoon_assignment: List[List[fsv.PlatoonVehicle]]):
        """
        Creates platoons and include vehicles in them
        :param platoon_assignment:
        :return:
        """
        for platoon_vehicles in platoon_assignment:
            new_platoon = platoon.Platoon()
            for veh in sorted(platoon_vehicles,
                              key=lambda x: x.get_x(),
                              reverse=True):
                new_platoon.add_vehicle(veh.id)
                veh.set_platoon(new_platoon)

    def run(self, final_time):
        self.tf = final_time
        dt = 1e-2
        time = np.arange(0, self.tf, dt)
        self.vehicle_group.initialize_state_matrices(len(time))
        self.vehicle_group.update_surrounding_vehicles()
        for i in range(len(time) - 1):
            if i == 0:
                self.vehicle_group.set_single_vehicle_lane_change_direction(
                    self.lc_veh_id, 1)
            self.simulate_one_time_step(time[i + 1])


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
        self.controller.find_lane_change_trajectory(
            0.0, self.vehicle_group.vehicles, [])
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
        self.vehicle_group.initialize_state_matrices(len(time))
        for i in range(len(time) - 1):
            current_inputs = self.controller.get_input(time[i], veh_ids)
            self.vehicle_group.update_vehicle_modes()
            self.vehicle_group.determine_inputs(current_inputs)
            self.vehicle_group.compute_derivatives()
            self.vehicle_group.update_states(time[i + 1])
            self.vehicle_group.update_surrounding_vehicles()

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
            veh.update_target_y()


class LaneChangeWithConstraints(ExternalOptimalControlScenario):
    """
    Used to test how to code safety constraints
    """

    def __init__(self):
        super().__init__()
        self.lc_veh_id = 1  # lane changing vehicle

        n_dest_lane_vehs = 2
        veh_classes = (
            [[fsv.ClosedLoopVehicle,
              fsv.SafeAccelOpenLoopLCVehicle,  # lane changing veh
              fsv.ClosedLoopVehicle],
             [fsv.ClosedLoopVehicle] * n_dest_lane_vehs])
        self.create_vehicle_group(veh_classes)

        self.min_lc_x = 20  # only used for initial tests

    def create_initial_state(self):
        self.test_scenario()

    def set_desired_lane_changes(self):
        lc_vehicle = self.vehicle_group.vehicles[self.lc_veh_id]
        lc_vehicle.set_lane_change_direction(1)
        lc_vehicle.update_target_y()
