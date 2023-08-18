from typing import List, Type

import numpy as np
import pandas as pd

from constants import lane_width
from vehicle_handler import BaseVehicle


class VehicleArray:
    """ Class to help manage groups of vehicles """

    def __init__(self):
        self.vehicles: List[BaseVehicle] = []
        self.n_vehs = 0
        self.n_states, self.n_inputs = 0, 0
        # Maps the vehicle id (position in the 'vehicles' list) to the index of
        # its states in the state vector containing all vehicles
        self.state_idx_map = []
        # Maps the vehicle id (position in the 'vehicles' list) to the index of
        # its inputs in the full input vector
        self.input_idx_map = []
        # self.initial_state = None

    def get_input_limits(self):
        lower_bounds = []
        upper_bounds = []
        for vehicle in self.vehicles:
            veh_lower_input, veh_upper_input = vehicle.get_input_limits()
            lower_bounds.extend(veh_lower_input)
            upper_bounds.extend(veh_upper_input)
        return lower_bounds, upper_bounds

    def get_desired_input(self):
        desired_inputs = []
        for vehicle in self.vehicles:
            veh_desired_inputs = vehicle.get_desired_input()
            desired_inputs.extend(veh_desired_inputs)
        return desired_inputs

    def get_state_indices(self, state_name: str) -> List[int]:
        state_indices = []
        for vehicle in self.vehicles:
            state_indices.append(self.state_idx_map[vehicle.id]
                                 + vehicle.state_idx[state_name])
        return state_indices

    def get_vehicle_state_vector_by_id(self, veh_id, states):
        start_idx = self.state_idx_map[veh_id]
        end_idx = self.state_idx_map[veh_id] + self.vehicles[veh_id].n_states
        return states[start_idx: end_idx]

    def get_vehicle_inputs_vector_by_id(self, veh_id, inputs):
        start_idx = self.input_idx_map[veh_id]
        end_idx = self.input_idx_map[veh_id] + self.vehicles[veh_id].n_inputs
        return inputs[start_idx: end_idx]

    def get_a_vehicle_state_by_id(self, veh_id, states, state_name):
        vehicle_states = self.get_vehicle_state_vector_by_id(veh_id, states)
        vehicle = self.vehicles[veh_id]
        return vehicle.get_state(vehicle_states, state_name)

    def get_a_vehicle_input_by_id(self, veh_id, inputs, input_name):
        vehicle_inputs = self.get_vehicle_inputs_vector_by_id(veh_id, inputs)
        vehicle = self.vehicles[veh_id]
        return vehicle.get_input(vehicle_inputs, input_name)

    def set_a_vehicle_free_flow_speed(self, veh_id, v_ff):
        self.vehicles[veh_id].set_free_flow_speed(v_ff)

    def create_vehicle_array(self, vehicle_classes: List[Type[BaseVehicle]],
                             free_flow_speeds: List[float]):
        """

        Populates the list of vehicles following the given classes
        :param vehicle_classes: Class of each vehicle instances
        :param free_flow_speeds: Desired free flow speed of each vehicle
        :return:
        """
        self.n_vehs = len(vehicle_classes)
        for i in range(len(vehicle_classes)):
            vehicle = vehicle_classes[i]()
            vehicle.set_free_flow_speed(free_flow_speeds[i])
            self.vehicles.append(vehicle)
            self.state_idx_map.append(self.n_states)
            self.input_idx_map.append(self.n_inputs)
            self.n_states += self.vehicles[-1].n_states
            self.n_inputs += self.vehicles[-1].n_inputs

    def create_uniform_array(self, n_vehs: int,
                             vehicle_class: Type[BaseVehicle],
                             free_flow_speed: float):
        """
        Populates a list of vehicles with instances of 'vehicle_class'. All
        vehicles have the same desired free flow speed.
        :param n_vehs: Number of vehicles
        :param vehicle_class: Class of all vehicle instances
        :param free_flow_speed: Desired free flow speed of all vehicles
        :return:
        """
        self.create_vehicle_array([vehicle_class] * n_vehs,
                                  [free_flow_speed] * n_vehs)

    def create_input_names(self) -> List[str]:
        input_names = []
        for vehicle in self.vehicles:
            str_id = str(vehicle.id)
            input_names.extend([name + str_id for name
                                in vehicle.input_names])
        return input_names

    def create_output_names(self) -> List[str]:
        output_names = []
        for vehicle in self.vehicles:
            str_id = str(vehicle.id)
            output_names.extend([name + str_id for name
                                in vehicle.state_names])
        return output_names

    def create_full_state_vector(self, x, y, theta, v=None):
        """
        Creates a single state vector

        :param x: Longitudinal position of each vehicle
        :param y: Lateral position of each vehicle
        :param theta: Orientation of each vehicle
        :param v: Initial speed of each vehicle. Only used if speed is one of
         the model states
        :return: The array with the states of all vehicles
        """

        full_state = []
        for i, vehicle in enumerate(self.vehicles):
            full_state.extend(vehicle.create_state_vector(
                x[i], y[i], theta[i], v[i]))
        return np.array(full_state)

    def assign_leaders(self, state):
        """
        Determines the leading vehicle for each vehicle, and saves the result
        at each vehicle

        :param state:
        :return:
        """

        for ego_vehicle in self.vehicles:
            ego_states = self.get_vehicle_state_vector_by_id(ego_vehicle.id,
                                                             state)
            ego_x = ego_vehicle.get_state(ego_states, 'x')
            ego_y = ego_vehicle.get_state(ego_states, 'y')
            leader_x = np.inf
            for other_vehicle in self.vehicles:
                other_states = self.get_vehicle_state_vector_by_id(
                    other_vehicle.id, state)
                other_x = other_vehicle.get_state(other_states, 'x')
                other_y = other_vehicle.get_state(other_states, 'y')
                if (np.abs(other_y - ego_y) < lane_width / 2  # same lane
                        and ego_x < other_x < leader_x):  # ahead and close
                    leader_x = other_x
                    ego_vehicle.leader_id = other_vehicle.id

    def assign_dest_lane_vehicles(self, state, lc_veh_id, y_target):
        ego_vehicle = self.vehicles[lc_veh_id]
        ego_states = self.get_vehicle_state_vector_by_id(ego_vehicle.id,
                                                         state)
        ego_x = ego_vehicle.get_state(ego_states, 'x')
        dest_lane_follower_x = -np.inf
        dest_lane_leader_x = np.inf
        for other_vehicle in self.vehicles:
            other_states = self.get_vehicle_state_vector_by_id(
                other_vehicle.id, state)
            other_x = other_vehicle.get_state(other_states, 'x')
            other_y = other_vehicle.get_state(other_states, 'y')
            if np.abs(other_y - y_target) < lane_width / 2:
                if ego_x < other_x < dest_lane_leader_x:
                    dest_lane_leader_x = other_x
                    ego_vehicle.destination_leader_id = other_vehicle.id
                elif dest_lane_follower_x < other_x < ego_x:
                    dest_lane_follower_x = other_x
                    ego_vehicle.destination_follower_id = other_vehicle.id

    def compute_free_flow_displacement(self, tf: float) -> np.ndarray:
        delta_x = np.zeros(len(self.vehicles))
        for i, veh in enumerate(self.vehicles):
            delta_x[i] = veh.free_flow_speed * tf
        return delta_x

    def compute_gap_to_leader(self, ego_id, states):
        """
        Computes the gap between the vehicle with id ego_id and its leader.
        If ego doesn't have a leader, returns infinity

        :param ego_id: ID of the ego vehicle
        :param states: Full state vector of all vehicles
        :return:
        """
        ego_vehicle = self.vehicles[ego_id]
        if not ego_vehicle.has_leader():
            return np.inf
        ego_x = self.get_a_vehicle_state_by_id(ego_id, states, 'x')
        leader_x = self.get_a_vehicle_state_by_id(ego_vehicle.leader_id,
                                                  states, 'x')
        return leader_x - ego_x

    def update(self, states, inputs, params):
        """
        Computes the states derivatives
        :param states: Current states of all vehicles
        :param inputs: Current inputs of all vehicles
        :param params:
        :return:
        """
        dxdt = []
        for vehicle in self.vehicles:
            ego_states = self.get_vehicle_state_vector_by_id(vehicle.id, states)
            ego_inputs = self.get_vehicle_inputs_vector_by_id(
                vehicle.id, inputs)
            if vehicle.has_leader():
                leader_states = self.get_vehicle_state_vector_by_id(
                    vehicle.leader_id, states)
            else:
                leader_states = None
            dxdt.extend(vehicle.dynamics(ego_states, ego_inputs, leader_states))
        return np.array(dxdt)

    def to_dataframe(self, time: np.ndarray,
                     states: np.ndarray, inputs: np.ndarray) -> pd.DataFrame:
        """

        :param time: Array with all simulation time points
        :param states: Matrix of states where each column contains the full
         system state at one time point
        :param inputs: Matrix of inputs where each column contains all the
         system's inputs at one time point
        :return:
        """
        data_per_vehicle = []
        for vehicle in self.vehicles:
            ego_states = self.get_vehicle_state_vector_by_id(vehicle.id,
                                                             states)
            ego_inputs = self.get_vehicle_inputs_vector_by_id(vehicle.id,
                                                              inputs)
            vehicle_df = vehicle.to_dataframe(time, ego_states, ego_inputs)
            data_per_vehicle.append(vehicle_df)
        all_data = pd.concat(data_per_vehicle).reset_index()
        return all_data.fillna(0)
