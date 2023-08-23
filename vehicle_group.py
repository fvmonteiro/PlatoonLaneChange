from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd

from vehicle_models import BaseVehicle


class VehicleGroup:
    """ Class to help manage groups of vehicles """

    def __init__(self):
        self.vehicles: Dict[int, BaseVehicle] = {}
        # Often, we need to iterate over all vehicles in the order they were
        # created. The list below make that easy
        self.sorted_vehicle_ids = None
        self.n_vehs = 0
        # Maps the vehicle id (position in the 'vehicles' list) to the index of
        # its inputs in the full input vector
        self.input_idx_map = []
        # self.initial_state = None

    def get_vehicle_inputs_vector_by_id(self, veh_id, inputs):
        start_idx = self.input_idx_map[veh_id]
        end_idx = self.input_idx_map[veh_id] + self.vehicles[veh_id].n_inputs
        return inputs[start_idx: end_idx]

    def get_free_flow_speeds(self):
        v_ff = np.zeros(self.n_vehs)
        for veh_id in self.sorted_vehicle_ids:
            v_ff[veh_id] = self.vehicles[veh_id].free_flow_speed
        return v_ff

    def get_full_initial_state_vector(self):
        initial_state = []
        for veh_id in self.sorted_vehicle_ids:
            initial_state.extend(self.vehicles[veh_id].initial_state)
        return initial_state

    def get_all_vehicles(self):
        return self.vehicles.values()

    def set_a_vehicle_free_flow_speed(self, veh_id, v_ff):
        self.vehicles[veh_id].set_free_flow_speed(v_ff)

    def set_free_flow_speeds(self, values: Union[float, List, np.ndarray]):
        if np.isscalar(values):
            values = [values] * self.n_vehs
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.set_free_flow_speed(values[veh_id])

    def set_vehicles_initial_states(self, x0, y0, theta0, v0):
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.set_initial_state(x0[veh_id], y0[veh_id],
                                      theta0[veh_id], v0[veh_id])

    def set_lane_change_direction_by_id(self, veh_id: int, lc_direction: int):
        self.vehicles[veh_id].set_lane_change_direction(lc_direction)

    def initialize_state_matrices(self, n_samples: int):
        for vehicle in self.vehicles.values():
            vehicle.initialize_states(n_samples)

    def create_vehicle_array(self, vehicle_classes: List[Type[BaseVehicle]]):
        """

        Populates the list of vehicles following the given classes
        :param vehicle_classes: Class of each vehicle instances
        :return:
        """
        self.vehicles = {}
        self.sorted_vehicle_ids = []
        self.n_vehs = len(vehicle_classes)
        n_states, n_inputs = 0, 0
        for veh_class in vehicle_classes:
            vehicle = veh_class()
            self.sorted_vehicle_ids.append(vehicle.id)
            self.vehicles[vehicle.id] = vehicle
            self.input_idx_map.append(n_inputs)
            n_states += vehicle.n_states
            n_inputs += vehicle.n_inputs

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
        self.create_vehicle_array([vehicle_class] * n_vehs)
        self.set_free_flow_speeds(free_flow_speed)

    def create_input_names(self) -> List[str]:
        input_names = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            str_id = str(veh_id)
            input_names.extend([name + str_id for name
                                in vehicle.input_names])
        return input_names

    def create_output_names(self) -> List[str]:
        output_names = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            str_id = str(veh_id)
            output_names.extend([name + str_id for name
                                in vehicle.state_names])
        return output_names

    def create_full_state_vector(self, x, y, theta, v=None):
        """
        Creates a single state vector.

        :param x: Longitudinal position of each vehicle
        :param y: Lateral position of each vehicle
        :param theta: Orientation of each vehicle
        :param v: Initial speed of each vehicle. Only used if speed is one of
         the model states
        :return: The array with the states of all vehicles
        """

        full_state = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            full_state.extend(vehicle.create_state_vector(
                x[veh_id], y[veh_id], theta[veh_id], v[veh_id]))
        return np.array(full_state)

    def update_surrounding_vehicles(self):
        for ego_vehicle in self.vehicles.values():
            ego_vehicle.find_orig_lane_leader(self.vehicles.values())
            ego_vehicle.find_dest_lane_vehicles(self.vehicles.values())
            ego_vehicle.update_target_leader(self.vehicles)

    def compute_steering_wheel_angle(self):
        delta = np.zeros(self.n_vehs)
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            delta[veh_id] = (
                    vehicle.compute_steering_wheel_angle())
        return delta

    def update_modes(self):
        for vehicle in self.vehicles.values():
            vehicle.update_mode(self.vehicles)

    def compute_derivatives(self, inputs):
        """
        Computes the states derivatives
        :param inputs: Current inputs of all vehicles
        :return:
        """
        dxdt = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            ego_inputs = inputs[veh_id]
            vehicle.compute_derivatives(ego_inputs, self.vehicles)
            dxdt.extend(vehicle.get_derivatives())
        return np.array(dxdt)

    def update_states(self, new_time):
        for vehicle in self.vehicles.values():
            vehicle.update_states(new_time)

    def to_dataframe(self, inputs: np.ndarray) -> pd.DataFrame:
        """

        :param inputs: Matrix of inputs where each column contains all the
         system's inputs at one time point
        :return:
        """
        data_per_vehicle = []
        for vehicle in self.vehicles.values():
            ego_inputs = self.get_vehicle_inputs_vector_by_id(vehicle.id,
                                                              inputs)
            vehicle_df = vehicle.to_dataframe(ego_inputs)
            data_per_vehicle.append(vehicle_df)
        all_data = pd.concat(data_per_vehicle).reset_index()
        return all_data.fillna(0)
