from typing import List, Type

import numpy as np
import pandas as pd

from constants import lane_width
from vehicle_handler import BaseVehicle


class VehicleArray:
    """ Class to help manage groups of vehicles """

    def __init__(self):
        self.vehicles = None
        self.n_per_lane = None
        self.n_vehs = 0
        self.n_states, self.n_inputs = 0, 0
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
        shift = 0
        for vehicle in self.vehicles:
            state_indices.append(shift + vehicle.state_idx[state_name])
            shift += vehicle.n_states
        return state_indices

    def create_uniform_array(self, n_per_lane: List[int],
                             vehicle_class: Type[BaseVehicle],
                             free_flow_speed: float):
        """
        Populates a list of vehicles with instances of 'vehicle_class'. All
        vehicles have the same desired free flow speed.
        :param n_per_lane: Number of vehicles per lane, starting by the
         leftmost lane
        :param vehicle_class: Class of all vehicle instances
        :param free_flow_speed: Desired free flow speed of all vehicles
        :return:
        """
        self.n_per_lane = n_per_lane
        self.n_vehs = sum(n_per_lane)
        self.vehicles = []
        for i in range(self.n_vehs):
            self.vehicles.append(vehicle_class(free_flow_speed))
            self.n_states += self.vehicles[-1].n_states
            self.n_inputs += self.vehicles[-1].n_inputs

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

    def set_initial_state(self, x0, y0, v0=None):
        """
        All vehicles start at the center of their respective lanes, at the given
        distance of the vehicle ahead, and with their desired free flow speed.

        :param x0: Longitudinal position of each vehicle
        :param y0: Lateral position of each vehicle
        :param v0: Initial speed of each vehicle. Only used if speed is one of
         the model states
        :return: The array with the initial states of all vehicles
        """

        initial_state = []
        for i, vehicle in enumerate(self.vehicles):
            vehicle.set_initial_state(x0[i], y0[i], 0, v0[i])
            initial_state.extend(vehicle.initial_state)
        return initial_state

    def create_desired_final_state(self, tf):
        """
        Creates a vector of desired final states.

        :param tf: Final time
        :return:
        """
        state_final = []
        for i, vehicle in enumerate(self.vehicles):
            x0 = vehicle.initial_state[vehicle.state_idx['x']]
            y0 = vehicle.initial_state[vehicle.state_idx['y']]
            v_ff = vehicle.free_flow_speed
            xf = x0 + v_ff * tf
            initial_lane = y0 // 4
            # yf = y0 + lane_change_directions[i] * lane_width
            yf = y0 + (-1) ** initial_lane * lane_width
            state_vector = vehicle.create_state_vector(
                xf, yf, 0, v_ff)
            state_final.append(state_vector)
        return np.concatenate(state_final)

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
        return full_state

    def update(self, states, inputs, params):
        """
        Computes the states derivatives
        :param states: Current states of all vehicles
        :param inputs: Current inputs of all vehicles
        :param params:
        :return:
        """
        dxdt = np.zeros(len(states))
        state_init = 0
        input_init = 0
        for vehicle in self.vehicles:
            state_end = state_init + vehicle.n_states
            input_end = input_init + vehicle.n_inputs
            ego_states = states[state_init: state_end]
            ego_inputs = inputs[input_init: input_end]
            leader_states = vehicle.get_leader_states(ego_states, states)
            dxdt[state_init: state_end] = vehicle.dynamics(
                ego_states, ego_inputs, leader_states)
            state_init = state_end
            input_init = input_end
        return dxdt

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
        all_data = []
        state_init = 0
        input_init = 0
        for vehicle in self.vehicles:
            state_end = state_init + vehicle.n_states
            input_end = input_init + vehicle.n_inputs
            ego_states = states[state_init: state_end]
            ego_inputs = inputs[input_init: input_end]
            vehicle_df = vehicle.to_dataframe(time, ego_states, ego_inputs)
            vehicle_df['id'] = vehicle.id
            state_init = state_end
            input_init = input_end
            all_data.append(vehicle_df)
        return pd.concat(all_data)
