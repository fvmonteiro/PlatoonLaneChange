from typing import List

import numpy as np
import pandas as pd

from constants import lane_width
from vehicle_handler import BaseVehicle


class VehicleArray:
    """ Class to help manage groups of vehicles """

    def __init__(self, n_per_lane: List[int], vehicle_class: BaseVehicle):
        self.n_per_lane = n_per_lane
        self.n_vehs = sum(self.n_per_lane)
        self.vehicle_class = vehicle_class
        self.initial_state = None

    def get_state_for_single_vehicle(self, full_state, veh_number: int,
                                     state_name: str) -> float:
        return self.vehicle_class.get_state(
            self.select_single_vehicle_states(full_state, veh_number),
            state_name)

    def get_input_for_single_vehicle(self, full_input, veh_number: int,
                                     input_name: str) -> float:
        return self.vehicle_class.get_input(
            self.select_single_vehicle_inputs(full_input, veh_number),
            input_name)

    def select_single_vehicle_states(self, full_state: np.ndarray,
                                     veh_number: int):
        starting_idx = veh_number * self.vehicle_class.n_states
        return full_state[starting_idx:
                          starting_idx + self.vehicle_class.n_states]

    def select_single_vehicle_inputs(self, full_state: np.ndarray,
                                     veh_number: int):
        starting_idx = veh_number * self.vehicle_class.n_inputs
        return full_state[starting_idx:
                          starting_idx + self.vehicle_class.n_inputs]

    def get_state_for_all_vehicles(self, full_state, state_name: str):
        state_value = np.zeros(self.n_vehs)
        for i in range(self.n_vehs):
            state_value[i] = self.get_state_for_single_vehicle(full_state, i,
                                                               state_name)
        return state_value

    def get_input_for_all_vehicles(self, full_input, input_name: str):
        input_value = np.zeros(self.n_vehs)
        for i in range(self.n_vehs):
            input_value[i] = self.get_input_for_single_vehicle(full_input, i,
                                                               input_name)
        return input_value

    def create_input_names(self) -> List[str]:
        input_names = []
        for i in range(self.n_vehs):
            str_i = str(i)
            input_names.extend([name + str_i for name
                                in self.vehicle_class.input_names])
        return input_names

    def create_output_names(self) -> List[str]:
        output_names = []
        for i in range(self.n_vehs):
            str_i = str(i)
            output_names.extend([name + str_i for name
                                 in self.vehicle_class.state_names])
        return output_names

    def create_initial_state(self, gap: float, v0: float = None):
        """
        Creates a vector of initial conditions where vehicles are organized by
        lanes. Within a lane, vehicle i+1 is 'gap' meters behind vehicle i.

        :param gap: Gap between consecutive vehicles on the same lane
        :param v0: initial speed. Only used if speed is one of the model states
        :return:
        """
        y0 = [-2 + lane_width * i for i in range(len(self.n_per_lane))]

        state0 = []
        for i in range(len(self.n_per_lane)):
            n = self.n_per_lane[i]
            position = gap * (n - 1)
            for j in range(n):
                state_vector = self.vehicle_class.create_state_vector(
                    position, y0[i], 0, v0)
                position -= gap
                state0.append(state_vector)

        self.initial_state = np.concatenate(state0)
        return self.initial_state

    def create_desired_final_state(self, v_ff, tf):
        """
        Creates a vector of desired final states.

        :param v_ff: Free-flow (desired) velocity
        :param tf: Final time
        :return:
        """
        n_states = self.vehicle_class.n_states
        state_final = []
        n_previous = 0
        for i in range(len(self.n_per_lane)):
            n = self.n_per_lane[i]
            for j in range(n):
                starting_idx = (n_previous + j) * n_states
                x0 = self.initial_state[starting_idx
                                        + self.vehicle_class.state_idx['x']]
                y0 = self.initial_state[starting_idx
                                        + self.vehicle_class.state_idx['y']]
                x = v_ff * tf + x0
                y = y0 + (-1) ** i * lane_width
                state_vector = self.vehicle_class.create_state_vector(
                    x, y, 0, v_ff)
                state_final.append(state_vector)
            n_previous += n
        return np.concatenate(state_final)

    def update(self, states, inputs, params):
        """
        Computes the states derivatives
        :param states: Current states of all vehicles
        :param inputs: Current inputs of all vehicles
        :param params:
        :return:
        """
        dxdt = np.zeros(len(states))
        n_states = self.vehicle_class.n_states
        n_inputs = self.vehicle_class.n_inputs
        for i in range(self.n_vehs):
            states_idx = [j for j in range(i * n_states, (i + 1) * n_states)]
            inputs_idx = [j for j in range(i * n_inputs, (i + 1) * n_inputs)]
            ego_states = states[states_idx]
            leader_states = self.vehicle_class.get_leader_states(ego_states,
                                                                 states)
            dxdt[states_idx] = self.vehicle_class.dynamics(
                ego_states, inputs[inputs_idx], leader_states)
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
        for i in range(self.n_vehs):
            vehicle_data = self.vehicle_class.to_dataframe(
                time, self.select_single_vehicle_states(states, i),
                self.select_single_vehicle_inputs(inputs, i))
            vehicle_data['id'] = i
            all_data.append(vehicle_data)
        return pd.concat(all_data)
