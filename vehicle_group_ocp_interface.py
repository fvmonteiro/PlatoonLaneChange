from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

import system_operating_mode as som
import vehicle_models.base_vehicle as base


# ========= Functions passed to the optimal control library methods ========== #
def vehicles_derivatives(t, states, inputs, params):
    """
    Implements the kinematic bicycle model with reference at the vehicles C.G.
    Follows the model of updfcn of the control package.
    :param t: time
    :param states: Array with states of all vehicles [x1, y1, ..., xN, yN]
    :param inputs: Array with inputs of all vehicles [u11, u12, ..., u1N, u2N]
    :param params: Dictionary which must contain the vehicle type
    :return: state update function
    """
    vehicle_group_interface: VehicleGroupInterface = params['vehicle_group']

    return vehicle_group_interface.compute_derivatives(t, states, inputs)


def vehicle_output(t, x, u, params):
    return x  # return (full state)


class VehicleGroupInterface:
    """ Class to help manage groups of vehicles """

    def __init__(self, vehicles: Dict[int, base.BaseVehicle],
                 ego_id: int = None):
        """

        :param vehicles: All simulation vehicles
        :param ego_id: If given, assumes this vehicle as the center of
         the system, i.e., its (x, y) = (0, 0)
        """
        self.vehicles: Dict[int, base.BaseVehicleInterface] = {}
        # Often, we need to iterate over all vehicles in the order they were
        # created. The list below make that easy
        self.sorted_vehicle_ids = None
        self.n_vehs = 0
        self.n_states, self.n_inputs = 0, 0
        # Maps the vehicle id to the index of its states in the state vector
        # containing all vehicles
        self.state_idx_map: Dict[int, int] = {}
        # Maps the vehicle id to the index of its inputs in the full input
        # vector
        self.input_idx_map: Dict[int, int] = {}

        self.create_vehicle_interfaces(vehicles, ego_id)

    def get_input_limits(self):
        lower_bounds = []
        upper_bounds = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            veh_lower_input, veh_upper_input = vehicle.get_input_limits()
            lower_bounds.extend(veh_lower_input)
            upper_bounds.extend(veh_upper_input)
        return lower_bounds, upper_bounds

    def get_desired_input(self):
        desired_inputs = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            veh_desired_inputs = vehicle.get_desired_input()
            desired_inputs.extend(veh_desired_inputs)
        return desired_inputs

    def get_initial_state(self):
        """
        Gets all vehicles' current states, which are used as the starting
        point for the ocp
        """
        initial_state = []
        for veh_id in self.sorted_vehicle_ids:
            initial_state.extend(self.vehicles[veh_id].get_initial_state())
        return np.array(initial_state)

    def get_initial_guess(self, time_points: np.ndarray):
        # estimated_safe_time: float):
        n_ctrl_points = len(time_points)
        initial_guess = []
        samples_in_a_second = sum(time_points < 1.0)
        for veh_id in self.sorted_vehicle_ids:
            min_phi, max_phi = self.vehicles[veh_id].get_input_limits()
            # only vehs with optimal control have bounds, but we should find a
            # better way of testing here since this might change
            if len(min_phi) > 0:
                initial_guess.extend(
                    max_phi * samples_in_a_second
                    + min_phi * samples_in_a_second
                    + [0.0] * (len(time_points) - 2 * samples_in_a_second))
        return np.array(initial_guess).reshape(-1, n_ctrl_points)

    def get_state_indices(self, state_name: str) -> List[int]:
        """
        Get all the indices for a certain state within the full state vector
        """
        state_indices = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            state_indices.append(self.state_idx_map[vehicle.get_id()]
                                 + vehicle.state_idx[state_name])
        return state_indices

    def get_a_vehicle_state_index(self, veh_id: int, state_name: str) -> int:
        """
        Get the index of a state for a vehicle within the full state vector
        """
        return (self.state_idx_map[veh_id]
                + self.vehicles[veh_id].state_idx[state_name])

    def get_vehicle_state_vector_by_id(self, veh_id: int, states: np.ndarray
                                       ) -> np.ndarray:
        start_idx = self.state_idx_map[veh_id]
        end_idx = self.state_idx_map[veh_id] + self.vehicles[veh_id].n_states
        return states[start_idx: end_idx]

    def get_vehicle_inputs_vector_by_id(self, veh_id: int, inputs: np.ndarray
                                        ) -> np.ndarray:
        start_idx = self.input_idx_map[veh_id]
        end_idx = self.input_idx_map[veh_id] + self.vehicles[veh_id].n_inputs
        return inputs[start_idx: end_idx]

    def get_a_vehicle_state_by_id(self, veh_id, states, state_name):
        vehicle_states = self.get_vehicle_state_vector_by_id(veh_id, states)
        vehicle = self.vehicles[veh_id]
        return vehicle.select_state_from_vector(vehicle_states, state_name)

    def get_a_vehicle_input_by_id(self, veh_id, inputs, input_name):
        vehicle_inputs = self.get_vehicle_inputs_vector_by_id(veh_id, inputs)
        vehicle = self.vehicles[veh_id]
        return vehicle.select_input_from_vector(vehicle_inputs, input_name)

    def set_mode_sequence(self, mode_sequence: som.ModeSequence):
        leader_sequences = som.mode_sequence_to_leader_sequence(mode_sequence)
        for foll_id, sequence in leader_sequences.items():
            self.vehicles[foll_id].set_leader_sequence(sequence)

    def create_input_names(self) -> List[str]:
        input_names = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            str_id = str(veh_id)
            input_names.extend([name + str_id for name
                                in vehicle.input_names])
        return input_names

    def create_vehicle_interfaces(
            self, vehicles: Dict[int, base.BaseVehicle], ego_id: int = None
    ) -> None:
        """

        :param vehicles: All simulation vehicles
        :param ego_id: If given, assumes this vehicle as the center of
         the system, i.e., its (x, y) = (0, 0)
        :return:
        """
        if ego_id is None:
            shift_map = {'x': 0., 'y': 0.}
        else:
            shift_map = {'x': -vehicles[ego_id].get_x(),
                         'y': -vehicles[ego_id].get_y()}

        self.sorted_vehicle_ids = []
        for veh_id in sorted(vehicles.keys()):
            vehicle_interface = vehicles[veh_id].get_ocp_interface()
            vehicle_interface.shift_initial_state(shift_map)
            self.sorted_vehicle_ids.append(vehicle_interface.get_id())
            self.vehicles[veh_id] = vehicle_interface
            self.state_idx_map[veh_id] = self.n_states
            self.input_idx_map[veh_id] = self.n_inputs
            self.n_states += vehicle_interface.n_states
            self.n_inputs += vehicle_interface.n_inputs
        self.n_vehs = len(self.vehicles)

    def create_output_names(self) -> List[str]:
        output_names = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            str_id = str(veh_id)
            output_names.extend([name + str_id for name
                                 in vehicle.state_names])
        return output_names

    def create_desired_state(self, tf: float):
        """
        Creates a full (for all vehicles) state vector where all vehicles
        travel at their current speeds, and are at their desired lateral
        positions
        """
        # Note: xf and vf are irrelevant with the accel feedback model
        desired_state = []
        for veh_id in self.sorted_vehicle_ids:
            veh = self.vehicles[veh_id]
            veh_states = veh.get_initial_state()
            try:
                vf = veh.select_state_from_vector(veh_states, 'v')
            except KeyError:  # three-state vehicles have v as an input
                vf = veh.select_input_from_vector(veh.get_desired_input(), 'v')
            xf = veh.select_state_from_vector(veh_states, 'x') + vf * tf
            yf = veh.get_target_y()
            thetaf = 0.0
            desired_state.extend(veh.create_state_vector(xf, yf, thetaf, vf))
        return np.array(desired_state)

    def create_state_cost_matrix(self, x_cost: float = 0, y_cost: float = 0,
                                 theta_cost: float = 0, v_cost: float = 0):
        """
        Creates a diagonal cost function where each state of all vehicles gets
        the same weight
        :param x_cost:
        :param y_cost:
        :param theta_cost:
        :param v_cost:
        :return:
        """
        veh_costs = []
        for veh_id in self.sorted_vehicle_ids:
            veh = self.vehicles[veh_id]
            if veh.n_inputs > 0 or True:  # TODO: cost only for controlled vehs?
                veh_costs.extend(self.vehicles[veh_id].create_state_vector(
                    x_cost, y_cost, theta_cost, v_cost))
            else:
                veh_costs.extend([0.0] * veh.n_states)
        return np.diag(veh_costs)

    def create_terminal_cost_matrix(self, cost):
        veh_costs = []
        for veh_id in self.sorted_vehicle_ids:
            veh = self.vehicles[veh_id]
            if veh.n_inputs > 0 or True:  # TODO: cost only for controlled vehs?
                if 'v' in veh.input_names or 'a' in veh.input_names:
                    veh_costs.extend([cost] * veh.n_states)
                else:
                    veh_costs.extend([0, cost, cost, 0])
            else:
                veh_costs.extend([0.0] * veh.n_states)
        return np.diag(veh_costs)

    def compute_derivatives(self, t, states, inputs):
        """
        Computes the states derivatives
        :param t: used to get the current leader
        :param states: Current states of all vehicles
        :param inputs: Current inputs of all vehicles
        :return:
        """
        dxdt = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            ego_states = self.get_vehicle_state_vector_by_id(vehicle.get_id(),
                                                             states)
            ego_inputs = self.get_vehicle_inputs_vector_by_id(
                vehicle.get_id(), inputs)
            if vehicle.has_leader(t):
                leader_states = self.get_vehicle_state_vector_by_id(
                    vehicle.get_current_leader_id(t), states)
            else:
                leader_states = None
            dxdt.extend(vehicle.compute_derivatives(ego_states, ego_inputs,
                                                    leader_states))
        return np.array(dxdt)

    def lane_changing_safety_constraint(self, states, inputs, lc_veh_id: int,
                                        other_id: int, is_other_behind: bool,
                                        make_smooth: bool = True):
        if other_id < 0:  # no risk
            return 0
        if is_other_behind:
            follower_id, leader_id = other_id, lc_veh_id
        else:
            follower_id, leader_id = lc_veh_id, other_id
        follower_veh = self.vehicles[follower_id]
        follower_states = (
            self.get_vehicle_state_vector_by_id(
                follower_id, states))
        leader_states = (
            self.get_vehicle_state_vector_by_id(
                leader_id, states))
        gap_error = follower_veh.compute_error_to_safe_gap(follower_states,
                                                           leader_states)
        phi = self.get_a_vehicle_input_by_id(lc_veh_id, inputs, 'phi')
        margin = 1e-1
        if make_smooth:
            return self._smooth_min_0(gap_error + margin) * phi
        else:
            return min(gap_error + margin, 0) * phi

    @staticmethod
    def _smooth_min_0(x, epsilon: float = 1e-5):
        if x < -epsilon:
            return x
        elif x > epsilon:
            return 0
        else:
            return -(x - epsilon) ** 2 / 4 / epsilon

    def map_input_to_vehicle_ids(self, array) -> Dict[int, np.ndarray]:
        """
        Creates a dictionary mapping the vehicle id to chunks of the given array
        :param array:
        :return:
        """
        return self.map_array_to_vehicle_ids(array, False)

    def map_state_to_vehicle_ids(self, array) -> Dict[int, np.ndarray]:
        """
        Creates a dictionary mapping the vehicle id to chunks of the given array
        :param array:
        :return:
        """
        return self.map_array_to_vehicle_ids(array, True)

    def map_array_to_vehicle_ids(self, array: np.ndarray,
                                 is_state: bool) -> Dict[int, np.ndarray]:
        """
        Creates a dictionary mapping the vehicle id to chunks of the given array
        :param array:
        :param is_state: Whether it is an array of states (if True) or inputs
         (if False)
        :return:
        """
        get_function = (self.get_vehicle_state_vector_by_id if is_state
                        else self.get_vehicle_inputs_vector_by_id)
        mapped_array = {}
        for veh_id in self.vehicles.keys():
            # it's alright, pycharm is tripping
            ego_values = get_function(veh_id, array)
            mapped_array[veh_id] = ego_values
        return mapped_array

    # TODO: maybe delete?
    def to_dataframe(self, time, states, inputs) -> pd.DataFrame:
        data_per_vehicle = []
        for vehicle in self.vehicles.values():
            ego_states = self.get_vehicle_state_vector_by_id(vehicle.get_id(),
                                                             states)
            ego_inputs = self.get_vehicle_inputs_vector_by_id(vehicle.get_id(),
                                                              inputs)
            vehicle_df = vehicle.to_dataframe(time, ego_states, ego_inputs)
            data_per_vehicle.append(vehicle_df)
        all_data = pd.concat(data_per_vehicle).reset_index()
        return all_data.fillna(0)
