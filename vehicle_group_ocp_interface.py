from __future__ import annotations

from typing import Callable, Iterable, Mapping, Union
import warnings

import numpy as np
import pandas as pd

import configuration
from operating_modes import system_operating_mode as som
import vehicle_models.base_vehicle as base


# ========= Functions passed to the optimal control library methods ========== #
def update_vehicles(t, states, inputs, params):
    """
    Follows the model of updfcn of the control package.
    :param t: time
    :param states: Array with states of all vehicles [x1, y1, ..., xN, yN]
    :param inputs: Array with inputs of all vehicles [u11, u12, ..., u1N, u2N]
    :param params: dictionary which must contain the vehicle type
    :return: the derivative of the state
    """
    vehicle_group_interface: VehicleGroupInterface = (
        params['vehicle_group_interface']
    )
    # vehicle_group_interface.update_surrounding_vehicles(t)
    return vehicle_group_interface.update_vehicles(states, inputs)


def vehicle_output(t, x, u, params):
    return x  # return (full state)


class VehicleGroupInterface:
    """ Class to help manage groups of vehicles """

    def __init__(self, vehicles: Mapping[int, base.BaseVehicle],
                 ego_id: int = None):
        """

        :param vehicles: All simulation vehicles
        :param ego_id: If given, assumes this vehicle as the center of
         the system, i.e., its (x, y) = (0, 0)
        """
        self.vehicles: dict[int, base.BaseVehicleInterface] = {}
        # Often, we need to iterate over all vehicles in the order they were
        # created. The list below make that easy
        self.sorted_vehicle_ids = None
        self.n_vehs = 0
        self.n_states, self.n_inputs = 0, 0
        # Maps the vehicle id to the index of its states in the state vector
        # containing all vehicles
        self.state_idx_map: dict[int, int] = {}
        # Maps the vehicle id to the index of its inputs in the full input
        # vector
        self.input_idx_map: dict[int, int] = {}

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

    def get_desired_input(self) -> np.ndarray:
        desired_inputs_array = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            veh_desired_inputs = vehicle.get_desired_input()
            desired_inputs_array.extend(veh_desired_inputs)
        return np.array(desired_inputs_array)

    def get_initial_inputs_guess(
            self, accel_guess: Union[str, float], n_time_steps: int,
    ) -> list[dict[int, np.ndarray]]:
        # initial_guess_array = []
        # initial_guess_dict = {}
        input_guess = []
        for i in range(n_time_steps):
            current_guess = {}
            for veh_id in self.sorted_vehicle_ids:
                vehicle = self.vehicles[veh_id]
                veh_desired_inputs = vehicle.get_desired_input()
                if vehicle.is_long_control_optimal():
                    if isinstance(accel_guess, str):
                        if accel_guess == 'zero':
                            accel = 0.
                        elif accel_guess == 'max':
                            accel = vehicle.get_accel_max()
                        elif accel_guess == 'min':
                            accel = vehicle.get_brake_max()
                        elif accel_guess == 'random':
                            # TODO: mean zero seems useless, but then what?
                            accel = np.random.uniform(-2.0, 1.0)
                        else:
                            raise ValueError(
                                'Trying to get initial acceleration guess, but '
                                'initial_input_guess was set to None')
                        veh_desired_inputs[
                            vehicle.optimal_input_idx['a']] = accel
                        # initial_guess_array.extend(veh_desired_inputs)
                    current_guess[veh_id] = veh_desired_inputs
                    input_guess.append(current_guess)
                return input_guess
                # return (initial_guess_dict if as_dict
                #         else np.array(initial_guess_array))

    def get_initial_state(self):
        """
        Gets all vehicles' current states, which are used as the starting
        point for the ocp
        """
        initial_state = []
        for veh_id in self.sorted_vehicle_ids:
            initial_state.extend(self.vehicles[veh_id].get_initial_state())
        return np.array(initial_state + [0.])  # time is the last state

    def get_state_indices(self, state_name: str) -> list[int]:
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

    @staticmethod
    def get_time(states):
        return states[-1]

    def set_mode_sequence(self, mode_sequence: som.ModeSequence):
        # optimal_veh_ids = [veh.get_id() for veh in self.vehicles.values()
        #                    if veh.is_long_control_optimal()]
        # mode_sequence.remove_leader_from_modes(optimal_veh_ids)
        mode_sequence.remove_overlapping_modes(
            configuration.Configuration.discretization_step)
        sv_sequences = mode_sequence.to_sv_sequence()
        for foll_id, sequence in sv_sequences.items():
            self.vehicles[foll_id].set_leader_sequence(sequence)

    def create_input_names(self) -> list[str]:
        input_names = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            str_id = str(veh_id)
            input_names.extend([name + str_id for name
                                in vehicle.get_input_names()])
        return input_names

    def create_vehicle_interfaces(
            self, vehicles: Mapping[int, base.BaseVehicle], ego_id: int = None
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
        self.n_states += 1  # time
        self.n_vehs = len(self.vehicles)

    def create_output_names(self) -> list[str]:
        output_names = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            str_id = str(veh_id)
            output_names.extend([name + str_id for name
                                 in vehicle.get_state_names()])
        return output_names + ['time']

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
                v0 = veh.select_state_from_vector(veh_states, 'v')
            except KeyError:  # three-state vehicles have v as an input
                v0 = veh.select_input_from_vector(veh.get_desired_input(), 'v')
            vf = veh.get_free_flow_speed()  # v0? Do we want const speed LC?
            xf = veh.select_state_from_vector(veh_states, 'x') + v0 * tf
            yf = veh.get_target_y()
            thetaf = 0.0
            desired_state.extend(veh.create_state_vector(xf, yf, thetaf, vf))
        return np.array(desired_state + [0.])  # time is the last state

    def create_state_cost_matrix(self, x_cost: float = 0, y_cost: float = 0,
                                 theta_cost: float = 0, v_cost: float = 0):
        """
        Creates a diagonal cost function where each state of all controlled
        vehicles gets the same weight
        :param x_cost:
        :param y_cost:
        :param theta_cost:
        :param v_cost:
        :return:
        """
        veh_costs = []
        for veh_id in self.sorted_vehicle_ids:
            veh = self.vehicles[veh_id]
            if veh.n_inputs == 0:
                veh_costs.extend([0.0] * veh.n_states)
            elif not veh.is_long_control_optimal():
                # if x_cost > 0 or v_cost > 0:
                #     warnings.warn('Trying to pass non-zero position or '
                #                   'velocity cost to a vehicle with feedback '
                #                   'acceleration. Setting both to zero')
                veh_costs.extend(veh.create_state_vector(
                    0., y_cost, theta_cost, 0.))
            else:  # steering wheel and accel optimal control
                veh_costs.extend(veh.create_state_vector(
                    x_cost, y_cost, theta_cost, v_cost))

        return np.diag(veh_costs + [0.])  # time is the last state

    def create_input_cost_matrix(self, accel_cost: float = 0,
                                 phi_cost: float = 0):
        """
        Creates a diagonal cost function where each input of all controlled
        vehicles gets the same weight
        :param accel_cost:
        :param phi_cost:
        :return:
        """
        veh_costs = []
        for veh_id in self.sorted_vehicle_ids:
            veh = self.vehicles[veh_id]
            # if veh.n_inputs == 0:
            #     continue
            if veh.is_long_control_optimal():
                veh_costs.append(accel_cost)
            if veh.is_lat_control_optimal():
                veh_costs.append(phi_cost)

        return np.diag(veh_costs)

    def update_vehicles(self, states, inputs):
        """
        Computes the states derivatives
        :param states: Current states of all vehicles
        :param inputs: Current inputs of all vehicles
        :return: state derivative
        """
        dxdt = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            ego_states = self.get_vehicle_state_vector_by_id(vehicle.get_id(),
                                                             states)
            ego_inputs = self.get_vehicle_inputs_vector_by_id(
                vehicle.get_id(), inputs)
            t = self.get_time(states)
            if vehicle.has_leader(t):
                leader_states = self.get_vehicle_state_vector_by_id(
                    vehicle.get_current_leader_id(t), states)
            else:
                leader_states = None
            dxdt.extend(vehicle.compute_derivatives(ego_states, ego_inputs,
                                                    leader_states))
        return np.array(dxdt + [1])  # time is the last state

    def cost_function(self, controlled_veh_ids: Iterable[int]) -> Callable:
        tf = 0.0  # only relevant for desired final position
        states_ref = self.create_desired_state(tf)
        u_ref = self.get_desired_input()
        Q = self.create_state_cost_matrix(x_cost=0.0, y_cost=0.2,
                                          theta_cost=0.0, v_cost=0.1)
        R = self.create_input_cost_matrix(accel_cost=0.1, phi_cost=0.1)

        vehs, x_idx, v_idx = [], [], []
        for veh_id in controlled_veh_ids:
            vehs.append(self.vehicles[veh_id])
            x_idx.append(self.get_a_vehicle_state_index(veh_id, 'x'))
            v_idx.append(self.get_a_vehicle_state_index(veh_id, 'v'))

        def support(states, inputs) -> float:
            t = self.get_time(states)

            # w_eg = 1.  # gap error weight
            for i in range(len(vehs)):
                leader_id = vehs[i].get_destination_lane_leader_id(t)
                if leader_id > -1:
                    leader_x = self.get_a_vehicle_state_by_id(leader_id, states,
                                                              'x')
                    gap_ref = vehs[i].compute_lane_keeping_safe_gap(
                        states[v_idx[i]])
                    states_ref[x_idx[i]] = leader_x - gap_ref
                else:
                    states_ref[x_idx[i]] = states[x_idx[i]]

            cost = ((states - states_ref) @ Q @ (states - states_ref)
                    + (inputs - u_ref) @ R @ (inputs - u_ref)).item()
            # eg_cost = []
            # for ego_id in controlled_veh_ids:
            #     ego_veh = self.vehicles[ego_id]
            #     gap_error = self.compute_gap_error(
            #         states, ego_id, ego_veh.get_dest_lane_follower_id(t),
            #         is_other_behind=True, is_follower_lane_changing=False)
            #     eg_cost.append(w_eg * min(gap_error, 0) ** 2)
            # cost += sum(eg_cost)
            return cost

        return support

    def lane_changing_safety_constraints(self, ego_id: int) -> list[Callable]:
        ego_veh = self.vehicles[ego_id]
        margin = 1e-3

        def to_lo(states, inputs) -> float:
            t = self.get_time(states)
            gap_error = self.compute_gap_error(
                    states, ego_id, ego_veh.get_origin_lane_leader_id(t),
                    False, is_follower_lane_changing=True)
            phi = self.get_a_vehicle_input_by_id(ego_id, inputs, 'phi')
            return min(gap_error + margin, 0) ** 2 * phi

        def to_ld(states, inputs) -> float:
            t = self.get_time(states)
            gap_error = self.compute_gap_error(
                    states, ego_id, ego_veh.get_destination_lane_leader_id(t),
                    False, is_follower_lane_changing=True)
            phi = self.get_a_vehicle_input_by_id(ego_id, inputs, 'phi')
            return min(gap_error + margin, 0) ** 2 * phi

        def to_fd(states, inputs) -> float:
            t = self.get_time(states)
            gap_error = self.compute_gap_error(
                    states, ego_id, ego_veh.get_destination_lane_follower_id(t),
                    True, is_follower_lane_changing=False)
            phi = self.get_a_vehicle_input_by_id(ego_id, inputs, 'phi')
            return min(gap_error + margin, 0) ** 2 * phi

        return [to_lo, to_ld, to_fd]

    # def lane_changing_constraint_to_fd(self, ego_id: int) -> Callable:
    #     ego_veh = self.vehicles[ego_id]
    #     margin = 1e-3
    #
    #     def to_fd(states, inputs) -> float:
    #         t = self.get_time(states)
    #         foll_id = ego_veh.get_dest_lane_follower_id(t)
    #         follower_veh = self.vehicles[foll_id]
    #         x_follower = (
    #             self.get_a_vehicle_state_by_id(foll_id, states, 'x'))
    #         v_follower = (
    #             self.get_a_vehicle_state_by_id(foll_id, states, 'v'))
    #         x_leader = (
    #             self.get_a_vehicle_state_by_id(ego_id, states, 'x'))
    #         gap_error = follower_veh.compute_error_to_safe_gap(
    #             x_follower, v_follower, x_leader, False)
    #         phi = self.get_a_vehicle_input_by_id(ego_id, inputs, 'phi')
    #         # theta = self.get_a_vehicle_state_by_id(ego_id, states, 'theta')
    #         return min(gap_error + margin, 0) ** 2 * phi
    #
    #     return to_fd

    def vehicle_following_safety_constraint(self, ego_id: int) -> Callable:
        ego_veh = self.vehicles[ego_id]

        def support(states, inputs) -> float:
            t = self.get_time(states)
            if ego_veh.has_origin_lane_leader(t):
                gap_error = self.compute_gap_error(
                    states, ego_id, ego_veh.get_origin_lane_leader_id(t), False,
                    is_follower_lane_changing=False)
                return gap_error
            return 1.0e-3  # anything greater or equal to zero

        return support

    def platoon_constraint(self, platoon_ids: tuple[int, int], other_id: int
                           ) -> Callable:
        def support(states, inputs) -> float:
            other_x = self.get_a_vehicle_state_by_id(other_id, states, 'x')
            ret = 1.
            for p_id in platoon_ids:
                platoon_veh_x = self.get_a_vehicle_state_by_id(p_id, states,
                                                               'x')
                gap = other_x - platoon_veh_x
                ret *= gap
            return ret
        return support

    def compute_gap_error(
            self, states, ego_id: int, other_id: int, is_other_behind: bool,
            is_follower_lane_changing: bool
    ) -> float:
        if other_id < 0:  # no risk
            return 0
        if is_other_behind:
            follower_id, leader_id = other_id, ego_id
        else:
            follower_id, leader_id = ego_id, other_id
        follower_veh = self.vehicles[follower_id]
        x_follower = (
            self.get_a_vehicle_state_by_id(follower_id, states, 'x'))
        v_follower = (
            self.get_a_vehicle_state_by_id(follower_id, states, 'v'))
        x_leader = (
            self.get_a_vehicle_state_by_id(leader_id, states, 'x'))
        return follower_veh.compute_error_to_safe_gap(
            x_follower, v_follower, x_leader, is_follower_lane_changing)

    def map_input_to_vehicle_ids(self, array) -> dict[int, np.ndarray]:
        """
        Creates a dictionary mapping the vehicle id to chunks of the given array
        :param array:
        :return:
        """
        return self.map_array_to_vehicle_ids(array, False)

    def map_state_to_vehicle_ids(self, array) -> dict[int, np.ndarray]:
        """
        Creates a dictionary mapping the vehicle id to chunks of the given array
        :param array:
        :return:
        """
        return self.map_array_to_vehicle_ids(array, True)

    def map_array_to_vehicle_ids(self, array: np.ndarray,
                                 is_state: bool) -> dict[int, np.ndarray]:
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


def _smooth_min_0(x, epsilon: float = 1e-5):
    if x < -epsilon:
        return x
    elif x > epsilon:
        return 0
    else:
        return -(x - epsilon) ** 2 / 4 / epsilon
