from typing import Dict, List, Type, Union

import numpy as np

import vehicle_models as vm
import vehicle_group as vg
import vehicle_ocp_interface as vi


def get_interface_for_vehicle(vehicle: vm.BaseVehicle):
    interface_map = {
        vm.ThreeStateVehicleRearWheel: vi.ThreeStateVehicleRearWheelInterface,
        vm.ThreeStateVehicleCG: vi.ThreeStateVehicleCGInterface,
        vm.FourStateVehicle: vi.FourStateVehicleInterfaceInterface,
        vm.FourStateVehicleAccelFB:
            vi.FourStateVehicleInterfaceAccelFBInterface,
        vm.LongitudinalVehicle: vi.LongitudinalVehicleInterface}
    return interface_map[type(vehicle)](vehicle)


class VehicleGroupInterface:
    """ Class to help manage groups of vehicles """

    def __init__(self, vehicle_group: vg.VehicleGroup):
        self.vehicles: Dict[int, vi.BaseVehicleInterface] = {}
        # Often, we need to iterate over all vehicles in the order they were
        # created. The list below make that easy
        self.sorted_vehicle_ids = None
        self.n_vehs = 0
        self.n_states, self.n_inputs = 0, 0
        # Maps the vehicle id (position in the 'vehicles' list) to the index of
        # its states in the state vector containing all vehicles
        self.state_idx_map = []
        # Maps the vehicle id (position in the 'vehicles' list) to the index of
        # its inputs in the full input vector
        self.input_idx_map = []

        self.create_vehicle_interfaces(vehicle_group)

    def create_vehicle_interfaces(self, vehicle_group: vg.VehicleGroup):
        self.sorted_vehicle_ids = []
        for veh_id in sorted(vehicle_group.vehicles.keys()):
            vehicle = vehicle_group.vehicles[veh_id]
            self.sorted_vehicle_ids.append(vehicle.id)
            self.vehicles[veh_id] = get_interface_for_vehicle(vehicle)
            self.state_idx_map.append(self.n_states)
            self.input_idx_map.append(self.n_inputs)
            self.n_states += vehicle.n_states
            self.n_inputs += vehicle.n_inputs
        self.n_vehs = len(self.vehicles)

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

    def get_state_indices(self, state_name: str) -> List[int]:
        state_indices = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
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
        return vehicle.select_state_from_vector(vehicle_states, state_name)

    def get_a_vehicle_input_by_id(self, veh_id, inputs, input_name):
        vehicle_inputs = self.get_vehicle_inputs_vector_by_id(veh_id, inputs)
        vehicle = self.vehicles[veh_id]
        return vehicle.select_input_from_vector(vehicle_inputs, input_name)

    def get_free_flow_speeds(self):
        v_ff = np.zeros(self.n_vehs)
        for veh_id in self.sorted_vehicle_ids:
            v_ff[veh_id] = self.vehicles[veh_id].free_flow_speed
        return v_ff

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
        leader_x = self.get_a_vehicle_state_by_id(
            ego_vehicle.get_current_leader_id(), states, 'x')
        return leader_x - ego_x

    def compute_derivatives(self, states, inputs, params):
        """
        Computes the states derivatives
        :param states: Current states of all vehicles
        :param inputs: Current inputs of all vehicles
        :param params:
        :return:
        """
        dxdt = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            ego_states = self.get_vehicle_state_vector_by_id(vehicle.id, states)
            ego_inputs = self.get_vehicle_inputs_vector_by_id(
                vehicle.id, inputs)
            if vehicle.has_leader():
                leader_states = self.get_vehicle_state_vector_by_id(
                    vehicle.get_current_leader_id(), states)
            else:
                leader_states = None
            dxdt.extend(vehicle.compute_derivatives(ego_states, ego_inputs,
                                                    leader_states))
        return np.array(dxdt)
