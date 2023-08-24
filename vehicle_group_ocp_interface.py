from typing import Dict, List, Union

import numpy as np
import pandas as pd

import vehicle_models as vm
import vehicle_group as vg
import vehicle_ocp_interface as vi


def get_interface_for_vehicle(
        vehicle: Union[vm.BaseVehicle, vm.ThreeStateVehicle,
                       vm.OpenLoopVehicle]):
    interface_map = {
        vm.ThreeStateVehicleRearWheel: vi.ThreeStateVehicleRearWheelInterface,
        vm.ThreeStateVehicleCG: vi.ThreeStateVehicleCGInterface,
        vm.OpenLoopVehicle: vi.FourStateVehicleInterfaceInterface,
        vm.SafeAccelOptimalLCVehicle:
            vi.FourStateVehicleInterfaceAccelFBInterface,
        vm.ClosedLoopVehicle: vi.LongitudinalVehicleInterface}
    return interface_map[type(vehicle)](vehicle)


class VehicleGroupInterface:
    """ Class to help manage groups of vehicles """

    def __init__(self, vehicle_group: vg.VehicleGroup):
        # TODO: make vehicles a  list?
        #  The order of vehicles must be fixed anyway
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
        self.mode = vehicle_group.mode

    def create_vehicle_interfaces(self, vehicle_group: vg.VehicleGroup):
        self.sorted_vehicle_ids = []
        for veh_id in sorted(vehicle_group.vehicles.keys()):
            vehicle_interface = get_interface_for_vehicle(
                vehicle_group.vehicles[veh_id])
            self.sorted_vehicle_ids.append(vehicle_interface.id)
            self.vehicles[veh_id] = vehicle_interface
            self.state_idx_map.append(self.n_states)
            self.input_idx_map.append(self.n_inputs)
            self.n_states += vehicle_interface.n_states
            self.n_inputs += vehicle_interface.n_inputs
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

    def map_ocp_solution_to_vehicle_inputs(self, ocp_inputs):
        inputs_map = {}
        for veh_id, vehicle in self.vehicles.items():
            ego_inputs = self.get_vehicle_inputs_vector_by_id(veh_id,
                                                              ocp_inputs)
            inputs_map[veh_id] = ego_inputs
        return inputs_map

    # TODO: maybe delete?
    def to_dataframe(self, time, states, inputs) -> pd.DataFrame:
        data_per_vehicle = []
        for vehicle in self.vehicles.values():
            ego_states = self.get_vehicle_state_vector_by_id(vehicle.id,
                                                             states)
            ego_inputs = self.get_vehicle_inputs_vector_by_id(vehicle.id,
                                                              inputs)
            vehicle_df = vehicle.to_dataframe(time, ego_states, ego_inputs)
            data_per_vehicle.append(vehicle_df)
        all_data = pd.concat(data_per_vehicle).reset_index()
        return all_data.fillna(0)
