from typing import Any, Dict, Iterable, List, Type, Union

import numpy as np
import pandas as pd

import platoon
import vehicle_models.base_vehicle as base
import system_operating_mode as som


class VehicleGroup:
    """ Class to help manage groups of vehicles """

    # The full system (all vehicles) mode is defined by follower/leader
    # pairs.
    mode: som.SystemMode

    def __init__(self):
        self.vehicles: Dict[int, base.BaseVehicle] = {}
        # Often, we need to iterate over all vehicles in the order they were
        # created. The list below makes that easy
        self.sorted_vehicle_ids: List[int] = []
        self.n_vehs = 0
        self.name_to_id: Dict[str, int] = {}

    def __str__(self):
        veh_strings = []
        for veh_id in self.sorted_vehicle_ids:
            veh_strings.append(str(self.vehicles[veh_id]))
        return ', '.join(veh_strings)

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

    def get_all_vehicles(self) -> Iterable[base.BaseVehicle]:
        return self.vehicles.values()

    def get_all_states(self):
        states = []
        for veh_id in self.sorted_vehicle_ids:
            states.append(self.vehicles[veh_id].get_state_history())
        return np.vstack(states)

    def get_all_inputs(self):
        inputs = []
        for veh_id in self.sorted_vehicle_ids:
            inputs.append(self.vehicles[veh_id].get_input_history())
        return np.vstack(inputs)

    def get_vehicle_by_name(self, name: str) -> base.BaseVehicle:
        return self.vehicles[self.name_to_id[name]]

    def get_initial_desired_gaps(self, v_ref: List[float] = None):
        gaps = []
        for veh_id in self.sorted_vehicle_ids:
            if v_ref is None:
                v = self.vehicles[veh_id].free_flow_speed
            else:
                v = v_ref[veh_id]
            gaps.append(self.vehicles[veh_id].compute_desired_gap(v))
        return gaps

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

    def set_single_vehicle_lane_change_direction(
            self, veh_id_or_name: Union[int, str], lc_direction: int):
        if isinstance(veh_id_or_name, str):
            veh_id = self.name_to_id[veh_id_or_name]
        else:
            veh_id = veh_id_or_name
        self.vehicles[veh_id].set_lane_change_direction(lc_direction)

    def set_vehicle_names(self, names: List[str]):
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.name = names[veh_id]
            self.name_to_id[names[veh_id]] = veh_id

    def map_values_to_names(self, values) -> Dict[str, Any]:
        """
        Receives variables ordered in the same order as the vehicles were
        created and returns the variables in a dictionary with vehicle names as
        keys
        """
        d = {}
        for veh_id in self.sorted_vehicle_ids:
            d[self.vehicles[veh_id].name] = values[veh_id]
        return d

    def initialize_state_matrices(self, n_samples: int):
        for vehicle in self.vehicles.values():
            vehicle.initialize_simulation_logs(n_samples)

    def make_all_connected(self):
        for vehicle in self.vehicles.values():
            vehicle.make_connected()

    def create_vehicle_array(self,
                             vehicle_classes: List[Type[base.BaseVehicle]]):
        """

        Populates the list of vehicles following the given classes
        :param vehicle_classes: Class of each vehicle instances
        :return:
        """
        self.vehicles = {}
        self.sorted_vehicle_ids = []
        self.n_vehs = len(vehicle_classes)
        for veh_class in vehicle_classes:
            vehicle = veh_class()
            self.sorted_vehicle_ids.append(vehicle.id)
            self.vehicles[vehicle.id] = vehicle

    # def create_platoons(self,
    #                     platoon_assignment: List[List[fsv.PlatoonVehicle]]):
    #     """
    #     Creates platoons and include vehicles in them
    #     :param platoon_assignment:
    #     :return:
    #     """
    #     for platoon_vehicles in platoon_assignment:
    #         new_platoon = platoon.Platoon()
    #         for veh in sorted(platoon_vehicles,
    #                           key=lambda x: x.get_a_state_by_name('x'),
    #                           reverse=True):
    #             new_platoon.add_vehicle(veh.id)
    #             veh.set_platoon(new_platoon)

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
            ego_vehicle.find_cooperation_requests(self.vehicles.values())
            ego_vehicle.update_target_leader(self.vehicles)
        new_mode = som.SystemMode(self.vehicles)
        try:
            if new_mode != self.mode:
                print("t={:.2f}. Mode update\nold: {}\nnew: {}".format(
                    self.vehicles[0].get_current_time(),
                    self.mode, new_mode))
        except AttributeError:
            print("Initial mode:\n{}".format(new_mode))
        self.mode = new_mode

    def determine_inputs(
            self, open_loop_controls: Dict[int, Dict[str, float]]):
        """
        Sets the open loop controls and computes the closed loop controls for
        all vehicles.
        :param open_loop_controls: Dictionary whose keys are the vehicle id
         and values are dictionaries with input name/value pairs.
        :return: Nothing. Each vehicle stores the computed input values
        """
        for veh_id, vehicle in self.vehicles.items():
            vehicle.determine_inputs(open_loop_controls.get(veh_id, {}),
                                     self.vehicles)

    def update_vehicle_modes(self):
        for vehicle in self.vehicles.values():
            vehicle.update_mode(self.vehicles)

    def compute_derivatives(self):
        """
        Computes the states derivatives
        :return:
        """
        dxdt = []
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.compute_derivatives()
            dxdt.extend(vehicle.get_derivatives())
        return np.array(dxdt)

    def update_states(self, new_time):
        for vehicle in self.vehicles.values():
            vehicle.update_states(new_time)

    def to_dataframe(self) -> pd.DataFrame:
        """

        :return:
        """
        data_per_vehicle = []
        for vehicle in self.vehicles.values():
            vehicle_df = vehicle.to_dataframe()
            data_per_vehicle.append(vehicle_df)
        all_data = pd.concat(data_per_vehicle).reset_index()
        return all_data.fillna(0)
