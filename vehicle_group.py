from typing import Any, Dict, Iterable, List, Type, Union, Tuple

import numpy as np
import pandas as pd

# import platoon
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv
import system_operating_mode as som


class VehicleGroup:
    """ Class to help manage groups of vehicles """

    def __init__(self):
        self.vehicles: Dict[int, base.BaseVehicle] = {}
        # Often, we need to iterate over all vehicles in the order they were
        # created. The list below makes that easy
        self.sorted_vehicle_ids: List[int] = []
        self.name_to_id: Dict[str, int] = {}
        # The full system (all vehicles) mode is defined by follower/leader
        # pairs.
        self.mode_sequence: som.ModeSequence = []
        self._is_verbose = True

    def get_n_vehicles(self):
        return len(self.vehicles)

    def get_current_mode(self) -> som.SystemMode:
        try:
            return self.mode_sequence[-1][1]
        except IndexError:
            return som.SystemMode({})

    def get_free_flow_speeds(self):
        v_ff = np.zeros(self.get_n_vehicles())
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

    def get_mode_sequence(self):
        return self.mode_sequence

    def get_vehicle_id_by_name(self, name: str) -> int:
        """
        Returns the id of the vehicle with given name. Returns -1 if the name
        is not found.
        """
        return self.name_to_id.get(name, -1)

    def get_vehicle_by_name(self, name: str) -> base.BaseVehicle:
        return self.vehicles[self.name_to_id[name]]

    def get_platoon_vehicles(self) -> List[fsv.PlatoonVehicle]:
        platoon_vehs: List[fsv.PlatoonVehicle] = []
        for veh_id in self.sorted_vehicle_ids:
            veh = self.vehicles[veh_id]
            if isinstance(veh, fsv.PlatoonVehicle):
                platoon_vehs.append(veh)
        return platoon_vehs

    def get_platoon_leader(self) -> fsv.PlatoonVehicle:
        for veh in self.get_platoon_vehicles():
            if veh.is_platoon_leader():
                return veh
        raise AttributeError("This vehicle group doesn't have any platoons")

    def get_initial_desired_gaps(self, v_ref: List[float] = None):
        gaps = []
        for veh_id in self.sorted_vehicle_ids:
            if v_ref is None:
                v = self.vehicles[veh_id].free_flow_speed
            else:
                v = v_ref[veh_id]
            gaps.append(self.vehicles[veh_id].compute_desired_gap(v))
        return gaps

    def set_verbose(self, value: bool):
        self._is_verbose = value
        for vehicle in self.vehicles.values():
            vehicle.set_verbose(value)

    def set_a_vehicle_free_flow_speed(self, veh_id, v_ff):
        self.vehicles[veh_id].set_free_flow_speed(v_ff)

    def set_free_flow_speeds(self, values: Union[float, List, np.ndarray]):
        if np.isscalar(values):
            values = [values] * self.get_n_vehicles()
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.set_free_flow_speed(values[veh_id])

    def set_free_flow_speeds_by_name(self, values: Dict[str, float]):
        for vehicle in self.vehicles.values():
            vehicle.set_free_flow_speed(values[vehicle.get_name()])

    def set_vehicles_initial_states(self, x0, y0, theta0, v0):
        for veh_id in self.sorted_vehicle_ids:
            vehicle = self.vehicles[veh_id]
            vehicle.set_initial_state(x0[veh_id], y0[veh_id],
                                      theta0[veh_id], v0[veh_id])

    def set_vehicles_lane_change_direction(
            self, ids_or_names: List[Union[int, str]],
            lc_direction: Union[int, List[int]]):
        if np.isscalar(lc_direction):
            lc_direction = [lc_direction] * len(ids_or_names)
        for i in range(len(ids_or_names)):
            veh_id = ids_or_names[i]
            self.set_single_vehicle_lane_change_direction(veh_id,
                                                          lc_direction[i])

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
            vehicle.set_name(names[veh_id])
            self.name_to_id[names[veh_id]] = veh_id

    def has_vehicle_with_name(self, veh_name: str):
        return veh_name in self.name_to_id

    def make_all_connected(self):
        for vehicle in self.vehicles.values():
            vehicle.make_connected()

    def map_values_to_names(self, values) -> Dict[str, Any]:
        """
        Receives variables ordered in the same order as the vehicles were
        created and returns the variables in a dictionary with vehicle names as
        keys
        """
        d = {}
        for veh_id in self.sorted_vehicle_ids:
            d[self.vehicles[veh_id].get_name()] = values[veh_id]
        return d

    def prepare_to_start_simulation(self, n_samples: int):
        """
        Sets all internal states, inputs and other simulation-related variables
        to zero.
        """
        self.mode_sequence = []
        for vehicle in self.vehicles.values():
            vehicle.prepare_to_start_simulation(n_samples)

    def create_vehicle_array(self,
                             vehicle_classes: List[Type[base.BaseVehicle]]):
        """

        Populates the list of vehicles following the given classes
        :param vehicle_classes: Class of each vehicle instances
        :return:
        """
        self.vehicles = {}
        self.sorted_vehicle_ids = []
        # self.n_vehs = len(vehicle_classes)
        for veh_class in vehicle_classes:
            vehicle = veh_class()
            self.sorted_vehicle_ids.append(vehicle.get_id())
            self.vehicles[vehicle.get_id()] = vehicle

    # TODO: rename to indicate vehicles are reset to restart simulation
    def populate_with_vehicles(self, vehicles: Dict[int, base.BaseVehicle],
                               initial_state_per_vehicle=None):
        if self.get_n_vehicles() > 0:
            raise AttributeError("Cannot set vehicles to a vehicle group "
                                 "that was already initialized")
        for veh_id in sorted(vehicles.keys()):
            if initial_state_per_vehicle:
                initial_state = initial_state_per_vehicle[veh_id]
            else:
                initial_state = None
            vehicle = vehicles[veh_id].make_reset_copy(initial_state)
            self.sorted_vehicle_ids.append(veh_id)
            self.vehicles[veh_id] = vehicle
            self.name_to_id[vehicle.get_name()] = veh_id

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

    def simulate_one_time_step(self, new_time, open_loop_controls=None):
        if open_loop_controls is None:
            open_loop_controls = {}
        self.update_platoons()
        self.update_vehicle_modes()
        self.determine_inputs(open_loop_controls)
        self.compute_derivatives()
        self.update_states(new_time)
        self.update_surrounding_vehicles()

    def set_vehicle_states(self, time, state_vectors):
        """
        Used to set vehicle states when they were computed by the optimal
        control solver
        :param time:
        :param state_vectors:
        :return:
        """
        for veh_id in self.sorted_vehicle_ids:
            self.vehicles[veh_id].set_new_time_and_state(time,
                                                         state_vectors[veh_id])

    def update_surrounding_vehicles(self):
        for ego_vehicle in self.vehicles.values():
            ego_vehicle.find_orig_lane_leader(self.vehicles.values())
            ego_vehicle.find_dest_lane_vehicles(self.vehicles.values())
            ego_vehicle.find_cooperation_requests(self.vehicles.values())
            ego_vehicle.update_target_leader(self.vehicles)
        new_mode = som.SystemMode(self.vehicles)
        if self.get_current_mode() != new_mode:
            time = self.vehicles[0].get_current_time()
            if self._is_verbose:
                if len(self.mode_sequence) == 0:
                    # print("Initial mode: {}".format(
                    #     new_mode))
                    pass
                else:
                    print("t={:.2f}. Mode update\nold: {}\nnew: {}".format(
                        time, self.get_current_mode(), new_mode))
            self.mode_sequence.append((time, new_mode))

    def update_platoons(self):
        for ego_vehicle in self.vehicles.values():
            ego_vehicle.analyze_platoons(self.vehicles)

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
        all_data = pd.concat(data_per_vehicle).reset_index(drop=True)
        return all_data.fillna(0)
