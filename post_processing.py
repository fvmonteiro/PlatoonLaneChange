from __future__ import annotations

from collections.abc import Iterable
# from typing import Union

import numpy as np
import pandas as pd

import configuration as config
# import vehicle_models.base_vehicle as base


# TODO: still deciding if these 'to_dataframe' methods belong here or within
#  the vehicle/vehicle_group classes.
#  A: easier to keep been in the vehicle classes to avoid circular imports

# def vehicle_to_dataframe(vehicle: base.BaseVehicle):
#     data = np.concatenate([vehicle.get_simulated_time().reshape(1, -1),
#                            vehicle.get_state_history(),
#                            vehicle.get_input_history()])
#     columns = (['t'] + [s for s in vehicle.get_state_names()]
#                + [i for i in vehicle.get_input_names()])
#     veh_data = pd.DataFrame(data=np.transpose(data), columns=columns)
#     veh_data['id'] = vehicle.get_id()
#     veh_data['name'] = vehicle.get_name()
#
#     def _set_surrounding_vehicles_ids_to_df(df, col_name, col_value):
#         if len(col_value) == 1:
#             df[col_name] = col_value[0]
#         else:
#             df[col_name] = col_value
#
#     _set_surrounding_vehicles_ids_to_df(
#         veh_data, 'orig_lane_leader_id',
#         vehicle.get_origin_lane_leader_id_history())
#     _set_surrounding_vehicles_ids_to_df(
#         veh_data, 'dest_lane_leader_id',
#         vehicle.get_destination_lane_leader_id_history())
#     _set_surrounding_vehicles_ids_to_df(
#         veh_data, 'dest_lane_follower_id',
#         vehicle.get_destination_lane_follower_id_history())
#     return veh_data


# def vehicles_to_dataframe(vehicles: Union[Iterable[base.BaseVehicle],
#                                           vg.VehicleGroup]):
#     if isinstance(vehicles, vg.VehicleGroup):
#         vehicles = vehicles.get_all_vehicles()
#
#     data_per_vehicle = []
#     for vehicle in vehicles:
#         vehicle_df = vehicle_to_dataframe(vehicle)
#         data_per_vehicle.append(vehicle_df)
#     all_data = pd.concat(data_per_vehicle).reset_index(drop=True)
#     return all_data.fillna(0)


# def find_maneuver_completion_time(vehicles: Iterable[base.BaseVehicle]
#                                   ) -> list[float]:
#     final_times = []
#     for veh in vehicles:
#         y_error = veh.get_target_y() - veh.get_y_history()
#         # argmax returns the idx of the first True (1)
#         idx = np.argmax(np.abs(y_error) < 0.5)
#         # argmax returns 0 if there is no True value, so we check to see if
#         # the vehicle ended at the target lane
#         if idx == 0 and veh.get_target_lane() != veh.get_current_lane():
#             idx = - 1
#         final_times.append(veh.get_simulated_time()[idx])
#
#     return final_times


def compute_acceleration_costs(
        data: pd.DataFrame, relevant_vehicle_names: Iterable[str] = None
) -> dict[str, float]:
    if relevant_vehicle_names is None:
        relevant_vehicle_names = data['name'].unique()

    group = data.groupby('name')
    accel_costs = {}
    for name in relevant_vehicle_names:
        veh_data = group.get_group(name)
        accel_costs[name] = np.trapz(veh_data['a'], veh_data['t'])
    return accel_costs


def compute_values_relative_to_other_vehicle(
        data: pd.DataFrame, other_name: str):
    """
    Computes gap relative to the relevant surrounding vehicle
    :param data: dataframe with position of all vehicles
    :param other_name: Options: leader, dest_lane_leader, dest_lane_follower
    :return:
    """
    other_id = other_name + '_id'
    merged = data[['t', 'id', other_id, 'v', 'x']].merge(
        data[['t', 'id', 'v', 'x']], how='left',
        left_on=['t', other_id], right_on=['t', 'id'],
        suffixes=(None, '_' + other_name)).drop(columns='id_' + other_name)
    data['gap_to_' + other_name] = merged['x_' + other_name] - merged['x']
    # data['rel_vel_to_' + other_name] = merged['v'] - merged['v_' + other_name]


def compute_values_to_fixed_nearby_vehicle(
        data: pd.DataFrame, other_name: str, do_renaming: bool = False):
    """
    Computes gap relative to the *initial* relevant surrounding vehicle
    :param data: dataframe with position of all vehicles
    :param other_name: Options: leader, dest_lane_leader, dest_lane_follower
    :param do_renaming: If true removes '_initial_' from the newly created
     column names
    :return:
    """
    data["_".join(['initial', other_name, 'id'])] = (
        data[other_name + '_id'].groupby(data['id']).transform('first'))
    compute_values_relative_to_other_vehicle(data, 'initial_' + other_name)
    # Remove 'initial_' from column names
    if do_renaming:
        new_names = {col: col.replace('_initial_', '_') for col in data.columns
                     if 'initial_' + other_name in col}
        data.rename(columns=new_names, inplace=True)


def compute_values_to_orig_lane_leader(data: pd.DataFrame):
    """
    Computes gap relative to the *initial* relevant surrounding vehicle
    :param data:
    :return:
    """
    compute_values_relative_to_other_vehicle(data, 'orig_lane_leader')


def compute_values_to_future_leader(data: pd.DataFrame):
    compute_values_relative_to_other_vehicle(data, 'dest_lane_leader')
    # compute_values_to_fixed_nearby_vehicle(data, 'dest_lane_leader',
    #                                        do_renaming=True)


def compute_values_to_future_follower(data: pd.DataFrame):
    compute_values_relative_to_other_vehicle(data, 'dest_lane_follower')
    # compute_values_to_fixed_nearby_vehicle(data, 'dest_lane_follower',
    #                                        do_renaming=True)
    data['gap_to_dest_lane_follower'] = -data['gap_to_dest_lane_follower']


def compute_all_relative_values(data: pd.DataFrame):
    compute_values_to_orig_lane_leader(data)
    compute_values_to_fixed_nearby_vehicle(data, 'orig_lane_leader')
    compute_values_to_future_leader(data)
    compute_values_to_future_follower(data)


def compute_default_safe_gap(vel):
    return (config.get_lane_changing_time_headway() * vel
            + config.STANDSTILL_DISTANCE)
