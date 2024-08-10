from __future__ import annotations

from collections.abc import Iterable
import os
import shutil

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import configuration as config
from platoon_functionalities import traffic_state_graph


def process_graph_exploration_results(
        n_platoon: Iterable[int], cost_type: Iterable[str],
        epsilon: Iterable[float]):
    data = []
    for cost in cost_type:
        for n in n_platoon:
            for eps in epsilon:
                data.append(process_single_graph_exploration_result(
                    n, cost, eps))
    return pd.concat(data)


def process_single_graph_exploration_result(
        n_platoon: int, cost_type: str, epsilon: float):
    data = traffic_state_graph.read_from_json(n_platoon, cost_type, epsilon)
    # first_times, end_times = [], []  # TODO: de we want these stats?
    max_time = -np.inf
    for d in data:
        # first_times.append(d["time"][0])
        # end_times.append(d["time"][-1])
        # min_time = min(min_time, d["time"][0])
        # Force first time to zero
        d["time"] = np.array(d["time"]) - d["time"][0]
        max_time = max(max_time, d["time"][-1])
        # Normalize costs
        min_cost = np.min(d["cost"])
        max_cost = np.max(d["cost"])
        if min_cost != max_cost:
            d["norm_cost"] = ((np.array(d["cost"]) - min_cost)
                              / (max_cost - min_cost))
        else:
            d["norm_cost"] = np.array([0])
    common_x = np.linspace(0., max_time, num=100)
    # Interpolate y values to common x values using zero-order hold
    all_y_interp = []
    for d in data:
        f = interp1d(d["time"], d["norm_cost"], kind='previous',
                     fill_value="extrapolate",
                     bounds_error=False)
        y_interp = f(common_x)
        all_y_interp.append(y_interp)

    # Create a DataFrame suitable for Seaborn
    df = pd.DataFrame(all_y_interp, columns=common_x)

    # Convert DataFrame from wide to long format
    df_long = df.melt(var_name='time', value_name='norm_cost')
    df_long["cost_type"] = cost_type
    df_long["n_platoon"] = n_platoon
    df_long["epsilon"] = epsilon
    return df_long


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


def compute_values_relative_to_other_vehicle(data: pd.DataFrame,
                                             other_name: str):
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


def export_strategy_maps_to_cloud(n_platoon: Iterable[int] = None,
                                  cost_names: Iterable[str] = None):
    _exchange_strategy_maps_with_cloud(True, n_platoon, cost_names)


def import_strategy_maps_from_cloud(n_platoon: Iterable[int] = None,
                                    cost_names: Iterable[str] = None):
    _exchange_strategy_maps_with_cloud(False, n_platoon, cost_names)


def _exchange_strategy_maps_with_cloud(is_exporting: bool,
                                       n_platoon: Iterable[int] = None,
                                       cost_names: Iterable[str] = None):
    if n_platoon is None:
        n_platoon = [2, 3, 4, 5]
    if cost_names is None:
        cost_names = ['time', 'accel']

    local_dir = os.path.join(config.DATA_FOLDER_PATH, 'strategy_maps')
    cloud_dir = os.path.join(config.SHARED_DATA_PATH, 'strategy_maps')
    if is_exporting:
        source_dir, dest_dir = local_dir, cloud_dir
    else:
        source_dir, dest_dir = cloud_dir, local_dir
    for n in n_platoon:
        for c in cost_names:
            file_name = '_'.join(['min', c, 'strategies_for',
                                  str(n), 'vehicles.json'])
            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy2(source_path, dest_path)
            print(f'File {file_name} copied from folder {source_dir} to '
                  f'{dest_dir}')


def export_results_to_cloud():
    _exchange_results_with_cloud(True)


def import_results_from_cloud():
    _exchange_results_with_cloud(False)


def _exchange_results_with_cloud(is_exporting: bool):
    local_dir = os.path.join(config.DATA_FOLDER_PATH,
                             "platoon_strategy_results")
    cloud_dir = os.path.join(config.SHARED_DATA_PATH,
                             "platoon_strategy_results")
    if is_exporting:
        source_dir, dest_dir = local_dir, cloud_dir
    else:
        source_dir, dest_dir = cloud_dir, local_dir
    file_name = "result_summary.csv"
    source_path = os.path.join(source_dir, file_name)
    dest_path = os.path.join(dest_dir, file_name)
    shutil.copy2(source_path, dest_path)
    print(f'File {file_name} copied from folder {source_dir} to '
          f'{dest_dir}')
