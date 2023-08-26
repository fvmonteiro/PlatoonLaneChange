import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import vehicle_models
import constants as const


def compute_values_relative_to_other_vehicle(
        data: pd.DataFrame, other_name):
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
        data: pd.DataFrame, other_name, do_renaming: bool = False):
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


def check_constraint_satisfaction(data: pd.DataFrame, lc_id: int):
    compute_all_relative_values(data)
    data['safe_gap'] = data['v'] + 1
    data['gap_error'] = data['gap_to_orig_lane_leader'] - data['safe_gap']
    data['constraint'] = np.minimum(data['gap_error'], 0) * data['phi']
    plot_scenario_results(['t', 't', 't'], ['constraint', 'phi', 'gap_error'],
                          data[data['id'] == lc_id])


def load_simulated_scenario(pickle_file_name: str):
    with open(pickle_file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_default_safe_gap(vel):
    return const.safe_time_headway * vel + const.standstill_distance


def compare_desired_and_actual_final_states(desired_data, simulated_data):
    fig, ax = plt.subplots(2, 1)
    plot_initial_and_final_states(desired_data, ax[0])
    ax[0].set_title("Desired")
    ax[0].set_aspect('equal', adjustable='box')
    plot_initial_and_final_states(simulated_data, ax[1])
    ax[1].set_title("Simulated")
    ax[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.show()


def plot_initial_and_final_states(data: pd.DataFrame, axis=None):
    """

    :param data: Dataframe containing only initial and desired final state
     for each vehicle
    """
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = plt.gcf()

    cmap = plt.get_cmap("tab10")
    n_vehs = data['id'].nunique()
    ax.scatter(data=data[data['t'] == data['t'].min()], x='x', y='y',
               marker='>', c=cmap.colors[0:n_vehs])
    ax.scatter(data=data[data['t'] == data['t'].max()], x='x', y='y',
               marker='>', c=cmap.colors[0:n_vehs], alpha=0.6)
    min_y = data['y'].min()
    max_y = data['y'].max()
    ax.axhline(y=(min_y + max_y) / 2, linestyle='--', color='black')
    ax.set(xlabel=_get_variable_with_unit('x'),
           ylabel=_get_variable_with_unit('y'),
           ylim=(min_y - 2, max_y + 2))
    ax.set_aspect('equal', adjustable='box')

    if axis is None:
        fig.tight_layout()
        fig.show()


def plot_lane_change(data: pd.DataFrame):
    """
    Plots the lane change in a y vs x plot along with speed and steering wheel
     angle vs time
    :param data:
    :return:
    """
    sns.set_style('whitegrid')
    x_axes = ['x', 't', 't']
    y_axes = ['y', 'v', 'phi']
    plot_scenario_results(x_axes, y_axes, data)


def plot_constrained_lane_change(data: pd.DataFrame, lc_veh_id: int):
    compute_all_relative_values(data)

    sns.set_style('whitegrid')
    x_axes = ['x', 't', 't']
    y_axes = ['y', 'v', 'phi']

    lc_vehicle_data = data[data['id'] == lc_veh_id]

    fig, ax = plt.subplots(len(y_axes) + 1)
    fig.set_size_inches(9, 6)
    ego_safe_gap = compute_default_safe_gap(lc_vehicle_data['v'].to_numpy())

    gap = lc_vehicle_data['gap_to_orig_lane_leader'].to_numpy()
    orig_lane_error = gap - ego_safe_gap
    ax[0].plot(lc_vehicle_data['t'], orig_lane_error, label='ego to lo')

    gap = lc_vehicle_data['gap_to_dest_lane_leader'].to_numpy()
    dest_lane_error = gap - ego_safe_gap
    ax[0].plot(lc_vehicle_data['t'], dest_lane_error, label='ego to ld')

    dest_follower_ids = lc_vehicle_data['dest_lane_follower_id'].unique()
    dest_follower_id = [veh_id for veh_id in dest_follower_ids if veh_id >= 0]
    if len(dest_follower_id) > 1:
        print("Hey! Time to deal with multiple dest lane followers")
    follower_data = data[data['id'] == dest_follower_id[0]]
    foll_safe_gap = compute_default_safe_gap(follower_data['v'].to_numpy())
    gap = lc_vehicle_data['gap_to_dest_lane_follower'].to_numpy()
    dest_lane_error = gap - foll_safe_gap
    ax[0].plot(lc_vehicle_data['t'], dest_lane_error, label='fd to ego')

    ax[0].legend()
    low, high = ax[0].get_ylim()
    low, high = max(low, -2), min(high, 2)
    ax[0].set(xlabel=_get_variable_with_unit('t'),
              ylabel=_get_variable_with_unit('gap_error'),
              ylim=(low, high))

    for i, (x, y) in enumerate(zip(x_axes, y_axes)):
        show_legend = True if i == len(x_axes) - 1 else False
        sns.lineplot(data, x=x, y=y, hue='name', ax=ax[i + 1], palette='tab10',
                     legend=show_legend)
        low, high = ax[i + 1].get_ylim()
        if y == 'v' and high - low < 1:
            low, high = np.floor(low - 0.5), np.ceil(high + 0.5)
        ax[i + 1].set(xlabel=_get_variable_with_unit(x),
                      ylabel=_get_variable_with_unit(y),
                      ylim=(low, high))
    fig.tight_layout()
    fig.show()


def plot_vehicle_following(data: pd.DataFrame):
    """
    Plots the x and vel vs time
    :param data:
    :return:
    """
    sns.set_style('whitegrid')
    compute_all_relative_values(data)
    x_axes = ['t', 't']
    y_axes = ['gap_to_orig_lane_leader', 'v']
    plot_scenario_results(x_axes, y_axes, data)


def plot_scenario_results(x_axes: List[str], y_axes: List[str],
                          data: pd.DataFrame):
    """

    :param x_axes: Name of the variable on the x-axis for each plot
    :param y_axes: Name of the variable on the y-axis for each plot
    :param data:
    :return:
    """
    fig, ax = plt.subplots(len(y_axes))
    for i, (x, y) in enumerate(zip(x_axes, y_axes)):
        sns.lineplot(data, x=x, y=y, hue='id', ax=ax[i], palette='tab10')
        low, high = ax[i].get_ylim()
        if y == 'v' and high - low < 1:
            low, high = np.floor(low - 0.5), np.ceil(high + 0.5)
        ax[i].set(xlabel=_get_variable_with_unit(x),
                  ylabel=_get_variable_with_unit(y),
                  ylim=(low, high))
    fig.tight_layout()
    fig.show()


def _get_variable_with_unit(variable: str):
    try:
        return variable + ' [' + const.units[variable] + ']'
    except KeyError:
        return variable
