import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import vehicle_handler
from constants import units


def compute_values_relative_to_leader(data: pd.DataFrame):
    merged = data[['t', 'id', 'leader_id', 'v', 'x']].merge(
        data[['t', 'id', 'v', 'x']], how='left',
        left_on=['t', 'leader_id'], right_on=['t', 'id'],
        suffixes=(None, '_leader')).drop(columns='id_leader')
    data['gap_to_leader'] = merged['x_leader'] - merged['x']
    # data['rel_vel_to_leader'] = merged['v'] - merged['v_leader']


def compute_values_relative_to_future_leader(data: pd.DataFrame):
    merged = data[['t', 'id', 'dest_lane_leader_id', 'v', 'x']].merge(
        data[['t', 'id', 'v', 'x']], how='left',
        left_on=['t', 'dest_lane_leader_id'], right_on=['t', 'id'],
        suffixes=(None, '_leader')).drop(columns='id_leader')
    data['gap_to_dest_lane_leader'] = merged['x_leader'] - merged['x']
    # data['rel_vel_to_dest_lane_leader'] = merged['v'] - merged['v_leader']


def compute_values_relative_to_future_follower(data: pd.DataFrame):
    merged = data[['t', 'id', 'dest_lane_follower_id', 'v', 'x']].merge(
        data[['t', 'id', 'v', 'x']], how='left',
        left_on=['t', 'dest_lane_follower_id'], right_on=['t', 'id'],
        suffixes=(None, '_follower')).drop(columns='id_follower')
    data['gap_to_dest_lane_follower'] = merged['x'] - merged['x_follower']


def check_constraint_satisfaction(data: pd.DataFrame, lc_id: int):
    compute_values_relative_to_leader(data)
    data['safe_gap'] = data['v'] + 1
    data['gap_error'] = data['gap_to_leader'] - data['safe_gap']
    data['constraint'] = np.minimum(data['gap_error'], 0) * data['phi']
    plot_scenario_results(['t', 't', 't'], ['constraint', 'phi', 'gap_error'],
                          data[data['id'] == lc_id])


def load_simulated_scenario(pickle_file_name: str):
    with open(pickle_file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_initial_and_final_states(data: pd.DataFrame):
    """

    :param data: Dataframe containing only initial and desired final state
     for each vehicle
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    n_vehs = data['id'].nunique()
    ax.scatter(data=data[data['t'] == data['t'].min()], x='x', y='y',
               marker='>', c=cmap.colors[0:n_vehs])
    ax.scatter(data=data[data['t'] == data['t'].max()], x='x', y='y',
               marker='>', c=cmap.colors[0:n_vehs], alpha=0.7)
    min_y = data['y'].min()
    max_y = data['y'].max()
    ax.axhline(y=(min_y + max_y) / 2, linestyle='--', color='black')
    ax.set(ylim=(min_y - 2, max_y + 2))
    ax.set_aspect('equal', adjustable='box')
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
    compute_values_relative_to_leader(data)
    compute_values_relative_to_future_leader(data)
    compute_values_relative_to_future_follower(data)

    sns.set_style('whitegrid')
    x_axes = ['x', 't', 't']
    y_axes = ['y', 'v', 'phi']

    lc_vehicle = vehicle_handler.FourStateVehicleAccelFB()
    lc_vehicle_data = data[data['id'] == lc_veh_id]
    # We assume single leader and dest lane leader during optimization phase
    orig_leader_id = lc_vehicle_data['leader_id'].iloc[0]
    dest_leader_id = lc_vehicle_data['dest_lane_leader_id'].iloc[0]
    dest_follower_id = lc_vehicle_data['dest_lane_follower_id'].iloc[0]
    fig, ax = plt.subplots(len(y_axes) + 1)
    fig.set_size_inches(9, 6)
    ego_safe_gap = (lc_vehicle_data['v'].to_numpy() * lc_vehicle.safe_h
                    + lc_vehicle.c)
    if orig_leader_id >= 0:
        # ax[0].plot('t', 'gap_to_leader', data=lc_vehicle_data, label='gap')
        gap = lc_vehicle_data['gap_to_leader'].to_numpy()
        orig_lane_error = gap - ego_safe_gap
        ax[0].plot(lc_vehicle_data['t'], orig_lane_error, label='lo')
        ax[0].legend()
    if dest_leader_id >= 0:
        gap = lc_vehicle_data['gap_to_dest_lane_leader'].to_numpy()
        dest_lane_error = gap - ego_safe_gap
        ax[0].plot(lc_vehicle_data['t'], dest_lane_error, label='ld')
        ax[0].legend()
    if dest_follower_id >= 0:
        follower_data = data[data['id'] == dest_follower_id]
        foll_safe_gap = follower_data['v'] * lc_vehicle.safe_h + lc_vehicle.c
        gap = lc_vehicle_data['gap_to_dest_lane_follower'].to_numpy()
        dest_lane_error = gap - foll_safe_gap
        ax[0].plot(lc_vehicle_data['t'], dest_lane_error, label='fd')
        ax[0].legend()
    low, high = ax[0].get_ylim()
    low, high = max(low, -5), min(high, 5)
    ax[0].set(xlabel=_get_variable_with_unit('t'),
              ylabel=_get_variable_with_unit('gap'),
              ylim=(low, high))

    for i, (x, y) in enumerate(zip(x_axes, y_axes)):
        show_legend = True if i == len(x_axes) - 1 else False
        sns.lineplot(data, x=x, y=y, hue='id', ax=ax[i + 1], palette='tab10',
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
    data['gap'] = 0
    data.loc[data['id'] == 1, 'gap'] = (data.loc[data['id'] == 0, 'x']
                                        - data.loc[data['id'] == 1, 'x'])
    x_axes = ['t', 't']
    y_axes = ['gap', 'v']
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
        return variable + ' [' + units[variable] + ']'
    except KeyError:
        return variable
