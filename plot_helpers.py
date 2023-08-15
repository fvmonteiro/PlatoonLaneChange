import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import units


def load_simulated_scenario(pickle_file_name: str):
    with open(pickle_file_name, 'rb') as f:
        data = pickle.load(f)
    return data


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


def plot_constrained_lane_change(data: pd.DataFrame, lc_veh_id: int,
                                 orig_leader_id: int = None,
                                 dest_leader_id: int = None):
    sns.set_style('whitegrid')
    x_axes = ['x', 't', 't']
    y_axes = ['y', 'v', 'phi']
    data['gap'] = 0

    fig, ax = plt.subplots(len(y_axes) + 1)
    fig.set_size_inches(9, 6)
    if orig_leader_id is not None:
        data.loc[data['id'] == lc_veh_id, 'gap_orig'] = (
                data.loc[data['id'] == orig_leader_id, 'x']
                - data.loc[data['id'] == lc_veh_id, 'x'])
        ax[0].plot('t', 'gap_orig', data=data)
    if dest_leader_id is not None:
        data.loc[data['id'] == lc_veh_id, 'gap_dest'] = (
                data.loc[data['id'] == dest_leader_id, 'x']
                - data.loc[data['id'] == lc_veh_id, 'x'])
        data.plot('t', 'gap_dest', ax=ax[0], data=data)
        ax[1].legend()

    for i, (x, y) in enumerate(zip(x_axes, y_axes)):
        sns.lineplot(data, x=x, y=y, hue='id', ax=ax[i + 1], palette='tab10')
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
    return variable + ' [' + units[variable] + ']'
