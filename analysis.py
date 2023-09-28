import pickle
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import constants as const
import post_processing as pp


def load_latest_simulated_scenario(pickle_file_name: str):
    with open(pickle_file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_costs_vs_iteration(running_costs, terminal_costs,
                            plot_separately: bool = False):
    """

    :param running_costs: 2D list with costs vs iterations
    :param terminal_costs: 2D list with costs vs iterations. Could be an empty
     list.
    :return:
    """
    n = len(running_costs)
    has_terminal_cost = len(terminal_costs) > 0

    rc_color = 'tab:blue'
    tc_color = 'tab:orange'
    fig, axs_2d = plt.subplots(n, 1, squeeze=False)
    axs = [ax[0] for ax in axs_2d]
    for i in range(n):
        if plot_separately:
            axs[i].plot(running_costs[i], label='running costs', color=rc_color)
            _, y_high = axs[i].get_ylim()
            axs[i].set_ylim([0, min(running_costs[i][0] + 0.5, y_high)])
            axs[i].set_ylabel('running costs', color=rc_color)
            axs[i].tick_params(axis='y', labelcolor=rc_color)
            if has_terminal_cost:
                ax_secondary = axs[i].twinx()
                ax_secondary.plot(terminal_costs[i], label='terminal costs',
                                  color=tc_color)
                ax_secondary.set_ylabel('terminal costs', color=tc_color)
                ax_secondary.tick_params(axis='y', labelcolor=tc_color)
                ax_secondary.set_yscale('log')
            # axs[i].legend()
        else:
            cost = running_costs[i]
            cost += (terminal_costs[i] if has_terminal_cost else 0)
            axs[i].plot(cost)
            _, y_high = axs[i].get_ylim()
            axs[i].set_ylim([1, 1.2])
            axs[i].set_ylabel('cost')
            axs[i].set_yscale('log')

    axs[-1].set_xlabel('iteration')
    fig.tight_layout()
    plt.show()


def plot_trajectory(data: pd.DataFrame):
    # min_y, max_y = data['y'].min(), data['y'].max()
    min_x, max_x = data['x'].min(), data['x'].max()
    dt = 1.0  # [s]
    time = np.arange(data['t'].min(), data['t'].max(), dt)
    step = round(dt / (data['t'].iloc[1] - data['t'].iloc[0]))
    fig, ax = plt.subplots(len(time), 1)
    # fig.set_size_inches(6, 9)
    for i in range(len(time)):
        for veh_id in data['id'].unique():
            veh_data = data[data['id'] == veh_id]
            veh_name = veh_data['name'].iloc[0]
            color = _get_color_by_name(veh_name)
            ax[i].scatter(data=veh_data.iloc[i * step],
                          x='x', y='y', marker='>', color=color,)
        # sns.scatterplot(data, x='x', y='y', hue='name', palette=colors,
        #                 markers='>', legend=False, marker='>')

        ax[i].axhline(y=const.LANE_WIDTH / 2, linestyle='--', color='black')
        # ax[i].set_aspect('equal', adjustable='box')
        ax[i].set(xlim=(min_x - 2, max_x),
                  ylim=(-const.LANE_WIDTH / 2, 3 * const.LANE_WIDTH / 2))
        if i == len(time) - 1:
            ax[i].set(xlabel=_get_variable_with_unit('x'))
        else:
            ax[i].set_xticks([])

    fig.tight_layout()
    fig.show()


def plot_initial_and_final_states(data: pd.DataFrame, axis=None,
                                  custom_colors: bool = False):
    """

    :param data: Dataframe containing only initial and desired final state
     for each vehicle
    :param axis: Axis on which to do the plot [optional]
    :param custom_colors: Whether to use the default colors (one per vehicle
     and up to 10 vehicles) or our custom defined colors based on the vehicle's
     name
    """
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = plt.gcf()

    if not custom_colors:
        cmap = plt.get_cmap("tab10")
        n_vehs = data['id'].nunique()
        ax.scatter(data=data[data['t'] == data['t'].min()], x='x', y='y',
                   marker='>', c=cmap.colors[0:n_vehs])
        ax.scatter(data=data[data['t'] == data['t'].max()], x='x', y='y',
                   marker='>', c=cmap.colors[0:n_vehs], alpha=0.6)
    else:
        for veh_id in data['id'].unique():
            veh_data = data[data['id'] == veh_id]
            veh_name = veh_data['name'].iloc[0]
            color = _get_color_by_name(veh_name)
            ax.scatter(data=veh_data[veh_data['t'] == veh_data['t'].min()],
                       x='x', y='y', marker='>', color=color)
            ax.scatter(data=veh_data[veh_data['t'] == veh_data['t'].max()],
                       x='x', y='y', marker='>', color=color, alpha=0.6)
    min_y = data['y'].min()
    max_y = data['y'].max()
    ax.axhline(y=const.LANE_WIDTH / 2, linestyle='--', color='black')
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


def plot_constrained_lane_change(data: pd.DataFrame,
                                 lc_veh_id_or_name: Union[int, str]):
    pp.compute_all_relative_values(data)

    sns.set_style('whitegrid')
    x_axes = ['t', 't', 't', 't']
    y_axes = ['y', 'theta', 'v', 'phi']

    fig, ax = plt.subplots(len(y_axes) + 1)
    fig.set_size_inches(12, 8)

    plot_gap_errors(data, lc_veh_id_or_name, ax[0])
    final_t = data['t'].max()
    for i, (x, y) in enumerate(zip(x_axes, y_axes)):
        show_legend = True if i == len(x_axes) - 1 else False
        sns.lineplot(data, x=x, y=y, hue='name', ax=ax[i + 1], palette='tab10',
                     legend=show_legend)
        y_low, y_high = ax[i + 1].get_ylim()
        if y == 'v' and y_high - y_low < 1:
            y_low, y_high = np.floor(y_low - 0.5), np.ceil(y_high + 0.5)
        ax[i + 1].set(xlabel=_get_variable_with_unit(x),
                      ylabel=_get_variable_with_unit(y),
                      xlim=(0, final_t),
                      ylim=(y_low, y_high))
    fig.tight_layout()
    fig.show()


def plot_gap_errors(data: pd.DataFrame,
                    veh_id_or_name: Union[int, str], ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(veh_id_or_name, str):
        lc_vehicle_data = data[data['name'] == veh_id_or_name]
        veh_name = veh_id_or_name
    else:
        lc_vehicle_data = data[data['id'] == veh_id_or_name]
        veh_name = lc_vehicle_data['name'].iloc[0]
    ego_safe_gap = pp.compute_default_safe_gap(lc_vehicle_data['v'].to_numpy())

    gap = lc_vehicle_data['gap_to_orig_lane_leader'].to_numpy()
    orig_lane_error = gap - ego_safe_gap
    ax.plot(lc_vehicle_data['t'], orig_lane_error, label=veh_name + ' to lo')

    gap = lc_vehicle_data['gap_to_dest_lane_leader'].to_numpy()
    dest_lane_error = gap - ego_safe_gap
    ax.plot(lc_vehicle_data['t'], dest_lane_error, label=veh_name + ' to ld')

    dest_follower_ids = lc_vehicle_data['dest_lane_follower_id'].unique()
    dest_follower_id = [veh_id for veh_id in dest_follower_ids if veh_id >= 0]
    if len(dest_follower_id) > 1:
        print("Hey! Time to deal with multiple dest lane followers")
    if len(dest_follower_id) > 0:
        follower_data = data[data['id'] == dest_follower_id[0]]
        foll_safe_gap = pp.compute_default_safe_gap(follower_data['v'].to_numpy())
        gap = lc_vehicle_data['gap_to_dest_lane_follower'].to_numpy()
        dest_lane_error = gap - foll_safe_gap
        ax.plot(lc_vehicle_data['t'], dest_lane_error,
                label='fd to ' + veh_name)

    ax.legend()
    y_low, y_high = ax.get_ylim()
    final_t = data['t'].max()
    ax.set(xlabel=_get_variable_with_unit('t'),
           ylabel=_get_variable_with_unit('gap_error'),
           xlim=(0, final_t), ylim=(y_low, y_high))


def plot_vehicle_following(data: pd.DataFrame):
    """
    Plots the x and vel vs time
    :param data:
    :return:
    """
    sns.set_style('whitegrid')
    pp.compute_all_relative_values(data)
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


def check_constraint_satisfaction(data: pd.DataFrame, lc_id: int):
    data['safe_gap'] = pp.compute_default_safe_gap(data['v'])
    data['gap_error'] = data['gap_to_orig_lane_leader'] - data['safe_gap']
    data['constraint'] = np.minimum(data['gap_error'], 0) * data['phi']
    plot_scenario_results(['t', 't', 't'], ['constraint', 'phi', 'gap_error'],
                          data[data['id'] == lc_id])


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


def _get_variable_with_unit(variable: str):
    try:
        return variable + ' [' + const.UNIT_MAP[variable] + ']'
    except KeyError:
        return variable


def _get_color_by_name(veh_name: str):
    if veh_name == 'p1':
        color = const.COLORS['dark_blue']
    elif veh_name == 'ego' or veh_name.startswith('p'):
        color = const.COLORS['blue']
    elif veh_name in {'lo', 'lo1', 'ld', 'ld1', 'fd', 'fd1'}:
        color = const.COLORS['red']
    elif veh_name.startswith('ld') or veh_name.startswith('fd'):
        color = const.COLORS['orange']
    else:
        color = const.COLORS['gray']
    return color
