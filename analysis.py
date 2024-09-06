from __future__ import annotations

from collections.abc import Mapping, Iterable
import pickle
import os
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import configuration
import helper
import post_processing
from platoon_functionalities import graph_tools, traffic_state_graph
import post_processing as pp


# ============================================================================ #
# Analysis tools for new graph exploration approach


def compare_bfs_and_dfs(n_platoon: int, cost_type: str):
    """
    Compare the results from the original approach (BFS full graph exploration
    then shortest path) to the newer DFS with epsilon-greedy exploration
    """

    # simulator_name = "vissim"

    warnings.warn("Function is outdated.")

    results = traffic_state_graph.load_best_results(n_platoon, cost_type, 1.0)
    strategy_map = graph_tools.LaneChangeStrategyManager.load_strategy_map(
        n_platoon, cost_type)

    disagreement_count = 0
    for initial_state, outer_dict in results.items():
        for first_mover_set, query_result in outer_dict.items():
            dfs_solution = (query_result["lc_order"], query_result["coop_order"])
            try:
                bfs_solution = strategy_map[initial_state][first_mover_set][0]
                if not dfs_solution == bfs_solution:
                    print(f"BFS sol.: {bfs_solution}")
                    print(f"DFS sol.: {dfs_solution}")
                    initial_node = traffic_state_graph.TrafficStateNode(
                        initial_state)
                    bfs_commands = helper.tuple_of_lists_to_list(bfs_solution)
                    bfs_cost, _ = initial_node.traverse_action_sequence(
                        bfs_commands)
                    dfs_commands = helper.tuple_of_lists_to_list(dfs_solution)
                    dfs_cost, _ = initial_node.traverse_action_sequence(
                        dfs_commands)
                    print(f"BFS cost: {bfs_cost}")
                    print(f"DFS cost: {dfs_cost}")
                    disagreement_count += 1
            except KeyError:
                print(f"Query x0 = {initial_state}, L0 = {first_mover_set} "
                      f"not found in bfs solution.")
                disagreement_count += 1
    if disagreement_count > 0:
        print(f"Disagreement count: {disagreement_count}")
    else:
        print("All results agree")


def plot_several_cost_vs_computation_time(
        n_platoon: Iterable[int], cost_type: Iterable[str],
        epsilon: Iterable[float], simulator: str = "all",
        save_fig: bool = False):
    data = post_processing.process_graph_exploration_results(
        n_platoon, cost_type, epsilon, simulator)
    data['$\epsilon$'] = data['epsilon'].astype(str)

    sns.set(style="whitegrid")
    font_size = 18
    plt.rc("font", size=font_size)  # not working
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rcParams['legend.title_fontsize'] = font_size
    # plt.rcParams.update({'font.size': 22})
    cost_pretty_name = {"time": "Maneuver Time", "accel": "Control Effort"}
    for cost in cost_type:
        for n in n_platoon:
            data_to_plot = data[(data["cost_type"] == cost)
                                & (data["n_platoon"] == n)]
            fig, ax = plt.subplots(1, 1)
            # plt.figure(figsize=(10, 6))
            sns.lineplot(x='time', y='norm_cost', data=data_to_plot,
                         hue='$\epsilon$', errorbar=None,
                         estimator=np.mean, ax=ax, linewidth=2.5
                        )
            ax.set_xlabel('Computation Time [s]')
            ax.set_ylabel(f'Normalized {cost_pretty_name[cost]}')

            # Because plt.rc("font", size=font_size) is not working
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_size)

            # ax.set_title(f'platoon size: {n}')
            fig.tight_layout()
            fig.set_dpi(400)
            # Show the plot
            fig.show()
            if save_fig:
                ResultAnalyzer.save_fig_at_shared_folder(
                    fig, "estimated_" + cost + "_vs_computation_time")


def plot_estimated_cost_vs_computation_time(
        n_platoon: int, cost_type: str, epsilon: float):
    data = post_processing.process_single_graph_exploration_result(
        n_platoon, cost_type, epsilon)

    # === Outputs === #
    # print(f"Avg. first sol. time: {np.mean(first_times)}")
    # print(f"Worst. first sol. time: {np.max(first_times)}")
    # print(f"Avg. total time: {np.mean(end_times)}")
    # print(f"Worst. total time: {np.max(end_times)}")

    sns.set(style="whitegrid")
    pi = 90
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(x='time', y='norm_cost', data=data,
                      # errorbar=("pi", pi),
                      estimator=np.mean, marker='o')

    ax.set_xlabel('time')
    ax.set_ylabel(f'normalized {cost_type}')
    ax.set_title(f'n={n_platoon}, epsilon={epsilon}')
    # ax.legend([f'Mean with {pi} percentile interval'])

    # Show the plot
    plt.show()


# ============================================================================ #
# TODO: create a class for optimal control results and one for
#  plotting simulations


class ResultAnalyzer:
    _category_plot_names = {
        "Graph-based_time": "Graph Min Time",
        "Graph-based_accel": "Graph Min Control ",
        "Last First": "Last Vehicle First",
        "Synchronous": "Single-Body Platoon"
    }
    _column_plot_names = {
        "completion_time": "Maneuver Time [s]",
        "accel_cost": "Control Effort [m/s]",
        "n_platoon": "Platoon Size",
        "strategy": "Strategy"
    }
    _simulation_identifiers = ["n_platoon", "vo", "vd", "gap_position"]
    _cost_names = ["completion_time", "accel_cost"]

    def __init__(self, is_bfs: bool, save_figs: bool = False):
        if is_bfs:
            self.results = ResultAnalyzer.load_bfs_result_summary()
        else:
            self.results = ResultAnalyzer.load_result_summary()
        self.save_figs = save_figs

    @staticmethod
    def load_bfs_result_summary():
        file_name = "result_summary_bfs.csv"
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "platoon_strategy_results", file_name)
        results = pd.read_csv(file_path)
        # Pretty names
        results["strategy_orig"] = results["strategy"]
        results["strategy"] = results["strategy"].map(
            ResultAnalyzer._category_plot_names).fillna(results["strategy"])

        simulation_identifiers = ["n_platoon", "vo", "vd", "gap_position",
                                  "strategy"]
        # For each simulated configuration, get only the latest result
        latest_results = results.loc[results.groupby(
            simulation_identifiers)["experiment_counter"].idxmax()]
        return latest_results

    @staticmethod
    def load_result_summary():
        file_name = "result_summary.csv"
        file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                                 "platoon_strategy_results", file_name)
        results = pd.read_csv(file_path)
        # Pretty names
        results["strategy_orig"] = results["strategy"]
        results["strategy"] = results["strategy"].map(
            ResultAnalyzer._category_plot_names).fillna(results["strategy"])

        simulation_identifiers = ["n_platoon", "vo", "vd", "gap_position",
                                  "strategy", "epsilon", "max_computation_time"]
        # For each simulated configuration, get only the latest result
        latest_results = results.loc[results.groupby(
            simulation_identifiers)["experiment_counter"].idxmax()]
        return latest_results

    def get_python_results_for_paper(self) -> None:
        self.print_average_number_of_maneuver_steps()
        self.compare_approaches()
        # self.compare_graph_to_best_heuristic()

    def print_average_number_of_maneuver_steps(self):
        self.results["K"] = self.results["cooperation_order"].str.strip(
            "[]").str.split(",").apply(len)
        print(self.results.groupby(["n_platoon", "strategy"])[
                  ["success", "K"]].mean())

    def compare_graph_to_best_heuristic(self) -> None:

        graph_results, other_results = self.split_result_df(
            ignore_synchronous=True)
        other_results.loc[
            ~other_results["success"], self._cost_names] = np.inf

        # Success rates
        best_results = other_results.loc[other_results.groupby(
            self._simulation_identifiers)["success"].idxmax()]
        best_results["strategy"] = "Best Fixed Order"
        all_results = pd.concat([graph_results, other_results, best_results])
        success_rate = all_results.groupby(["n_platoon", "strategy"])[
            ["success"]].mean()
        print(success_rate)

        # Time and Accel
        sns.set_style("whitegrid")
        plt.rcParams.update({"font.size": 16})
        for c in self._cost_names:
            best_results = other_results.loc[other_results.groupby(
                self._simulation_identifiers)[c].idxmin()]
            best_results["strategy"] = "Best Fixed Order"
            all_results = pd.concat([graph_results, best_results])
            drop_sims = best_results.loc[~best_results["success"],
                                         self._simulation_identifiers]
            successful_sims = all_results.set_index(
                self._simulation_identifiers).drop(index=drop_sims.to_numpy()
                                                   ).reset_index()
            avg_costs = successful_sims.groupby(["n_platoon", "strategy"])[
                c].mean()
            print(avg_costs)

            x, y, hue = self._get_plot_names(["n_platoon", c, "strategy"])
            ax = sns.boxplot(
                data=successful_sims.rename(
                    columns=self._column_plot_names),
                x=x, y=y, hue=hue)
            fig: plt.Figure = ax.get_figure()
            fig.tight_layout()
            fig.show()
            if self.save_figs:
                self.save_fig_at_shared_folder(
                    fig, c + "_comparison_to_best_heuristic")

    def compare_approaches(self) -> None:

        sns.set_style("whitegrid")
        plt.rcParams.update({"font.size": 16})

        graph_strategies = []
        other_strategies = []
        for s in self.results["strategy"].unique():
            if s.lower().startswith("graph"):
                graph_strategies.append(s)
            elif s.lower() != "single-body platoon":
                other_strategies.append(s)

        # TODO: reorganize to use these two dfs
        graph_results, other_results = self.split_result_df(
            ignore_synchronous=True)

        cost_names = ["completion_time", "accel_cost"]
        platoon_sizes = self.results["n_platoon"].unique()
        for other in other_strategies:
            relevant_results_per_n = []
            for n in platoon_sizes:
                # print(f"n={n}")
                results_n = self.results[self.results["n_platoon"] == n]
                other_results = results_n.loc[
                    (results_n["strategy"] == other)].reset_index(drop=True)
                other_success_idx = other_results["success"]
                graph_results = pd.concat(
                    [results_n.loc[results_n["strategy"] == gs
                                   ].reset_index(drop=True).loc[
                         other_success_idx]
                     for gs in graph_strategies])
                relevant_results_per_n.append(pd.concat(
                    [graph_results, other_results.loc[other_success_idx]]))
            relevant_results = pd.concat(relevant_results_per_n)
            if relevant_results.empty:
                continue
            grouped_results = relevant_results.groupby(
                ["n_platoon", "strategy"])
            avg_costs = grouped_results[cost_names].mean()
            print(avg_costs)

            for c in cost_names:
                x, y, hue = self._get_plot_names(["n_platoon", c, "strategy"])
                ax: plt.Axes = sns.pointplot(
                    data=relevant_results.rename(
                        columns=self._column_plot_names),
                    x=x, y=y, hue=hue)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[0:], labels=labels[0:])
                fig: plt.Figure = ax.get_figure()
                fig.tight_layout()
                fig.show()
                if self.save_figs:
                    self.save_fig_at_shared_folder(
                        fig, c + "_comparison_to_" + other)

    def compare_to_approach(self, cost_name: str) -> None:
        accepted_cost_name = {"time", "control"}
        if cost_name.lower() not in accepted_cost_name:
            raise ValueError(f"cost_name must be one of {accepted_cost_name}")
        our_approach = "Graph Min " + cost_name.title()
        strategies = self.results["strategy"].unique()
        platoon_sizes = self.results["n_platoon"].unique()
        all_main_results = self.results.loc[
            self.results["strategy"] == our_approach].sort_values(
            by=["vd", "gap_position"]).reset_index(drop=True)
        for strat in strategies:
            if strat == our_approach:
                continue
            for n in platoon_sizes:
                main_results = all_main_results.loc[
                    all_main_results["n_platoon"] == n].reset_index(drop=True)
                other_results = self.results.loc[
                    (self.results["strategy"] == strat)
                    & (self.results["n_platoon"] == n)
                    ].sort_values(by=["vd", "gap_position"]).reset_index(
                    drop=True)
                # other_success = other_results.loc[other_results["success"]]
                main_success_count = main_results["success"].sum()
                other_success_count = other_results["success"].sum()
                success_diff = ((main_success_count - other_success_count)
                                / other_success_count)
                both_success_idx = (other_results["success"]
                                    & main_results["success"])
                time_results = self._compute_variation(
                    other_results.loc[both_success_idx],
                    main_results.loc[both_success_idx],
                    "completion_time")
                accel_results = self._compute_variation(
                    other_results.loc[both_success_idx],
                    main_results.loc[both_success_idx],
                    "accel_cost")
                print(f"Comparing {our_approach} to {strat}, n={n}")
                print(f"Absolute results:\n"
                      f"\tsuccess: {main_results['success'].sum()} vs "
                      f"{other_results['success'].sum()}\n"
                      f"\ttime: {time_results['new_avg']:.1f} vs "
                      f"{time_results['base_avg']:.1f}\n"
                      f"\taccel: {accel_results['new_avg']:.1f} vs "
                      f"{accel_results['base_avg']:.1f}"
                      )
                print("Comparative:\n"
                      f"\tsuccess: {success_diff * 100:+.1f}%\n"
                      f"\ttime: {time_results['avg_of_changes'] * 100:+.1f}%\n"
                      f"\taccel: {accel_results['avg_of_changes'] * 100:+.1f}%")

    def split_result_df(self, ignore_synchronous=False
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataframe in two: the first with the graph approach results
        and the other with the rest.
        :return:
        """
        graph_strategies = []
        other_strategies = []
        for s in self.results["strategy"].unique():
            if s.lower().startswith("graph"):
                graph_strategies.append(s)
            elif ignore_synchronous and s.lower() != "single-body platoon":
                other_strategies.append(s)

        other_results = self.results[
            self.results["strategy"].isin(other_strategies)
        ].sort_values(by=self._simulation_identifiers)
        graph_results = self.results[
            self.results["strategy"].isin(graph_strategies)
        ].sort_values(by=self._simulation_identifiers)
        return graph_results, other_results

    def plot_cost_vs_max_computation_time(self, n_platoon: int):
        for strategy in self.results["strategy_orig"].unique():
            data = self.results.loc[
                (self.results["n_platoon"] == n_platoon)
                & (self.results["strategy_orig"] == strategy)]
            cost_type = strategy.split("_")[1]
            if cost_type == "time":
                cost_col = "completion_time"
            elif cost_type == "accel":
                cost_col = "accel_cost"
            else:
                raise RuntimeError(f"Unknown cost_type {cost_type}")
            sns.lineplot(data=data, x="max_computation_time", y=cost_col)
            plt.show()

    def _get_plot_names(self, var_names: Iterable[str]) -> list[str]:
        res = []
        for name in var_names:
            if name in self._column_plot_names:
                res.append(self._column_plot_names[name])
            elif name in self._category_plot_names:
                res.append(self._category_plot_names[name])
            else:
                raise KeyError("Variable not in any 'plot_name' map")
        return res

    @staticmethod
    def _compute_variation(base_results, new_results, variable):
        base_avg = base_results[variable].mean()
        new_avg = new_results[variable].mean()
        change_in_avg = (new_avg - base_avg) / base_avg
        avg_of_changes = ((new_results[variable] - base_results[variable])
                          / base_results[variable]).mean()
        return {"base_avg": base_avg, "new_avg": new_avg,
                "change_in_avg": change_in_avg,
                "avg_of_changes": avg_of_changes}

    @staticmethod
    def save_fig_at_shared_folder(figure: plt.Figure, fig_name: str):
        file_folder = configuration.SHARED_IMAGES_PATH
        figure.savefig(os.path.join(file_folder, fig_name))
        print(f"Figure {fig_name} saved at {file_folder}")


def load_latest_simulated_scenario(pickle_file_name: str):
    with open(pickle_file_name, "rb") as f:
        data = pickle.load(f)
    return data


def plot_cost_vs_ordering(cost: Iterable[float],
                          completion_times: Iterable[float],
                          named_orderings: dict[str, int]):
    # Analysis ideas:
    # - Differentiate successful from non-finished maneuvers
    # - Differentiate based on which vehicle starts the maneuver

    cost = np.array(cost)
    completion_times = np.array(completion_times)
    special_x_ticks = [""] * len(cost)
    for key, value in named_orderings.items():
        special_x_ticks[value] = key

    idx_sort = np.argsort(cost)[::-1]
    special_x_ticks = np.array(special_x_ticks)[idx_sort]

    blue = "tab:blue"
    orange = "tab:orange"
    fig, ax = plt.subplots(1, 1)
    ax.plot(cost[idx_sort], label="cost", color=blue)
    ax.set_ylabel("cost", color=blue)
    ax.set_xlabel("move/coop order")
    ax.set_xticks([i for i in range(len(cost))], labels=special_x_ticks)
    ax.tick_params(axis="y", labelcolor=blue)
    ax_right = ax.twinx()
    ax_right.plot(completion_times[idx_sort], label="LC time", color=orange)
    ax_right.set_ylabel("LC time", color=orange)
    ax_right.tick_params(axis="y", labelcolor=orange)

    fig.tight_layout()
    fig.show()


def plot_trajectory(data: pd.DataFrame, plot_title: str = None,
                    n_plots: int = 8):
    sns.set_style("white")
    # min_y, max_y = data["y"].min(), data["y"].max()
    min_x, max_x = data["x"].min(), data["x"].max()
    tf = data["t"].max()
    dt = tf / n_plots
    time = np.arange(data["t"].min(), tf + dt / 2, dt)
    step = int(np.floor(dt / (data["t"].iloc[1] - data["t"].iloc[0])))
    fig, ax = plt.subplots(len(time), 1)
    fig.set_size_inches(6, n_plots - 1)
    for i in range(len(time)):
        k = i * step
        for veh_id in data["id"].unique():
            veh_data = data[data["id"] == veh_id]
            veh_name = veh_data["name"].iloc[0]
            color = _get_color_by_name(veh_name)
            # if i == len(time) - 1:  # temp
            #     k = veh_data.shape[0] - 1
            ax[i].scatter(data=veh_data.iloc[k],
                          x="x", y="y", marker=">", color=color, )

        ax[i].set_title("t = {:.1f}".format(time[i]), loc="left")
        ax[i].axhline(y=configuration.LANE_WIDTH / 2, linestyle="--",
                      color="black")
        # ax[i].set_aspect("equal", adjustable="box")
        ax[i].set(xlim=(min_x - 2, max_x + 3),
                  ylim=(-configuration.LANE_WIDTH / 2,
                        3 * configuration.LANE_WIDTH / 2))
        if i == len(time) - 1:
            ax[i].set(xlabel=_get_variable_with_unit("x"))
        else:
            ax[i].set_xticks([])

    if plot_title:
        fig.suptitle(plot_title, fontsize=14)
    fig.tight_layout()
    fig.show()


def plot_initial_and_final_states(data: pd.DataFrame, axis=None,
                                  custom_colors: bool = False):
    """

    :param data: Dataframe containing only initial and desired final state
     for each vehicle
    :param axis: Axis on which to do the plot [optional]
    :param custom_colors: Whether to use the default colors (one per vehicle
     and up to 10 vehicles) or our custom defined colors based on the vehicle"s
     name
    """
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = plt.gcf()

    if not custom_colors:
        cmap = plt.get_cmap("tab10")
        n_vehs = data["id"].nunique()
        ax.scatter(data=data[data["t"] == data["t"].min()], x="x", y="y",
                   marker=">", c=cmap.colors[0:n_vehs])
        ax.scatter(data=data[data["t"] == data["t"].max()], x="x", y="y",
                   marker=">", c=cmap.colors[0:n_vehs], alpha=0.6)
    else:
        for veh_id in data["id"].unique():
            veh_data = data[data["id"] == veh_id]
            veh_name = veh_data["name"].iloc[0]
            color = _get_color_by_name(veh_name)
            ax.scatter(data=veh_data[veh_data["t"] == veh_data["t"].min()],
                       x="x", y="y", marker=">", color=color)
            ax.scatter(data=veh_data[veh_data["t"] == veh_data["t"].max()],
                       x="x", y="y", marker=">", color=color, alpha=0.6)
    min_y = data["y"].min()
    max_y = data["y"].max()
    ax.axhline(y=configuration.LANE_WIDTH / 2, linestyle="--", color="black")
    ax.set(xlabel=_get_variable_with_unit("x"),
           ylabel=_get_variable_with_unit("y"),
           ylim=(min_y - 2, max_y + 2))
    ax.set_aspect("equal", adjustable="box")

    if axis is None:
        fig.tight_layout()
        fig.show()


def plot_initial_state(data: pd.DataFrame, axis=None):
    """

    :param data: Dataframe containing only initial and desired final state
     for each vehicle
    :param axis: Axis on which to do the plot [optional]
    """
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = plt.gcf()

    for veh_id in data["id"].unique():
        veh_data = data[data["id"] == veh_id]
        veh_name = veh_data["name"].iloc[0]
        color = _get_color_by_name(veh_name)
        ax.scatter(data=veh_data.iloc[0],
                   x="x", y="y", marker=">", color=color)
    min_y = data["y"].min()
    max_y = data["y"].max()
    ax.axhline(y=configuration.LANE_WIDTH / 2, linestyle="--", color="black")
    ax.set(xlabel=_get_variable_with_unit("x"),
           ylabel=_get_variable_with_unit("y"),
           ylim=(min_y - 2, max_y + 2))
    ax.set_aspect("equal", adjustable="box")

    if axis is None:
        fig.tight_layout()
        fig.show()


def plot_state_vector(state: Iterable[float], title: str = None):
    state = np.array(state).reshape(-1, 4).transpose()
    x = state[0, :]
    y = state[1, :]

    fig, ax = plt.subplots()
    ax.scatter(x, y, marker=">")
    ax.axhline(y=configuration.LANE_WIDTH / 2, linestyle="--", color="black")
    ax.set(xlabel=_get_variable_with_unit("x"),
           ylabel=_get_variable_with_unit("y"),
           ylim=(np.min(y) - 2, np.max(y) + 2))
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.show()


def plot_lane_change(data: pd.DataFrame):
    """
    Plots the lane change in a y vs x plot along with speed and steering wheel
     angle vs time
    :param data:
    :return:
    """
    sns.set_style("whitegrid")
    x_axes = ["x", "t", "t"]
    y_axes = ["y", "v", "phi"]
    plot_scenario_results(x_axes, y_axes, data)


def plot_platoon_lane_change(data: pd.DataFrame):
    pp.compute_all_relative_values(data)

    n_platoon = np.max([int(name[1]) for name in data["name"].unique()
                        if name[0] == "p"])
    vehicle_pairs = {"p1": ["ld0", "lo0"]}
    for i in range(2, n_platoon + 1):
        vehicle_pairs["p" + str(i)] = ["p" + str(i - 1)]
    vehicle_pairs["p" + str(n_platoon)].append("fd0")
    if n_platoon > 1:
        vehicle_pairs["p" + str(n_platoon)].append("ld0")

    sns.set_style("whitegrid")
    x_axes = ["t", "t", "t", "t"]
    y_axes = ["y", "v", "phi", "a"]

    fig, ax = plt.subplots(len(y_axes) + 1)
    fig.set_size_inches(12, 8)
    plot_gap_errors(data, vehicle_pairs, ax[0])
    plot_scenario_results(x_axes, y_axes, data, ax[1:])
    fig.tight_layout()
    fig.show()


def plot_gap_errors(
        data: pd.DataFrame,
        vehicle_pairs: Mapping[Union[int, str], list[Union[int, str]]],
        ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for ego_identifier, others in vehicle_pairs.items():
        ego_data = _get_veh_data(data, ego_identifier)
        if ego_data.empty:
            warnings.warn(f"No data for vehicle {ego_identifier}")
            continue
        ego_name = ego_data["name"].iloc[0]
        for other_identifier in others:
            other_data = _get_veh_data(data, other_identifier)
            if not other_data.empty:
                if other_data["x"].iloc[0] < ego_data["x"].iloc[0]:
                    follower_data = other_data
                    leader_data = ego_data
                else:
                    follower_data = ego_data
                    leader_data = other_data
                safe_gap = pp.compute_default_safe_gap(
                    follower_data["v"].to_numpy())
                gap = (leader_data["x"].to_numpy()
                       - follower_data["x"].to_numpy())
                gap_error = gap - safe_gap
                other_name = other_data["name"].iloc[0]
                if np.any(gap_error < 5):
                    ax.plot(ego_data["t"].to_numpy(), gap_error,
                            label=f"{ego_name} to {other_name}")
    ax.legend()
    y_low, y_high = ax.get_ylim()
    final_t = data["t"].max()
    ax.set(xlabel=_get_variable_with_unit("t"),
           ylabel=_get_variable_with_unit("gap_error"),
           xlim=(0, final_t), ylim=(max(-5, y_low), min(5, y_high)))


def plot_single_vehicle_lane_change_gap_errors(
        data: pd.DataFrame, veh_id_or_name: Union[int, str], ax=None):
    """
    Plots the gaps between a vehicle and all the relevant surrounding vehicles:
    leader at the origin lane, leader at the destination lane, and follower at
    the destination lane
    """
    if ax is None:
        fig, ax = plt.subplots()

    lc_vehicle_data = _get_veh_data(data, veh_id_or_name)
    veh_name = lc_vehicle_data["name"].iloc[0]
    ego_safe_gap = pp.compute_default_safe_gap(lc_vehicle_data["v"].to_numpy())

    gap = lc_vehicle_data["gap_to_orig_lane_leader"].to_numpy()
    orig_lane_error = gap - ego_safe_gap
    if np.any(orig_lane_error < 5):
        ax.plot(lc_vehicle_data["t"].to_numpy(), orig_lane_error,
                label=f"{veh_name} to lo")

    gap = lc_vehicle_data["gap_to_dest_lane_leader"].to_numpy()
    dest_lane_error = gap - ego_safe_gap
    if np.any(dest_lane_error < 5):
        ax.plot(lc_vehicle_data["t"].to_numpy(), dest_lane_error,
                label=f"{veh_name} to ld")

    dest_follower_ids = lc_vehicle_data["dest_lane_follower_id"].unique()
    dest_follower_id = [veh_id for veh_id in dest_follower_ids if veh_id >= 0]
    if len(dest_follower_id) > 1:
        print("Hey! Time to deal with multiple dest lane followers")
    if len(dest_follower_id) > 0:
        follower_data = data[data["id"] == dest_follower_id[0]]
        foll_safe_gap = pp.compute_default_safe_gap(
            follower_data["v"].to_numpy())
        gap = lc_vehicle_data["gap_to_dest_lane_follower"].to_numpy()
        dest_lane_error = gap - foll_safe_gap
        if np.any(dest_lane_error < 5):
            ax.plot(lc_vehicle_data["t"].to_numpy(), dest_lane_error,
                    label=f"fd to {veh_name}")

    ax.legend()
    y_low, y_high = ax.get_ylim()
    final_t = data["t"].max()
    ax.set(xlabel=_get_variable_with_unit("t"),
           ylabel=_get_variable_with_unit("gap_error"),
           xlim=(0, final_t), ylim=(max(-5, y_low), min(5, y_high))
           )


def plot_vehicle_following(data: pd.DataFrame):
    """
    Plots the x and vel vs time
    :param data:
    :return:
    """
    sns.set_style("whitegrid")
    pp.compute_all_relative_values(data)
    x_axes = ["t", "t"]
    y_axes = ["gap_to_orig_lane_leader", "v"]
    plot_scenario_results(x_axes, y_axes, data)


def plot_scenario_results(x_axes: list[str], y_axes: list[str],
                          data: pd.DataFrame, axs: list = None):
    """

    :param x_axes: Name of the variable on the x-axis for each plot
    :param y_axes: Name of the variable on the y-axis for each plot
    :param data:
    :param axs:
    :return:
    """
    if axs is None:
        fig, ax = plt.subplots(len(y_axes))
    else:
        fig = None
        ax = axs

    relevant_ids = get_relevant_vehicle_ids(data)
    data_to_plot = data.loc[data["id"].isin(relevant_ids)]

    for i, (x, y) in enumerate(zip(x_axes, y_axes)):
        final_x = data_to_plot[x].max()
        show_legend = True if i == len(x_axes) - 1 else False
        sns.lineplot(data_to_plot, x=x, y=y, hue="name", ax=ax[i],
                     palette="tab10", legend=show_legend)
        if show_legend:
            sns.move_legend(ax[i], "right")
        low, high = ax[i].get_ylim()
        if y == "v" and high - low < 1:
            low, high = np.floor(low - 0.5), np.ceil(high + 0.5)
        ax[i].set(xlabel=_get_variable_with_unit(x),
                  ylabel=_get_variable_with_unit(y),
                  xlim=(0, final_x), ylim=(low, high))

    if axs is None:
        fig.tight_layout()
        fig.show()


def get_relevant_vehicle_ids(data: pd.DataFrame) -> np.ndarray:
    platoon_ids = data[data["name"].str.startswith("p")]["id"].unique()
    ld_id = pd.unique(data.loc[
                          (data["id"].isin(platoon_ids)) & (
                                  data["y"] > 2), "orig_lane_leader_id"
                      ].values.ravel("K"))
    fd_id = pd.unique(data.loc[
                          (data["orig_lane_leader_id"].isin(platoon_ids) & (
                                  data["y"] > 2), "id")
                      ].values.ravel("K"))
    return np.unique(np.concatenate((platoon_ids, ld_id, fd_id)))


def _get_veh_data(data: pd.DataFrame, id_or_name: Union[int, str]
                  ) -> pd.DataFrame:
    if isinstance(id_or_name, str):
        ego_data = data[data["name"] == id_or_name]
    else:
        ego_data = data[data["id"] == id_or_name]
    return ego_data


def _get_variable_with_unit(variable: str):
    try:
        return variable + " [" + configuration.UNIT_MAP[variable] + "]"
    except KeyError:
        return variable


def _get_color_by_name(veh_name: str):
    if veh_name == "p1":
        color = configuration.COLORS["dark_blue"]
    elif veh_name == "ego" or veh_name.startswith("p"):
        color = configuration.COLORS["blue"]
    # elif veh_name in {"lo", "lo0", "ld", "ld0", "fd", "fd0"}:
    #     color = configuration.COLORS["red"]
    elif veh_name.startswith("ld") or veh_name.startswith("fd"):
        color = configuration.COLORS["orange"]
    else:
        color = configuration.COLORS["gray"]
    return color
