from collections.abc import Iterable, Sequence
import os
from typing import Any

import pandas as pd
import numpy as np

import configuration


def order_values(lo_value: Any, platoon_value: Iterable[Any],
                 ld_value: Any, fd_value: Any = None) -> np.ndarray:
    """
    Used to ensure any stacked vectors always places the vehicles in the
    same order
    :param lo_value:
    :param platoon_value:
    :param ld_value:
    :param fd_value:
    :return:
    """
    if fd_value is None:
        fd_value = []
    platoon_value_array = np.array(platoon_value).flatten()
    return np.hstack((lo_value, platoon_value_array, ld_value, fd_value))


def split_state_vector(values: Sequence[Any]) -> dict[str, Any]:
    """
    Splits a sequence of values into a dictionary where keys are vehicle
    names and values are their respective states. Assumes that the sequence is
    ordered as [lo, p1, ..., pN, ld]
    """
    # TODO remove hard coded indices?
    n_states = 4
    n_vehicles = len(values) // n_states
    by_vehicle = {"lo": values[:n_states], "ld": values[-n_states:]}
    for idx in range(n_vehicles - 2):
        by_vehicle["p" + str(idx + 1)] = values[n_states * (idx + 1)
                                                : n_states * (idx + 2)]
    return by_vehicle


def load_queries_from_simulations(
        n_platoon: int, simulator_name: str) -> tuple[pd.DataFrame, str]:
    """
    Reads the file containing the queries encountered during simulations
    and returns the dataframe with the data together with the file path.
    We remove duplicates before returning the dataframe.
    """
    if simulator_name not in {"python", "vissim"}:
        raise ValueError(
            "Invalid value for parameter 'simulator'. Unsolved nodes must "
            "be loaded either from python or vissim simulations.")
    file_name = "_".join([simulator_name + "_queries", str(n_platoon),
                          "vehicles.csv"])
    file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                             "queries", file_name)
    # Load the initial states found during simulation, and keep only
    # unique values in the file
    df: pd.DataFrame = pd.read_csv(file_path, skipinitialspace=True)
    df = df.drop_duplicates(ignore_index=True)
    df.to_csv(file_path, index=False)
    return df, file_path


def load_initial_states_seen_in_simulations(
        n_platoon: int, simulator_name: str) -> tuple[pd.DataFrame, str]:
    """
    Reads the file containing the initial states (at lane change intention
    time) encountered during simulations and returns the dataframe with
    the data together with the file path. We remove duplicates before
    returning the dataframe.
    """
    if simulator_name not in {"python", "vissim"}:
        raise ValueError(
            "Invalid value for parameter 'simulator'. Unsolved nodes must "
            "be loaded either from python or vissim simulations.")
    file_name = "_".join([simulator_name + "_x0", str(n_platoon),
                          "vehicles.csv"])
    file_path = os.path.join(configuration.DATA_FOLDER_PATH,
                             "vehicle_state_graphs", file_name)
    # Load the initial states found during simulation, and keep only
    # unique values in the file
    df: pd.DataFrame = pd.read_csv(file_path)
    df = df.drop_duplicates(ignore_index=True)
    df.to_csv(file_path, index=False)
    return df, file_path


def tuple_of_lists_to_list(input_tuple: tuple[list, ...]) -> list[tuple]:
    ret = []
    for i in range(len(input_tuple[0])):
        ret.append(tuple([input_tuple[j][i] for j in range(len(input_tuple))]))
    return ret
