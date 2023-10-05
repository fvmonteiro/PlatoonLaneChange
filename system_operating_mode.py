from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np

import vehicle_models.base_vehicle as base


class SystemMode:
    """
    The system operating mode is described by all follower/leader pairs, where
    leader is the vehicle used by the follower to determine its own acceleration.
    The leader might be:
    - the origin lane leader (for all vehicles)
    - the destination lane leader (for lane changing vehicles)
    - the vehicle moving into our lane (for cooperating vehicles)
    """
    def __init__(self, vehicles: Dict[int, base.BaseVehicle]):
        self.vehicle_pairs: Dict[int, int] = {}
        self.id_to_name_map: Dict[int, str] = {}  # for easier debugging
        # self.names: Dict[str, str] = {}  # for easier debugging
        for veh_id, vehicle in vehicles.items():
            leader_id = vehicle.get_current_leader_id()
            self.vehicle_pairs[veh_id] = leader_id
            self.id_to_name_map[veh_id] = vehicle.get_name()
            # self.names[vehicle.get_name()] = (vehicles[leader_id].get_name()
            #                                   if leader_id >= 0 else ' ')

    def __eq__(self, other: SystemMode):
        return self.vehicle_pairs == other.vehicle_pairs

    def __str__(self):
        pairs = []
        for foll_id, lead_id in self.vehicle_pairs.items():
            pairs.append(self.id_to_name_map[foll_id] + '->'
                         + (self.id_to_name_map[lead_id] if lead_id > -1
                         else ' '))
        # pairs = []
        # for foll_name, lead_name in self.names.items():
        #     pairs.append(foll_name + '->' + lead_name)
        res = ', '.join(pairs)
        return '{' + res + '}'

    def create_altered_mode(
            self, follower_leader_pairs: List[Tuple[str, Union[str, None]]]
    ) -> SystemMode:
        """
        Creates another SystemMode instance similar to the original one
        except for the given follower/leader pair
        :param follower_leader_pairs:
        :return:
        """
        new_mode = copy.deepcopy(self)
        name_to_id: Dict[str, int] = {}
        for veh_id, veh_name in self.id_to_name_map.items():
            name_to_id[veh_name] = veh_id
        for follower, leader in follower_leader_pairs:
            foll_id = name_to_id[follower]
            lead_id = name_to_id[leader] if leader is not None else -1
            new_mode.vehicle_pairs[foll_id] = lead_id
        return new_mode


# Alias for easier typing hints
ModeSequence = List[Tuple[float, SystemMode]]


def append_mode_to_sequence(input_sequence: ModeSequence, time: float,
                            follower_leader_changes: List[Tuple[str, str]]
                            ) -> None:
    last_mode = input_sequence[-1][1]
    new_mode = last_mode.create_altered_mode(follower_leader_changes)
    input_sequence.append((time, new_mode))


def mode_sequence_to_leader_sequence(mode_sequence: ModeSequence
                                     ) -> Dict[int, List[Tuple[float, int]]]:
    """
    Transforms a list of (time, mode) tuples into a dictionary where vehicle
    ids are keys and values are lists of (time, leader id) tuples
    """
    leader_sequence: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for time, mode in mode_sequence:
        for foll_id, lead_id in mode.vehicle_pairs.items():
            leader_sequence[foll_id].append((time, lead_id))
    return leader_sequence


def mode_sequence_to_str(s: ModeSequence) -> str:
    ret = []
    for t, m in s:
        ret.append("(" + str(t) + ": " + str(m) + ")")
    return " ".join(ret)


def compare_mode_sequences(s1: ModeSequence, s2: ModeSequence, dt: float
                           ) -> bool:
    """

    :param s1: First mode sequence
    :param s2: Second mode sequence
    :param dt: The time tolerance for mode switches in each sequence to be
     considered the same.
    :return:
    """
    if len(s1) != len(s2):
        return False
    for i in range(len(s1)):
        t1, mode1 = s1[i]
        t2, mode2 = s2[i]
        if not (np.abs(t1 - t2) <= dt and mode1 == mode2):
            return False
    return True


def print_vehicles_leader_sequences(
        vehicles: Dict[int, base.BaseVehicleInterface]):
    print("OCP leader sequences:")
    for veh in vehicles.values():
        if len(veh.ocp_leader_switch_times) == 0:
            continue
        print(veh.get_name(), end=": ")
        for t, lead_id in zip(veh.ocp_leader_switch_times,
                              veh.ocp_leader_sequence):
            print('(t={}, l={})'.format(
                t, vehicles[lead_id].get_name()
                if lead_id >= 0 else lead_id), end="; ")
        print()
