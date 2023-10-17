from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np

import vehicle_models.base_vehicle as base

# TODO: with the inclusion of all surrounding vehicles in the mode definition
#  this is starting to look too convoluted. We transform the current
#  surrounding vehicles of each vehicle in a mode only to translate if back
#  to sequences of surrounding vehicles afterwards


class SystemMode:
    """
    Each vehicle can have a varying number of relevant surrounding vehicles.
    The combination of all relevant vehicles for all vehicles describes the
    system operation mode.
    """
    def __init__(self, vehicles: Dict[int, base.BaseVehicle]):
        self.mode: Dict[int, Dict[str, int]] = {}
        self.id_to_name_map: Dict[int, str] = {-1: 'x'}  # for easier debugging
        for veh_id, vehicle in vehicles.items():
            # leader_id = vehicle.get_current_leader_id()
            surrounding_veh_ids = vehicle.get_relevant_surrounding_vehicle_ids()
            self.mode[veh_id] = surrounding_veh_ids
            self.id_to_name_map[veh_id] = vehicle.get_name()

    def __eq__(self, other: SystemMode):
        return self.mode == other.mode

    def __str__(self):
        pairs = []
        for ego_id, surrounding_ids in self.mode.items():
            surrounding_names = {name: self.id_to_name_map[veh_id]
                                 for name, veh_id in surrounding_ids.items()
                                 # if veh_id > -1
                                 }
            pairs.append(
                self.id_to_name_map[ego_id] + ': '
                + str(surrounding_names))
        res = ', '.join(pairs)
        return '{' + res + '}'

    def create_altered_mode(
            self, ego_surrounding_map: Dict[str, Dict[str, str]]
    ) -> SystemMode:
        """
        Creates another SystemMode instance similar to the original one
        except for the given follower/leader pair
        :param ego_surrounding_map:
        :return:
        """
        new_mode = copy.deepcopy(self)
        # Get id per name
        name_to_id: Dict[str, int] = {}
        for veh_id, veh_name in self.id_to_name_map.items():
            name_to_id[veh_name] = veh_id
        # Change only the requested pairs
        for ego, surrounding_veh_map in ego_surrounding_map.items():
            ego_id = name_to_id[ego]
            for surrounding_pos, veh_name in surrounding_veh_map.items():
                surrounding_veh_id = name_to_id[veh_name]
                new_mode.mode[ego_id][surrounding_pos] = surrounding_veh_id
        return new_mode


# Aliases for easier typing hints
ModeSequence = List[Tuple[float, SystemMode]]
# Surrounding Vehicle Sequence:
# a sequence of times and the surrounding vehicles at each time
SVSequence = List[Tuple[float, Dict[str, int]]]


def append_mode_to_sequence(input_sequence: ModeSequence, time: float,
                            follower_leader_changes: Dict[str, Dict[str, str]]
                            ) -> None:
    last_mode = input_sequence[-1][1]
    new_mode = last_mode.create_altered_mode(follower_leader_changes)
    input_sequence.append((time, new_mode))


def mode_sequence_to_sv_sequence(mode_sequence: ModeSequence
                                 ) -> Dict[int, SVSequence]:
    """
    Transforms a list of (time, mode) tuples into a dictionary where vehicle
    ids are keys and values are lists of (time, leader id) tuples
    """
    surrounding_vehicle_sequences: Dict[int, SVSequence] = defaultdict(list)
    for time, mode in mode_sequence:
        for ego_id, surrounding_ids in mode.mode.items():
            surrounding_vehicle_sequences[ego_id].append(
                (time, surrounding_ids))
    return surrounding_vehicle_sequences


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
        if len(veh.ocp_mode_switch_times) == 0:
            continue
        print(veh.get_name(), end=": ")
        for t, lead_id in zip(veh.ocp_mode_switch_times,
                              veh.ocp_target_leader_sequence):
            print('(t={}, l={})'.format(
                t, vehicles[lead_id].get_name()
                if lead_id >= 0 else lead_id), end="; ")
        print()
