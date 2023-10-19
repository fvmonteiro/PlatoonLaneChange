from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np

import vehicle_models.base_vehicle as base


# TODO: with the inclusion of all surrounding vehicles in the mode definition
#  this is starting to look too convoluted. We transform the current
#  surrounding vehicles of each vehicle in a mode only to translate it back
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


# Surrounding Vehicle Sequence:
# a sequence of times and the surrounding vehicles at each time
SVSequence = List[Tuple[float, Dict[str, int]]]


class ModeSequence:

    def __init__(self):
        self.sequence: List[Tuple[float, SystemMode]] = []

    def __str__(self):
        ret = []
        for t, m in self.sequence:
            ret.append("(" + str(t) + ": " + str(m) + ")")
        return " ".join(ret)

    def is_equal_to(self, other: ModeSequence, dt: float) -> bool:
        """

        :param other: Mode sequence being compared with self
        :param dt: The time tolerance for mode switches in each sequence to be
         considered the same.
        :return:
        """
        s1 = self.sequence
        s2 = other.sequence
        if len(s1) != len(s2):
            return False
        for i in range(len(s1)):
            t1, mode1 = s1[i]
            t2, mode2 = s2[i]
            if not (np.abs(t1 - t2) <= dt and mode1 == mode2):
                return False
        return True

    def is_empty(self):
        return len(self.sequence) == 0

    def get_latest_switch_time(self):
        return self.sequence[-1][0]

    def get_latest_mode(self) -> SystemMode:
        return self.sequence[-1][1]

    def add_mode(self, time: float, mode: SystemMode):
        self.sequence.append((time, mode))

    def alter_and_add_mode(self, time: float,
                           follower_leader_changes: Dict[str, Dict[str, str]]):
        """
        Copies the latest mode, change it according to the given parameter,
        and adds it to the sequence
        :param time:
        :param follower_leader_changes:
        :return:
        """
        last_mode = self.get_latest_mode()
        new_mode = last_mode.create_altered_mode(follower_leader_changes)
        self.add_mode(time, new_mode)

    def to_sv_sequence(self) -> Dict[int, SVSequence]:
        """
        Transforms a list of (time, mode) tuples into a dictionary where vehicle
        ids are keys and values are lists of (time, leader id) tuples
        """
        surrounding_vehicle_sequences: Dict[int, SVSequence] = defaultdict(list)
        for time, mode in self.sequence:
            for ego_id, surrounding_ids in mode.mode.items():
                surrounding_vehicle_sequences[ego_id].append(
                    (time, surrounding_ids))
        return surrounding_vehicle_sequences


def create_synchronous_lane_change_mode_sequence(mode_sequence: ModeSequence,
                                                 platoon_size: int):
    latest_mode = mode_sequence.get_latest_mode()
    time = mode_sequence.get_latest_switch_time()
    switch_time = time + 1.0  # TODO: no idea what value to add here
    pN = 'p' + str(platoon_size)
    changes = {
        'p1': {'lo': 'ld1'},
        'fd1': {'lo': pN, 'leader': pN}
        }
    mode_sequence.alter_and_add_mode(switch_time, changes)
