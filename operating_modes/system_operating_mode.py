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
        # TODO: wouldn't it be easier to work only with vehicle names (str)
        #  instead of moving back and forth between ids and names?
        self.mode: Dict[int, Dict[str, int]] = {}
        self.id_to_name_map: Dict[int, str] = {-1: 'x'}  # for easier debugging
        self.name_to_id: Dict[str, int] = {}  # makes editing/getting easier
        for veh_id, vehicle in vehicles.items():
            surrounding_veh_ids = vehicle.get_relevant_surrounding_vehicle_ids()
            self.mode[veh_id] = surrounding_veh_ids
            veh_name = vehicle.get_name()
            self.id_to_name_map[veh_id] = veh_name
            self.name_to_id[veh_name] = veh_id

    def __eq__(self, other: SystemMode):
        return self.mode == other.mode

    def __str__(self):
        return self.to_string(show_all=False)

    def to_string(self, show_all: bool = False):
        """
        For nice printing out with option to hide non-existing surrounding
        vehicles
        :param show_all: If true, prints out all possible surrounding vehicles
         position with 'x' if there's no vehicle at some position. If false,
         only prints existing vehicles
        :return:
        """
        pairs = []
        for ego_id, surrounding_ids in self.mode.items():

            surrounding_names = {name: self.id_to_name_map[veh_id]
                                 for name, veh_id in surrounding_ids.items()
                                 if (veh_id > -1 or show_all)
                                 }
            pairs.append(
                self.id_to_name_map[ego_id] + ': '
                + str(surrounding_names))
        res = ', '.join(pairs)
        return '{' + res + '}'

    def get_surrounding_vehicle_id(self, ego_identifier: Union[str, int],
                                   surrounding_position: str) -> int:

        return self.mode[self._get_id(ego_identifier)][surrounding_position]

    def get_surrounding_vehicle_name(self, ego_identifier: Union[str, int],
                                     surrounding_position: str) -> str:
        return self.id_to_name_map[self.get_surrounding_vehicle_id(
            ego_identifier, surrounding_position)]

    def create_altered_mode(
            self,
            mode_changes: Dict[Union[str, int], Dict[str, Union[str, int]]]
    ) -> SystemMode:
        """
        Creates another SystemMode instance similar to the original one
        except for the given follower/leader pair
        :param mode_changes:
        :return:
        """
        new_mode = copy.deepcopy(self)
        # Change only the requested pairs
        for ego, surrounding_veh_map in mode_changes.items():
            ego_id = self._get_id(ego)
            for surrounding_pos, other_veh in surrounding_veh_map.items():
                surrounding_veh_id = self._get_id(other_veh)
                new_mode.mode[ego_id][surrounding_pos] = surrounding_veh_id
        return new_mode

    def _get_id(self, identifier: Union[str, int]) -> int:
        """
        Ensures that we get the vehicle id as an int whether the identifier
        is a name or an id
        :param identifier: Name (str) or id (int) of vehicle
        :return:
        """
        if isinstance(identifier, str):
            return self.name_to_id[identifier]
        else:
            return identifier


# Surrounding Vehicle Sequence:
# a sequence of times and the surrounding vehicles at each time
SVSequence = List[Tuple[float, Dict[str, int]]]


class ModeSequence:

    def __init__(self):
        self.sequence: List[Tuple[float, SystemMode]] = []

    def __str__(self):
        return self.to_string(skip_lines=False)

    def to_string(self, skip_lines: bool):
        ret = []
        for t, m in self.sequence:
            ret.append("(" + str(t) + ": " + str(m) + ")")
        sep = "\n" if skip_lines else " "
        return sep.join(ret)

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

    def alter_and_add_mode(
            self, time: float,
            mode_changes: Dict[Union[str, int], Dict[str, Union[str, int]]]):
        """
        Copies the latest mode, change it according to the given parameter,
        and adds it to the sequence
        :param time:
        :param mode_changes:
        :return:
        """
        if len(mode_changes) == 0:
            return
        last_mode = self.get_latest_mode()
        new_mode = last_mode.create_altered_mode(mode_changes)
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
