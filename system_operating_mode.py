from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Union

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
        self.names: Dict[str, str] = {}  # for easier debugging
        for veh_id, vehicle in vehicles.items():
            leader_id = vehicle.get_current_leader_id()
            self.vehicle_pairs[veh_id] = leader_id
            self.names[vehicle.get_name()] = (vehicles[leader_id].get_name()
                                              if leader_id >= 0 else ' ')

    def __eq__(self, other: SystemMode):
        return self.vehicle_pairs == other.vehicle_pairs

    def __str__(self):
        pairs = []
        for foll_name, lead_name in self.names.items():
            pairs.append(foll_name + '->' + lead_name)
        res = ', '.join(pairs)
        return '{' + res + '}'


def mode_sequence_to_leader_sequence(
        mode_sequence: ModeSequence
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


# Alias for easier typing hints
ModeSequence = List[Tuple[float, SystemMode]]

