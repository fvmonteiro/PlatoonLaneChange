from typing import Dict

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
        self.id: Dict[int, int] = {}
        self.names: Dict[str, str] = {}  # same as id but with vehicle names
        for veh_id, vehicle in vehicles.items():
            leader_id = vehicle.get_current_leader_id()
            self.id[veh_id] = leader_id
            self.names[vehicle.name] = (vehicles[leader_id].name
                                        if leader_id >= 0 else ' ')

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        pairs = []
        for foll_name, lead_name in self.names.items():
            pairs.append(foll_name + '->' + lead_name)
        res = ', '.join(pairs)
        return '{' + res + '}'
