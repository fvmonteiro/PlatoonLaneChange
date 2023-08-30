from typing import Dict, List


class Platoon:
    _counter = 0

    vehicle_ids: List[int]

    def __init__(self, platoon_leader_id: int):
        self.id = Platoon._counter
        Platoon._counter += 1
        self.vehicle_ids.append(platoon_leader_id)

    def get_platoon_leader_id(self):
        return self.vehicle_ids[0]

    def get_platoon_last_vehicle_id(self):
        return self.vehicle_ids[-1]

    def add_vehicle(self, veh_id):
        self.vehicle_ids.append(veh_id)
