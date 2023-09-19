from typing import Dict, List, Union

import controllers.optimal_controller as opt_ctrl
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv
import system_operating_mode as som


class BiLevelLCController:
    """
    The bi-level optimal controller follows the steps:
    1. Set an operating mode sequence
    2. Solve the OPC with given mode sequence
    3. Apply optimal input and simulate system
    4. Compare assumed operating mode sequence to obtained mode sequence
    5. Repeat steps 1-4 in case mode sequences don't match
    """

    def __init__(self, ocp_horizon: float):
        self.optimal_controller: opt_ctrl.VehicleOptimalController = (
            opt_ctrl.VehicleOptimalController(ocp_horizon)
        )

    # TODO: change names to not overlap with opt controller methods?
    def find_single_vehicle_trajectory(self,
                                       vehicles: Dict[int, base.BaseVehicle],
                                       ego_veh_id: int
                                       ):
        current_mode = som.SystemMode(vehicles)
        mode_sequence: som.ModeSequence = [(0.0, current_mode)]
        # TODO: guess a mode change?
        self.optimal_controller.find_single_vehicle_trajectory(
            vehicles, ego_veh_id, mode_sequence)

    def get_input(self, time: float, veh_ids: Union[int, List[int]]):
        return self.optimal_controller.get_input(time, veh_ids)
