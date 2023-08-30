from abc import ABC
from typing import Dict

import numpy as np

import vehicle_models.base_vehicle as base


# =========================== Three-State Vehicles =========================== #
# Three-state vehicles are used in initial tests with the optimization tool
# since they are simpler and were used in the tool's example.
class ThreeStateVehicle(base.BaseVehicle, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]):
        pass

    def get_vel(self):
        try:
            return self.get_an_input_by_name('v')
        except AttributeError:
            # trying to read vehicle's speed before any input is computed
            return self.free_flow_speed

    def _set_speed(self, v0, state):
        # Does nothing because velocity is an input for this model
        pass

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop controls v (velocity) and phi (steering wheel
        :param open_loop_controls: Dictionary containing 'v' and 'phi' values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['v']] = open_loop_controls[
            self._input_idx['v']]
        self._inputs[self._input_idx['phi']] = open_loop_controls[
            self._input_idx['phi']]

    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, base.BaseVehicle]):
        pass

    def _set_up_lane_change_control(self):
        pass

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        """
        Does nothing, since this vehicle class does not have autonomous
        longitudinal control
        """
        pass


class ThreeStateVehicleRearWheel(ThreeStateVehicle):
    """ From the library's example.
    States: [x, y, theta], inputs: [v, phi], centered at the rear wheels """

    def __init__(self):
        super().__init__()

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_rear_wheels(vel, theta, phi)


class ThreeStateVehicleCG(ThreeStateVehicle):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self):
        super().__init__()

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_cg(vel, theta, phi)