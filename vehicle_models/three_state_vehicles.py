from __future__ import annotations

from abc import ABC

import numpy as np

import vehicle_models.base_vehicle as base
# import vehicle_ocp_interface as vi


# =========================== Three-State Vehicles =========================== #
# Three-state vehicles are used in initial tests with the optimization tool
# since they are simpler and were used in the tool's example.
class ThreeStateVehicle(base.BaseVehicle, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model()

    def set_ocp_leader_sequence(self, leader_sequence):
        """
        Does nothing because these vehicles do not have optimal controllers
        """
        pass

    def update_mode(self, vehicles: dict[int, base.BaseVehicle]):
        pass

    def get_has_open_loop_acceleration(self) -> bool:
        return False

    def get_optimal_input_history(self) -> np.ndarray:
        return self.get_input_history()

    def get_external_input_idx(self) -> dict[str, int]:
        return self._input_idx

    def get_vel(self) -> float:
        try:
            return self.get_an_input_by_name('v')
        except AttributeError:
            # trying to read vehicle's speed before any input is computed
            return self._free_flow_speed

    def _set_speed(self, v0, state):
        # Does nothing because velocity is an input for this model
        pass

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: dict[int, base.BaseVehicle]):
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

    def update_target_leader(self, vehicles: dict[int, base.BaseVehicle]):
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
        self._ocp_interface = ThreeStateVehicleRearWheelInterface

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_rear_wheels(vel, theta, phi)


class ThreeStateVehicleCG(ThreeStateVehicle):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self):
        super().__init__()
        self._ocp_interface = ThreeStateVehicleCGInterface

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_cg(vel, theta, phi)


class ThreeStateVehicleInterface(base.BaseVehicleInterface, ABC):
    """ States: [x, y, theta], inputs: [v, phi] """
    _state_names = ['x', 'y', 'theta']
    _input_names = ['v', 'phi']

    def __init__(self, vehicle: ThreeStateVehicle):
        super().__init__(vehicle)
        self._set_model()

    def _set_speed(self, v0, state):
        # Does nothing because velocity is an input for this model
        pass

    def get_accel(self, ego_states, inputs, leader_states):
        # Does nothing because velocity is an input for this model
        pass

    def get_desired_input(self) -> np.ndarray:
        return np.array([self.get_free_flow_speed(), 0])

    def get_input_limits(self) -> (list[float], list[float]):
        return ([0, -self.get_phi_max()],
                [self.get_free_flow_speed() + 5, self.get_phi_max()])


class ThreeStateVehicleRearWheelInterface(ThreeStateVehicleInterface):
    """ From the library's example.
    States: [x, y, theta], inputs: [v, phi], centered at the rear wheels """

    def __init__(self, vehicle: ThreeStateVehicleRearWheel):
        super().__init__(vehicle)

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self._position_derivative_rear_wheels(vel, theta, phi, derivatives)


class ThreeStateVehicleCGInterface(ThreeStateVehicleInterface):
    """ States: [x, y, theta], inputs: [v, phi], centered at the C.G. """

    def __init__(self, vehicle: ThreeStateVehicleRearWheel):
        super().__init__(vehicle)

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self._position_derivative_cg(vel, theta, phi, derivatives)
