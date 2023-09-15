from abc import ABC
from typing import Dict

import numpy as np

import constants as const
import controllers.longitudinal_controller as long_ctrl
import controllers.lateral_controller as lat_ctrl
import controllers.optimal_controller as opt_ctrl
import platoon
import vehicle_models.base_vehicle as base
import vehicle_operating_modes.concrete_vehicle_modes as modes
from vehicle_models.base_vehicle import BaseVehicle


class FourStateVehicle(base.BaseVehicle, ABC):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    """

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)
        self.brake_max = -4
        self.accel_max = 2

        # Controller
        self.lk_controller: lat_ctrl.LaneKeepingController = (
            lat_ctrl.LaneKeepingController(self)
        )
        self.long_controller: long_ctrl.LongitudinalController = (
            long_ctrl.LongitudinalController(self)
        )

    def get_vel(self):
        return self.get_a_state_by_name('v')

    def compute_gap_to_a_leader(self, a_leader: BaseVehicle):
        return BaseVehicle.compute_a_gap(a_leader, self)

    def _set_speed(self, v0, state):
        state[self._state_idx['v']] = v0

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_cg(vel, theta, phi)
        self._derivatives[self._state_idx['v']] = self.get_an_input_by_name('a')


class OpenLoopVehicle(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Does not compute any inputs internally. This class and its derivatives
    are useful for testing inputs computed by a centralized controller """

    def __init__(self):
        super().__init__()

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]):
        pass

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = open_loop_controls[
            self._input_idx['a']]
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


class SafeAccelOpenLoopLCVehicle(OpenLoopVehicle):
    """ Safe acceleration (internally computed) and externally determined phi.
    States: [x, y, theta, v], external input: [phi], centered at the C.G.
    and accel is computed by a feedback law"""

    def __init__(self):
        super().__init__()

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Array with phi value
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self.long_controller.compute_acceleration(vehicles))
        self._inputs[self._input_idx['phi']] = open_loop_controls[0]

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        self._leader_id.append(
            self.long_controller.get_more_critical_leader(vehicles))


class OptimalControlVehicle(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Accel and phi computed by optimal control when there is lane change
    intention. Otherwise, zero accel and lane keeping phi. """

    _solver_wait_time = 20.0  # [s] time between attempts to solve an ocp
    _n_optimal_inputs = 2

    def __init__(self):
        super().__init__()
        self.set_mode(modes.OCPLaneKeepingMode())

        self._n_feedback_inputs = self._n_inputs - self._n_optimal_inputs
        self._ocp_has_solution = False
        self._ocp_horizon = 10.0  # [s]
        self._solver_attempt_time = -np.inf

        self.opt_controller = opt_ctrl.VehicleOptimalController(
            self._ocp_horizon)

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]):
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

    def can_start_lane_change(self, vehicles: Dict[int, base.BaseVehicle]
                              ) -> bool:
        t = self.get_current_time()

        # The OPC solver should be run again every time the vehicle starts
        # following a new leader or when the dest lane foll starts cooperating
        if self.has_requested_cooperation():
            cooperating_vehicle = vehicles[
                self.get_desired_future_follower_id()]
            has_coop_just_started = (
                    cooperating_vehicle.get_current_leader_id() == self.id
                    and cooperating_vehicle.has_changed_leader())
        else:
            has_coop_just_started = False
        has_vehicle_configuration_changed = (
                self.has_changed_leader() or has_coop_just_started
        )
        # If the OPC solver didn't find a solution at first, we do not want to
        # run it again too soon.
        is_cool_down_period_done = (
                t - self._solver_attempt_time
                >= OptimalControlVehicle._solver_wait_time
        )
        if has_vehicle_configuration_changed or is_cool_down_period_done:
            self._solver_attempt_time = t
            print("t={:.2f}, veh:{}. Calling ocp solver...".format(t, self.id))
            self.opt_controller.find_lane_change_trajectory(t, vehicles,
                                                            [self.id])
        return True  # TODO: self.opt_controller.has_solution()

    def is_lane_changing(self):
        delta_t = self.get_current_time() - self._lc_start_time
        return delta_t <= self._ocp_horizon

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        if self.is_lane_changing():
            self._inputs = self.opt_controller.get_input(
                self.get_current_time(), [self.id])
        else:
            self._inputs[self._input_idx['a']] = 0.0
            self._inputs[self._input_idx['phi']] = (
                self.lk_controller.compute_steering_wheel_angle())

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        """
        Does nothing, since this vehicle class does not have autonomous
        longitudinal control
        """
        pass

    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, base.BaseVehicle]) -> None:
        self.update_target_y()

    def _set_up_lane_change_control(self):
        pass


class SafeAccelOptimalLCVehicle(OptimalControlVehicle):
    """ Safe acceleration (internally computed) and optimal control computed
    lane changes.
    States: [x, y, theta, v], external input: [phi], centered at the C.G.
    and accel is computed by a feedback law"""

    _n_optimal_inputs = 1

    def __init__(self):
        super().__init__()

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop control and phi (steering wheel angle) and
        computes the acceleration
        :param open_loop_controls: Dictionary containing 'phi' value
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self.long_controller.compute_acceleration(vehicles))
        if self.is_lane_changing():
            t = self.get_current_time()
            phi = self.opt_controller.get_input(t, self.id)[0]
        else:
            phi = self.lk_controller.compute_steering_wheel_angle()
        self._inputs[self._input_idx['phi']] = phi

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        self._leader_id.append(
            self.long_controller.get_more_critical_leader(vehicles))


class ClosedLoopVehicle(FourStateVehicle):
    """ Vehicle that computes all of its inputs by feedback laws.
     States: [x, y, theta, v], external input: None, centered at the C.G. """

    # delete after figuring out a better class organization
    _n_optimal_inputs = 0

    def __init__(self):
        super().__init__()
        self.set_mode(modes.CLLaneKeepingMode())
        self.lc_controller = lat_ctrl.LaneChangingController(self)

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]):
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

    def is_lane_change_safe(self, vehicles: Dict[int, base.BaseVehicle]):
        is_safe_to_orig_lane_leader = True
        if self.has_orig_lane_leader():
            orig_lane_leader = vehicles[self._orig_leader_id[-1]]
            is_safe_to_orig_lane_leader = (
                ClosedLoopVehicle.is_gap_safe_for_lane_change(
                    orig_lane_leader, self))

        is_safe_to_dest_lane_leader = True
        if self.has_dest_lane_leader():
            dest_lane_leader = vehicles[self._destination_leader_id[-1]]
            is_safe_to_dest_lane_leader = (
                ClosedLoopVehicle.is_gap_safe_for_lane_change(
                    dest_lane_leader, self))

        is_safe_to_dest_lane_follower = True
        if self.has_dest_lane_follower():
            dest_lane_follower = vehicles[self._destination_follower_id[-1]]
            is_safe_to_dest_lane_follower = (
                ClosedLoopVehicle.is_gap_safe_for_lane_change(
                    self, dest_lane_follower))

        return (is_safe_to_orig_lane_leader
                and is_safe_to_dest_lane_leader
                and is_safe_to_dest_lane_follower)

    def is_lane_change_complete(self):
        return (np.abs(self.get_y() - self.target_y) < 1e-2
                and np.abs(self.get_theta()) < 1e-3)

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Computes the acceleration and phi (steering wheel angle)
        :param open_loop_controls: irrelevant
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self.long_controller.compute_acceleration(vehicles))
        delta_t = self.get_current_time() - self._lc_start_time
        if delta_t <= self._lc_duration:
            phi = self.lc_controller.compute_steering_wheel_angle()
        else:
            phi = self.lk_controller.compute_steering_wheel_angle()

        self._inputs[self._input_idx['phi']] = phi

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        self._leader_id.append(
            self.long_controller.get_more_critical_leader(vehicles))

    def _set_up_longitudinal_adjustments_control(
            self, vehicles: Dict[int, base.BaseVehicle]):
        pass

    def _set_up_lane_change_control(self):
        self.update_target_y()
        self.lc_controller.start(self._lc_start_time,
                                 self._lc_duration)

    @staticmethod
    def is_gap_safe_for_lane_change(leading_vehicle: BaseVehicle,
                                    following_vehicle: BaseVehicle):
        margin = 1e-2
        gap = BaseVehicle.compute_a_gap(leading_vehicle, following_vehicle)
        safe_gap = following_vehicle.compute_safe_gap(
            following_vehicle.get_vel())
        return gap + margin >= safe_gap


class PlatoonVehicle(SafeAccelOpenLoopLCVehicle):
    """
    Vehicles belonging to a platoon. Each vehicle computes its own acceleration
    but the lane change control is defined by the platoon.
    Note: not yet sure who this class should inherit from
    """

    _platoon: platoon.Platoon

    def __init__(self):
        super().__init__()

    def set_platoon(self, new_platoon: platoon.Platoon):
        self._platoon = new_platoon

    def update_mode(self, vehicles: Dict[int, BaseVehicle]):
        pass  # TODO

    def can_start_lane_change(self, vehicles: Dict[int, base.BaseVehicle]
                              ) -> bool:
        t = self.get_current_time()
        if self._is_platoon_leader():
            self._platoon.compute_lane_change_trajectory(t, vehicles)
        return self._platoon.trajectory_exists

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self.long_controller.compute_acceleration(vehicles))

        t = self.get_current_time()
        if self._platoon.is_lane_changing(t):
            if self._is_platoon_leader():
                self._platoon.retrieve_all_inputs(t)
            phi = self._platoon.get_input_for_vehicle(self.id)
        else:
            phi = self.lk_controller.compute_steering_wheel_angle()
        self._inputs[self._input_idx['phi']] = phi

    def _set_up_lane_change_control(self):
        self._platoon.set_lc_start_time(self._lc_start_time)

    def _is_platoon_leader(self):
        return self.id == self._platoon.get_platoon_leader_id()

    # ========================= Not in use =================================== #
    # Methods if we want platoon vehicles to manage platoons.
    # [Aug 23] For now, it is easier to create platoons in the scenarios
    def analyze_platoons(self, vehicles: Dict[int, base.BaseVehicle],
                         platoons: Dict[int, platoon.Platoon],
                         platoon_lc_strategy=None):

        # [Aug 23] We are only simulating simple scenarios. At the start of the
        # simulation, every vehicle will either create its own platoon
        # or join the platoon of the vehicle ahead. Vehicles do not leave or
        # join platoons afterward

        if not self.is_in_a_platoon():
            if (self.has_orig_lane_leader()
                    and vehicles[
                        self.get_orig_lane_leader_id()].is_in_a_platoon()):
                leader_platoon_id = vehicles[
                    self.get_orig_lane_leader_id()].get_platoon_id()
                platoons[leader_platoon_id].add_vehicle(self.id)
            else:
                self._create_platoon(platoons)

    def _create_platoon(self, platoons: Dict[int, platoon.Platoon]):
        new_platoon = platoon.Platoon(self.id)
        platoons[new_platoon.id] = new_platoon
