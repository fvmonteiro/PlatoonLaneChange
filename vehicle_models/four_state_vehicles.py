from __future__ import annotations

from abc import ABC
import copy
from typing import Dict, List, TypeVar, Union, Iterable

import numpy as np

import controllers.longitudinal_controller as long_ctrl
import controllers.lateral_controller as lat_ctrl
import controllers.optimal_controller as opt_ctrl
import platoon
import vehicle_models.base_vehicle as base
import operating_modes.concrete_vehicle_modes as modes


class FourStateVehicle(base.BaseVehicle, ABC):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    """

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']
    _platoon: platoon.Platoon

    static_attribute_names = base.BaseVehicle.static_attribute_names.union(
        {'brake_max', 'accel_max'}
    )

    def __init__(self):
        super().__init__()
        self._set_model(self._state_names, self._input_names)

        # Controllers
        self.lk_controller: lat_ctrl.LaneKeepingController = (
            lat_ctrl.LaneKeepingController(self)
        )
        self.long_controller: long_ctrl.LongitudinalController = (
            long_ctrl.LongitudinalController(self)
        )

    def get_vel(self):
        return self.get_a_state_by_name('v')

    def get_platoon(self) -> Union[None, platoon.Platoon]:
        try:
            return self._platoon
        except AttributeError:
            return None

    def get_desired_dest_lane_leader_id(self) -> int:
        if not self.is_in_a_platoon():
            return self.get_dest_lane_leader_id()
        return self.get_platoon().get_dest_lane_leader_id(self.get_id())

    def find_cooperation_requests(self, vehicles: Iterable[base.BaseVehicle]
                                  ) -> None:
        # We assume platoon vehicles do not cooperate with other vehicles
        if not self.is_in_a_platoon():
            super().find_cooperation_requests(vehicles)
        else:
            incoming_vehicle_id = self.get_platoon().get_incoming_vehicle_id(
                self.get_id())
            self._incoming_vehicle_id[self._iter_counter] = incoming_vehicle_id

    def compute_gap_to_a_leader(self, a_leader: base.BaseVehicle):
        return base.BaseVehicle.compute_a_gap(a_leader, self)

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
        # self._ocp_controllable_interface = OpenLoopVehicleInterface
        # self._ocp_non_controllable_interface = ClosedLoopVehicleInterface

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

    def _write_optimal_inputs(self, optimal_inputs):
        self._inputs = optimal_inputs

    # def _set_up_longitudinal_adjustments_control(
    #         self, vehicles: Dict[int, base.BaseVehicle]):
    #     pass

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
        # self._ocp_controllable_interface = SafeAccelVehicleInterface
        # self._ocp_non_controllable_interface = ClosedLoopVehicleInterface

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Array with phi value
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self.long_controller.compute_acceleration(vehicles))
        self._inputs[self._input_idx['phi']] = open_loop_controls[0]

    def _write_optimal_inputs(self, optimal_phi):
        self._inputs[self._input_idx['phi']] = optimal_phi[0]

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        self._leader_id[self._iter_counter] = (
            self.long_controller.get_more_critical_leader(vehicles))


class OpenLoopNoLCVehicle(OpenLoopVehicle):

    def __init__(self):
        super().__init__()
        # self._ocp_controllable_interface = OpenLoopNoLCVehicleInterface

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        self._inputs[self._input_idx['a']] = open_loop_controls[0]
        self._inputs[self._input_idx['phi']] = 0.

    def _write_optimal_inputs(self, optimal_accel):
        self._inputs[self._input_idx['a']] = optimal_accel[0]


class OptimalControlVehicle(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Accel and phi computed by optimal control when there is lane change
    intention. Otherwise, constant time headway policy for accel and
    a CBF computed lane keeping phi. """

    _solver_wait_time = 20.0  # [s] time between attempts to solve an ocp
    _n_optimal_inputs = 2

    def __init__(self):
        super().__init__()
        self._ocp_controllable_interface = OpenLoopVehicleInterface
        self._ocp_non_controllable_interface = ClosedLoopVehicleInterface
        self.set_mode(modes.OCPLaneKeepingMode())

        # self._n_feedback_inputs = self._n_inputs - self._n_optimal_inputs
        # self._ocp_has_solution = False
        # self._ocp_initial_time = -np.inf

        self.opt_controller: opt_ctrl.VehicleOptimalController = (
            opt_ctrl.VehicleOptimalController()
        )
        self.opt_controller.set_controlled_vehicles_ids(self.get_id())

    def set_centralized_controller(
            self, centralized_controller: opt_ctrl.VehicleOptimalController):
        centralized_controller.add_controlled_vehicle_id(self.get_id())
        self.opt_controller = centralized_controller

    # def reset_ocp_initial_time(self):
    #     self._ocp_initial_time = 0.0

    def make_open_loop_copy(self, initial_state=None):
        return self.make_reset_copy(initial_state, OpenLoopVehicle)

    def get_intermediate_steps_data(self):
        return self.opt_controller.get_data_per_iteration()

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]):
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

    def can_start_lane_change(self, vehicles: Dict[int, base.BaseVehicle]
                              ) -> bool:
        if self.opt_controller.has_solution():
            return True

        # The OPC solver should be run again every time the vehicle starts
        # following a new leader or when the dest lane foll starts cooperating
        if self.has_requested_cooperation():
            cooperating_vehicle = vehicles[
                self.get_desired_future_follower_id()]
            has_coop_just_started = (
                    cooperating_vehicle.get_current_leader_id() == self._id
                    and cooperating_vehicle.has_changed_leader())
        else:
            has_coop_just_started = False
        has_vehicle_configuration_changed = (
                self.has_changed_leader() or has_coop_just_started
        )
        # If the OPC solver didn't find a solution at first, we do not want to
        # run it again too soon.
        t = self.get_current_time()
        is_cool_down_period_done = (
                t - self.opt_controller.get_activation_time()
                >= OptimalControlVehicle._solver_wait_time
        )
        if has_vehicle_configuration_changed or is_cool_down_period_done:
            # self._ocp_initial_time = t
            if self._is_verbose:
                print("t={:.2f}, veh:{}. Calling ocp solver...".format(
                    t, self._id))
            self.opt_controller.find_trajectory(vehicles)
        return True  # TODO self.opt_controller.has_solution()

    def is_lane_changing(self) -> bool:
        delta_t = self.get_current_time() - self._lc_start_time
        return delta_t <= self.opt_controller.get_time_horizon()

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        t = self.get_current_time()
        if self.opt_controller.is_active(t):
            self._inputs = self.opt_controller.get_input(t, [self._id])
        else:
            accel = self.long_controller.compute_acceleration(vehicles)
            phi = self.lk_controller.compute_steering_wheel_angle()
            self._inputs[self._input_idx['a']] = accel
            self._inputs[self._input_idx['phi']] = phi

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        """
        Does nothing, since this vehicle class does not have autonomous
        longitudinal control
        """
        pass

    # def _set_up_longitudinal_adjustments_control(
    #         self, vehicles: Dict[int, base.BaseVehicle]) -> None:
    #     pass

    def _set_up_lane_change_control(self):
        pass

    def _write_optimal_inputs(self, optimal_inputs):
        self._inputs = optimal_inputs


class SafeAccelOptimalLCVehicle(OptimalControlVehicle):
    """ Safe acceleration (internally computed) and optimal control computed
    lane changes.
    States: [x, y, theta, v], external input: [phi], centered at the C.G.
    and accel is computed by a feedback law"""

    _n_optimal_inputs = 1

    def __init__(self):
        super().__init__()
        self._ocp_controllable_interface = SafeAccelVehicleInterface

    def make_open_loop_copy(self, initial_state=None):
        return self.make_reset_copy(initial_state, SafeAccelOpenLoopLCVehicle)

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]):
        """
        Sets the open loop control and phi (steering wheel angle) and
        computes the acceleration
        :param open_loop_controls: Irrelevant in this derived class
         implementation
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        self._inputs[self._input_idx['a']] = (
            self.long_controller.compute_acceleration(vehicles))
        t = self.get_current_time()
        if self.opt_controller.is_active(t):
            phi = self.opt_controller.get_input(t, self._id)[0]
        else:
            phi = self.lk_controller.compute_steering_wheel_angle()
        self._inputs[self._input_idx['phi']] = phi

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]):
        self._leader_id[self._iter_counter] = (
            self.long_controller.get_more_critical_leader(vehicles))

    def _write_optimal_inputs(self, optimal_phi):
        self._inputs[self._input_idx['phi']] = optimal_phi[0]


class OptimalCoopNoLCVehicle(OptimalControlVehicle):
    """
    In scenarios with a centralized controller, this vehicle type can be used
    to represent surrounding cooperating vehicles.
    States: [x, y, theta, v], optimal input: [a], centered at the C.G.
    """

    _n_optimal_inputs = 1

    def __init__(self):
        super().__init__()
        self._ocp_controllable_interface = OpenLoopNoLCVehicleInterface
        self._ocp_non_controllable_interface = SafeLongitudinalVehicleInterface
        # self.set_mode(modes.CLLaneKeepingMode())  # TODO

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]):
        pass  # TODO: modes could be cooperating or not

    def make_open_loop_copy(self, initial_state=None):
        return self.make_reset_copy(initial_state, OpenLoopNoLCVehicle)

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_longitudinal_only(vel)
        self._derivatives[self._state_idx['v']] = self.get_an_input_by_name('a')

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]) -> None:
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        t = self.get_current_time()
        if self.opt_controller.is_active(t):
            optimal_inputs = self.opt_controller.get_input(t, self.get_id())
            accel = optimal_inputs[self._input_idx['a']]
        else:
            accel = self.long_controller.compute_acceleration(vehicles)

        self._inputs[self._input_idx['a']] = accel

    def _set_up_lane_change_control(self):
        pass


class ClosedLoopVehicle(FourStateVehicle):
    """ Vehicle that computes all of its inputs by feedback laws.
     States: [x, y, theta, v], external input: None, centered at the C.G. """

    def __init__(self):
        super().__init__()
        self._ocp_non_controllable_interface = ClosedLoopVehicleInterface
        # self.long_controller = long_ctrl.LongitudinalController(self)
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
            orig_lane_leader = vehicles[self.get_orig_lane_leader_id()]
            is_safe_to_orig_lane_leader = (
                ClosedLoopVehicle.is_gap_safe_for_lane_change(
                    orig_lane_leader, self))

        is_safe_to_dest_lane_leader = True
        if self.has_dest_lane_leader():
            dest_lane_leader = vehicles[self.get_dest_lane_leader_id()]
            is_safe_to_dest_lane_leader = (
                ClosedLoopVehicle.is_gap_safe_for_lane_change(
                    dest_lane_leader, self))

        is_safe_to_dest_lane_follower = True
        if self.has_dest_lane_follower():
            dest_lane_follower = vehicles[self.get_dest_lane_follower_id()]
            is_safe_to_dest_lane_follower = (
                ClosedLoopVehicle.is_gap_safe_for_lane_change(
                    self, dest_lane_follower))

        return (is_safe_to_orig_lane_leader
                and is_safe_to_dest_lane_leader
                and is_safe_to_dest_lane_follower)

    def is_lane_change_complete(self):
        return (np.abs(self.get_y() - self.get_target_y()) < 1e-2
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
        self._leader_id[self._iter_counter] = (
            self.long_controller.get_more_critical_leader(vehicles))

    # def _set_up_longitudinal_adjustments_control(
    #         self, vehicles: Dict[int, base.BaseVehicle]):
    #     pass

    def _set_up_lane_change_control(self):
        self.lc_controller.start(self._lc_start_time, self._lc_duration)

    @staticmethod
    def is_gap_safe_for_lane_change(leading_vehicle: base.BaseVehicle,
                                    following_vehicle: base.BaseVehicle):
        margin = 1e-2
        gap = base.BaseVehicle.compute_a_gap(leading_vehicle, following_vehicle)
        safe_gap = following_vehicle.compute_safe_lane_change_gap(
            following_vehicle.get_vel())
        return gap + margin >= safe_gap


class SafeLongitudinalVehicle(ClosedLoopVehicle):
    def __init__(self):
        super().__init__()
        self._ocp_non_controllable_interface = SafeLongitudinalVehicleInterface
        # self.set_mode(modes.CLLaneKeepingMode())  # TODO

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]):
        pass  # TODO: modes could be cooperating or not

    def _compute_derivatives(self, vel, theta, phi):
        self._position_derivative_longitudinal_only(vel)
        self._derivatives[self._state_idx['v']] = self.get_an_input_by_name('a')

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

    def _set_up_lane_change_control(self):
        pass


class PlatoonVehicle(OptimalControlVehicle):
    """
    Vehicle belonging to a platoon. The platoon vehicle has longitudinal
    and lateral feedback controllers while it has no lane change intention.
    Once it has lane change intention, we can decide whether acceleration
    is computed by an optimal controller. For now, the lane-changing steering
    wheel angle is always computed by an optimal controller

    """

    _is_acceleration_optimal: bool

    static_attribute_names = FourStateVehicle.static_attribute_names.union(
        {'_is_acceleration_optimal'}
    )

    def __init__(self):
        super().__init__()
        self.set_mode(modes.PlatoonVehicleLaneKeepingMode())

    def make_open_loop_copy(self, initial_state=None) -> base.V:
        if self._is_acceleration_optimal:
            return self.make_reset_copy(initial_state, OpenLoopVehicle)
        else:
            new_vehicle = self.make_reset_copy(initial_state,
                                               SafeAccelOpenLoopLCVehicle)
            new_vehicle._platoon = self.get_platoon()
            return new_vehicle

    # def get_possible_target_leader_ids(self) -> List[int]:
    #     ids = super().get_possible_target_leader_ids()
    #     if self.is_in_a_platoon():
    #         ids.append(self._platoon.get_preceding_vehicle_id(self.get_id()))
    #     return ids

    # def get_platoon(self) -> Union[None, platoon.Platoon]:
    #     try:
    #         return self._platoon
    #     except AttributeError:
    #         return None

    def set_acceleration_controller_type(
            self, is_acceleration_optimal: bool = False) -> None:
        """
        Must be called after instantiation of an object (but we don't want to
        add a parameter to the constructor)
        :param is_acceleration_optimal:
        :return:
        """
        if is_acceleration_optimal:
            self._n_optimal_inputs = 2
            self._ocp_controllable_interface = OpenLoopVehicleInterface
        else:
            self._n_optimal_inputs = 1
            self._ocp_controllable_interface = SafeAccelVehicleInterface
        self._is_acceleration_optimal = is_acceleration_optimal

    def set_platoon(self, new_platoon: platoon.Platoon) -> None:
        self._platoon = new_platoon

    def update_mode(self, vehicles: Dict[int, base.BaseVehicle]) -> None:
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

    def is_platoon_leader(self) -> bool:
        return self._id == self._platoon.get_platoon_leader_id()

    def can_start_lane_change(self, vehicles: Dict[int, base.BaseVehicle]
                              ) -> bool:
        if self.opt_controller.has_solution():
            return True

        if self._is_verbose:
            t = self.get_current_time()
            print("t={:.2f}, veh:{}. Calling ocp solver...".format(
                t, self._id))
        self.opt_controller.find_trajectory(vehicles)
        return self.opt_controller.has_solution()

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Dict[int, base.BaseVehicle]) -> None:
        """
        Sets the open loop controls a (acceleration) and phi (steering wheel
        angle)
        :param open_loop_controls: Vector with accel and phi values
        :param vehicles: Surrounding vehicles
        :return: Nothing. The vehicle stores the computed input values
        """
        t = self.get_current_time()
        if self.opt_controller.is_active(t):
            if self._is_acceleration_optimal:
                optimal_inputs = self.opt_controller.get_input(t, self.get_id())
                accel = optimal_inputs[self._input_idx['a']]
                phi = optimal_inputs[self._input_idx['phi']]
            else:
                accel = self.long_controller.compute_acceleration(vehicles)
                phi = self.opt_controller.get_input(t, self._id)
        else:
            accel = self.long_controller.compute_acceleration(vehicles)
            phi = self.lk_controller.compute_steering_wheel_angle()

        self._inputs[self._input_idx['a']] = accel
        self._inputs[self._input_idx['phi']] = phi

    def _update_target_leader(self, vehicles: Dict[int, base.BaseVehicle]
                              ) -> None:
        if (self._is_acceleration_optimal
                and self.opt_controller.is_active(self.get_current_time())):
            # During the lane change, there is no target leader for the
            # longitudinal controller since the optimal controller takes over
            self._leader_id[self._iter_counter] = -1
        else:
            self._leader_id[self._iter_counter] = (
                self.long_controller.get_more_critical_leader(vehicles))

    def analyze_platoons(self, vehicles: Dict[int, base.BaseVehicle]
                         ) -> None:

        # [Aug 23] We are only simulating simple scenarios. At the start of
        # the simulation, every vehicle will either create its own platoon
        # or join the platoon of the vehicle ahead. Vehicles do not leave or
        # join platoons afterward

        is_leader_in_a_platoon = (
                self.has_orig_lane_leader()
                and vehicles[self.get_orig_lane_leader_id()].is_in_a_platoon()
        )
        if is_leader_in_a_platoon:
            leader_platoon: platoon.Platoon = vehicles[
                self.get_orig_lane_leader_id()].get_platoon()
            if not self.is_in_a_platoon() or self._platoon != leader_platoon:
                leader_platoon.add_vehicle(self)
                self.set_platoon(leader_platoon)
        else:
            if not self.is_in_a_platoon():
                self.set_platoon(platoon.Platoon(self))

        # if not self.is_in_a_platoon():
        #     if is_leader_in_a_platoon:
        #         leader_platoon = vehicles[
        #             self.get_orig_lane_leader_id()].get_platoon()
        #         leader_platoon.add_vehicle(self)
        #         self.set_platoon(leader_platoon)
        #     else:
        #         self.set_platoon(platoon.Platoon(self))
        # else:
        #     if is_leader_in_a_platoon:
        #         leader_platoon = vehicles[
        #             self.get_orig_lane_leader_id()].get_platoon()
        #         if self._platoon != leader_platoon:
        #             leader_platoon.add_vehicle(self)
        #             self.set_platoon(leader_platoon)

    # def join_centrally_controlled_platoon(
    #         self, vehicles: Dict[int, base.BaseVehicle]) -> None:
    #     # Find the "central" platoon
    #     central_platoon = None
    #     for veh in vehicles.values():
    #         if veh.is_in_a_platoon():
    #             central_platoon = veh.get_platoon()
    #             break
    #     if central_platoon is None:
    #         self._create_platoon()
    #     else:
    #         central_platoon.add_vehicle(self)
    #         self.set_platoon(central_platoon)

    def _create_platoon(self) -> None:
        self.set_platoon(platoon.Platoon(self))
        # platoons[new_platoon.id] = new_platoon
        # self._platoon

    def _write_optimal_inputs(self, optimal_inputs) -> None:
        if self._is_acceleration_optimal:
            self._inputs = optimal_inputs
        else:
            self._inputs[self._input_idx['phi']] = optimal_inputs[0]


class FourStateVehicleInterface(base.BaseVehicleInterface, ABC):

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    def __init__(self, vehicle: FourStateVehicle):
        super().__init__(vehicle)
        self._set_model()
        self.brake_max = vehicle.brake_max
        self.accel_max = vehicle.accel_max

    def get_desired_input(self) -> np.ndarray:
        return np.array([0] * self.n_inputs)

    @classmethod
    def _set_speed(cls, v0, state) -> None:
        state[cls.state_idx['v']] = v0

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives) -> None:
        self._position_derivative_cg(vel, theta, phi, derivatives)
        derivatives[self.state_idx['v']] = accel


class OpenLoopVehicleInterface(FourStateVehicleInterface):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G. """

    def __init__(self, vehicle: OpenLoopVehicle):
        super().__init__(vehicle)

    def get_accel(self, ego_states, inputs, leader_states):
        return self.select_input_from_vector(inputs, 'a')

    def get_input_limits(self) -> (List[float], List[float]):
        return ([self.brake_max, -self.get_phi_max()],
                [self.accel_max, self.get_phi_max()])


class OpenLoopNoLCVehicleInterface(FourStateVehicleInterface):
    """ States: [x, y, theta, v], inputs: [a], centered at the C.G. """

    _input_names = ['a']

    def __init__(self, vehicle: OptimalCoopNoLCVehicle):
        super().__init__(vehicle)

    def get_phi(cls, optimal_inputs):
        return 0.0

    def get_accel(self, ego_states, inputs, leader_states):
        return self.select_input_from_vector(inputs, 'a')

    def get_input_limits(self) -> (List[float], List[float]):
        return [self.brake_max], [self.accel_max]


class SafeAccelVehicleInterface(FourStateVehicleInterface):
    """ States: [x, y, theta, v], inputs: [phi], centered at the C.G.
     and accel is computed by a feedback law"""

    _input_names = ['phi']

    def __init__(self, vehicle: Union[SafeAccelOpenLoopLCVehicle,
                                      SafeAccelOptimalLCVehicle,
                                      ClosedLoopVehicle]):
        super().__init__(vehicle)
        # Controller parameters
        # self.h: float = vehicle.h  # time headway [s]
        self.long_controller = vehicle.long_controller

    def get_input_limits(self) -> (List[float], List[float]):
        return [-self.get_phi_max()], [self.get_phi_max()]

    def get_accel(self, ego_states, inputs, leader_states) -> float:
        """
        Computes acceleration for the ego vehicle following a leader
        """
        return self.long_controller.compute_acceleration_from_interface(
            type(self), ego_states, self.get_free_flow_speed(), leader_states
        )
        # v_ego = self.select_state_from_vector(ego_states, 'v')
        # v_ff = self.get_free_flow_speed()
        # if leader_states is None or len(leader_states) == 0:
        #     return self.long_controller.compute_velocity_control(v_ff, v_ego)
        # else:
        #     gap = (self.select_state_from_vector(leader_states, 'x')
        #            - self.select_state_from_vector(ego_states, 'x'))
        #     v_leader = self.select_state_from_vector(leader_states, 'v')
        #     accel = self.long_controller.compute_gap_control(gap, v_ego,
        #                                                      v_leader)
        #     if v_ego >= v_ff and accel > 0:
        #         return self.long_controller.compute_velocity_control(
        #             v_ff, v_ego)
        #     return accel


class ClosedLoopVehicleInterface(SafeAccelVehicleInterface):
    """ Vehicle that does not perform lane change and computes its own
    acceleration """

    _input_names = []

    def __init__(self, vehicle: ClosedLoopVehicle):
        super().__init__(vehicle)

    def select_input_from_vector(cls, optimal_inputs: List, input_name: str
                                 ) -> float:
        return 0.0  # all inputs are computed internally

    def get_input_limits(self) -> (List[float], List[float]):
        return [], []


class SafeLongitudinalVehicleInterface(ClosedLoopVehicleInterface):
    def __init__(self, vehicle: SafeLongitudinalVehicle):
        super().__init__(vehicle)

    def get_phi(cls, optimal_inputs):
        return 0.0

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives):
        self._position_derivative_longitudinal_only(vel, derivatives)
        derivatives[self.state_idx['v']] = accel

# OCV = TypeVar('OCV', bound=OptimalControlVehicle)
