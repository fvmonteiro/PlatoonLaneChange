from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Iterable
from typing import Union

import numpy as np

import controllers.longitudinal_controller as long_ctrl
import controllers.optimal_controller as opt_ctrl
import controllers.vehicle_controller as veh_ctrl
import graph_tools
import operating_modes.concrete_vehicle_modes as modes
import platoon
import vehicle_models.base_vehicle as base


class FourStateVehicle(base.BaseVehicle, ABC):
    """
    States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    """

    _controller_type: type[veh_ctrl.VehicleController]
    _platoon: platoon.Platoon

    _state_names = ['x', 'y', 'theta', 'v']
    _input_names = ['a', 'phi']

    static_attribute_names = base.BaseVehicle.static_attribute_names.union(
        {'brake_max', 'accel_max'}
    )

    def __init__(self, can_change_lanes: bool,
                 has_open_loop_acceleration: bool,
                 is_connected: bool):
        super().__init__(is_connected)
        self._ocp_interface = FourStateVehicleInterface

        self._can_change_lanes = can_change_lanes
        self._has_open_loop_acceleration = has_open_loop_acceleration
        # TODO: bad naming.
        self._external_input_idx = {}
        if self._has_open_loop_acceleration:
            self._external_input_idx['a'] = 0
        if self._can_change_lanes:
            self._external_input_idx['phi'] = (
                    self._external_input_idx.get('a', -1) + 1)

        self._set_model()

        self._controller = self._controller_type(
            self, self._can_change_lanes, self._has_open_loop_acceleration)

    def get_can_change_lanes(self) -> bool:
        return self._can_change_lanes

    def get_has_open_loop_acceleration(self) -> bool:
        return self._has_open_loop_acceleration

    def get_vel(self):
        return self.get_a_state_by_name('v')

    def get_platoon(self) -> Union[None, platoon.Platoon]:
        try:
            return self._platoon
        except AttributeError:
            return None

    def get_external_input_idx(self) -> dict[str, int]:
        return self._external_input_idx

    def get_optimal_input_history(self) -> np.ndarray:
        ret = []
        for key in self._external_input_idx:
            ret.append(self._inputs_history[self._input_idx[key]])
        return np.array(ret)

    def get_desired_destination_lane_leader_id(self) -> int:
        if not self.is_in_a_platoon():
            return self.get_destination_lane_leader_id()
        return self.get_platoon().get_desired_dest_lane_leader_id(self.get_id())

    def get_is_lane_change_safe(self):
        return self._is_lane_change_safe

    def set_platoon(self, new_platoon: platoon.Platoon) -> None:
        self._platoon = new_platoon

    def make_reset_copy(self, initial_state: np.ndarray = None) -> base.V:
        """
        Creates copies of vehicles used in internal iterations of our optimal
        controller. For vehicles without optimal control, the method returns
        an "empty" copy of the vehicle (without any state history). For
        vehicles with optimal control, the method returns the equivalent
        open loop type vehicle.
        :param initial_state: If None, sets the new vehicle's initial state
        equal to the most recent state of this instance
        as this instance
        :return:
        """
        new_vehicle_type = type(self)
        new_vehicle = new_vehicle_type(self._can_change_lanes,
                                       self._has_open_loop_acceleration,
                                       self._is_connected)
        self.copy_attributes(new_vehicle, initial_state)
        return new_vehicle

    def make_closed_loop_copy(self, initial_state: np.ndarray = None
                              ) -> ClosedLoopVehicle:
        new_vehicle = ClosedLoopVehicle(
            self._can_change_lanes, False, self._is_connected)
        self.copy_attributes(new_vehicle, initial_state)
        new_vehicle._desired_future_follower_id = (
            self._desired_future_follower_id)
        return new_vehicle

    def find_cooperation_requests(self, vehicles: Iterable[base.BaseVehicle]
                                  ) -> None:
        # Note that any optimal vehicle might be in a platoon, even if it
        # is not a PlatoonVehicle. Possible code smell, but it works so far
        # We assume platoon vehicles do not cooperate with other vehicles
        if not self.is_in_a_platoon():
            super().find_cooperation_requests(vehicles)
        else:
            incoming_vehicle_id = self.get_platoon().get_incoming_vehicle_id(
                self.get_id())
            self._incoming_vehicle_id[self._iter_counter] = incoming_vehicle_id

    def update_target_leader(self, vehicles: Mapping[int, FourStateVehicle]
                             ) -> None:
        self._leader_id[self._iter_counter] = (
            self._controller.get_target_leader_id(vehicles))

    def compute_gap_to_a_leader(self, a_leader: base.BaseVehicle):
        return base.BaseVehicle.compute_a_gap(a_leader, self)

    def is_platoon_leader(self) -> bool:
        return self.get_id() == self._platoon.get_platoon_leader_id()

    def update_platoons(
            self, vehicles: Mapping[int, base.BaseVehicle],
            platoon_lane_change_strategy: int,
            # strategy_parameters: tuple[list[int], list[int]] = None
    ) -> None:

        # [Aug 23] We are only simulating simple scenarios. At the start of
        # the simulation, every vehicle will either create its own platoon
        # or join the platoon of the vehicle ahead. Vehicles do not leave or
        # join platoons afterward

        # We're only interested in lane changing platoons
        if not self._is_connected or not self._can_change_lanes:
            return

        is_leader_in_a_platoon = (
                self.has_origin_lane_leader()
                and vehicles[self.get_origin_lane_leader_id()].is_in_a_platoon()
        )
        if is_leader_in_a_platoon:
            leader_platoon: platoon.Platoon = vehicles[
                self.get_origin_lane_leader_id()].get_platoon()
            if not self.is_in_a_platoon() or self._platoon != leader_platoon:
                leader_platoon.add_vehicle(self)
                # opt_control = leader_platoon.get_optimal_controller()
                # self.set_centralized_controller(opt_control)
                self.set_platoon(leader_platoon)
        else:
            if not self.is_in_a_platoon():
                self.set_platoon(self._create_platoon(
                    platoon_lane_change_strategy))

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

    def set_platoon_strategy_order(
            self, strategy_parameters: tuple[list[int], list[int]] = None
    ) -> None:
        if self.is_in_a_platoon():
            self.get_platoon().set_strategy_parameters(strategy_parameters)

    def set_platoon_strategy_states_graph(
            self, states_graph: graph_tools.VehicleStatesGraph) -> None:
        if self.is_in_a_platoon():
            self.get_platoon().set_strategy_states_graph(states_graph)

    def _determine_inputs(self, open_loop_controls: np.ndarray,
                          vehicles: Mapping[int, FourStateVehicle]):
        accel, phi = self._controller.determine_inputs(open_loop_controls,
                                                       vehicles)
        self._inputs[self._input_idx['a']] = accel
        self._inputs[self._input_idx['phi']] = phi

    def _set_speed(self, v0, state):
        state[self._state_idx['v']] = v0

    def _compute_derivatives(self, vel, theta, phi):
        # TODO: can we avoid the if here? maybe 'save' which function is called
        if self._can_change_lanes:
            self._position_derivative_cg(vel, theta, phi)
        else:
            self._position_derivative_longitudinal_only(vel)
        self._derivatives[self._state_idx['v']] = self.get_an_input_by_name('a')

    def _write_optimal_inputs(self, optimal_inputs):
        for i_name, i_idx in self._external_input_idx.items():
            self._inputs[self._input_idx[i_name]] = optimal_inputs[i_idx]

    def _create_platoon(
            self, platoon_lane_change_strategy: int,
            # strategy_parameters: tuple[list[int], list[int]] = None
    ) -> platoon.Platoon:
        pass


class OpenLoopVehicle(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Does not compute any inputs internally. This class and its derivatives
    are useful for testing inputs computed by a centralized controller """

    _controller_type = veh_ctrl.ExternalControl

    def __init__(self, can_change_lanes: bool,
                 has_open_loop_acceleration: bool,
                 is_connected: bool = False):
        super().__init__(can_change_lanes, has_open_loop_acceleration,
                         is_connected)

    def update_mode(self, vehicles: Mapping[int, base.BaseVehicle]):
        pass


class OptimalControlVehicle(FourStateVehicle):
    """ States: [x, y, theta, v], inputs: [a, phi], centered at the C.G.
    Accel and phi can be computed by optimal control when there is lane change
    intention. Otherwise, constant time headway policy for accel and
    a CBF computed lane keeping phi. """

    _controller_type = veh_ctrl.OptimalControl
    _platoon: platoon.OptimalPlatoon

    def __init__(self, can_change_lanes: bool = True,
                 has_open_loop_acceleration: bool = True,
                 is_connected: bool = False):
        super().__init__(can_change_lanes, has_open_loop_acceleration,
                         is_connected)

        self._platoon_type = platoon.OptimalPlatoon
        self.set_mode(modes.OCPLaneKeepingMode())
        self.get_opt_controller().set_controlled_vehicles_ids(self.get_id())

    def get_opt_controller(self) -> opt_ctrl.VehicleOptimalController:
        return self._controller.get_opt_controller()

    def get_platoon(self) -> Union[None, platoon.OptimalPlatoon]:
        try:
            return self._platoon
        except AttributeError:
            return None

    def set_centralized_controller(
            self, centralized_controller: opt_ctrl.VehicleOptimalController):
        centralized_controller.add_controlled_vehicle_id(self.get_id())
        self._controller.set_opt_controller(centralized_controller)

    def make_open_loop_copy(self, initial_state: np.ndarray = None
                            ) -> OpenLoopVehicle:
        new_vehicle = OpenLoopVehicle(
            self._can_change_lanes, self._has_open_loop_acceleration,
            self._is_connected)
        self.copy_attributes(new_vehicle, initial_state)
        new_vehicle._platoon = self.get_platoon()
        new_vehicle._desired_future_follower_id = (
            self._desired_future_follower_id)
        return new_vehicle

    def get_intermediate_results(self):
        return self.get_opt_controller().get_simulation_per_iteration()

    def update_mode(self, vehicles: Mapping[int, base.BaseVehicle]):
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

    def can_start_lane_change(self, vehicles: Mapping[int, base.BaseVehicle]
                              ) -> bool:
        if self.get_opt_controller().has_solution():
            return True

        if self.is_in_a_platoon():
            dest_lane_veh_ids = [veh.get_id() for veh in vehicles.values()
                                 if veh.get_current_lane() == self._target_lane]
            self.get_opt_controller().\
                set_platoon_formation_constraint_parameters(
                self.get_platoon().get_vehicle_ids(), dest_lane_veh_ids
            )
        if self._is_verbose:
            t = self.get_current_time()
            print("t={:.2f}, veh:{}. Calling optimal controller".format(
                t, self._id))
        self.get_opt_controller().find_trajectory(vehicles)
        return self.get_opt_controller().has_solution()

    # Outdated. We're not considering the persistent scenario.
    def can_start_lane_change_with_checks(
            self, vehicles: Mapping[int, base.BaseVehicle]) -> bool:
        if self.get_opt_controller().has_solution():
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
        solver_wait_time = 20.0  # [s] time between attempts to solve an ocp
        is_cool_down_period_done = (
                t - self.get_opt_controller().get_activation_time()
                >= solver_wait_time
        )

        if has_vehicle_configuration_changed or is_cool_down_period_done:
            return self.can_start_lane_change(vehicles)

    def is_lane_changing(self) -> bool:
        delta_t = self.get_current_time() - self._lc_start_time
        return delta_t <= self.get_opt_controller().get_time_horizon()

    def _create_platoon(self, platoon_lane_change_strategy: int
                        ) -> platoon.OptimalPlatoon:
        return platoon.OptimalPlatoon(self, platoon_lane_change_strategy)


class ClosedLoopVehicle(FourStateVehicle):
    """ Vehicle that computes all of its inputs by feedback laws.
     States: [x, y, theta, v], external input: None, centered at the C.G. """

    _controller_type = veh_ctrl.ClosedLoopControl
    _platoon: platoon.ClosedLoopPlatoon
    _platoon_lane_change_graph: graph_tools.VehicleStatesGraph

    def __init__(self, can_change_lanes: bool,
                 has_open_loop_acceleration: bool = False,
                 is_connected: bool = False):
        super().__init__(can_change_lanes, has_open_loop_acceleration,
                         is_connected)
        self._platoon_type = platoon.ClosedLoopPlatoon
        self.set_mode(modes.CLLaneKeepingMode())

    def set_platon_lane_change_graph(
            self, platoon_lane_change_graph: graph_tools.VehicleStatesGraph):
        self._platoon_lane_change_graph = platoon_lane_change_graph

    def get_platoon(self) -> Union[None, platoon.ClosedLoopPlatoon]:
        try:
            return self._platoon
        except AttributeError:
            return None

    def update_mode(self, vehicles: Mapping[int, base.BaseVehicle]):
        if self.has_lane_change_intention():
            self._mode.handle_lane_changing_intention(vehicles)
        else:
            self._mode.handle_lane_keeping_intention(vehicles)

    def can_start_lane_change(self, vehicles: Mapping[int, base.BaseVehicle]
                              ) -> bool:
        is_safe = self.check_is_lane_change_safe(vehicles)
        is_my_turn = (
                not self.is_in_a_platoon()
                or self.get_platoon().can_start_lane_change(self.get_id()))
        # We can't short-circuit the evaluation because the platoon's method
        # (can_start_lane_change) may update internal values.
        return is_safe and is_my_turn

    def is_lane_change_complete(self):
        return (np.abs(self.get_y() - self.get_target_y()) < 1e-1
                and np.abs(self.get_theta()) < 1e-1)

    def _set_up_lane_change_control(self):
        self._controller.set_up_lane_change_control(self._lc_start_time)

    def _create_platoon(self, platoon_lane_change_strategy: int
                        ) -> platoon.ClosedLoopPlatoon:
        return platoon.ClosedLoopPlatoon(self, platoon_lane_change_strategy)


class ShortSimulationVehicle(ClosedLoopVehicle):
    """
    Class used when computing costs between nodes (which are defined as
    quantized states)
    """
    def __init__(self, can_change_lanes: bool,
                 has_open_loop_acceleration: bool = False,
                 is_connected: bool = False):
        super().__init__(can_change_lanes, has_open_loop_acceleration,
                         is_connected)
        self._desired_dest_lane_leader_id = -1
        self._fixed_incoming_vehicle_id = -1

    def set_desired_dest_lane_leader_id(self, value):
        self._desired_dest_lane_leader_id = value

    def set_incoming_vehicle_id(self, value):
        self._fixed_incoming_vehicle_id = value

    def get_desired_destination_lane_leader_id(self) -> int:
        return self._desired_dest_lane_leader_id

    def find_cooperation_requests(self, vehicles: Iterable[base.BaseVehicle]
                                  ) -> None:
        self._incoming_vehicle_id[self._iter_counter] = (
            self._fixed_incoming_vehicle_id
        )

    def can_start_lane_change(self, vehicles: Mapping[int, base.BaseVehicle]
                              ) -> bool:
        is_safe = self.check_is_lane_change_safe(vehicles)
        is_my_turn = (self.get_desired_destination_lane_leader_id()
                      == self.get_destination_lane_leader_id())
        return is_safe and is_my_turn

    def make_closed_loop_copy(self, initial_state: np.ndarray = None
                              ) -> ShortSimulationVehicle:
        new_vehicle = ShortSimulationVehicle(
            self._can_change_lanes, self._is_connected)
        self.copy_attributes(new_vehicle, initial_state)
        new_vehicle._desired_future_follower_id = (
            self._desired_future_follower_id)
        return new_vehicle


class FourStateVehicleInterface(base.BaseVehicleInterface, ABC):
    # _state_names = ['x', 'y', 'theta', 'v']
    _state_names = FourStateVehicle.get_state_names()

    def __init__(self, vehicle: FourStateVehicle):
        super().__init__(vehicle)
        self._can_change_lanes = vehicle.get_can_change_lanes()
        self._has_open_loop_acceleration = (
            vehicle.get_has_open_loop_acceleration())
        self._input_names = []
        if self._has_open_loop_acceleration:
            self._input_names.append('a')
        if self._can_change_lanes:
            self._input_names.append('phi')
        self._set_model()
        self.long_controller = long_ctrl.LongitudinalController(vehicle)

    def _set_speed(self, v0, state) -> None:
        state[self.state_idx['v']] = v0

    def get_input_limits(self) -> (list[float], list[float]):
        lb, ub = [], []
        if self._has_open_loop_acceleration:
            lb.append(self.get_brake_max())
            ub.append(self.get_accel_max())
        if self._can_change_lanes:
            lb.append(-self.get_phi_max())
            ub.append(self.get_phi_max())
        return lb, ub

    def get_desired_input(self) -> np.ndarray:
        return np.array([0] * self.n_inputs)

    def get_accel(self, ego_states, inputs, leader_states) -> float:
        if self._has_open_loop_acceleration:
            return self.select_input_from_vector(inputs, 'a')
        else:
            return self.long_controller.compute_acceleration_from_interface(
                self, ego_states, self.get_free_flow_speed(),
                leader_states
            )

    def get_phi(self, optimal_inputs) -> float:
        if self._can_change_lanes:
            return self.select_input_from_vector(optimal_inputs, 'phi')
        else:
            return 0.0

    def _compute_derivatives(self, vel, theta, phi, accel, derivatives) -> None:
        # TODO: can we avoid the if here? maybe 'save' which function is called
        if self._can_change_lanes:
            self._position_derivative_cg(vel, theta, phi, derivatives)
        else:
            self._position_derivative_longitudinal_only(vel, derivatives)
        # self._position_derivative_cg(vel, theta, phi, derivatives)
        derivatives[self.state_idx['v']] = accel
