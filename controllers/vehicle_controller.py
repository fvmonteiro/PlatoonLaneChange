from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
import warnings

import numpy as np

import controllers.lateral_controller as lat_ctrl
import controllers.longitudinal_controller as long_ctrl
import controllers.optimal_controller as opt_ctrl
import vehicle_models.four_state_vehicles as fsv

# TODO: reevaluate naming choices (specially wrt VehicleOptimalController class)


class VehicleController(ABC):

    class LongMode(Enum):
        REAL_LEADER = 1
        VIRTUAL_LEADER = 2
        OPTIMAL = 3

    _opt_controller: opt_ctrl.VehicleOptimalController
    _lk_controller: lat_ctrl.LaneKeepingController
    _lc_controller: lat_ctrl.LaneChangingController

    def __init__(self, ego_vehicle: fsv.FourStateVehicle,
                 can_change_lanes: bool, has_open_loop_acceleration: bool):
        self._ego_vehicle = ego_vehicle
        self._can_change_lanes = can_change_lanes
        self._has_open_loop_acceleration = has_open_loop_acceleration
        self.long_mode = (self.LongMode.OPTIMAL if has_open_loop_acceleration
                          else self.LongMode.REAL_LEADER)

        self._real_leader_long_controller = long_ctrl.LongitudinalController(
            ego_vehicle, ego_vehicle.brake_max)
        self._virtual_leader_long_controller = (
            long_ctrl.LongitudinalController(
                ego_vehicle, ego_vehicle.brake_comfort_max))

        self._external_input_idx = ego_vehicle.get_external_input_idx()

    def get_opt_controller(self) -> opt_ctrl.VehicleOptimalController:
        return self._opt_controller

    def set_opt_controller(
            self, new_opt_controller: opt_ctrl.VehicleOptimalController
    ) -> None:
        self._opt_controller = new_opt_controller

    def update_real_leader_time_headway(self, new_h):
        self._real_leader_long_controller.set_time_headway(new_h)

    def update_virtual_leader_time_headway(self, new_h):
        self._virtual_leader_long_controller.set_time_headway(new_h)

    def determine_inputs(self, external_controls: np.ndarray,
                         vehicles: Mapping[int, fsv.FourStateVehicle]
                         ) -> (float, float):
        if self._has_open_loop_acceleration:
            accel = self._get_open_loop_acceleration(external_controls,
                                                     vehicles)
        else:
            accel = self._compute_closed_loop_acceleration(vehicles)
            # accel = self._real_leader_long_controller.compute_acceleration(
            #     vehicles)
        if self._can_change_lanes:
            phi = self._determine_steering_wheel_angle(external_controls,
                                                       vehicles)
        else:
            phi = 0.
        return accel, phi

    # def get_target_leader_id(self,
    # vehicles: Mapping[int, fsv.FourStateVehicle]
    #                          ) -> int:
    #     if self._has_open_loop_acceleration:
    #         return self._get_target_leader_id(vehicles)
    #     else:
    #         return self._ego_vehicle.get_current_leader_id()

    # TODO: make virtual? abstract?
    def set_up_lane_change_control(self, start_time):
        self._lc_controller.compute_lc_trajectory(
            start_time)

    def _compute_closed_loop_acceleration(
            self, vehicles: Mapping[int, fsv.FourStateVehicle]) -> float:
        accel_real_leader = (
            self._real_leader_long_controller.compute_acceleration(
                vehicles, self._ego_vehicle.get_origin_lane_leader_id()))
        if self._ego_vehicle.has_virtual_leader():
            accel_virtual_leader = (
                self._virtual_leader_long_controller.compute_acceleration(
                    vehicles, self._ego_vehicle.get_virtual_leader_id()))
        else:
            accel_virtual_leader = np.inf

        if accel_real_leader <= accel_virtual_leader:
            new_mode = self.LongMode.REAL_LEADER
            self._ego_vehicle.target_origin_lane_leader()
            desired_accel = accel_real_leader
        else:
            new_mode = self.LongMode.VIRTUAL_LEADER
            self._ego_vehicle.target_virtual_leader()
            desired_accel = accel_virtual_leader
        # if self.long_mode != new_mode:
        #     print(
        #         f"[VehicleController] "
        #         f"t={self._ego_vehicle.get_current_time()} "
        #         f"veh {self._ego_vehicle.get_name()} long mode from: "
        #         f"{self.long_mode.name} to {new_mode.name}")
        self.long_mode = new_mode
        return desired_accel

    @abstractmethod
    def _get_open_loop_acceleration(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        pass

    @abstractmethod
    def _determine_steering_wheel_angle(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        pass

    @abstractmethod
    def _get_target_leader_id(self, vehicles: Mapping[int, fsv.FourStateVehicle]
                              ) -> int:
        pass


class ExternalControl(VehicleController):
    """
    Reads inputs determined externally and passes them to the vehicle. Useful
    for open loop tests, controlling the front-most vehicle on a lane, or,
    most often, within the iterations of the optimal controller.
    """

    def _get_open_loop_acceleration(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        return external_controls[self._external_input_idx['a']]

    def _determine_steering_wheel_angle(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        return external_controls[self._external_input_idx['phi']]

    def _get_target_leader_id(self,
                              vehicles: Mapping[int, fsv.FourStateVehicle]):
        """
        No target leader, since this vehicle class does not have autonomous
        longitudinal control
        """
        return -1


class OptimalControl(VehicleController):

    def __init__(self, ego_vehicle: fsv.FourStateVehicle,
                 can_change_lanes: bool,
                 has_open_loop_acceleration: bool):
        super().__init__(ego_vehicle, can_change_lanes,
                         has_open_loop_acceleration)
        self._opt_controller = opt_ctrl.VehicleOptimalController()
        self._lk_controller = lat_ctrl.LaneKeepingController(ego_vehicle)

    def _get_open_loop_acceleration(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        t = self._ego_vehicle.get_current_time()
        if self._opt_controller.is_active(t):
            return self._opt_controller.get_input(
                t, self._ego_vehicle.id)[self._external_input_idx['a']]
        else:
            return self._compute_closed_loop_acceleration(vehicles)

    def _determine_steering_wheel_angle(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        t = self._ego_vehicle.get_current_time()
        if self._opt_controller.is_active(t):
            return self._opt_controller.get_input(
                t, self._ego_vehicle.id)[self._external_input_idx['phi']]
        else:
            return self._lk_controller.compute_steering_wheel_angle()

    def _get_target_leader_id(self, vehicles: Mapping[int, fsv.FourStateVehicle]
                              ) -> int:
        if self._opt_controller.is_active(
                self._ego_vehicle.get_current_time()):
            # During the lane change, there is no target leader for the
            # long controller since the optimal controller takes over
            return -1
        else:
            # At all other times, we only look at the origin lane leader
            return self._ego_vehicle.get_origin_lane_leader_id()


class ClosedLoopControl(VehicleController):

    def __init__(self, ego_vehicle: fsv.FourStateVehicle,
                 can_change_lanes: bool, has_open_loop_acceleration: bool):
        if has_open_loop_acceleration:
            warnings.warn('You cannot construct a ClosedLoopController with'
                          'has_open_loop_acceleration=True. It will be '
                          'set to False')
        super().__init__(ego_vehicle, can_change_lanes,
                         False)
        self._lk_controller = lat_ctrl.LaneKeepingController(ego_vehicle)
        self._lc_controller = lat_ctrl.LaneChangingController(ego_vehicle)

    def _get_open_loop_acceleration(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        # we should never reach this
        warnings.warn('ClosedLoopController was constructed with'
                      'has_open_loop_acceleration True')
        return self._compute_closed_loop_acceleration(vehicles)

    def _determine_steering_wheel_angle(
            self, external_controls: np.ndarray,
            vehicles: Mapping[int, fsv.FourStateVehicle]
    ) -> float:
        delta_t = (self._ego_vehicle.get_current_time()
                   - self._lc_controller.get_start_time())
        if delta_t <= self._lc_controller.get_lc_duration():
            return self._lc_controller.compute_steering_wheel_angle()
        else:
            return self._lk_controller.compute_steering_wheel_angle()

    def _get_target_leader_id(self, vehicles: Mapping[int, fsv.FourStateVehicle]
                              ) -> int:
        # we should never reach this
        warnings.warn('ClosedLoopController was constructed with'
                      'has_open_loop_acceleration True')
        if self.long_mode == self.LongMode.REAL_LEADER:
            return self._ego_vehicle.get_current_leader_id()
        else:
            return self._ego_vehicle.get_virtual_leader_id()
