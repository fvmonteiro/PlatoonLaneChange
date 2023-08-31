import numpy as np

import constants as const
import vehicle_models.base_vehicle as base


class LateralController:

    def __init__(self, vehicle: base.BaseVehicle):
        self._lateral_gain = 1.0
        self.vehicle = vehicle

    def _translate_slip_to_steering_wheel_angle(self, slip_angle: float):
        return np.arctan(self.vehicle.lr / (self.vehicle.lf + self.vehicle.lr)
                         * np.tan(slip_angle))

    def _compute_cbf_slip_angle(self, y_ref: float, vy_ref: float):
        lat_error = y_ref - self.vehicle.get_y()
        theta = self.vehicle.get_theta()
        vel = self.vehicle.get_vel()
        return ((vy_ref + self._lateral_gain * lat_error)
                / (vel * np.cos(theta)) - np.tan(theta))


class LaneKeepingController(LateralController):
    def compute_steering_wheel_angle(self):
        lane_center = self.vehicle.get_current_lane() * const.lane_width
        slip_angle = self._compute_cbf_slip_angle(lane_center, 0.0)
        return self._translate_slip_to_steering_wheel_angle(slip_angle)


class LaneChangingController(LateralController):

    def __init__(self, vehicle: base.BaseVehicle):
        super().__init__(vehicle)
        self._lc_start_time = -np.inf

    def start(self, lc_start_time: float, lc_duration: float):
        self._lc_start_time = lc_start_time
        self._compute_polynomial_lc_trajectory(lc_duration)

    def compute_steering_wheel_angle(self):
        delta_t = self.vehicle.get_current_time() - self._lc_start_time
        yr = sum([self._polynomial_lc_coeffs[i] * delta_t ** i
                  for i in range(len(self._polynomial_lc_coeffs))])
        vyr = sum([i * self._polynomial_lc_coeffs[i] * delta_t ** (i - 1)
                   for i in range(1, len(self._polynomial_lc_coeffs))])
        slip_angle = self._compute_cbf_slip_angle(yr, vyr)
        return self._translate_slip_to_steering_wheel_angle(slip_angle)

    def _compute_polynomial_lc_trajectory(self, lc_duration):
        y0 = self.vehicle.get_y()
        vy0 = 0
        ay0 = 0
        yf = self.vehicle.target_y
        vyf = vy0
        ayf = ay0

        tf = lc_duration
        a = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 2, 0, 0, 0],
                      [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                      [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                      [0, 0, 2, 6 * tf, 12 * tf * 2, 20 * tf ** 3]])
        b = np.array([[y0], [vy0], [ay0], [yf], [vyf], [ayf]])
        self._polynomial_lc_coeffs = np.linalg.solve(a, b)
