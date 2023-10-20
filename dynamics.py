from typing import Tuple

import numpy as np

# Not being used for now. Goal is to reduce code duplication between
# vehicle models and vehicle to ocp interface


def position_derivative_cg(vel: float, theta: float, phi: float,
                           lf: float, lr: float) -> Tuple[float, float, float]:
    beta = np.arctan(lr * np.tan(phi) / (lf + lr))
    dx = vel * np.cos(theta + beta)
    dy = vel * np.sin(theta + beta)
    dtheta = (vel * np.sin(beta) / lr)
    return dx, dy, dtheta


def position_derivative_rear_wheels(vel: float, theta: float, phi: float,
                                    wheelbase: float
                                    ) -> Tuple[float, float, float]:
    dx = vel * np.cos(theta)
    dy = vel * np.sin(theta)
    dtheta = vel * np.tan(phi) / wheelbase
    return dx, dy, dtheta


def position_derivative_longitudinal_only(vel):
    return vel, 0., 0.
