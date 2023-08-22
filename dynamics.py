from vehicle_group import VehicleGroup


# === Functions passed to the optimal control library methods === #
def vehicles_derivatives(t, states, inputs, params):
    """
    Implements the kinematic bicycle model with reference at the vehicles C.G.
    Follows the model of updfcn of the control package.
    :param t: time
    :param states: Array with states of all vehicles [x1, y1, ..., xN, yN]
    :param inputs: Array with inputs of all vehicles [u11, u12, ..., u1N, u2N]
    :param params: Dictionary which must contain the vehicle type
    :return: state update function
    """
    vehicle_group: VehicleGroup = params['vehicle_array']

    return vehicle_group.compute_derivatives(states, inputs, params)


def vehicle_output(t, x, u, params):
    return x  # return (full state)
