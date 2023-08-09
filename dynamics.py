# === Functions passed to the optimal control library methods === #
def vehicle_update(t, states, inputs, params):
    """
    Implements the kinematic bicycle model with reference at the vehicles C.G.
    Follows the model of updfcn of the control package.
    :param t: time
    :param states: Array with states of all vehicles [x1, y1, ..., xN, yN]
    :param inputs: Array with inputs of all vehicles [u11, u12, ..., u1N, u2N]
    :param params: Dictionary which must contain the vehicle type
    :return: state update function
    """
    vehicle_array = params['vehicle_array']

    if params['test']:
        print('------ once? ------')
        params['test'] = False

    # TODO: save vehicle_array instead of vehicle_class in the params dict.
    #  Create and call a method 'compute_internal_inputs' which will set the
    #  accelerations. The dynamics will be the same for all vehicle with the
    #  same states

    return vehicle_array.update(states, inputs, params)


def vehicle_output(t, x, u, params):
    return x  # return x, y, theta (full state)
