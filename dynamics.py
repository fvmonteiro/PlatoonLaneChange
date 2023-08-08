import numpy as np


def run_test_simulation(x0, time, n_vehs):
    states = np.zeros([len(x0), len(time)])
    states[:, 0] = x0
    params = {'n_vehs': n_vehs}
    steering_angle = np.zeros([n_vehs, len(time)])
    for i in range(len(time)-1):
        if time[i] <= 2:
            steering_angle[:, i] = 0.005 - 0.005/2 * time[i]
        elif time[i] <= 8:
            steering_angle[:, i] = 0
        else:
            steering_angle[:, i] = - 0.005/2 * (time[i] - 8)
        dxdt = vehicle_update(time[i], states[:, i], steering_angle[:, i],
                              params)
        states[:, i+1] = states[:, i] + dxdt*(time[i+1] - time[i])
    return states, steering_angle


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
    vehicle_type = params['vehicle_type']
    n_vehs = sum(params['n_per_lane'])

    # Return the derivative of the state
    dxdt = np.zeros(len(states))
    n_states = vehicle_type.n_states
    n_inputs = vehicle_type.n_inputs
    for i in range(n_vehs):
        states_idx = [j for j in range(i * n_states, (i + 1) * n_states)]
        inputs_idx = [j for j in range(i * n_inputs, (i + 1) * n_inputs)]
        dxdt[states_idx] = vehicle_type.dynamics(
            states[states_idx], inputs[inputs_idx], params)
    return dxdt


def vehicle_output(t, x, u, params):
    return x  # return x, y, theta (full state)
