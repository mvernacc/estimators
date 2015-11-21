''' Sensor bias estimation.

Augment the system state to include estimation of sensor biases.

Parafoil GNC
MIT Rocket Team
2015 Nov 20
'''

import numpy as np

from dynamics.dubin import discrete_dubin_dynamics


def discrete_dubin_dynamics_with_bias(x, **kwargs):
    '''Augmented dynamics with bias.

    Arguments:
        x (real vector): Bias-augmented state
            [LNED_north, LNED_east, LNED_down, heading,
            baro.bias_pressure, magneto.h_bias_total_in_sensor_frame[0:3]].
    '''
    x = np.array(x)
    x_next = np.zeros(x.shape)

    # The un-augmented state changes according to the normal dynamics
    x_next[0:4] = discrete_dubin_dynamics(x[0:4], **kwargs)
    # The bias state is constant.
    x_next[4:] = x[4:]

    return x_next