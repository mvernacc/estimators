''' Rate gyroscope sensor test.

Parafoil GNC
Matt Vernacchia
MIT Rocket Team
2015 Nov 22
'''

import numpy as np
from matplotlib import pyplot as plt
import time
import cPickle as pickle
import argparse

from rate_gyro import RateGyro

def main(args):
    rg = RateGyro()

    n_steps = args.n_steps
    y = np.zeros((n_steps, 3))
    y_noisy = np.zeros((n_steps, 3))
    bias = np.zeros((n_steps, 3))
    w_true = np.zeros((n_steps, 3))

    for i in xrange(n_steps):
        bias[i] = rg.bias
        y[i] = rg.measurement_function(w_true[i])
        y_noisy[i] = rg.add_noise(y[i])
        rg.update_sensor_state(w_true[i])

    if args.s:
        # Save to a pickle
        filename = 'rate_gyro_test_{:s}.p'.format(
            time.strftime('%Y-%m-%d_%H-%M-%S'))
        with open(filename, 'wb') as f:
            pickle.dump({
                'y': y,
                'y_noisy': y_noisy,
                'bias': bias,
                'rate_gyro': rg
                }, f)

    plt.subplot(2, 1, 1)
    axis_colors = ['red', 'green', 'blue']
    for i in xrange(3):
        plt.plot(y[:, i], color=axis_colors[i], label='meas {:d} no noise'.format(i))
        plt.plot(y_noisy[:, i], color=axis_colors[i], marker='x',
            linestyle=':', label='meas {:d} noisy'.format(i))
    plt.xlabel('step')
    plt.ylabel('Measurement [rad / s]')
    plt.legend(framealpha=0.5)

    plt.subplot(2, 1, 2)
    for i in xrange(3):
        plt.plot(bias[:, i], color=axis_colors[i], label='bias {:d}'.format(i))
    plt.xlabel('step')
    plt.ylabel('Bias [rad / s]')
    plt.legend(framealpha=0.5)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the rate_gyro; optinally save the data to a pickle for later Allan Variance testing.')
    parser.add_argument('-s', action='store_true', help='Save the data to a pickle file.')
    parser.add_argument('-n', '--n_steps', type=int, help='Number of timesteps.', default=1000)
    args = parser.parse_args()
    main(args)