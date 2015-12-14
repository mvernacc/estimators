'''Magnetometer calibration demo.

Matt Vernacchia
2015 Dec 14.
'''

import magnetometer
import numpy as np
from matplotlib import pyplot as plt
import transforms3d.quaternions as quat
from estimators.utils import quat_utils
from estimators.utils.plot_utils_16322 import plot_single_state_vs_time



def main():    
    mag = magnetometer.Magnetometer()
    magcal = magnetometer.MagCalUKF(mag.noise_cov, np.linalg.norm(mag.h_earth_ned))

    n_steps = 50
    x = np.zeros((n_steps, 9))
    Q = np.zeros((n_steps, 9, 9))
    for i in xrange(n_steps):
        # Measure in a random orientation
        q = quat_utils.random_quat()
        # q = [1,0,0,0]
        noise = np.random.multivariate_normal(np.zeros(3), mag.noise_cov)
        h_meas = mag.measurement_function(q) + noise

        # Update the estimator
        magcal.update(h_meas)

        # Record the estimator state
        x[i] = magcal.ukf.x_est
        Q[i] = magcal.ukf.Q

    print 'True b, D:'
    print mag.b
    print mag.D
    print '\nEstimated b, D:'
    print magcal.b
    print magcal.D

    plot_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange']
    t = range(n_steps)
    ax1 = plt.subplot(2,1,1)
    for i in xrange(3):
        plt.axhline(y=mag.b[i], color=plot_colors[i], label='b[{:d}] true'.format(i))
        plot_single_state_vs_time(ax1, t, x, Q, i,
            color=plot_colors[i], label='b[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Magnetometer bias [uT]')
    plt.legend(framealpha=0.5)

    ax2 = plt.subplot(2,1,2)
    index_map = {
        0: (0,0),
        1: (1,1),
        2: (2,2),
        3: (0,1),
        4: (0,2),
        5: (1,2)
    }
    for i in xrange(6):
        plt.axhline(y=mag.D[index_map[i]], color=plot_colors[i], label='D[{:d},{:d}] true'.format(*index_map[i]))
        plot_single_state_vs_time(ax2, t, x, Q, i+3,
            color=plot_colors[i], label='D[{:d},{:d}] est'.format(*index_map[i]),
            linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Magnetometer scale factors [-]')
    plt.legend(framealpha=0.5)

    plt.show()


if __name__ == '__main__':
    main()