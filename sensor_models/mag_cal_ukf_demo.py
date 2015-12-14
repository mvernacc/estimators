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
import argparse
import cPickle as pickle



def main(args):    
    mag = magnetometer.Magnetometer()
    magcal = magnetometer.MagCalUKF(mag.noise_cov, np.linalg.norm(mag.h_earth_ned))

    if args.meas_source == 'pickle':
        with open(args.pkl_file, 'rb') as f:
            data = pickle.load(f)
        mag_data = data['mag_data']
        np.random.shuffle(mag_data)
        n_steps = mag_data.shape[0]
    elif args.meas_source == 'sim':
        n_steps = 50

    x = np.zeros((n_steps, 9))
    Q = np.zeros((n_steps, 9, 9))
    for i in xrange(n_steps):
        if args.meas_source == 'pickle':
            h_meas = mag_data[i]
        elif args.meas_source == 'sim':                
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

    if args.meas_source == 'sim':
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
        if args.meas_source == 'sim':
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
        if args.meas_source == 'sim':
            plt.axhline(y=mag.D[index_map[i]], color=plot_colors[i],
                label='D[{:d},{:d}] true'.format(*index_map[i]))
        plot_single_state_vs_time(ax2, t, x, Q, i+3,
            color=plot_colors[i], label='D[{:d},{:d}] est'.format(*index_map[i]),
            linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Magnetometer scale factors [-]')
    plt.legend(framealpha=0.5)

    # Save the magnetometer calibration data.
    with open('mag_cal.p', 'wb') as f:
        pickle.dump({
            'b': magcal.b,
            'D': magcal.D,
            'covar': magcal.ukf.Q
            }, f)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Magnetometer calibration.')
    parser.add_argument('--meas_source', type=str, choices=['sim', 'pickle'],
        required=True)
    parser.add_argument('--pkl_file', type=str, required=False,
        help='The pickle file containing the measurement data. Required if --meas_source is "pickle".')
    args = parser.parse_args()
    if args.meas_source == 'pickle' and args.pkl_file is None:
        parser.error('--pkl_file is required if --meas_source is "pickle"')
    main(args)