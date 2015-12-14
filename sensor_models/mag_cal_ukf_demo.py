'''Magnetometer calibration demo.

Matt Vernacchia
2015 Dec 14.
'''

import magnetometer
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        mag_data = np.zeros((n_steps, 3))
        for i in xrange(n_steps):
            # Measure in a random orientation
            q = quat_utils.random_quat()
            # q = [1,0,0,0]
            noise = np.random.multivariate_normal(np.zeros(3), mag.noise_cov)
            mag_data[i] = mag.measurement_function(q) + noise


    x = np.zeros((n_steps, 9))
    Q = np.zeros((n_steps, 9, 9))
    for i in xrange(n_steps):
        h_meas = mag_data[i]

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

    # 3D plot of pre- and post-cal data
    mag_data_post_cal = np.zeros(mag_data.shape)
    for i in xrange(n_steps):
        mag_data_post_cal[i] = np.squeeze(np.dot(np.eye(3) + magcal.D, np.array([mag_data[i]]).T))\
            - magcal.b

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot([0,50], [0,0], [0,0], color='red')
    ax3d.plot([0,0], [0,50], [0,0], color='green')
    ax3d.plot([0,0], [0,0], [0,50], color='blue')
    ax3d.scatter(mag_data[:,0], mag_data[:,1], mag_data[:,2], color='cyan', label='Uncalibrated')
    ax3d.scatter(mag_data_post_cal[:,0], mag_data_post_cal[:,1], mag_data_post_cal[:,2], color='magenta', label='Calibrated')
    plt.legend()
    plt.xlabel('Magnetic field [uT]')
    plt.ylabel('Magnetic field [uT]')

    # HACK for equal axes
    # http://stackoverflow.com/a/9349255
    MAX = 80
    for direction in (-1, 1):
        for point in np.diag(direction * MAX * np.array([1,1,1])):
            ax3d.plot([point[0]], [point[1]], [point[2]], 'w')

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