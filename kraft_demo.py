''' Sensor simulation test.

Matt Vernacchia
2015 Nov 30
'''

from matplotlib import pyplot as plt
import numpy as np
import pykalman
from scipy.integrate import odeint
import transforms3d.quaternions as quat

from sensors.sensor_interface import KalmanSensors
from sensors.magnetometer import Magnetometer
from sensors.rate_gyro import RateGyro

from kraft_quat_ukf import KraftQautUKF
from utils import quat_utils
from utils.plot_utils_16322 import plot_single_state_vs_time


def rotation_dynamics(x, u, dt=0.01):
    # Attitude quaternion
    q = x[0:4]
    # Angular rate vector
    w = x[4:7]
    
    q_next = quat_utils.quat_propagate(q, w, dt)
    w_next = w + u * dt

    x_next = np.hstack((q_next, w_next))
    return x_next


def main():
    np.set_printoptions(precision=3)
    dt=0.01
    # Create the sensors for the Kalman filter estimator (known bias parameters).
    magneto_est = Magnetometer(h_bias_ned=[0, 0, 0], h_bias_sensor=[0, 0, 0])
    gyro_est = RateGyro(constant_bias=[0,0,0], bias_walk_process_std_dev=0,
        dt=dt)

    # System process noise covariance
    process_std_dev = np.hstack((np.deg2rad([1, 1, 1])*dt, 
        np.deg2rad([10, 10, 10])*dt))
    W = np.diag(process_std_dev)**2

    # Number of system states.
    n_system_states = 7

    est_sensors = KalmanSensors([gyro_est],
        [[4, 5, 6]], n_system_states)

    # Create the sensors for the simulation (unknown, random bias parameters). 
    gyro_sim = RateGyro(dt=dt)
    sim_sensors = KalmanSensors([gyro_sim],
        [[4, 5, 6]], n_system_states)

    # Initial true state
    x_init = np.array([1., 0., 0., 0., 0., 0., 0.])

    # Initial state estimate. Set the sensor bias states
    # to an initial estimate of zero.
    x_est_init = np.array([1., 0., 0., 0., 0., 0., 0.])

    # Initial estimate covariance.
    # The std. dev. uncertainty of the initial system state
    # estimate
    system_state_init_std_dev = np.hstack((np.deg2rad([1., 1., 1.]), 
        np.deg2rad([1., 1., 1.])))
    # The std. dev. uncertainty of the sensor bias states.
    sensor_state_init_std_dev = []
    Q_init = np.diag(system_state_init_std_dev)**2

    # Create the Kalman Filter
    ukf = KraftQautUKF(
        x_est_init,
        Q_init,
        lambda x, u: rotation_dynamics(x, u, dt),
        W,
        est_sensors.measurement_function,
        est_sensors.noise_cov,
        )

    u = np.deg2rad([10.0, 0, 0])

    n_steps = 100
    x_traj = np.zeros((n_steps, len(x_init)))
    x_traj[0] = x_init
    x_est_traj = np.zeros((n_steps, len(x_est_init)))
    x_est_traj[0] = x_est_init
    t_traj = np.zeros(n_steps)
    Q_traj = np.zeros((n_steps, len(x_est_init)-1, len(x_est_init)-1))

    x_est = x_est_init
    Q = Q_init
    for i in xrange(1, n_steps):
        # Get measurements.
        y = sim_sensors.add_noise(sim_sensors.measurement_function(x_traj[i-1]))
        # Update Kalman filter estimate.
        ukf.propagate_dynamics(u)
        ukf.update_measurement(y)
        x_est_traj[i] = ukf.x_est
        Q_traj[i] = ukf.Q
        # Simulate the true dynamics.
        x_traj[i] = rotation_dynamics(x_traj[i-1], u)
        t_traj[i] = t_traj[i-1] + dt


    print 'Final state est = '
    print ukf.x_est
    print 'Final estimate covariance Q = '
    print ukf.Q

    ax = plt.subplot(2, 1, 1)
    colors = ['black', 'red', 'green', 'blue']
    for i in xrange(4):
        plt.plot(t_traj, x_traj[:, i], color=colors[i], linestyle='-',
            label='q[{:d}] true'.format(i))
        plt.plot(t_traj, x_est_traj[:, i], color=colors[i], linestyle='--',
            label='q[{:d}] est'.format(i), marker='x')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.legend(framealpha=0.5)
    
    ax = plt.subplot(2, 1, 2)
    Q_traj_padded = np.concatenate((
        np.zeros((n_steps, len(x_est_init), 1)),
        np.concatenate((
            np.zeros((n_steps, 1, len(x_est_init)-1)),
            Q_traj), axis=1)
        ), axis=2)
    for i in [0, 1, 2]:
        plt.plot(t_traj, x_traj[:, i+4], color=colors[i+1], linestyle='-',
            label='w[{:d}] true'.format(i))
        plot_single_state_vs_time(ax, t_traj, x_est_traj, Q_traj_padded, i+4,
            color=colors[i+1], label='w[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular rate [rad / s]')
    plt.legend(framealpha=0.5)

    plt.show()


if __name__ == '__main__':
    main()