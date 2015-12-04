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
    dt=0.1
    # Create the sensors for the Kalman filter estimator (known bias parameters).
    magneto_est = Magnetometer(h_bias_ned=[0, 0, 0], h_bias_sensor=[0, 0, 0])
    magneto_est.is_stateful = False
    gyro_est = RateGyro(constant_bias=[0,0,0],
        dt=dt)

    # System process noise covariance
    process_std_dev = np.hstack((np.deg2rad([10, 10, 10])*dt, 
        np.deg2rad([10, 10, 10])*dt))
    W = np.diag(process_std_dev)**2

    # Number of system states.
    n_system_states = 7
    # Number of sensor bias states.
    n_sensor_states = 3


    est_sensors = KalmanSensors([gyro_est, magneto_est],
        [[4, 5, 6], [0, 1, 2, 3]], n_system_states,
        [[7, 8, 9], []], n_sensor_states,
        lambda x, u: rotation_dynamics(x, u, dt),
        W,
        1)

    # Create the sensors for the simulation (unknown, random bias parameters). 
    gyro_sim = RateGyro(dt=dt)
    magneto_sim = Magnetometer(h_bias_ned=[0, 0, 0], h_bias_sensor=[0, 0, 0])
    magneto_sim.is_stateful = False
    sim_sensors = KalmanSensors([gyro_sim, magneto_sim],
        [[4, 5, 6], [0, 1, 2, 3]], n_system_states)

    # Initial true state
    x_init = np.array([1., 0., 0., 0., 0., 0., 0.])

    # Initial state estimate. Set the sensor bias states
    # to an initial estimate of zero.
    x_est_init = np.concatenate(([0.76, 0.76, 0., 0., 0., 0., 0.],
        np.zeros(n_sensor_states)))

    # Initial estimate covariance.
    # The std. dev. uncertainty of the initial system state
    # estimate
    system_state_init_std_dev = np.hstack((np.deg2rad([30., 30., 30.]), 
        np.deg2rad([0.1, 0.1, 0.1])))
    # The std. dev. uncertainty of the sensor bias states.
    sensor_state_init_std_dev = np.deg2rad([5., 5., 5.])
    Q_init = np.diag(np.concatenate((
        system_state_init_std_dev,
        sensor_state_init_std_dev
        )))**2

    # Create the Kalman Filter
    ukf = KraftQautUKF(
        x_est_init,
        Q_init,
        est_sensors.augmented_transition_function,
        est_sensors.augmented_process_covariance,
        est_sensors.measurement_function,
        est_sensors.noise_cov,
        )

    n_steps = 500
    x_traj = np.zeros((n_steps, len(x_init)))
    x_traj[0] = x_init
    x_est_traj = np.zeros((n_steps, len(x_est_init)))
    x_est_traj[0] = x_est_init
    t_traj = np.zeros(n_steps)
    Q_traj = np.zeros((n_steps, len(x_est_init)-1, len(x_est_init)-1))

    y_traj = np.zeros((n_steps, len(sim_sensors.measurement_function(x_init))))

    gyro_bias_traj = np.zeros((n_steps, 3))
    x_est = x_est_init
    Q = Q_init

    u_traj = np.zeros((n_steps, 3))
    for i in xrange(100, 110):
        u_traj[i] = np.deg2rad([10.0, 0, 0])

    for i in xrange(1, n_steps):
        # Update sensor state
        sim_sensors.update_sensor_state(x_traj[i-1])
        gyro_bias_traj[i] = gyro_sim.bias + gyro_sim.constant_bias
        # Get measurements.
        y_traj[i] = sim_sensors.add_noise(sim_sensors.measurement_function(x_traj[i-1]))
        # Update Kalman filter estimate.
        ukf.propagate_dynamics(u_traj[i])
        ukf.update_measurement(y_traj[i])
        x_est_traj[i] = ukf.x_est
        Q_traj[i] = ukf.Q
        # Simulate the true dynamics.
        x_traj[i] = rotation_dynamics(x_traj[i-1], u_traj[i], dt)
        t_traj[i] = t_traj[i-1] + dt


    print 'Final state est = '
    print ukf.x_est
    print 'Final estimate covariance Q = '
    print ukf.Q

    ax = plt.subplot(2, 2, 1)
    colors = ['black', 'red', 'green', 'blue']
    for i in xrange(4):
        plt.plot(t_traj, x_traj[:, i], color=colors[i], linestyle='-',
            label='q[{:d}] true'.format(i))
        plt.plot(t_traj, x_est_traj[:, i], color=colors[i], linestyle='--',
            label='q[{:d}] est'.format(i), marker='x')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.legend(framealpha=0.5)
    
    ax2 = plt.subplot(2, 2, 2, sharex=ax)
    Q_traj_padded = np.concatenate((
        np.zeros((n_steps, len(x_est_init), 1)),
        np.concatenate((
            np.zeros((n_steps, 1, len(x_est_init)-1)),
            Q_traj), axis=1)
        ), axis=2)
    for i in [0, 1, 2]:
        plt.plot(t_traj, x_traj[:, i+4], color=colors[i+1], linestyle='-',
            label='w[{:d}] true'.format(i))
        plot_single_state_vs_time(ax2, t_traj, x_est_traj, Q_traj_padded, i+4,
            color=colors[i+1], label='w[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular rate [rad / s]')
    plt.legend(framealpha=0.5)

    ax3 = plt.subplot(2, 2, 3, sharex=ax)
    for i in xrange(3):
        plt.plot(t_traj, y_traj[:, i + 3], color=colors[i+1], marker='x',
            label='mag[{:d}]'.format(i))
    plt.xlabel('Time [s]')
    plt.ylabel('Mag Field [uT]')
    plt.legend(framealpha=0.5)

    ax4 = plt.subplot(2, 2, 4, sharex=ax)
    for i in [0, 1, 2]:
        plt.plot(t_traj, gyro_bias_traj[:, i], color=colors[i+1], linestyle='-',
            label='b[{:d}] true'.format(i))
        plot_single_state_vs_time(ax4, t_traj, x_est_traj, Q_traj_padded, i+7,
            color=colors[i+1], label='b[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Gyro bias [rad / s]')
    plt.legend(framealpha=0.5)

    plt.show()


if __name__ == '__main__':
    main()