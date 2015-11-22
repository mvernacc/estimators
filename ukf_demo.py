''' Sensor simulation test.

Parafoil GNC
MIT Rocket Team
2015 Nov 20
'''

from matplotlib import pyplot as plt
import numpy as np
import pykalman
from scipy.integrate import odeint
import transforms3d.quaternions as quat

from sensors.sensor_interface import KalmanSensors
from sensors.gps import GPS
from sensors.barometer import Barometer
from sensors.magnetometer import Magnetometer
from dynamics.dubin import continuous_dubin_parafoil, discrete_dubin_dynamics
from utils.plot_utils_16322 import plot_single_state_vs_time
from estimators.bias_est  import discrete_dubin_dynamics_with_bias

def main():
    # Create the sensors for the Kalman filter estimator (known bias parameters).
    gps_est = GPS()
    baro_est = Barometer(bias_pressure=0)
    magneto_est = Magnetometer(h_bias_ned=[0, 0, 0], h_bias_sensor=[0, 0, 0])
    # Introduce fictitious sensor noise to make the estimator better
    # able to deal with unknown sensor bias
    baro_est.noise_cov *= 4
    magneto_est.noise_cov *= 4

    # System process noise covariance
    dubin_dynamics_cov = np.diag([0.1, 0.1, 0.1, 0.01])**2

    # Number of system states.
    n_system_states = 4
    # Number of sensor bias states.
    n_sensor_states = 4

    est_sensors = KalmanSensors([gps_est, baro_est, magneto_est],
        [[0, 1, 2], [2], [3]], n_system_states,
        [[], [4], [5, 6, 7]], n_sensor_states,
        discrete_dubin_dynamics,
        dubin_dynamics_cov)

    # Create the sensors for the simulation (unknown, random bias parameters). 
    gps_sim = GPS()
    baro_sim = Barometer()
    magneto_sim = Magnetometer()
    sim_sensors = KalmanSensors([gps_sim, baro_sim, magneto_sim],
        [[0, 1, 2], [2], [3]], n_system_states)

    # Initial true state
    x_init = np.array([0, 0, -1000, 0])

    # Initial state estimate. Set the sensor bias states
    # to an initial estimate of zero.
    x_est_init = np.concatenate(([100, -100, -500, 1],
        np.zeros(n_sensor_states)))

    # Initial estimate covariance.
    # The std. dev. uncertainty of the initial system state
    # estimate
    system_state_init_std_dev = [100, 100, 100, 1]
    # The std. dev. uncertainty of the sensor bias states.
    sensor_state_init_std_dev = [1e3, 5e-6, 5e-6, 5e-6]
    Q_init = np.diag(np.concatenate((
        system_state_init_std_dev,
        sensor_state_init_std_dev
        )))**2

    # Create the Kalman Filter
    ukf = pykalman.AdditiveUnscentedKalmanFilter(
        est_sensors.augmented_transition_function,
        est_sensors.measurement_function,
        est_sensors.augmented_process_covariance,
        est_sensors.noise_cov,
        x_est_init,
        Q_init
        )

    dt = 0.1
    n_steps = 100
    x_traj = np.zeros((n_steps, len(x_init)))
    x_traj[0] = x_init
    x_est_traj = np.zeros((n_steps, len(x_est_init)))
    x_est_traj[0] = x_est_init
    t_traj = np.zeros(n_steps)
    Q_traj = np.zeros((n_steps, len(x_est_init), len(x_est_init)))

    x_est = x_est_init
    Q = Q_init
    for i in xrange(1, n_steps):
        # Get measurements.
        y = sim_sensors.add_noise(sim_sensors.measurement_function(x_traj[i-1]))
        # Update Kalman filter estimate.
        (x_est, Q) = ukf.filter_update(x_est, Q, y)
        x_est_traj[i] = x_est
        Q_traj[i] = Q
        # Simulate the true dynamics.
        x_soln = odeint(continuous_dubin_parafoil, x_traj[i-1],
            [0, dt], args=(0.01,))
        x_traj[i] = x_soln[-1]
        t_traj[i] = t_traj[i-1] + dt


    ax = plt.subplot(2, 3, 1)
    plt.plot(t_traj, x_traj[:, 3] * 180 / np.pi, color='blue', label='true')
    plot_single_state_vs_time(ax, t_traj,
        x_est_traj * 180 / np.pi, Q_traj * (180 / np.pi)**2, 3,
        color='red', label='est')
    plt.xlabel('time')
    plt.ylabel('heading [degree]')
    plt.legend()

    ax = plt.subplot(2, 3, 2)
    plt.plot(t_traj, x_traj[:, 2], color='blue', label='true')
    plot_single_state_vs_time(ax, t_traj, x_est_traj, Q_traj, 2,
        color='red', label='est')
    plt.xlabel('time')
    plt.ylabel('altitude [meter]')
    plt.legend(framealpha=0.5)

    plt.subplot(2, 3, 3)
    plt.plot(x_traj[:, 1], x_traj[:, 0], color='blue', label='true')
    plt.plot(x_traj[:, 1], x_est_traj[:, 0], color='red', label='est')
    plt.xlabel('x1 (West)')
    plt.ylabel('x0 (North)')
    plt.axis('equal')
    plt.legend(framealpha=0.5)

    ax = plt.subplot(2, 3, 4)
    plt.plot(t_traj, [baro_sim.bias_pressure]*len(t_traj), color='blue', label='true')
    plot_single_state_vs_time(ax, t_traj, x_est_traj, Q_traj, 4,
        color='red', label='est')
    plt.xlabel('time')
    plt.ylabel('Barometer bias [pascal]')
    plt.legend(framealpha=0.5)

    ax = plt.subplot(2, 3, 5)
    h_bias_sensor = np.zeros((len(t_traj), 3))
    for i in xrange(len(t_traj)):
        heading = x_traj[i,3]
        q_ned2sensor = quat.axangle2quat([0, 0, 1], heading)
        h_bias_sensor[i] = magneto_sim.h_bias_sensor \
            + quat.rotate_vector(magneto_sim.h_bias_ned, q_ned2sensor)

    plt.plot(t_traj, h_bias_sensor[:,0] * 1e6, color='red', label='true[0]')
    plot_single_state_vs_time(ax, t_traj, x_est_traj * 1e6, Q_traj * 1e12, 5,
        color='red', label='est[0]', linestyle='--')

    plt.plot(t_traj, h_bias_sensor[:,1] * 1e6, color='green', label='true[1]')
    plot_single_state_vs_time(ax, t_traj, x_est_traj * 1e6, Q_traj * 1e12, 6,
        color='green', label='est[1]', linestyle='--')

    plt.plot(t_traj, h_bias_sensor[:,2] * 1e6, color='blue', label='true[2]')
    plot_single_state_vs_time(ax, t_traj, x_est_traj * 1e6, Q_traj * 1e12, 7,
        color='blue', label='est[2]', linestyle='--')

    plt.xlabel('time')
    plt.ylabel('Magneto total bias in sensor frame [microtesla]')
    plt.legend(framealpha=0.5)

    plt.show()


if __name__ == '__main__':
    main()