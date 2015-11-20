''' Sensor simulation test.

Parafoil GNC
MIT Rocket Team
2015 Nov 20
'''

from matplotlib import pyplot as plt
import numpy as np
import pykalman
from scipy.integrate import odeint

from sensors.sensor_interface import KalmanSensors
from sensors.gps import GPS
from sensors.barometer import Barometer
from sensors.magnetometer import Magnetometer
from dynamics.dubin import continuous_dubin_parafoil, discrete_dubin_dynamics
from utils.plot_utils_16322 import plot_single_state_vs_time


def main():
    # Create the sensors for the kalman filter estimator (known bias parameters).
    gps_est = GPS()
    baro_est = Barometer(bias_pressure=0)
    magneto_est = Magnetometer(h_bias_ned=[0, 0, 0], h_bias_sensor=[0, 0, 0])
    # Introduce ficticious sensor noise to make the estimator better
    # able to deal with unknown sensor bias
    baro_est.noise_cov *= 4
    magneto_est.noise_cov *= 4

    est_sensors = KalmanSensors([gps_est, baro_est, magneto_est], [[0, 1, 2], [2], [3]])

    # Create the sensors for the simulation (unknown, random bias parameters). 
    gps_sim = GPS()
    baro_sim = Barometer()
    magneto_sim = Magnetometer()
    sim_sensors = KalmanSensors([gps_sim, baro_sim, magneto_sim], [[0, 1, 2], [2], [3]])

    # Intitial true state
    x_init = np.array([0, 0, -1000, 0])
    # Initial state estimate
    x_est_init = np.array([100, -100, -500, -1])
    # Initial estimate covariance
    Q_init = np.diag([100, 100, 100, 1])**2

    ukf = pykalman.AdditiveUnscentedKalmanFilter(
        discrete_dubin_dynamics,
        est_sensors.measurement_function,
        np.diag([0.1, 0.1, 0.1, 0.01])**2,
        est_sensors.noise_cov,
        x_est_init,
        Q_init
        )

    dt = 0.1
    n_steps = 100
    x_traj = np.zeros((n_steps, len(x_init)))
    x_traj[0] = x_init
    x_est_traj = np.zeros((n_steps, len(x_init)))
    x_est_traj[0] = x_est_init
    t_traj = np.zeros(n_steps)
    Q_traj = np.zeros((n_steps, len(x_init), len(x_init)))

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


    ax = plt.subplot(3, 1, 1)
    plt.plot(t_traj, x_traj[:, 3], color='blue', label='true')
    plot_single_state_vs_time(ax, t_traj, x_est_traj, Q_traj, 3,
        color='red', label='est')
    plt.xlabel('time')
    plt.ylabel('heading [radia]')
    plt.legend()

    ax = plt.subplot(3, 1, 2)
    plt.plot(t_traj, x_traj[:, 2], color='blue', label='true')
    plot_single_state_vs_time(ax, t_traj, x_est_traj, Q_traj, 2,
        color='red', label='est')
    plt.xlabel('time')
    plt.ylabel('altitude [meter]')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x_traj[:, 1], x_traj[:, 0], color='blue', label='true')
    plt.plot(x_traj[:, 1], x_est_traj[:, 0], color='red', label='est')
    plt.xlabel('x1 (West)')
    plt.ylabel('x0 (North)')
    plt.axis('equal')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()