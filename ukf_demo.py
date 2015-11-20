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

def main():
    gps = GPS()
    baro = Barometer()
    magneto = Magnetometer()
    ks = KalmanSensors([gps, baro, magneto], [[0, 1, 2], [2], [3]])

    # Intitial true state
    x_init = np.array([0, 0, -1000, 0])
    # Initial state estimate
    x_est_init = np.array([100, -100, -500, -1])
    # Initial estimate covariance
    Q_init = np.diag([10, 10, 10, 0.1])**2

    ukf = pykalman.AdditiveUnscentedKalmanFilter(
        discrete_dubin_dynamics,
        ks.measurement_function,
        np.diag([0.1, 0.1, 0.1, 0.01])**2,
        ks.noise_cov,
        x_est_init,
        Q_init
        )

    dt = 0.1
    n_steps = 1000
    x_traj = np.zeros((n_steps, len(x_init)))
    x_traj[0] = x_init
    x_est_traj = np.zeros((n_steps, len(x_init)))
    x_est_traj[0] = x_est_init
    t_traj = np.zeros(n_steps)

    x_est = x_est_init
    Q = Q_init
    for i in xrange(1, n_steps):
        # Get measurements.
        y = ks.add_noise(ks.measurement_function(x_traj[i-1]))
        # Update Kalman filter estimate.
        (x_est, Q) = ukf.filter_update(x_est, Q, y)
        x_est_traj[i] = x_est
        # Simulate the true dynamics.
        x_soln = odeint(continuous_dubin_parafoil, x_traj[i-1],
            [0, dt], args=(0.01,))
        x_traj[i] = x_soln[-1]
        t_traj[i] = t_traj[i-1] + dt


    plt.subplot(3, 1, 1)
    plt.plot(t_traj, x_traj[:, 3], color='blue', label='true')
    plt.plot(t_traj, x_est_traj[:, 3], color='red', label='est')
    plt.xlabel('time')
    plt.ylabel('heading')

    plt.subplot(3, 1, 2)
    plt.plot(t_traj, x_traj[:, 2], color='blue', label='true')
    plt.plot(t_traj, x_est_traj[:, 2], color='red', label='est')
    plt.xlabel('time')
    plt.ylabel('altitude [meter]')

    plt.subplot(3, 1, 3)
    plt.plot(x_traj[:, 1], x_traj[:, 0], color='blue', label='true')
    plt.plot(x_traj[:, 1], x_est_traj[:, 0], color='red', label='est')
    plt.xlabel('x1 (West)')
    plt.ylabel('x0 (North)')
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    main()