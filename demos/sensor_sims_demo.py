"""Sensor simulation test.

Parafoil GNC
MIT Rocket Team
2015 Nov 19
"""

from matplotlib import pyplot as plt
import numpy as np

from estimators.sensor_models.sensor_interface import KalmanSensors
from estimators.sensor_models.gps import GPS
from estimators.sensor_models.barometer import Barometer
from estimators.sensor_models.magnetometer import Magnetometer


def main():
    gps = GPS()
    baro = Barometer()
    magneto = Magnetometer()
    ks = KalmanSensors([gps, baro, magneto], [[0, 1, 2], [2], [3]], 4)

    x = [0, 0, 0, 0]
    y = np.zeros((1000, ks.noise_cov.shape[0]))

    for i in range(1000):
        y[i] = ks.add_noise(ks.measurement_function(x))

    plt.subplot(2, 2, 1)
    plt.scatter(y[:, 0], y[:, 1], marker="x")
    plt.xlabel("y0: GPS horiz [meter]")
    plt.ylabel("y1: GPS horiz [meter]")
    plt.axis("equal")

    plt.subplot(2, 2, 2)
    plt.scatter(y[:, 0], y[:, 2], marker="x")
    plt.xlabel("y0: GPS horiz [meter]")
    plt.ylabel("y2: GPS vertical [meter]")
    plt.axis("equal")

    plt.subplot(2, 2, 3)
    plt.scatter(y[:, 2], y[:, 3], marker="x")
    plt.xlabel("y2: GPS vertical [meter]")
    plt.ylabel("y3: baro vertical [pascal]")

    plt.subplot(2, 2, 4)
    plt.scatter(y[:, 4] * 1e6, y[:, 5] * 1e6, marker="x")
    plt.xlabel("y4: magneto north [microtesla]")
    plt.ylabel("y5: magneto east [microtesla]")

    plt.show()


if __name__ == "__main__":
    main()
