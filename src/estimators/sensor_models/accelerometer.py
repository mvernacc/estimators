"""Accelerometer sensor.

Parafoil GNC
MIT Rocket Team
2015 Nov 19
"""

import numpy as np
from .sensor_interface import StatefulSensor
from estimators.utils import quat_utils


class Accelerometer(StatefulSensor):
    def __init__(self, a_bias_sensor=None, noise_std_dev=4 * 9.81e-3):
        """Accelerometer sensor model.

        Arguments:
            a_bias_sensor (3-vector): The acceleration bias vector in the sensor
                coordinate system. If h_bias_ned has the default value of None,
                it will be initialized to a random value.
                [units: meter second**-2].

            noise_std_dev (positive real): The standard deviation of the sensor noise.
                [units: meter second**-2]. The default value is taken from Section 6.2 of
                [1], which specifies and rms noise of 4 mg.


        References:
            [1] 'MPU-9150 Product Specification Revision 4.3,' InvenSense, Sunnyvale, CA, US.
                Online: https://cdn.sparkfun.com/datasheets/Sensors/IMU/MPU-9150-Datasheet.pdf
        """
        # Acceleration due to gravity in the North, East, Down frame.
        self.g_earth_ned = np.array([0, 0, 9.81])

        # The zero - G output spec in [1] is +/- 150 mg.
        if a_bias_sensor is None:
            self.a_bias_sensor = np.random.normal(0, 150 * 9.81e-3, size=3)
        else:
            self.a_bias_sensor = np.array(a_bias_sensor)
            assert len(self.a_bias_sensor) == 3

        noise_cov = np.diag([(noise_std_dev) ** 2] * 3)
        sensor_state_process_covariance = np.diag([0.5, 0.5, 0.5]) ** 2
        super(self.__class__, self).__init__(noise_cov, sensor_state_process_covariance)

    def measurement_function(self, x, sensor_state=None):
        """Magnetometer state-to-measurement function.

        Arguments:
            x (real vector): The system state, which contains the following values:
                If len(x) == 4:
                * q_LNED_to_sensor real component.
                * q_LNED_to_sensor i component.
                * q_LNED_to_sensor j component.
                * q_LNED_to_sensor k component.

            sensor_state (optional, real vector): The sensor state, which contains
                the following values:
                * The acceleration bias in the sensor frame x direction
                    [units: meter second**-2].
                * The acceleration bias in the sensor frame y direction
                    [units: meter second**-2].
                * The acceleration bias in the sensor frame z direction
                    [units: meter second**-2].


        Returns:
            (real vector): The measurement, acceleration vector in the sensor
                frame [units: meter second**-2].
        """
        x = np.array(x)
        if len(x) == 4:
            q_ned2sensor = x
        else:
            raise ValueError

        a_bias_sensor = self.a_bias_sensor
        if sensor_state is not None:
            assert len(sensor_state) == 3
            a_bias_sensor = sensor_state

        g_earth_sensor = quat_utils.rotate_frame(-self.g_earth_ned, q_ned2sensor)
        y = g_earth_sensor + a_bias_sensor
        return y

    def sensor_state_transition(self, x, sensor_state):
        """Transition function for the sensor's state.

        Arguments:
             x (real vector): The system state, which is not used by this function
                 and is included only for compatibility with the super-class.

             sensor_state (real vector): The sensor state, which contains
                 the following values:
                 * The acceleration bias in the sensor frame x direction
                     [units: meter second**-2].
                 * The acceleration bias in the sensor frame y direction
                     [units: meter second**-2].
                 * The acceleration bias in the sensor frame z direction
                     [units: meter second**-2].

         Returns:
             real vector: The new sensor state, which contains the following values:
                 * The acceleration bias in the sensor frame x direction
                     [units: meter second**-2].
                 * The acceleration bias in the sensor frame y direction
                     [units: meter second**-2].
                 * The acceleration bias in the sensor frame z direction
                     [units: meter second**-2].
        """
        a_bias_sensor = sensor_state
        return a_bias_sensor
