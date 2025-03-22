"""Rate gyroscope sensor.

Parafoil GNC
Matt Vernacchia
MIT Rocket Team
2015 Nov 22
"""

import numpy as np
from .sensor_interface import StatefulSensor


class RateGyro(StatefulSensor):
    def __init__(
        self,
        rate_noise_std_dev=5e-4,
        constant_bias=None,
        scale_factor=[0, 0, 0],
        correlation_time=7.0 * 3600,
        bias_walk_process_std_dev=4e-4,
        dt=1e-2,
    ):
        """Rate gyro sensor model.

        The sensor model is taken from the appendix of [1]:
            w_meas(t) = (1 + diag(k)) * w_true(t) + b(t) + d + n_v(t)
            db/dt = - a b(t) + n_u(t)
        where:
            k is the scale factor error,
            b(t) is the time-varying bias,
            d is the constant bias,
            n_v is the rate noise, a random variable drawn from a zero mean gaussian,
            a is the inverse of the correlation time,
            and n_u is the bias walk noise, a random variable drawn from a zero
                mean gaussian.

        In this model, the evolution of the time-varying bias is a Gauss-Markov
        process.

        The sensor model parameters in the arguments are in their continuous
        time form. The measurement interval is also given as an argument,
        and is used to convert the model parameters to their discrete form.

        Arguments:
            rate_noise_std_dev (positive real): The standard deviation of the
                rate noise [units: radian second**-0.5]. The default value is
                taken from the L3GD20 spec sheet [2]. This value is also known
                as the 'angle random walk'.

            constant_bias (real 3-vector): The constant angular rate bias [units:
                radian second**-1]. If the default value of None is given, the constant
                bias will be set to a random value.

            scale_factor (real 3-vector): The scale factor error on each axis
                [units: none].

            correlation_time (positive real): The correlation time of the bias
                Gauss-Markov process [units: second]. The default value is taken
                from table 3 of Bhatt et al [3].

            bias_walk_process_std_dev (positive real): The standard deviation
                of the bias Gauss-Markov process [units: radian second **-1].
                Note that the variance of n_u(t) is 2 / correlation_time *
                bias_walk_process_std_dev**2. The default value is taken
                from table 3 of Bhatt et al [3].

            dt (positive real): Measurement interval [units: second].

        References:
            [1] J. Crassidis, 'Sigma-Point Kalman Filtering for Integrated GPS
                and Inertial Navigation,' SUNY Buffalo, Amherst, NY, 2005.
                Online: http://www.acsu.buffalo.edu/~johnc/gpsins_gnc05.pdf
            [2] 'L3GD20 MEMS motion sensor: three-axis digital output gyroscope,'
                ST Microelectronics, Feb 2013.
                Online: http://www.st.com/web/en/resource/technical/document/datasheet/DM00036465.pdf
            [3] D. Bhatt, P. Aggarwal, P. Bhattacharya, V Devabhaktuni, 'An
                Enhanced MEMS Error Modeling Approach Based on Nu-Support Vector
                Regression,' MDPI Sensors, July 2012.
        """

        assert rate_noise_std_dev >= 0, "rate_noise_std_dev must be positive or zero."
        self.rate_noise_std_dev = rate_noise_std_dev

        if constant_bias is None:
            # The variance of the constant bias is taken from figure 1 in
            # Bhatt et al [3].
            constant_bias = np.random.multivariate_normal(
                np.zeros(3), np.diag([np.deg2rad(0.2)] * 3) ** 2
            )
        assert len(constant_bias) == 3, "constant_bias must be a 3-vector."
        self.constant_bias = np.array(constant_bias)

        assert len(scale_factor) == 3, "scale_factor must be a 3-vector."
        self.scale_factor = np.array(scale_factor)

        assert correlation_time > 0, "correlation_time must be positive."
        self.correlation_time = correlation_time

        assert bias_walk_process_std_dev >= 0, (
            "bias_walk_process_std_dev must be positive or zero."
        )
        self.bias_walk_process_std_dev = bias_walk_process_std_dev

        assert dt > 0, "dt must be positive."
        self.dt = dt

        # Initialize the internal sensor state.
        self.bias = np.zeros(3)

        # The covariance of the rate noise n_v(t).
        noise_cov = np.diag([self.rate_noise_std_dev] * 3) ** 2
        # Convert to discrete time.
        noise_cov *= 1 / self.dt

        # The covariance of the bias walk noise n_u(t).
        sensor_state_process_covariance = np.diag(
            [2.0 / self.correlation_time * self.bias_walk_process_std_dev**2] * 3
        )
        # Convert to discrete time.
        sensor_state_process_covariance *= self.dt

        # Pass the noise parameters to the parent class.
        super(self.__class__, self).__init__(noise_cov, sensor_state_process_covariance)

    def measurement_function(self, x, sensor_state=None):
        """Rate gyroscope state-to-measurement function.

        Arguments:
            x (real vector): The system state, which contains the following values:
                * rotational rate about the sensor x axis [units: radian second**-1].
                * rotational rate about the sensor y axis [units: radian second**-1].
                * rotational rate about the sensor z axis [units: radian second**-1].

            sensor_state (optional, real vector): The sensor state, which contains
                the following values:
                * bias about the sensor x axis [units: radian second**-1].
                * bias about the sensor y axis [units: radian second**-1].
                * bias about the sensor z axis [units: radian second**-1].

        Returns:
            (real vector): The measurement, which contains the following values:
                * rotational rate about the sensor x axis [units: radian second**-1].
                * rotational rate about the sensor y axis [units: radian second**-1].
                * rotational rate about the sensor z axis [units: radian second**-1].
        """
        # If no sensor state is passed, use the internal sensor state.
        b = self.bias
        if sensor_state is not None:
            b = sensor_state
        assert len(b) == 3, "sensor_state must be a 3-vector."

        assert len(x) == 3, "x must be a 3-vector."

        # Use equation 114 a from Crassidis [1], but without noise.
        y = (1 + self.scale_factor) * x + b + self.constant_bias
        return y

    def sensor_state_transition(self, x, sensor_state, noise=False):
        """Transition function for the sensor's state.

        Arguments:
             x (real vector): The system state, which is not used by this function
                 and is included only for compatibility with the super-class.

             sensor_state (real vector): The sensor state, which contains
                 the following values:
                 * bias about the sensor x axis [units: radian second**-1].
                 * bias about the sensor y axis [units: radian second**-1].
                 * bias about the sensor z axis [units: radian second**-1].

             noise (optional boolean): Include process noise in the transition.

         Returns:
             real vector: The new sensor state, which contains the following values:
                 * bias about the sensor x axis [units: radian second**-1].
                 * bias about the sensor y axis [units: radian second**-1].
                 * bias about the sensor z axis [units: radian second**-1].
        """
        # If no sensor state is passed, use the internal sensor state.
        b = self.bias
        if sensor_state is not None:
            b = sensor_state
        assert len(b) == 3, "sensor_state must be a 3-vector."

        # Use equation 114 b from Crassidis [1], but without noise.
        b_dot = -(1.0 / self.correlation_time) * b
        if noise:
            # convert bias walk covariance to continuous time.
            cov_contin = self.sensor_state_process_covariance / self.dt
            n_u = np.random.multivariate_normal(np.zeros(3), cov_contin)
            b_dot += n_u
        # Time integration with forward Euler because I'm lazy. TODO should
        # use Runge-Kutta instead.
        b_new = b + b_dot * self.dt
        return b_new

    def update_sensor_state(self, x):
        """Update the sensor's internal record of the sensor state.

        Arguments:
            x (real vector): The system state, which contains the following values:
                * rotational rate about the sensor x axis [units: radian second**-1].
                * rotational rate about the sensor y axis [units: radian second**-1].
                * rotational rate about the sensor z axis [units: radian second**-1].

        Returns:
            None.
        """
        self.bias = self.sensor_state_transition(
            x=None, sensor_state=self.bias, noise=True
        )
