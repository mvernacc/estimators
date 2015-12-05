''' Sensor interface.

Parafoil GNC
MIT Rocket Team
2015 Nov 19
'''


import numpy as np
import scipy.linalg
import itertools


class KalmanSensors:
    def __init__(self, sensors, system_state_indices, n_system_states,
            sensor_state_indices=None, n_sensor_states=0,
            system_transition_function=None, system_process_covariance=None,
            n_redundant_states=0):
        '''Create a representation of the sensors used in a Kalman Filter.

        This class groups the sensors used in a Kalman filter together into a single
        measurement function and a single sensor noise covariance matrix.

        If some of the sensors are stateful, this class can also create an augmented
        dynamics model, which contains both the system state dynamics and the sensor 
        state dynamics. The augmented dynamics model is represented by the augmented
        state transition function and the augmented process noise covariance matrix.

        Arguments:
            sensors (array-like of Sensor): The filter's sensors.

            system_state_indices (array of arrays of integers): For each sensor,
                the indices in the state vector of the states which that sensor 
                measures. Example: a system has state [a, b, c]. Sensor s1 
                measures a and b
                    y1 <-- s1.measurement_function([a,b])
                and sensor s2 measures c
                    y2 <-- s2.measurement_function([c])
                Then system_state_indices = [[0, 1], [2]].

            n_system_states (positive integer): The number of system states.

            sensor_state_indices (optional, array of arrays of integers): For each
                sensor, the indices in the augmented state vector of that sensor's
                sensor state. Entries corresponding to non-stateful sensors are
                ignored. Example: a system has state [a, b, c]. Sensor s1 is
                stateless, sensor s2 has state d, and sensor s3 has state [e, f]. The
                augmented system state is [a, b, c, d, e, f] and the sensor_state_indices
                are [[], [3], [4,5]]. This argument is only needed if an augmented
                state is to be used for stateful sensors.

            system_transition_function (optional, function): The transition
                function which maps the current system state to the system
                state at the next time step. This argument is only needed if an
                augmented state is to be used for stateful sensors.

            system_process_covariance (optional, real matrix): The process noise
                covariance for the system state transition function.  This
                argument is only needed if an augmented state is to be used for
                stateful sensors.

            n_redundant_states (integer): The number of redunant system states,
                The covariance dimension should be less that the state dimension
                by this amount. e.g. this is 1 for a quaternion estimator.
        '''
        self.sensors = sensors
        self.system_state_indices = system_state_indices
        assert len(self.system_state_indices) == len(self.sensors), \
            'There must be one entry in system_state_indices for each sensor.'
        self.n_system_states = n_system_states
        self.sensor_state_indices = sensor_state_indices
        if self.sensor_state_indices is not None:
            assert len(self.sensor_state_indices) == len(self.sensors), \
            'There must be one entry in sensor_state_indices for each sensor.'
        self.n_sensor_states = n_sensor_states
        self.system_transition_function = system_transition_function
        self.noise_cov = scipy.linalg.block_diag(
            *[s.noise_cov for s in sensors])
        # Build the augmented (system state + sensor state) process noise
        # covariance matrix.
        if system_process_covariance is not None:
            assert system_process_covariance.shape[0] \
                == self.n_system_states - n_redundant_states, \
                'The system_process_covariance matrix must be square with' \
                + ' dimension equal to the number of system states.'
            assert system_process_covariance.shape[1] \
                == self.n_system_states - n_redundant_states, \
                'The system_process_covariance matrix must be square with' \
                + ' dimension equal to the number of system states.'
            cov_blocks = []
            cov_blocks.append(system_process_covariance)
            for sensor in self.sensors:
                if hasattr(sensor, 'is_stateful') and sensor.is_stateful:
                    cov_blocks.append(sensor.sensor_state_process_covariance)
            self.augmented_process_covariance = scipy.linalg.block_diag(
                *cov_blocks)
            assert(self.augmented_process_covariance.shape[0] ==
                self.n_system_states + self.n_sensor_states - n_redundant_states)
            assert(self.augmented_process_covariance.shape[1] ==
                self.n_system_states + self.n_sensor_states - n_redundant_states)


    def measurement_function(self, x):
        '''Kalman Filter measurement function.

        Calls the measurement function for each sensor, and concatenates the results.

        Arguments:
            x (real vector): The filter state. If some of the filter's sensors are
                stateful, the filter state is a concatenation of the system state
                and the sensor states.

        Returns:
            real vector: The sensor measurements, given the system and sensor
                states provided in x.

        '''
        x = np.array(x)
        y = []
        if self.sensor_state_indices is not None:
            # Some sensors are stateful.
            for (sensor, sys_states, sen_states) in \
                    zip(self.sensors, self.system_state_indices, self.sensor_state_indices):
                if hasattr(sensor, 'is_stateful') and sensor.is_stateful:
                    # Stateful sensor: measurement depends on system state and
                    # on sensor state.
                    y.append(sensor.measurement_function(
                        x[sys_states], x[sen_states]))
                else:
                    # Not stateful sensor: measurement only depends on system
                    # state.
                    y.append(sensor.measurement_function(x[sys_states]))
        else:
            # No sensors are stateful.
            for (sensor, sys_states) in zip(self.sensors, self.system_state_indices):
                y.append(sensor.measurement_function(x[sys_states]))
        y = list(itertools.chain(*y))
        return y


    def augmented_transition_function(self, x, u):
        '''State transition function of the augmented (system + sensor) filter state.

        Calls the system state transition function and the sensor state transition
        function for each sensor, and concatenates the results.

        Arguments:
            x (real vector): The filter state. If some of the filter's sensors are
                stateful, the filter state is a concatenation of the system state
                and the sensor states.

        Returns:
            real vector: The new filter state, given the system and sensor
                states provided in x.
        '''
        assert(len(x) == self.n_sensor_states + self.n_system_states)
        x_next = []
        x_next.append(self.system_transition_function(x[:self.n_system_states], u))
        if self.sensor_state_indices is not None:
            # Some sensors are stateful.
            for (sensor, sys_states, sen_states) in \
                    zip(self.sensors, self.system_state_indices, self.sensor_state_indices):
                if hasattr(sensor, 'is_stateful') and sensor.is_stateful:
                    x_next.append(sensor.sensor_state_transition(
                        x[sys_states], x[sen_states]))
        x_next = list(itertools.chain(*x_next))
        assert(len(x_next) == self.n_system_states + self.n_sensor_states)
        return x_next


    def get_real_measurement(self, x):
        y = []
        for sensor in self.sensors:
            y.append(sensor.get_real_measurement())
        y = list(itertools.chain(*y))
        return y


    def add_noise(self, y):
        y = np.array(y)
        return y + np.random.multivariate_normal(np.zeros(y.shape), self.noise_cov)


    def update_sensor_state(self, x):
        for (sensor, sys_states) in \
                zip(self.sensors, self.system_state_indices):
            if hasattr(sensor, 'is_stateful') and sensor.is_stateful:
                # Stateful sensor: we need to update its internal state.
                sensor.update_sensor_state(x[sys_states])


class Sensor(object):
    def __init__(self, noise_cov):
        self.noise_cov = noise_cov
        self.n_measure = noise_cov.shape[0]


    def measurement_function(self, x):
        pass


    def get_real_measurement(self):
        pass


    def add_noise(self, y):
        y = np.array(y)
        return y + np.random.multivariate_normal(np.zeros(y.shape), self.noise_cov)


class StatefulSensor(Sensor):
    def __init__(self, noise_cov, sensor_state_process_covariance):
        super(StatefulSensor, self).__init__(noise_cov)
        self.sensor_state_process_covariance = sensor_state_process_covariance
        self.is_stateful = True

    def sensor_state_transition(self, x, sensor_state):
        ''' Transition function for the sensor's state.
        '''
        pass


    def measurement_function(self, x , sensor_state=None):
        ''' Get a measurement from the sensor.

        If sensor state is passed, the passed value will be used as the sensor state.
        Otherwise, the sensor will use its internal record of its state.
        '''
        pass

    def update_sensor_state(self, x):
        '''Update the sensor's internal record of the sensor state.
        '''
        pass
