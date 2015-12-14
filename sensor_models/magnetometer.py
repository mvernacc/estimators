''' Magnetic compass sensor.

Parafoil GNC
MIT Rocket Team
2015 Nov 19
'''

import numpy as np
from sensor_interface import StatefulSensor
import transforms3d.quaternions as quat
from estimators.utils import quat_utils

class Magnetometer(StatefulSensor):
    def __init__(self,
        noise_std_dev=0.8,
        magnitude=52.2,
        declination=np.deg2rad(-14.76),
        inclination=np.deg2rad(67.31),
        b=None,
        D=None):
        '''Magnetic compass sensor model.

        In order to compute Earth-fixed orientation from the sensor measurements,
        we must know the magnetic magnetic field magnitude, declination and
        inclination in the location where we are operating.

        Parameters of the magnetic field at locations of interest, from [1]:
         
          * 52.2 microtesla, -14.76 deg declination, 67.31 deg inclination 
            at 42.3592 N, -71.0937 W, 3 meter EGM96 altitude
            (77 Mass Ave, Cambridge, MA)
          
          * 51.1 microtesla, 10.70 deg declination, 64.65 deg inclination
            at 38.8414 N, -110.2344 W, 1304 meter EGM96 altitude
            (IREC Launch Site, Green River, UT)
        
        These paramters are valid as of 2015 Nov 19. Note that the Earth's magnetic
        field changes with time, so these values should be updated every year or so.

        Arguments:
            noise_std_dev (real): Standard deviation of measurement noise
                [units: microtesla].
            magnitude (real): The magnitude of the Earth's magnetic field
                at the launch site [units: microtesla].
            declination (real): The magnetic declination at the launch site
                [units: radian]. This is the angle from geographic north to
                magnetic north.
            inclination (real): The magnetic inclination at the launch site
                [units: radian]. This is the angle from the local horizontal
                plane to the magnetic field vector. Positive values point down.
            b (real 3-vector): The bias vector b as defined in [3]. If b has the
                default value of None, it will be intialized to a random value.
            D (real 3x3 matrix): The symmetric scale factor and non-orthogonality
                matrix D as defined in [3]. If D has the default value of None,
                it will be intialized to a random value.

        References:
            [1] 'Magnetic Field Calculators,' National Oceanic and Atmospheric Administration.
                Online: http://www.ngdc.noaa.gov/geomag-web/#igrfwmm.
            [2] A. M. Sabatini, 'Quaternion-Based Extended Kalman Filter for
                Determining Orientation by Inertial and Magnetic Sensing,'
                IEEE Transactions on Biomedical Engineering, vol. 53, no. 7,
                July 2006.
            [3] J. L. Crassidis, K.-L. Lai, and R. R. Harman, 'Real-time attitude-
                independent three-axis magnetometer calibration,' Journal of Guidance,
                Control, and Dynamics, vol. 28, no. 1, pp. 115-120, Jan 2005. [Online].
                Available: http://arc.aiaa.org/doi/abs/10.2514/1.6278
        '''
        assert(magnitude > 0)
        assert(magnitude <  100)
        self.h_earth_ned = dec_inc_to_vector(magnitude, declination, inclination)
        
        # The sensor bias standard deviation [units: microtesla]
        bias_std_dev = 5
        # The sensor scale factor and non-orthogonality standard dev
        # [units: none]
        D_std_dev = 0.1
        if b is None:
            self.b = np.random.normal(0, bias_std_dev, size=3)
        else:
            self.b = np.array(b)
            assert(len(self.b) == 3)
        
        if D is None:
            theta = np.random.normal(0, D_std_dev, size=9)
            junk, self.D = sensor_state_vector_to_bD(theta)
            # Make D symmetric
            self.D = (self.D + self.D.T) / 2
        else:
            self.D = np.array(D)
            assert(self.D.shape == (3,3))

        # The noise covariance is based on the worst case std. dev. in table
        # 2 of reference [2].
        noise_cov = np.diag([(noise_std_dev)**2]*3)
        # I made up values for the process covaraince.
        sensor_state_process_covariance = np.diag(
            np.hstack(([0.5]*3, [1e-6]*6)))**2
        super(self.__class__, self).__init__(noise_cov,
            sensor_state_process_covariance)


    def measurement_function(self, x, sensor_state=None):
        '''Magnetometer state-to-measurement function.

        Arguments:
            x (real vector): The system state, which contains the following values:
                If len(x) == 1:
                * heading (rotation about LNED z+ axis to transform LNED frame
                    to sensor frame) [units: radian].
                If len(x) == 4:
                * q_LNED_to_sensor real component.
                * q_LNED_to_sensor i component.
                * q_LNED_to_sensor j component.
                * q_LNED_to_sensor k component.

            sensor_state (optional, real vector): The sensor state, which contains
                the following values:
                * The magnetic field bias in the sensor frame x direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame y direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame z direction
                    [units: microtesla].
                * The six parameters of Crassidis's D matrix [3] [units: none].

        Returns:
            (real vector): The measurement, magnetic field vector in the sensor
                frame [units: microtesla].
        '''
        x = np.array(x)
        if len(x) == 1:
            heading = x[0]
            q_ned2sensor = quat.axangle2quat([0, 0, 1], heading)
        elif len(x) == 4:
            q_ned2sensor = x
        else:
            raise ValueError

        b = self.b
        D = self.D
        if sensor_state is not None:
            assert(len(sensor_state) == 9)
            b, D = sensor_state_vector_to_bD(sensor_state)
        
        h_earth_sensor = quat_utils.rotate_frame(self.h_earth_ned, q_ned2sensor)
        y = np.dot(np.linalg.inv(np.eye(3) + D), np.array([h_earth_sensor + b]).T)
        return y


    def sensor_state_transition(self, x, sensor_state):
        '''Transition function for the sensor's state.

       Arguments:
            x (real vector): The system state, which is not used by this function
                and is included only for compatibility with the super-class.

            sensor_state (real vector): The sensor state, which contains
                the following values:
                * The magnetic field bias in the sensor frame x direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame y direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame z direction
                    [units: microtesla].
                * The six parameters of Crassidis's D matrix [3] [units: none].

        Returns:
            real vector: The new sensor state, which contains the following values:
                * The magnetic field bias in the sensor frame x direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame y direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame z direction
                    [units: microtesla].
                * The six parameters of Crassidis's D matrix [3] [units: none].
        '''
        return sensor_state


def dec_inc_to_vector(magnitude, declination, inclination):
    '''Convert the magnetic field magnitude, declination, inclination to a 3-vector in the NED frame.

    Arguments:
        magnitude (real): The magnitude of the Earth's magnetic field
            at the launch site [units: microtesla].
        declination (real): The magnetic declination at the launch site
            [units: radian]. This is the angle from geographic north to
            magnetic north.
        inclination (real): The magnetic inclination at the launch site
            [units: radian]. This is the angle from the local horizontal
            plane to the magnetic field vector. Positive values point down.

    Returns:
        real 3-vector: The magnetic field vector in a North-East-Down coordinate
            system [units: microtesla].
    '''
    h_ned = [0, 0, 0]
    h_ned[2] = magnitude * np.sin(inclination)
    h_horiz = magnitude * np.cos(inclination)
    h_ned[0] = h_horiz * np.cos(declination)
    h_ned[1] = h_horiz * np.sin(declination)
    assert(abs(np.linalg.norm(h_ned) - magnitude) < (magnitude * 1e-6))
    return h_ned


def sensor_state_vector_to_bD(theta):
    '''Convert the 9-element sensor state to the D matrix and b vector.

    Arguments:
        theta (real 9-vector): The paramter vector as defined in [1].

    Returns:
        real 3-vector: The bias vector b as defined in [1].
        real 3x3 matrix: The symmetric scale factor and non-orthogonality
            matrix D as defined in [1].

    References:
        [1] J. L. Crassidis, K.-L. Lai, and R. R. Harman, 'Real-time attitude-
            independent three-axis magnetometer calibration,' Journal of Guidance,
            Control, and Dynamics, vol. 28, no. 1, pp. 115-120, Jan 2005. [Online].
            Available: http://arc.aiaa.org/doi/abs/10.2514/1.6278
    '''
    b = theta[0:3]
    D = np.array([
        [theta[3], theta[6], theta[7]],
        [theta[6], theta[4], theta[8]],
        [theta[7], theta[8], theta[5]]
        ])
    return (b, D)


def bD_to_sensor_state_vector(b, D):
    '''Convert the D matrix and b vector to the 9-element sensor state.

    Arguments:
        b (real 3-vector): The bias vector b as defined in [1].
        D (real 3x3 matrix): The symmetric scale factor and non-orthogonality
            matrix D as defined in [1].

    Returns:
        real 9-vector: The paramter vector as defined in [1].
        
    References:
        [1] J. L. Crassidis, K.-L. Lai, and R. R. Harman, 'Real-time attitude-
            independent three-axis magnetometer calibration,' Journal of Guidance,
            Control, and Dynamics, vol. 28, no. 1, pp. 115-120, Jan 2005. [Online].
            Available: http://arc.aiaa.org/doi/abs/10.2514/1.6278
    '''
    return np.array([b[0], b[1], b[2], D[0,0], D[1,1], D[2,2], D[0,1], D[0,2], D[1,2]])