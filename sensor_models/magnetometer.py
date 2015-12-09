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
        h_bias_ned=None,
        h_bias_sensor=None,):
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
            h_bias_ned (real): The magnetic field bias vector in a North-East-Down
                coordinate system. If h_bias_ned has the default value of None,
                it will be initialized to a random value. This represents magnetic
                sources (ferromagnetic materials, currents) which move with the world.
                [units: microtesla].
            h_bias_sensor (real): The magnetic field bias vector in the sensor
                coordinate system. If h_bias_ned has the default value of None,
                it will be initialized to a random value. This represents magnetic
                sources (ferromagnetic materials, currents) which move with the sensor
                or vehicle. This bias is added in addition to h_bias_ned.
                [units: microtesla].

        References:
            [1] 'Magnetic Field Calculators,' National Oceanic and Atmospheric Administration.
                Online: http://www.ngdc.noaa.gov/geomag-web/#igrfwmm.
            [2] A. M. Sabatini, 'Quaternion-Based Extended Kalman Filter for
                Determining Orientation by Inertial and Magnetic Sensing,'
                IEEE Transactions on Biomedical Engineering, vol. 53, no. 7,
                July 2006.
        '''
        assert(magnitude > 0)
        assert(magnitude <  100)
        self.h_earth_ned = dec_inc_to_vector(magnitude, declination, inclination)
        
        if h_bias_ned is None:
            self.h_bias_ned = np.random.normal(0, 5, size=3)
        else:
            self.h_bias_ned = np.array(h_bias_ned)
            assert(len(self.h_bias_ned) == 3)
        
        if h_bias_sensor is None:
            self.h_bias_sensor = np.random.normal(0, 5, size=3)
        else:
            self.h_bias_sensor = np.array(h_bias_sensor)
            assert(len(self.h_bias_sensor) == 3)

        # The noise covariance is based on the worst case std. dev. in table
        # 2 of reference [2].
        noise_cov = np.diag([(noise_std_dev)**2]*3)
        sensor_state_process_covariance = np.diag([0.5, 0.5, 0.5])**2
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

        h_bias_sensor =  quat_utils.rotate_frame(self.h_bias_ned, q_ned2sensor) \
            + self.h_bias_sensor
        if sensor_state is not None:
            assert(len(sensor_state) == 3)
            h_bias_sensor = sensor_state
        
        h_earth_sensor = quat_utils.rotate_frame(self.h_earth_ned, q_ned2sensor)
        y = h_earth_sensor + h_bias_sensor
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

        Returns:
            real vector: The new sensor state, which contains the following values:
                * The magnetic field bias in the sensor frame x direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame y direction
                    [units: microtesla].
                * The magnetic field bias in the sensor frame z direction
                    [units: microtesla].
        '''
        h_bias_sensor = sensor_state
        return h_bias_sensor


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


