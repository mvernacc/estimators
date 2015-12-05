''' Barometric altitude sensor.

Parafoil GNC
MIT Rocket Team
2015 Nov 19
'''

import numpy as np
from sensor_interface import StatefulSensor

class Barometer(StatefulSensor):
    def __init__(self, geoid_height=0, sea_level_temp=288.15, bias_pressure=None):
        '''Barometric altitude sensor model.

        The position state of the parafoil in measured in the LNED frame,
        for which altitude is measured relative to the WGS84 datum. A Barometer
        measures the atmospheric pressure, which is a function of the height
        above mean sea level, the atmospheric temperature, and a bias pressure
        due to local weather. Finally, height above mean sea level is higher
        than the WGS84 altitude by the local geoid height.

        The noise level of 2 Pa is taken from the Bosch BMP180 data sheet [1].

        Arguments:
            geoid_height (real): The height of Mean Sea Level above the WGS84 ellipsoid
                at the LNED origin [units: meters]. This value depends on the geographic
                location of the launch site.
            sea_level_temp (real): The temperature at sea level [units: kelvin]. 
            bias_pressure (real): The difference of the pressure from the standard
                atmosphere model due to daily weather fluctuations [units: pascal].
                These fluctuations are typically 'a few kPa' [2]. If bias_pressure has the
                default value of None, self.bias_pressure will be set to a random sample from a
                representative normal distribution.

        References:
            [1] 'Data Sheet BMP180 Digital Pressure Sensor,' Bosch Sensortec GmbH,
                Germany, 2013. Online: http://cdn.sparkfun.com/datasheets/Sensors/Pressure/BMP180.pdf.
            [2] J. Coleman, 'Variations in Air Pressure,' Univ. of Oxford
                Phonetics Lab, Oxford, UK. Online: http://www.phon.ox.ac.uk/jcoleman/PRESSURE.htm.
            [3] 'A Quick Derivation Relating Altitude to Air Pressure,' Portland
                State Aerospace Society, Portland, Oregon, US, 2004.
                Online: http://psas.pdx.edu/RocketScience/PressureAltitude_Derived.pdf.
        '''
        self.bias_pressure = bias_pressure
        if self.bias_pressure is None:
            self.bias_pressure = np.random.normal(0, 1e3)
        self.sea_level_temp = sea_level_temp
        self.geoid_height = geoid_height

        # Lapse rate [units: kelvin meter**-1]
        self.L = -6.5e-3
        # Acceleration due to gravity [units: meter second**-2]
        self.g = 9.80665
        # Gas constant for air [units: joule kilogram**-1 kelvin**-1]
        self.R = 287.053
        # Pressure at Mean Sea Level [units: pascal]
        self.sea_level_pressure = 101325.0

        noise_cov = np.diag([2**2])
        sensor_state_process_covariance = np.diag([1e-3])
        super(self.__class__, self).__init__(noise_cov,
            sensor_state_process_covariance)


    def measurement_function(self, x, sensor_state=None):
        '''Barometer state-to-measurement function.

        Arguments:
            x (real vector): The system state, which contains the following values:
                * altitude in LNED frame [units: meter].

            sensor_state (optional, real vector): The sensor state, which contains
                the following values:
                * bias pressure [units: pascal].

        Returns:
            real: The measurement (barometric pressure) [units: pascal].
        '''
        alt_lned = x
        bias_pressure =  self.bias_pressure
        if sensor_state is not None:
            bias_pressure = sensor_state

        # Altitude above mean sea level [units: meter]
        alt_msl = (-alt_lned) - self.geoid_height
        # Standard atmosphere pressure for the given altitude, from eqn. 8 in [3].
        std_atm_pressure =  self.sea_level_pressure * (alt_msl * self.L / self.sea_level_temp + 1)\
            **(-self.g / (self.L * self.R))
        y = np.array([std_atm_pressure + bias_pressure])
        return y


    def sensor_state_transition(self, x, sensor_state):
        '''Transition function for the sensor's state.

       Arguments:
            x (real vector): The system state, which is not used by this function
                and is included only for compatibility with the super-class.

            sensor_state (real vector): The sensor state, which contains
                the following values:
                * bias pressure [units: pascal].

        Returns:
            real vector: The new sensor state, which contains the following values:
                * bias pressure [units: pascal].
        '''
        bias_pressure = sensor_state
        return bias_pressure