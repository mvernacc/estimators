''' Global Positioning System / Global Navigation Satellite System position sensor.

Parafoil GNC
MIT Rocket Team
2015 Nov 19
'''

import numpy as np
from sensor_interface import Sensor

class GPS(Sensor):
    # TODO should the measurement be in meters or degrees lat/lon?
    # TODO simulate time-to-first-fix

    def __init__(self):
        # GPS Noise levels from http://www.nstb.tc.faa.gov/reports/PAN86_0714.pdf#page=22
        # The report gives the 95% error accuracy, which is equal to 2 std. dev.,
        # assuming a gaussian distribution.
        noise_cov = np.diag([(3.35/2)**2, (3.35/2)**2, (4.68/2)**2])
        super(self.__class__, self).__init__(noise_cov)


    def measurement_function(self, x):
        '''GPS state-to-measurement function.

        Arguments:
            x (real 3 vector): The state, position in the LNED frame [units: meter].

        Returns:
            real 3 vector: The measurement, position in the LNED frame [units: meter].
        '''
        assert(len(x) == 3)
        return x
