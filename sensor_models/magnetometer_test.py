'''Unit test for magnetometer model.
'''

import unittest
import numpy as np
import magnetometer as mag

class MagTest(unittest.TestCase):
    def test_sensor_state_vector_to_bD_reversible(self):
        b = np.random.rand(3)
        D = np.random.rand(3,3)
        # Make D symmetric
        D = (D + D.T)/2

        theta = mag.bD_to_sensor_state_vector(b, D)
        b2, D2, = mag.sensor_state_vector_to_bD(theta)

        self.assertTrue(np.allclose(b, b2))
        self.assertTrue(np.allclose(D, D2))


if __name__ == '__main__':
    unittest.main()