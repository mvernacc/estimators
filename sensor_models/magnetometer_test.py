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

    def test_get_meas_variance_zero(self):
        b = np.zeros(3)
        D = np.zeros((3,3))
        h = np.ones(3)
        noise_cov = np.diag(np.ones(3))
        v = mag.get_meas_variance(noise_cov, b, D, h)
        self.assertTrue(v.shape == (1,1) or v.shape == ())
        self.assertTrue(np.allclose(v, 4*3 + 2*3))


if __name__ == '__main__':
    unittest.main()