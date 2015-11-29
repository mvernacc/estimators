import unittest

import numpy as np
import transforms3d.quaternions as quat
import usque

class QuatPropagateTest(unittest.TestCase):
    def test_zero(self):
        q1 = quat.qeye()
        w = [0, 0, 0]
        q2 = usque.quat_propagate(q1, w, 1)
        self.assertTrue(all(q1 == q2))

    def test_x(self):
        q1 = quat.qeye()
        w = [0.1, 0, 0]
        q2 = usque.quat_propagate(q1, w, 0.1)
        q3 = quat.axangle2quat([1,0,0], 0.01)
        self.assertTrue(all(abs(q3 - q2) < 1e-10))

class RodriguesTest(unittest.TestCase):
    def test_self_inverse(self):
        np.random.seed(33249)
        q1 = quat.axangle2quat(np.random.random(3), np.random.random())
        dp = usque.quat2rodrigues(q1)
        q2 = usque.rodrigues2quat(dp)
        self.assertTrue(all(abs(q1 - q2) < 1e-10))


class QuatDerivativeTest(unittest.TestCase):
    def test_zero(self):
        q1 = quat.qeye()
        w = [0, 0, 0]
        q2 = usque.quat_derivative(q1, w)
        self.assertTrue(all(q2 == 0))


    def test_positive_x(self):
        q1 = quat.qeye()
        w = [1, 0, 0]
        q2 = usque.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] > 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] == 0)


    def test_positive_y(self):
        q1 = quat.qeye()
        w = [0, 1, 0]
        q2 = usque.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] > 0)
        self.assertTrue(q2[3] == 0)


    def test_positive_z(self):
        q1 = quat.qeye()
        w = [0, 0, 1]
        q2 = usque.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] > 0)

    def test_negative_x(self):
        q1 = quat.qeye()
        w = [-1, 0, 0]
        q2 = usque.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] < 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] == 0)


    def test_negative_y(self):
        q1 = quat.qeye()
        w = [0, -1, 0]
        q2 = usque.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] < 0)
        self.assertTrue(q2[3] == 0)


    def test_negative_z(self):
        q1 = quat.qeye()
        w = [0, 0, -1]
        q2 = usque.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] < 0)

if __name__ == '__main__':
    unittest.main()
