import unittest

import numpy as np
import transforms3d.quaternions as quat
import quat_utils
import itertools


class QuatPropagateTest(unittest.TestCase):
    def test_zero(self):
        q1 = quat.qeye()
        w = [0, 0, 0]
        dt = 1
        q2 = quat_utils.quat_propagate(q1, w, dt)
        self.assertTrue(np.allclose(q1, q2))


    def test_x_ident(self):
        q1 = quat.qeye()
        w = [0.1, 0, 0]
        dt = 0.1
        q2 = quat_utils.quat_propagate(q1, w, dt)
        q3 = quat.axangle2quat([1,0,0], np.linalg.norm(w) * dt)
        self.assertTrue(np.allclose(q3, q2))


    def test_x_general(self):
        q1 = quat.axangle2quat([1,0,0], 1)
        w = [0.1, 0, 0]
        dt = 0.1
        q2 = quat_utils.quat_propagate(q1, w, dt)
        q3 = quat.axangle2quat([1,0,0], 1 + np.linalg.norm(w) * dt)
        self.assertTrue(np.allclose(q3, q2))


    def test_xyz_general(self):
        q1 = quat.axangle2quat([1,1,1], 1)
        w = [1, 1, 1]
        dt = 0.1
        q2 = quat_utils.quat_propagate(q1, w, dt)
        q3 = quat.axangle2quat([1,1,1], 1 + np.linalg.norm(w) * dt)
        self.assertTrue(np.allclose(q3, q2))


    def test_sequential_axes(self):
        q1 = quat.axangle2quat([0,1,0], np.pi/2)
        w = [-np.pi/2, 0, 0]
        dt = 1.0
        q2 = quat_utils.quat_propagate(q1, w, dt)

        # Q3 is 90 deg about y, followed by -90 deg about the new x.
        q3 = [ 0.5, -0.5,  0.5,  0.5]
        self.assertTrue(np.allclose(q3, q2))


class RodriguesTest(unittest.TestCase):
    def test_self_inverse(self):
        np.random.seed(33249)
        q1 = quat.axangle2quat(np.random.random(3), np.random.random())
        dp = quat_utils.quat2rodrigues(q1)
        q2 = quat_utils.rodrigues2quat(dp)
        self.assertTrue(np.allclose(q1, q2))


class QuatDerivativeTest(unittest.TestCase):
    def test_zero(self):
        q1 = quat.qeye()
        w = [0, 0, 0]
        q2 = quat_utils.quat_derivative(q1, w)
        self.assertTrue(all(q2 == 0))


    def test_positive_x(self):
        q1 = quat.qeye()
        w = [1, 0, 0]
        q2 = quat_utils.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] > 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] == 0)


    def test_positive_y(self):
        q1 = quat.qeye()
        w = [0, 1, 0]
        q2 = quat_utils.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] > 0)
        self.assertTrue(q2[3] == 0)


    def test_positive_z(self):
        q1 = quat.qeye()
        w = [0, 0, 1]
        q2 = quat_utils.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] > 0)

    def test_negative_x(self):
        q1 = quat.qeye()
        w = [-1, 0, 0]
        q2 = quat_utils.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] < 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] == 0)


    def test_negative_y(self):
        q1 = quat.qeye()
        w = [0, -1, 0]
        q2 = quat_utils.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] < 0)
        self.assertTrue(q2[3] == 0)


    def test_negative_z(self):
        q1 = quat.qeye()
        w = [0, 0, -1]
        q2 = quat_utils.quat_derivative(q1, w)
        self.assertTrue(q2[0] == 0)
        self.assertTrue(q2[1] == 0)
        self.assertTrue(q2[2] == 0)
        self.assertTrue(q2[3] < 0)


class QuatAverageTest(unittest.TestCase):
    def test_single(self):
        q1 = quat.qeye()
        q_avg = quat_utils.quat_average([q1])
        self.assertTrue(np.allclose(q1, q_avg))


    def test_same_negative(self):
        q1 = quat.qeye()
        q_avg = quat_utils.quat_average([q1, -q1])
        self.assertTrue(np.allclose(q1, q_avg))


    def test_same_orthogonal_pairs(self):
        q1 = np.array([0, 1, 0, 0])
        q2 = np.array([0, 0, 1, 0])

        for q_list in itertools.permutations([q1, -q1, q2, -q2]):
            q_avg = quat_utils.quat_average(q_list)
            print q_avg
            self.assertTrue(
                np.allclose([0, 0.70710678, 0.70710678, 0], q_avg))


if __name__ == '__main__':
    unittest.main()
