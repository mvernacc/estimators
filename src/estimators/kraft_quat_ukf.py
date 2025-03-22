import numpy as np
import scipy.linalg
from estimators.utils import quat_utils
import transforms3d.quaternions as quat


class KraftQautUKF(object):
    def __init__(self, x_est_0, Q_0, f, W, h, R):
        """Create an Unscented Kalman Filter for quaternion attitude estimation.

        The state vector is:
            index 0: attitude quaternion real component
            index 1: attitude quaternion i component
            index 2: attitude quaternion j component
            index 3: attitude quaternion k component
            index n > 3: arbitrary state

        The estimate covariance matrix is:
            row/col 0: covariance of attitude about vehicle x axis [units: radian**2].
            row/col 1: covariance of attitude about vehicle y axis [units: radian**2].
            row/col 2: covariance of attitude about vehicle z axis [units: radian**2].
            row/col n > 2: covariance of state n+1.

        The process noise covariance matrix is:
            row/col 0: covariance of attitude noise about vehicle x axis per
                time step [units: radian**2].
            row/col 1: covariance of attitude noise about vehicle y axis per
                time step [units: radian**2].
            row/col 2: covariance of attitude noise about vehicle z axis per
                time step [units: radian**2].
            row/col n > 2: covariance of state n+1.

        Arguments:
            x_est_0 (real vector): The initial state estimate.
            Q_0 (real matrix): The intial estimate covariance.
            f (function): State update function (x_{t-1}, u) --> x_t.
            W (real matrix): Process noise covariance.
            h (function): Measurement function (x_t) --> y.
            R (real matrix): Measurement noise covariance.

        References:
            [1] E. Kraft, 'A Quaternion-based Unscented Kalman Filter for
                Orientation Tracking,' Physikaliches Institut, Univ. of Bonn,
                Bonn, Germany.
                Online: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1257247
        """
        self.x_est = np.array(x_est_0)
        self.normalize_quat_state()
        self.n = len(self.x_est)
        self.Q = np.array(Q_0)
        self.f = f
        self.W = np.array(W)
        self.h = h
        self.R = np.array(R)

        assert len(self.x_est) - 1 == self.Q.shape[0]
        assert len(self.x_est) - 1 == self.Q.shape[1]
        assert len(self.x_est) - 1 == self.W.shape[0]
        assert len(self.x_est) - 1 == self.W.shape[1]

    def propagate_dynamics(self, u):
        """Propagate the estimate forward by one time step.

        In the notation used by N. Roy in 16.322, calling this function
        propagates self.x_est and self.Q as follows:
            x_{k-1|k-1} --> x_{k|k-1}
            Q_{k-1|k-1} --> Q_{k|k-1}

        Arguments:
            u (real vector): The control action for this time step.
        """
        # Form 2n sigma points from x_{k-1|k-1} and Q_{k-1|k-1}.
        sigma_points = self.get_sigma_points(self.x_est, self.Q + self.W)
        # Propagate each sigma point from step k-1 to step k.
        sigma_points_next = np.array([self.f(x, u) for x in sigma_points])
        # Use the sigma points to approximate x_{k|k-1} and Q_{k|k-1}
        self.x_est[0:4] = quat_utils.quat_average(
            sigma_points_next[:, 0:4], q_expected=self.x_est[0:4]
        )
        self.x_est[4:] = np.mean(sigma_points_next[:, 4:], axis=0)
        self.Q = self.get_Q_from_sigma_points(
            sigma_points_next, self.x_est, is_1_quat=True
        )
        self.normalize_quat_state()
        self.Q = self.force_positive_definite(self.Q)

    def update_measurement(self, y):
        """Update the estimate to incorporate information from a measurement.

        In the notation used by N. Roy in 16.322, calling this function
        propagates self.x_est and self.Q as follows:
            x_{k|k-1} --> x_{k|k}
            Q_{k|k-1} --> Q_{k|k}

        Arguments:
            y (real vector): The measurement.
        """
        # Create new sigma points from x_{k|k-1} and Q_{k|k-1}.
        sigma_points_x = self.get_sigma_points(self.x_est, self.Q)
        # Transform each sigma point through the measurement function.
        sigma_points_y = [self.h(x) for x in sigma_points_x]
        # Use the y sigma points to estimate the mean y est and Q_{yy}, Q_{xy}
        y_est = np.mean(sigma_points_y, axis=0)
        Q_yy = self.get_Q_from_sigma_points(sigma_points_y, y_est) + self.R
        Q_xy = self.get_Q_from_sigma_points(
            sigma_points_x, self.x_est, sigma_points_y, y_est, is_1_quat=True
        )  # + self.R
        # Compute the Kalman gain.
        K = np.dot(Q_xy, np.linalg.inv(Q_yy))
        # Update the state estimate using the Kalman gain.
        increment = np.squeeze(np.dot(K, np.array([y - y_est]).T))
        dq = quat.axangle2quat(increment[0:3], np.linalg.norm(increment[0:3]))
        self.x_est[0:4] = quat.qmult(self.x_est[0:4], dq)
        self.x_est[4:] = self.x_est[4:] + increment[3:]
        self.Q = self.Q - np.dot(K, np.dot(Q_yy, K.T))
        self.normalize_quat_state()
        self.Q = self.force_positive_definite(self.Q)

    def normalize_quat_state(self):
        self.x_est[0:4] = self.x_est[0:4] / np.linalg.norm(self.x_est[0:4])

    def get_sigma_points(self, x, Q):
        # Get the set of disturbance vectors, script-w in Kraft.
        n = self.n
        distrub = np.zeros((2 * (n - 1), n - 1))
        root_nQ = scipy.linalg.sqrtm((n - 1) * Q)
        for i in range(n - 1):
            distrub[2 * i] = root_nQ[i]
            distrub[2 * i + 1] = -root_nQ[i]

        sigma_points = np.zeros((2 * (n - 1), n))
        for i in range(2 * (n - 1)):
            # Disturb the quaternion state
            angle = np.linalg.norm(distrub[i, 0:3])
            if abs(angle) < 1e-6:
                q_disturb = quat.qeye()
            else:
                axis = distrub[i, 0:3] / angle
                q_disturb = quat.axangle2quat(axis, angle, is_normalized=True)
            q_old = x[0:4]
            sigma_points[i, 0:4] = quat.qmult(q_old, q_disturb)
            # Disturb the vector and scalar states
            sigma_points[i, 4:] = x[4:] + distrub[i, 3:]

        return sigma_points

    def get_Q_from_sigma_points(self, sp1, x1, sp2=None, x2=None, is_1_quat=False):
        n = self.n
        distrub1 = sp1 - x1
        if is_1_quat:
            distrub1 = np.zeros((2 * (n - 1), n - 1))
            for i in range(2 * (n - 1)):
                q_i = sp1[i, 0:4]
                q_avg = x1[0:4]
                q_disturb = quat.qmult(quat.qinverse(q_avg), q_i)
                axis, angle = quat.quat2axangle(q_disturb)
                distrub1[i, 0:3] = axis * angle
                distrub1[i, 3:] = sp1[i, 4:] - x1[4:]

        if sp2 is None and x2 is None:
            distrub2 = distrub1
        else:
            distrub2 = sp2 - x2

        Q = np.zeros((self.safe_len(distrub1[0]), self.safe_len(distrub2[0])))
        for i in range(2 * (n - 1)):
            Q += np.dot(np.array([distrub1[i]]).T, np.array([distrub2[i]]))
        Q = Q / (2.0 * (n - 1))
        return Q

    def safe_len(self, x):
        """Check the length of x, and return 1 of x is scalar.

        It seems like there should be a better way of doing this than
        defining my own function.
        """
        length = None
        try:
            length = len(x)
        except TypeError:
            length = 1
        return length

    def force_positive_definite(self, Q, min_eigenvalue=0, epsilon=1e-6):
        """Force a matrix to be positive definite using the SVD approach.

        Source: http://stats.stackexchange.com/a/6367.

        Arguments:
            Q (real matrix): Matrix which should be positive definite.
            min_eigenvalue (real): The lowest value an eigenvalue should have.
            epsilon (real): A small positive number. Eigenvalues less than
                min_eigenvalue will be replaced with epsilon.

        Returns:
            Q, but with all negative eigenvalues reset to a small positive value.
        """
        # Decompose into eigenvalues and vectors
        [d, V] = np.linalg.eig(Q)
        # Check if eigenvalues are sufficiently positive
        if all(d > min_eigenvalue):
            return Q
        else:
            # Replace bad (negative) eigenvalues.
            print("Warning: negative eigenvalues encountered, fixing...")
            for i in range(len(d)):
                if d[i] < min_eigenvalue:
                    d[i] = epsilon
            return np.dot(V, np.dot(np.diag(d), V.T))
