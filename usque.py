''' Unscented Quaternion Estimator for attitude state estimation.

Matt Vernacchia
2015 Nov 28
'''

import numpy as np
import transforms3d.quaternions as quat


class Usque(object):
    def __init__(self, quat_0, x_est_0, Q_0, f, W, h, R):
        ''' Create an UnScented QUaternion Estimator.

        The state vector is:
            index 0: rate gyro bias sensor frame x component
            index 1: rate gyro bias sensor frame y component
            index 2: rate gyro bias sensor frame z component
            index 3+: arbitrary state

        The covariance matrix is:
            row/col 0, 1, 2: covariance of attitude.
            row/col 3, 4, 5: covariance of gyro bias.
            row/col n > 5: covariance of state n-3.

        Arguments:
            quat_0: initial attitude quaternion.
            x_est_0 (real vector): The initial state estimate.
            Q_0 (real matrix): The intial estimate covariance.
            f (function): State update function (x_{t-1}, u) --> x_t.
            W (real matrix): Process noise covariance.
            h (function): Measurement function (x_t) --> y.
            R (real matrix): Measurement noise covariance.

        References:
            [1] J. L. Crassidis and F. Landis Markley, 'Unscented Filtering for
                Spacecraft Attitude Estimation,' SUNY Buffalo, Amherst, NY.
                Online: http://ancs.eng.buffalo.edu/pdf/ancs_papers/2003/uf_att.pdf
        ''' 
        self.x_est = np.array(x_est_0)
        self.n = len(self.x_est)
        self.Q = np.array(Q_0)
        self.f = f
        self.W = np.array(W)
        self.h = h
        self.R = np.array(R)

        assert(len(self.x_est) == self.Q.shape[0])
        assert(len(self.x_est) == self.Q.shape[1])

        self.dp = np.zeros(3)


    def propagate_dynamics(self, w, u):
        ''' Propagate the estimate forward by one time step.

        In the notation used by N. Roy in 16.322, calling this function
        propagates self.x_est and self.Q as follows:
            x_{k-1|k-1} --> x_{k|k-1}
            Q_{k-1|k-1} --> Q_{k|k-1}

        Arguments:
            w (3-vector): The sensed rotational velocity
                [units: radian second**-1].
            u (real vector): The control action for this time step.
        '''
        # Form 2n sigma points from x_{k-1|k-1} and Q_{k-1|k-1}.
        sigma_points = self.get_sigma_points(
            np.hstack((self.dp, self.x_est)),
            self.Q)
        # Propagate each sigma point from step k-1 to step k.
        sigma_points_next = np.array([self.f(x, u) for x in sigma_points])
        # Use the sigma points to approximate x_{k|k-1} and Q_{k|k-1}
        self.x_est = np.mean(sigma_points_next, axis=0)
        self.Q = self.get_Q_from_sigma_points(sigma_points_next, self.x_est) \
            + self.W
        self.force_positive_definite(self.Q)


    def update_measurement(self, y):
        ''' Update the estimate to incorporate information from a measurement.

        In the notation used by N. Roy in 16.322, calling this function
        propagates self.x_est and self.Q as follows:
            x_{k|k-1} --> x_{k|k}
            Q_{k|k-1} --> Q_{k|k}

        Arguments:
            y (real vector): The measurement.
        '''
        # Create new sigma points from x_{k|k-1} and Q_{k|k-1}.
        sigma_points_x = self.get_sigma_points(self.x_est, self.Q)
        # Transform each sigma point through the measurement function.
        sigma_points_y = [self.h(x) for x in sigma_points_x]
        # Use the y sigma points to estimate the mean y est and Q_{yy}, Q_{xy}
        y_est = np.mean(sigma_points_y, axis=0)
        Q_yy = self.get_Q_from_sigma_points(sigma_points_y, y_est) + self.R
        Q_xy = self.get_Q_from_sigma_points(sigma_points_x, self.x_est,
            sigma_points_y, y_est) # + self.R
        # Compute the Kalman gain.
        K = np.dot(Q_xy, np.linalg.inv(Q_yy))
        # Update the state estimate using the Kalman gain.
        self.x_est = self.x_est + np.squeeze(
            np.dot(K, np.array([y - y_est]).T))
        self.Q = self.Q - np.dot(K, np.dot(Q_yy, K.T))
        self.force_positive_definite(self.Q)


    def get_sigma_points(self, x, Q):
        n = self.n
        sigma_points = np.zeros((2*n, n))
        root_nQ = scipy.linalg.sqrtm(n*Q)
        for i in xrange(n):
            sigma_points[2*i] = x + root_nQ[i]
            sigma_points[2*i+1] = x - root_nQ[i]
        return sigma_points


    def get_Q_from_sigma_points(self, sp1, x1, sp2=None, x2=None):
        if sp2 is None and x2 is None:
            sp2 = sp1
            x2 = x1
        if not hasattr(x2, '__len__'):
            x2 = np.array([x2])
        n = self.n
        Q = np.zeros((self.safe_len(x1), self.safe_len(x2)))
        for i in xrange(2*n):
            Q += np.dot(
                np.array([sp1[i] - x1]).T,
                np.array([sp2[i] - x2])
                )
        Q = Q / (2.0 * n)
        return Q


    def safe_len(self, x):
        ''' Check the length of x, and return 1 of x is scalar.

        It seems like there should be a better way of doing this than
        defining my own function.
        '''
        length = None
        try:
            length = len(x)
        except TypeError:
            length = 1
        return length


    def force_positive_definite(self, Q, min_eigenvalue=0, epsilon=1e-6):
        ''' Force a matrix to be positive definite using the SVD approach.

        Source: http://stats.stackexchange.com/a/6367.

        Arguments:
            Q (real matrix): Matrix which should be positive definite.
            min_eigenvalue (real): The lowest value an eigenvalue should have.
            epsilon (real): A small positive number. Eigenvalues less than
                min_eigenvalue will be replaced with epsilon.

        Returns:
            Q, but with all negative eigenvalues reset to a small positive value.
        '''
        # Decompose into eigenvalues and vectors
        [d, V] = np.linalg.eig(Q)
        # Check if eigenvalues are sufficiently positive
        if all(d > min_eigenvalue):
            return Q
        else:
            # Replace bad (negative) eigenvalues.
            print 'Warning: negative eigenvalues encountered, fixing...'
            for i in xrange(len(d)):
                if d[i] < min_eigenvalue:
                    d[i] = epsilon
            return np.dot(V, np.dot(np.diag(d), V.T))
