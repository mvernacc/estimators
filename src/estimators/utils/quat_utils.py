"""Unscented Quaternion Estimator for attitude state estimation.

Matt Vernacchia
2015 Nov 28
"""

import numpy as np
import transforms3d.quaternions as quat


def cross_mat(v):
    """Cross product matrix.

    Implement eqn 17 in Crassidis [1].

    Arguments:
        v (3-vector)

    Returns:
        3x3 matrix

    References:
        [1] J. L. Crassidis and F. Landis Markley, 'Unscented Filtering for
            Spacecraft Attitude Estimation,' SUNY Buffalo, Amherst, NY.
            Online: http://ancs.eng.buffalo.edu/pdf/ancs_papers/2003/uf_att.pdf
    """
    v = np.squeeze(np.array(v))
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def quat_derivative(q, w):
    """Quaternion derivative dq / dt.

    Note this is slightly different from equation 16a in Crassidis [1]. He
    has q[3] as the real part of the the quaternion, I have q[0] as the real
    part for compatiblity with the transforms3d library.

    Arguments:
        q (quaternion): The attitude [units: none].
        w (3-vector): The angular velocity [units: radian second**-1].

    Returns:
        quaternion: dq / dt [units: second**-1].

    References:
        [1] J. L. Crassidis and F. Landis Markley, 'Unscented Filtering for
            Spacecraft Attitude Estimation,' SUNY Buffalo, Amherst, NY.
            Online: http://ancs.eng.buffalo.edu/pdf/ancs_papers/2003/uf_att.pdf
    """
    Xi = np.bmat(
        [
            [-np.matrix(q[1:])],
            [q[0] * np.eye(3) + cross_mat(q[1:])],
        ]
    ).A
    w = np.matrix(w)
    q_dot = 0.5 * np.dot(Xi, w.T)
    return np.squeeze(q_dot.A)


def quat_propagate(q, w, dt):
    """Propagate a quaternion forward in time.

    Arguments:
        q (quaternion): The attitude [units: none].
        w (3-vector): The angular velocity [units: radian second**-1].
        dt (real): The time step [units: second].

    Returns:
        quaternion: New attitude [units: none].

    References:
        [1] J. L. Crassidis and F. Landis Markley, 'Unscented Filtering for
            Spacecraft Attitude Estimation,' SUNY Buffalo, Amherst, NY.
            Online: http://ancs.eng.buffalo.edu/pdf/ancs_papers/2003/uf_att.pdf
    """
    w = np.array(w)
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-10:
        return q
    angle = norm_w * dt
    dq = quat.axangle2quat(w, angle)
    return quat.qmult(q, dq)


def quat2rodrigues(dq, a=1.0, f=4.0):
    """Convert an error quaternion to Rodrigues parameter vector.

    Arguments:
        dq (quaternion): Error quaternion [units: none].
        a (real, optional): scale factor. Must be in [0, 1].
        f (real, optional): scale factor.

    Returns:
        3-vector: Generalized Rodrigues paramter vector.

    References:
        [1] J. L. Crassidis and F. Landis Markley, 'Unscented Filtering for
            Spacecraft Attitude Estimation,' SUNY Buffalo, Amherst, NY.
            Online: http://ancs.eng.buffalo.edu/pdf/ancs_papers/2003/uf_att.pdf
    """
    dp = f / (a + dq[0]) * dq[1:]
    return dp


def rodrigues2quat(dp, a=1.0, f=4.0):
    """Convert an error quaternion to Rodrigues parameter vector.

    Arguments:
        dp (3-vector): Generalized Rodrigues paramter vector. [units: none].
        a (real, optional): scale factor. Must be in [0, 1].
        f (real, optional): scale factor.

    Returns:
        quaternion: Error quaternion [units: none].

    References:
        [1] J. L. Crassidis and F. Landis Markley, 'Unscented Filtering for
            Spacecraft Attitude Estimation,' SUNY Buffalo, Amherst, NY.
            Online: http://ancs.eng.buffalo.edu/pdf/ancs_papers/2003/uf_att.pdf
    """
    dq = np.zeros(4)
    n = np.linalg.norm(dp) ** 2
    dq[0] = (-a * n + f * (f**2 + (1 - a**2) * n) ** 0.5) / (f**2 + n)
    dq[1:] = (a + dq[0]) / f * dp
    return dq


def quat_log(q):
    """Natural logarithm of a Quaternion.

    References:
        https://en.wikipedia.org/wiki/Quaternion#Exponential.2C_logarithm.2C_and_power
        http://www.geometrictools.com/Documentation/Quaternions.pdf
    """
    nq = np.linalg.norm(q)
    nv = np.linalg.norm(q[1:])

    # if np.isclose(nq, 1):
    #     # Unit quaternion
    #     axis, angle = quat.quat2axangle(q)
    #     return np.hstack((
    #         0,
    #         axis * angle
    #         ))
    if np.isclose(nq, 1) and np.isclose(nv, 0):
        # Identity quaterion
        return np.zeros(4)
    else:
        return np.hstack(
            (np.log(nq), q[1:] / np.linalg.norm(q[1:]) * np.arccos(q[0] / nq))
        )


def quat_exp(q):
    """Exponential of a quaternion, e**q.

    References:
        https://en.wikipedia.org/wiki/Quaternion#Exponential.2C_logarithm.2C_and_power
    """
    nv = np.linalg.norm(q[1:])

    if np.isclose(q[0], 0):
        # e**q is a unit quaterniron
        if np.isclose(nv, 0):
            # e**q is the identity quaternion
            return quat.qeye()
        else:
            return quat.axangle2quat(q[1:], nv)
    else:
        return np.exp(q[0]) * np.hstack((np.cos(nv), q[1:] / nv * np.sin(nv)))


def quat_average(q_list, q_expected=None):
    """Average of a list of quaternions.

    Of the two equivalent average results (q_avg or -q_avg),
    return the one closer to q_expected.

    References:
        [1] Online: http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
    """
    # My & Trang's approach:
    q_avg = np.copy(q_list[0])
    if sum(q_avg) < 0:
        q_avg = -q_avg
    for q in q_list[1:]:
        if sum(q) < 0:
            q = -q
        if np.dot(q_avg, q) < 0:
            q = -q
        q_avg += q
    # Normalize
    q_avg = q_avg / np.linalg.norm(q_avg)

    # Point the quaternion towards q_expected.
    if q_expected is not None and np.dot(q_expected, q_avg) < 0:
        q_avg = -q_avg
    return q_avg

    #  Markley et al's approach in [1]. Breaks if two quaternions
    # in the list have zero dot product.
    # M = sum( np.matrix(q).T * np.matrix(q) for q in q_list)
    # print 'M = '
    # print M
    # [d, V] = np.linalg.eig(M)
    # i = np.argmax(d)
    # print i
    # q_avg = np.squeeze(V[i,:])
    # return q_avg

    # Jacob Panikulam's approach. Breaks when the list contains a
    # quaternion and its negative
    # q_log = np.zeros(4)
    # for q in q_list:
    #     print 'q = {:s}, log(q) = {:s}'.format(
    #         np.array_str(q), np.array_str(quat_log(q)))
    # q_log = sum(quat_log(q) for q in q_list)
    # q_avg = quat_exp(q_log)
    # return q_avg


def rotate_frame(v_A, q_A2B):
    """Rotate the frame in which a vector is written.

    Arguments:
        v_A (3-vector): A vector written in the coordinates of
            frame A.
        q_A2B (quaternion): the rotation from frame A to frame B.

    Returns:
        3-vector: The vector, written in the coordinates of frame
            B.
    """
    v_B = quat.rotate_vector(v_A, quat.qinverse(q_A2B))
    return v_B


def random_quat():
    """Get a random unit quaternion from a uniform distribution over
    SO(3).

    References:
        https://www-preview.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
    """
    s = np.random.rand()
    s1 = (1 - s) ** 0.5
    s2 = s**0.5
    t1 = 2 * np.pi * np.random.rand()
    t2 = 2 * np.pi * np.random.rand()
    w = np.cos(t2) * s2
    x = np.sin(t1) * s1
    y = np.cos(t1) * s1
    z = np.sin(t2) * s2
    return np.array([w, x, y, z])
