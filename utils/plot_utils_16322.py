'''Plotting utilities for 16.322.

Matt Vernacchia
Fall 2015
'''

import numpy as np
import math
from matplotlib.patches import Ellipse


def get_ellipse(mu, cov, **kwargs):
    ''' Turn a covariance into a drawable ellipse.

    Arguments:
        mu (real vector): The mean.
        cov (real matrix): The covariance.

    Source: N. Roy "16.322 Materials: L5 - nonlinear_least_squares.py," MIT,
        Cambridge, MA, 2015.
    '''
    cov = np.matrix(cov)
    mu = np.matrix(mu)
    (w, v) = np.linalg.eig(cov)

    # The angle of the covariance ellipse is the angle of the first eigenvector
    angle_in_degrees = math.atan2(v.A[1][0], v.A[0][0])*180/math.pi

    # Change from N. Roy's original function: make the dimension of the ellipse
    # the standard deviation, not the variance.
    width = (w[0]**0.5) * 2
    height = (w[1]**0.5) * 2

    ellipse = Ellipse(xy=mu.A[0,:], width=width, height=height, angle=angle_in_degrees,
                      **kwargs)
    return ellipse


def plot_single_state_vs_time(ax, time, state, covariance, which_state, **kwargs):
    mean = state[:,which_state]
    std_dev = [v**0.5 for v in covariance[:, which_state, which_state]]
    
    ax.plot(time, mean, **kwargs)
    # Remove the lable from the filled swath
    kwargs.pop('label', None)
    ax.fill_between(time, mean + std_dev, mean - std_dev, alpha=0.5, **kwargs)
