''' Allan variance analysis of gyroscope noise.

Matt Vernacchia
2015 Nov 27
'''

from __future__ import division
import numpy as np
import argparse
import cPickle as pickle
from matplotlib import pyplot as plt


def allan(times, values, n_points=100):
    ''' Allan variance analysis.

    Arguments:
        times (array of reals): Time of each data point [units: second]. The times
            should be evenly spaced.
        values (array of reals): Value of each data point [units: x].

    Returns:
        array of reals: Averaging times [units: second].
        array of reals: Allan deviation at each averaging time [units: x].

    References:
        [1] O. J. Woodman, 'An introduction to inertial navigation,' Univ. of
            Cambridge, Cambridge UK, Tech. Rep. UCAM-CL-TR-696, 2007.

    '''
    # Averaging times to evaluate.
    dt = np.mean(np.diff(times))
    t_range = times[-1] - times[0]
    # The shortest averaging time is long enough for 10 samples per bin.
    t_avg_short = dt * 10
    # The longest averaging time is long enough for a total of 9 bins, this is
    # the minimum valid number of bins according to Woodman [1].
    t_avg_long = t_range / 9.0
    assert t_avg_long > t_avg_short, 'Not enough data for Allan variance analysis.'
    # Build logarithmic array of averaging times
    t_avg = np.logspace(np.log2(t_avg_short), np.log2(t_avg_long), num=n_points,
        base=2.0)
    # Compute the Allan Variance for each averaging time.
    avar = np.zeros(len(t_avg))
    for (t, i) in zip(t_avg, range(len(t_avg))):
        # Divide the data into bins of time length t, step 1 in Woodman [1].
        n_bins = int(t_range // t)
        bins = np.array_split(values, n_bins)
        # Average the data in each bin, step 2 in Woodman [1].
        a = [np.mean(b) for b in bins]
        # Compute the Allan variance, step 3 in Woodman [1].
        avar[i] = 1.0 / (2 * (n_bins - 1)) * sum([(a[j + 1] - a[j])**2 \
            for j in xrange(n_bins - 1)])
    # The Allan deviation is the square root of the Allan variance, eqn
    # 20 in Woodman [1].
    adev = avar**0.5
    return (t_avg, adev)


def rate_noise_from_allan(t_avg, adev, fit_params=False):
    ''' Compute the rate noise (a.k.a. angle random walk) from the Allan Deviation.

    Arguments:
        t_avg (array of reals): Averaging times [units: second].
        adev (array of reals): Allan deviation at each averaging time [units: x].
        fit_params (optional, boolean): Return the fit parameters of the 
            linear portion of the log10-log10 Allan Deviation vs averaging
            time curve.

    Returns:
        real: The rate noise [units: x ** second**0.5]. This is the value
            a line fit to the linear portion of the loglog Allan Deviation
            at t_avg = 1.0 second.
        array of reals: Present only if fit_params is true. Array returned
            by np.polyfit.

    References:
        [1] O. J. Woodman, 'An introduction to inertial navigation,' Univ. of
            Cambridge, Cambridge UK, Tech. Rep. UCAM-CL-TR-696, 2007.
    '''
    # Find the index at which the linear region ends. Assume the linear region
    # goes from min(t_avg) to t_avg = 10 s.
    ind_start = 0
    ind_end = np.searchsorted(t_avg, 10)
    log_t = np.log10(t_avg[ind_start:ind_end])
    log_adev = np.log10(adev[ind_start:ind_end])
    p = np.polyfit(log_t, log_adev, 1)
    # p[0] is the slope, it should be -0.5 according to Woodman [1].
    assert abs(p[0] - -0.5) < 0.2, 'The log log slope is {:.3f}, it should be -0.5'\
        .format(p[0])
    # The rate noise is the linear fit evaluated at t_avg = 1, Woodman [1].
    log_rate_noise = p[0]*np.log10(1) + p[1]
    rate_noise = 10**log_rate_noise

    if fit_params:
        return (rate_noise, p)
    else:
        return rate_noise



def main(args):
    with open(args.in_file, 'rb') as f:
        data = pickle.load(f)

    if args.data_name == 'gyro_data':
        # Gyro data from MPU-9150 is in deg/s.
        y_units = 'deg/s'
    if args.data_name == 'accel_data':
        # Accel data from MPU-9150 is in g.
        y_units = 'g'
    if args.data_name == 'mag_data':
        # Magnetometer data from MPU-9150 is in microtesla.
        y_units = 'uT'
    y = data[args.data_name]



    if 'time' in data:
        time = data['time']
    else:
        n = y.shape[0]
        dt = 1e-2
        time = np.linspace(0, (n-1)*dt, n)

    plot_colors = ['red', 'green', 'blue']

    for i in xrange(3):
        t_avg, adev = allan(time, y[:,i])

        rate_noise, p = rate_noise_from_allan(t_avg, adev, fit_params=True)
        t_fit = np.array([t_avg[0], 10])
        fit_log = p[0]*np.log10(t_fit) + p[1]
        fit = np.power(10, fit_log)

        print 'Rate noise = {:.3e} {:s}'.format(rate_noise,
            y_units + ' s**0.5')

        plt.loglog(t_avg, adev, label='Data', color=plot_colors[i])
        plt.xlabel('Averaging time [s]')
        plt.ylabel('Allan Deviation [{:s}]'.format(y_units))
        plt.title('{:s} Allan Deviation'.format(args.in_file))
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor')

        plt.loglog(t_fit, fit, color=plot_colors[i], linestyle='--', label='Rate noise fit')
        plt.loglog(1, rate_noise, color=plot_colors[i], marker='o', label='Rate noise')

    plt.legend(framealpha=0.5)

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Allan variance analysis of gyroscope noise.')
    parser.add_argument('in_file', type=str, help='Pickle file of gyro data.')
    parser.add_argument('data_name', type=str, help='Key for data in the pickle dict.')
    args = parser.parse_args()
    main(args)
