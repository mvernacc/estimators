"""Autocorrelation analysis of gyroscope noise.

Matt Vernacchia
2015 Nov 28
"""

import numpy as np
import argparse
import pickle as pickle
from matplotlib import pyplot as plt
import scipy.signal


def autocorrelation(x, dt):
    """Estimate the autocorrelation function of a signal.

    Arguments:
        x (array of reals): signal values at constant time intervals.
        dt (positive real): The time interval between values in x.

    Returns:
        array of reals: time lags [units: same as dt].
        array of reals: normalized autocorrelation values for each time lag
            [units: none].

    References:
        [1] Unutbu, 'Answer to Estimate Autocorrelation using Python,' 2013.
            Online: http://stackoverflow.com/a/14298647.
        [2] A. B. Downey, 'Think DSP: Digital Signal Processing in Python,'
            Green Tea Press, Ch. 5.
            Online: http://greenteapress.com/thinkdsp/html/thinkdsp006.html.
    """
    n = len(x)
    var = np.var(x)
    xm = x - np.mean(x)
    r = np.correlate(xm, xm, mode="same")
    r = r[n // 2 :]
    autocorr = r / (var * np.arange(n, n // 2, -1))
    t_lag = dt * np.arange(np.floor(-n / 2.0), np.floor(n / 2.0), 1)
    return t_lag, autocorr


def butter_lowpass(cutoff, fs, order=5):
    """Source: http://stackoverflow.com/a/25192640"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Source: http://stackoverflow.com/a/25192640"""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def main(args):
    with open(args.in_file, "rb") as f:
        data = pickle.load(f)

    y_noisy = data["y_noisy"]

    if "time" in data:
        time = data["time"]
    else:
        n = y_noisy.shape[0]
        dt = 1e-2
        time = np.linspace(0, (n - 1) * dt, n)

    # Sampling frequency
    dt = time[1] - time[0]
    # Filter the data
    # fs = 1.0 / dt
    # y_filt = butter_lowpass_filter(y_noisy[:,0], 0.1, fs)
    y_filt = data["y"][:, 0]
    # Compute the autocorrelation of the filtered data.
    t_lag, autocorr = autocorrelation(y_filt, dt)

    plt.subplot(2, 1, 1)
    plt.plot(time, y_noisy, color="r", label="raw data")
    plt.plot(time, y_filt, color="b", label="filtered data")
    plt.legend(framealpha=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(autocorr)
    plt.xlabel("index")
    plt.ylabel("normalized autocorrelation [-]")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Allan variance analysis of gyroscope noise."
    )
    parser.add_argument("in_file", type=str, help="Pickle file of gyro data.")
    args = parser.parse_args()
    main(args)
