Estimators
==========

State estimators and sensor models.

Contents
--------
### `sensor_models`
Measurement functions and noise models of common sensors.

Also contains a class KalmanSensors for packaging several sensor models together into a single object, which can then provide a combined measurement function to a multi-sensor Kalman Filter. KalmanSensors also provides a framework for augmenting the system state with sensor bias estimates.

### `kraft_quat_ufk.py`
A quaternion-based Unscented Kalman Filter for estimation on a system with an attitude state. Based on a paper by E. Kraft [1].

### `usque.py`
A quaternion-based Unscented Kalman Filter for estimation on a system with an attitude state. Based on the USQUE filter by J. Crassidis. Not yet fully implemented.

### `utils`
Various plotting and quaternion math utilities.

Dependencies
------------
  * numpy
  * matplotlib
  * [transforms3d](https://github.com/matthew-brett/transforms3d)

Author
------
Matt Vernacchia
MIT Aero Astro
mvernacc@mit.edu

References
----------
[1] E. Kraft, 'A Quaternion-based Unscented Kalman Filter for 
    Orientation Tracking,' Physikaliches Institut, Univ. of Bonn,
    Bonn, Germany.
    Online: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1257247