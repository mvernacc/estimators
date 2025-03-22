# Estimators

State estimators (extended, unscented and quaternion Kalman filters) and sensor models.

## Project history

I originally wrote these in Fall 2015 for my MIT 16.322 class project in grad school.
Late in 2015 I also started to extend this project to do Guidance Navigation and Control for a guided parafoil recovery system for the MIT Rocket Team, but we abandoned that project before I got very far with the code.

In March 2025 I updated the now-decade-old code from python 2 -> 3. I packaged the project with `uv` and organized the directory structure.
I added code quality checks with `ruff`, `pre-commit` and GitHub actions.

If I were to put this library to serious use, I would also add type hints and static type analysis with `pyright` (or Astral's [new type checker](https://x.com/charliermarsh/status/1884651482009477368) 👀, once that is ready). But I did not think that was worthwhile at the time of the March 2025 update.

## Installation and running

First, install the [uv package manager for python](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Then clone this repository.

Then,
```
cd estimators/demos
uv run kraft_demo.py
```

`uv run` will first automatically download and install compatible versions of python and all dependencies in a virtual environment. The it will run a demonstration of Kraft's Unscented Kalman Filter for a quaternion attitude state[^1].

## Contents

### `src/estimators/sensor_models`

Measurement functions and noise models of common sensors.

Also contains a class `KalmanSensors` for packaging several sensor models together into a single object, which can then provide a combined measurement function to a multi-sensor Kalman Filter. `KalmanSensors` also provides a framework for augmenting the system state with sensor bias estimates.

### `src/estimators/nonlinear_kalman.py`

Extended and Uscented Kalman Filters for non-linear systems.

### `src/estimators/kraft_quat_ufk.py`

A quaternion-based Unscented Kalman Filter for estimation on a system with an attitude state. Based on a paper by E. Kraft[^1].

### `src/estimators/usque.py`

A quaternion-based Unscented Kalman Filter for estimation on a system with an attitude state. Based on the USQUE filter by J. Crassidis. Not yet fully implemented.

### `src/estimators/utils`

Various plotting and quaternion math utilities.

# References

[^1]: E. Kraft, 'A Quaternion-based Unscented Kalman Filter for 
    Orientation Tracking,' Physikaliches Institut, Univ. of Bonn,
    Bonn, Germany.
    Online: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1257247