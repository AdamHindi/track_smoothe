# track\_smoothe

**A collection of Python scripts for trajectory smoothing and analysis**, featuring both model-based and data-driven filters.

---

## Overview

This repository demonstrates two principal approaches to smoothing noisy 2D trajectories:

1. **Kalman Filter + Rauch–Tung–Striebel (RTS) Smoother**
2. **Savitzky–Golay Filter**
3. **Median Filter**

A simple synthetic data generator (`model_xy.py`) is provided for testing, along with example scripts that apply each smoothing technique and visualize the results.

---

## Repository Structure

* `model_xy.py`
  Generates synthetic
  $(x, y)$ trajectories of a 2D walk using Gaussian distribution for heading angles.
  **Features:**

  * Gaussian heading angle (var_psi), with cutoff over $[-a,a]$.
  * Gaussian acceleration (Esp = 0, var_acc).

* `smooth_traj_Kalman_RTS.py`
  Implements a constant-velocity Kalman filter followed by an RTS smoother to recover a smooth path from noisy $(x, y)$ measurements.
  **Features:**

  * Forward Kalman filter for real-time state estimation
  * Tuning of uncertainty in measurement or process
  * Backward RTS pass for offline smoothing
  * Plots raw vs. filtered trajectories

* `smoothe_traj_Savitzky.py`
  Applies a Savitzky–Golay filter to smooth $(x, y)$ time-series data.
  **Features:**

  * Configurable window length and polynomial order
  * Requires uniformly sampled data

* `smoothe_traj_median.py`
  Applies median filter to smoothe data.
  **Features:**

  * Configurable window length

---

## Configuration

* **Kalman filter** parameters in `smooth_traj_Kalman_RTS.py`:

  * `sigma_process`: process noise standard deviation
  * `sigma_measurement`: measurement noise standard deviation

* **Savitzky–Golay filter** parameters in `smoothe_traj_Savitzky.py`:

  * `window_length` (odd integer > polynomial order)
  * `polyorder` (degree of local polynomial)
 
* **Median filter** parameters in `smoothe_traj_median.py`:

  * `window_length` (odd integer)
 

Adjust these to balance smoothness versus fidelity.

