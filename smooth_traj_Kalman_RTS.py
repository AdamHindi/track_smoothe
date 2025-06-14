import matplotlib.pyplot as plt
import numpy as np
from random import randint
from numpy import sin, cos

# Reconstruct a continuous function:

def kalman_smoother(t, x_noisy, y_noisy, sigma_process=1.0, sigma_measurement=1.0):
    """
    Applies a constant-velocity Kalman filter + Rauch–Tung–Striebel smoother
    to 2D noisy measurements (x_noisy, y_noisy) observed at times t.

    Parameters
    ----------
    t : array_like, shape (n,)
        delta_t = cte.
    x_noisy : array_like, shape (n,)
        noisy x-coordinates.
    y_noisy : array_like, shape (n,)
        noisy y-coordinates.
    sigma_process : float
        Standard deviation of the process (acceleration) noise.
    sigma_measurement : float
        Standard deviation of the measurement noise (applied equally to x and y).

    Returns
    -------
    x_smooth : ndarray, shape (n,)
        Smoothed x-coordinates.
    y_smooth : ndarray, shape (n,)
        Smoothed y-coordinates.
    """
    n = len(t)
    # State vector: [x, vx, y, vy]
    # Allocate arrays 
    x_pred = np.zeros((n, 4))
    P_pred = np.zeros((n, 4, 4))
    x_filt = np.zeros((n, 4))
    P_filt = np.zeros((n, 4, 4))

    # Initialize filter with first measurement, zero velocity
    x_filt[0] = [x_noisy[0], 0, y_noisy[0], 0]
    P_filt[0] = np.eye(4)

    # Observation model H : we observe position only
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    # Covariance of obs model (sigma_measurement on step 0)
    R = np.eye(2) * (sigma_measurement**2)

    dt = t[1]-t[0]

     # State transition (constant velocity, x += dt vx, y += dt vy)
    A = np.array([[1, dt, 0,  0],
                    [0,  1, 0,  0],
                    [0,  0, 1, dt],
                    [0,  0, 0,  1]])
    
    # Process-noise covariance (assuming acceleration noise)
    q = sigma_process**2
    Q_block = np.array([[dt**3/3, dt**2/2],
                            [dt**2/2, dt]])
    Q = q * np.block([[Q_block, np.zeros((2, 2))],
                        [np.zeros((2, 2)), Q_block]])

    # Forward Kalman filter pass

    # Here we consider x_{k|k-1} == x_pred[k] then we update to get x_{k|k} == x_filt[k]
    # We keep both for the backward pass of RTS
    for i in range(1, n):
        # Predict (x_pred[k] = x_{k|k-1})
        x_pred[i] = A @ x_filt[i - 1]
        P_pred[i] = A @ P_filt[i - 1] @ A.T + Q

        # Update (filt[k] = x_{k|k})
        z = np.array([x_noisy[i], y_noisy[i]])      #
        S = H @ P_pred[i] @ H.T + R                 #Innovation Covariance
        K = P_pred[i] @ H.T @ np.linalg.inv(S)      #Optimal Gain
        y_resid = z - H @ x_pred[i]                 #Innovation 
        x_filt[i] = x_pred[i] + K @ y_resid         #Updated estimate estate
        P_filt[i] = (np.eye(4) - K @ H) @ P_pred[i] #Updated estimate covariance

    # Simple RTS smoothing (Backward phase/pass, while the forward phase is Kalman filter)
    # src : https://en.wikipedia.org/wiki/Kalman_filter
    # x_smooth[k] = x_{k|n}
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    # Backward phase, we start with x_{n|n} since it's what we have
    for i in range(n - 2, -1, -1):
        # C = P_{k|k} A^{t}_{k+1} (=A, cte) P^{-1}_{k+1|k}
        C = P_filt[i] @ A.T @ np.linalg.inv(P_pred[i + 1])
        x_smooth[i] = x_filt[i] + C @ (x_smooth[i + 1] - x_pred[i + 1])
        P_smooth[i] = P_filt[i] + C @ (P_smooth[i + 1] - P_pred[i + 1]) @ C.T

    return x_filt[:, 0], x_filt[:, 2]
def load_trajectory(filename='trajectory.npy'):
    """Load nested trajectory array from .npy."""
    try:
        arr = np.load(filename, allow_pickle=True)
    except:
        raise ValueError("Failed to load filename")
    if arr.shape == (2,) and arr.dtype == object:
        t = arr[1]
        pos = arr[0]
        x = pos[:, 0]
        y = pos[:, 1]
        return t, x, y
    raise ValueError("Shape doesn't fit")

t_loaded, x_loaded, y_loaded = load_trajectory('trajectory2.npy')
x_sm, y_sm = kalman_smoother(t_loaded, x_loaded, y_loaded,
                                sigma_process=4, sigma_measurement=20)
plt.scatter(x_loaded,y_loaded)
plt.plot()
plt.show()
plt.scatter(x_loaded, y_loaded, s=10, alpha=0.5, label='noisy')
plt.plot(x_sm, y_sm, 'r-', label='smoothed')
plt.legend(); plt.show()
