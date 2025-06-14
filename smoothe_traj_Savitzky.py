import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def savitzky_golay_coeffs(window_length,order):
    """
    Compute Savitzky-Golay filter coefficients.
    Parameters :
    ------------
        - window_length : odd integer
        - order : polynomial order (< window_length)
    Returns :
    ----------  
        - a 1D array of length window_length.
    """
    half = (window_length - 1) // 2
    # Design matrix: rows for each offset i = -half..half
    A = np.vander(np.arange(-half, half + 1), order + 1, increasing=True)
    # Coeff
    ATA = A.T @ A
    c = np.linalg.inv(ATA) @ A.T
    # Coefficients for smoothing (derivative order 0 is first row of pinvA)
    return c[0]


def savgol_filter(y, window_length, order):
    """
    Apply Savitzky-Golay smoothing to 1D array y.
    Uses mirror-padding at the ends.
    Parameters :
    ------------
        - y : perturbed data set
        - window_length : odd integer
        - order : polynomial order (< window_length)
    Returns :
    ---------
        - 1D array of length y
    """
    # Check conditions
    if window_length % 2 == 0 or window_length < 1:
        raise ValueError("window_length must be a positive odd integer")
    
    # Compute filter coefficients
    coeffs = savitzky_golay_coeffs(window_length, order)
    half = (window_length - 1) // 2
    # Mirror-pad the signal at both ends
    y_padded = np.concatenate([y[half:0:-1], y, y[-2:-half-2:-1]])
    n = len(y)
    y_smooth = np.empty(n)
    # Slide window and apply filter
    for i in range(n):
        window = y_padded[i:i + window_length]
        y_smooth[i] = np.dot(coeffs, window)
    return y_smooth

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


if __name__ == "__main__":
    # Load Trajectories
    t_loaded, x_noisy, y_noisy = load_trajectory('trajectory.npy')
    # Apply Savitzky–Golay filter
    window_length = 15 # must be odd, 1<w< len(y_noisy)
    polyorder = 8      # polyorder < window_length
    x_sg = savgol_filter(x_noisy, window_length, polyorder)
    y_sg = savgol_filter(y_noisy, window_length, polyorder)

    # Plot the noisy vs. smoothed trajectory
    plt.plot(x_noisy, y_noisy, '.', label='Noisy trajectory')
    plt.plot(x_sg, y_sg, '-', label='Savitzky–Golay smoothed')
    plt.axis('equal')
    plt.legend()
    plt.title('Savitzky–Golay Smoothing of a Single Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()