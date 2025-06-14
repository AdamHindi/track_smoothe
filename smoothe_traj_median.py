import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def median_filter_1d(y, window_length):
    """
    Applies a median filter to a 1D data with mirror-padding.
    Parameters :
    ------------
        - y: 1D numpy array.
        - window_length: odd integer >= 1.
    Returns :
    ----------
        - a filtered signal of the same length.
    """
    # Check conditions
    if window_length % 2 == 0 or window_length < 1:
        raise ValueError("window_length must be a positive odd integer")
    
    half = window_length // 2
    # Mirror padding
    y_padded = np.concatenate([y[half:0:-1], y, y[-2:-half-2:-1]])
    y_smooth = np.empty_like(y)
    
    # Slide window and compute median
    for i in range(len(y)):
        window = y_padded[i:i + window_length]
        y_smooth[i] = np.median(window)
    
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
    # Load trajectories
    t_loaded, x_noisy, y_noisy = load_trajectory('trajectory.npy')
    # Apply Median filter
    window_length = 5 # must be odd, 1<w< len(y_noisy)
    
    x_m = median_filter_1d(x_noisy, window_length)
    y_m = median_filter_1d(y_noisy, window_length)

    # Plot the noisy vs. smoothed trajectory
    plt.plot(x_noisy, y_noisy, '.', label='Noisy trajectory')
    plt.plot(x_m, y_m, '-', label='Median smoothed')
    plt.axis('equal')
    plt.legend()
    plt.title('Median Smoothing of a Single Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
