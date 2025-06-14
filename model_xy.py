import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

def sample_headings(n_steps: int, var_psi: float, a: float) -> np.ndarray:
    """
    Sample heading angle from a Gaussian distribution (Var = var_psi, Esp = 0) truncated to [-a, a].

    Parameters:
    - n_steps: number of angles to sample
    - var_psi: variance of the Gaussian (radians^2)
    - a: maximum absolute heading angle (radians)

    Returns:
    - Array of shape (n_steps,) of heading angles in radians
    """
    sigma = np.sqrt(var_psi)
    lower, upper = -a / sigma, a / sigma
    return truncnorm(lower, upper, loc=0, scale=sigma).rvs(n_steps)


def simulate_walk(
    velocity: float,
    var_psi: float,
    a: float,
    delta_t: float,
    total_time: float,
    sigma_acc: float,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """
    Simulate a walking trajectory in 2D with:
      - Heading psi drawn from truncated Gaussian (variance var_psi) on [-a, a]
      - Gaussian acceleration noise on dx, dy per step (Var = sigma_acc)

    Returns:
    - traj: array of shape (n_steps+1, 3) with columns [x, y, t]
    """
    n_steps = int(np.ceil(total_time / delta_t))
    headings = sample_headings(n_steps, var_psi, a)

    xs = np.zeros(n_steps + 1)
    ys = np.zeros(n_steps + 1)
    ts = np.linspace(0, n_steps * delta_t, n_steps + 1)

    xs[0], ys[0] = x0, y0

    for i in range(1, n_steps + 1):
        psi = headings[i - 1] 
        # deterministic displacement
        dx = velocity * np.cos(psi) * delta_t
        dy = velocity * np.sin(psi) * delta_t
        # Gaussian acceleration noise as displacement perturbation
        noise_x = np.random.normal(0, sigma_acc)
        noise_y = np.random.normal(0, sigma_acc)
        xs[i] = xs[i - 1] + dx + noise_x
        ys[i] = ys[i - 1] + dy + noise_y

    return np.vstack((xs, ys, ts)).T


def save_trajectory(traj: np.ndarray, filename: str = "trajectory.npy") -> None:
    """
    Save trajectory as [positions, times] to a .npy file using an object array.

    The stored object is a 1D numpy array of length 2 with dtype=object:
      - element 0: pos_array of shape (n_steps+1, 2)
      - element 1: t_array of shape (n_steps+1,)

    Parameters:
    - traj: array with columns [x, y, t]
    - filename: output filename, must end with .npy
    """
    if not filename.endswith('.npy'):
        raise ValueError("Unsupported file format. Use .npy for nested output format.")

    pos = traj[:, :2]  # shape (n_steps+1, 2)
    t = traj[:, 2]     # shape (n_steps+1,)

    # Create an object array to hold pos and t without shape conflicts
    arr = np.empty(2, dtype=object)
    arr[0] = pos
    arr[1] = t

    np.save(filename, arr)
    print(f"Trajectory saved to {filename}: [pos_array shape {pos.shape}, t_array shape {t.shape}]")


if __name__ == "__main__":
    # Simulation parameters
    velocity = 1.0       # units per second
    var_psi = 10      # variance of heading (rad^2)
    a = np.pi / 3        # max heading angle truncation (±45°)
    delta_t = 0.3        # seconds per step
    total_time = 20.0    # total simulation time (s)
    sigma_acc = 0.05     # std dev of Gaussian noise on dx, dy per step
    x0, y0 = 0.0, 0.0    # starting position

    # Run simulation
    traj = simulate_walk(velocity, var_psi, a, delta_t, total_time, sigma_acc, x0, y0)

    # Plot trajectory
    xs, ys, ts = traj[:,0], traj[:,1], traj[:,2]
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker='o', markersize=3, linestyle='-')
    plt.title('2D Random Walk with Direction Variance and Acceleration Noise')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Save to file in requested nested format
    save_trajectory(traj, 'trajectory.npy')
