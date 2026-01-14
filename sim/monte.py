import os
from collections import deque  # (kept if you later want to animate)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # (unused in this script, but handy)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from system_model import *
import particle_filter
import firefly_pf


# =========================
# Global Config / Constants
# =========================
np.random.seed(69)

# Monte Carlo
NUM_MONTE_RUNS = 10

sim_time = 100.0 # seconds
sim_hz = 200   # integrator rate (dt = 1/simulation_hz)
sim_dt = 1 / sim_hz


# =========================
# Dynamics & Integrator
# =========================
def drone_x_dot(x_n, acceleration_n):
    """
    6-state continuous-time dynamics:
      x = [x, y, z, vx, vy, vz]
    dx/dt = [vx, vy, vz, ax, ay, az]
    """
    return np.array([
        x_n[3],             # dx/dt = vx
        x_n[4],             # dy/dt = vy
        x_n[5],             # dz/dt = vz
        acceleration_n[0],  # dvx/dt = ax
        acceleration_n[1],  # dvy/dt = ay
        acceleration_n[2],  # dvz/dt = az
    ])

def runge_kutta(get_x_dot, x_0, t_0, t_f, dt, accel_fn):
    """
    4th-order Runge-Kutta integrator with an acceleration provider accel_fn(t).
    Returns a dict with:
      'state_sum' : (N, 6) array of states over time
      'acc_sum'   : (N, 3) array of accelerations used at each step
    """
    steps = int((t_f - t_0) / dt)
    state_summary = np.zeros((steps, NUM_STATES))
    acceleration_summary = np.zeros((steps, 3))

    t = t_0
    x_n = x_0.copy()

    for k in range(steps):
        a = accel_fn(t)  # possibly deterministic + noise
        state_summary[k] = x_n
        acceleration_summary[k] = a

        k1 = dt * get_x_dot(x_n, a)
        k2 = dt * get_x_dot(x_n + 0.5 * k1, a)
        k3 = dt * get_x_dot(x_n + 0.5 * k2, a)
        k4 = dt * get_x_dot(x_n + k3, a)
        x_n = x_n + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        t += dt

    return {'state_sum': state_summary, 'acc_sum': acceleration_summary}

# ====================================
# Ellipsoid Reference Motion (Center)
# ====================================
def ellipsoid_pos_vel_acc(t, rx, ry, rz, w_th, w_ph,
                          th0=0.0, ph0=0.0):
    """
    Elliptical x-y motion + independent z that starts at 0 and peaks at 12 m.
    """

    # -----------------
    # Horizontal motion (unchanged)
    # -----------------
    th = th0 + w_th * t
    ph = ph0 + w_ph * t

    cth, sth = np.cos(th), np.sin(th)
    cph, sph = np.cos(ph), np.sin(ph)

    x = rx * cth * cph
    y = ry * cth * sph

    vx = -rx * sth * cph * w_th - rx * cth * sph * w_ph
    vy = -ry * sth * sph * w_th + ry * cth * cph * w_ph

    ax = -rx * cth * cph * (w_th**2 + w_ph**2) + 2 * rx * sth * sph * w_th * w_ph
    ay = -ry * cth * sph * (w_th**2 + w_ph**2) - 2 * ry * sth * cph * w_th * w_ph

    # -----------------
    # Vertical motion (new)
    # -----------------
    z_max = 12.0  # meters
    wz = 0.5      # vertical frequency [rad/s], tweak for faster/slower bounce
    # z(t) = (z_max / 2) * (1 - cos(wz * t)) -> ranges 0..z_max
    z  = (z_max / 2.0) * (1.0 - np.cos(wz * t))
    vz = (z_max / 2.0) * (np.sin(wz * t) * wz)
    az = (z_max / 2.0) * (np.cos(wz * t) * (wz**2))

    pos = np.array([x, y, z])
    vel = np.array([vx, vy, vz])
    acc = np.array([ax, ay, az])
    return pos, vel, acc

def make_ellipsoid_accel_provider(rx, ry, rz, w_th, w_ph, th0=0.0, ph0=0.0, noise_std=process_noise_std):
    pos0_rel, vel0, _ = ellipsoid_pos_vel_acc(0.0, rx, ry, rz, w_th, w_ph, th0, ph0)

    x0 = np.zeros(6)
    x0[0:3] = pos0_rel + CENTER
    x0[3:6] = vel0

    def accel_fn(t):
        _, _, acc = ellipsoid_pos_vel_acc(t, rx, ry, rz, w_th, w_ph, th0, ph0)
        if noise_std > 0:
            acc = acc + np.random.normal(0.0, noise_std, size=3)
        return acc
    return accel_fn, x0

# ====================================
# Monte Carlo Trajectory Generation
# ====================================
def generate_trajectories():
    monte_state = []
    monte_acc   = []

    for _ in range(NUM_MONTE_RUNS):
        rx, ry, rz = 10.0, 10.0, 6.0
        w_th = np.random.uniform(0.05, 0.2)
        w_ph = np.random.uniform(0.05, 0.2)
        th0  = np.random.uniform(-np.pi/4, np.pi/4)
        ph0  = np.random.uniform(0, 2*np.pi)

        accel_fn, x0 = make_ellipsoid_accel_provider(
            rx, ry, rz, w_th, w_ph, th0, ph0
        )

        res = runge_kutta(
            drone_x_dot, x0,
            t_0=0.0, t_f=sim_time,
            dt=1.0/sim_hz,
            accel_fn=accel_fn
        )

        # pull from dict returned by integrator
        monte_state.append(res['state_sum'])  # (T, 6)
        monte_acc.append(res['acc_sum'])      # (T, 3)

    monte_data = {
        'state_sum': np.stack(monte_state, axis=0),  # (R, T, 6)
        'acc_sum':   np.stack(monte_acc,   axis=0),  # (R, T, 3)
    }
    return monte_data

# =========================
# Plotting
# =========================
def plot_trajectories(trajectories, fig_num=1, save_as_png=False, dpi=300):
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([CENTER[0]], [CENTER[1]], [CENTER[2]], s=80, marker='X', label='Beacon', zorder=5)

    # --- FIX: iterate over the runs in the stacked array ---
    S_all = np.asarray(trajectories['state_sum'])  # (R, T, 6)
    for r in range(S_all.shape[0]):
        S = S_all[r]  # (T, 6)
        ax.plot(S[:, 0], S[:, 1], S[:, 2], linewidth=1.0)

    ax.set_title('Spherical Trajectories around Beacon (0, 0, 1)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    _set_axes_equal(ax)  # optional: keeps aspect ratio sane

    if save_as_png:
        plt.savefig('rover_trajectories.png', format='png', dpi=dpi)

    plt.show()

def plot_pf_xyz_est_vs_truth(monte_data, run_idx=0, sim_hz=None, imu_hz=None):
    """
    Plots x,y,z PF estimate (solid) vs truth (dashed) over time for one run.

    Inputs:
      monte_data['x_estimate'] : (R, T_pf, NS) or (T_pf, NS)
      monte_data['state_sum']  : (R, T_sim, 6) or (T_sim, 6)
      Optional: monte_data['t_pf'] (T_pf,), or imu_hz / dt_pf for time axis.
                If sim_hz & imu_hz given (or in monte_data), truth is decimated;
                else it is interpolated to PF length.
    """
    X = np.asarray(monte_data['x_estimate'])
    S = np.asarray(monte_data['state_sum'])
    if X.ndim == 2: X = X[None, ...]
    if S.ndim == 2: S = S[None, ...]

    est = X[run_idx, :, :3]   # (T_pf, 3)
    tru = S[run_idx, :, :3]   # (T_sim, 3)
    T_pf, T_sim = est.shape[0], tru.shape[0]

    # time axis for PF
    if 't_pf' in monte_data:
        t = np.asarray(monte_data['t_pf'])[:T_pf]
        xlabel = 'Time (s)'
    else:
        if imu_hz is None: imu_hz = monte_data.get('imu_hz', None)
        if sim_hz is None: sim_hz = monte_data.get('sim_hz', None)
        if imu_hz:
            t = np.arange(T_pf, dtype=float) / float(imu_hz)
            xlabel = 'Time (s)'
        elif 'dt_pf' in monte_data and monte_data['dt_pf']:
            t = np.arange(T_pf, dtype=float) * float(monte_data['dt_pf'])
            xlabel = 'Time (s)'
        else:
            t = np.arange(T_pf, dtype=float)
            xlabel = 'Sample'

    # align truth to PF length
    if T_sim == T_pf:
        tru_pf = tru
    elif (sim_hz is not None) and (imu_hz is not None) and (sim_hz % imu_hz == 0):
        step = int(sim_hz // imu_hz)
        tru_pf = tru[::step][:T_pf]
    else:
        # interpolate truth by index onto PF grid
        idx_sim = np.arange(T_sim, dtype=float)
        idx_pf  = np.linspace(0, T_sim - 1, T_pf)
        tru_pf = np.column_stack([np.interp(idx_pf, idx_sim, tru[:, i]) for i in range(3)])

    # plot
    labels = ['x', 'y', 'z']
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, est[:, i], label=f'{labels[i]} estimate')
        ax.plot(t, tru_pf[:, i], linestyle='--', label=f'{labels[i]} truth')
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f'PF Position Estimate vs Truth (run {run_idx})')
    fig.tight_layout()
    plt.show()

    return {'t': t, 'est_xyz': est, 'truth_xyz': tru_pf}

def plot_pf_state_errors_all_runs(monte_data, output_dir='sim_results', sim_hz=sim_hz, imu_hz=imu_hz,
                                  state_labels=None, dpi=150, show=False):
    """
    For each state dimension s, plot ALL runs' errors e_r^s(t) on one chart and save as JPG.

    Expects (after PF run):
      monte_data['state_sum']  : (R, T_sim, S)  true states
      monte_data['x_estimate'] : (R, T_pf,  S)  estimated states

    Saves:
      <output_dir>/state_<label_or_index>_errors.jpg

    Also adds to monte_data:
      monte_data['err'] : (R, T_pf, S)  (est - true) aligned to PF timestamps
    """
    # ---- pull arrays ----
    x_true_all = monte_data['state_sum']      # (R, T_sim, S)
    x_hat_all  = monte_data['x_estimate']     # (R, T_pf,  S)

    R, T_sim, S = x_true_all.shape
    R2, T_pf, S2 = x_hat_all.shape
    if not (R == R2 and S == S2):
        raise ValueError(f"Shape mismatch: true {x_true_all.shape} vs est {x_hat_all.shape}")

    # ---- time alignment (nearest) ----
    t_pf  = np.arange(T_pf)  / float(imu_hz)
    sim_idx = np.clip(np.rint(t_pf * sim_hz).astype(int), 0, T_sim - 1)
    x_true_at_pf = x_true_all[:, sim_idx, :]  # (R, T_pf, S)

    # ---- errors ----
    err = x_hat_all - x_true_at_pf           # (R, T_pf, S)
    monte_data['err'] = err                   # stash for later use

    # ---- labels ----
    if state_labels is None:
        state_labels = [f"s{i}" for i in range(S)]
    elif len(state_labels) != S:
        raise ValueError(f"state_labels length {len(state_labels)} != S {S}")

    # ---- output dir ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- one figure per state: plot all runs ----
    for s in range(S):
        plt.figure()
        for r in range(R):
            plt.plot(t_pf, err[r, :, s], linewidth=1.0)
        plt.xlabel("Time [s]")
        plt.ylabel(f"{state_labels[s]} Error")
        plt.title(f"Per-Run Errors for state: {state_labels[s]}  (Num Runs={R})")
        plt.grid(True)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"state_{state_labels[s]}_errors.jpg")
        plt.savefig(out_path, dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close()

    return err, t_pf

def _set_axes_equal(ax):
    # Make 3D axes have equal scale
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits); y_mid = np.mean(y_limits); z_mid = np.mean(z_limits)
    ax.set_xlim3d(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim3d(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim3d(z_mid - max_range/2, z_mid + max_range/2)

def plot_trajectories_monte(monte_data, fig_num=1, save_as_png=False, dpi=300,
                            outfile="rover_trajectories.png",
                            CENTER=None, BEACONS=None, title="Trajectories"):
    """
    Plots all true trajectories from monte_data['state_sum'] (R, T_sim, 6).

    Args:
      monte_data: dict with key 'state_sum' of shape (R, T, 6)
      CENTER: optional (3,) beacon center to mark with an 'X'
      BEACONS: optional (M,3) array of beacon positions to scatter
    """
    S_all = monte_data['state_sum']  # (R, T, 6)
    R, T, S = S_all.shape
    assert S >= 3, "Expect at least x,y,z in state"

    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')

    # Plot beacons or center marker if provided
    if BEACONS is not None:
        BEACONS = np.asarray(BEACONS, float)
        ax.scatter(BEACONS[:,0], BEACONS[:,1], BEACONS[:,2],
                   s=40, marker='^', label='Beacons', zorder=5)
    elif CENTER is not None:
        ax.scatter([CENTER[0]], [CENTER[1]], [CENTER[2]],
                   s=80, marker='X', label='Beacon', zorder=5)

    # Plot each run's true trajectory
    for r in range(R):
        S_run = S_all[r]              # (T, 6)
        ax.plot(S_run[:,0], S_run[:,1], S_run[:,2], linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if (BEACONS is not None) or (CENTER is not None):
        ax.legend(loc='upper left')

    # Make axes equal so shapes aren’t distorted
    _set_axes_equal(ax)

    if save_as_png:
        plt.savefig(outfile, format='png', dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_pf_weight_pdfs_vs_time(
    monte_data,
    run_idx=0,
    output_dir='sim_results',
    dpi=200,
    sim_hz=sim_hz,
    imu_hz=imu_hz,
    show=False,
):
    """
    For a single run, plot where the particles actually are over time, with
    brightness (grayscale intensity) proportional to particle weight.

    Produces and saves:
        sim_results/state_x_pdf_run<idx>.jpg
        sim_results/state_y_pdf_run<idx>.jpg

    Each figure:
      - x-axis: time [s]
      - y-axis: state value (x or y)
      - points: particles (t, state_value) with brightness ~ weight
      - white dashed: truth
      - red: PF estimate
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    x_k_all = np.asarray(monte_data['x_k'])        # (R, T_pf, NUM_STATES, N)
    w_k_all = np.asarray(monte_data['w_k'])        # (R, T_pf, N)
    x_est_all = np.asarray(monte_data['x_estimate'])  # (R, T_pf, NUM_STATES)
    S_all = np.asarray(monte_data['state_sum'])    # (R, T_sim, 6)

    # Ensure run dimension
    if x_k_all.ndim == 3:
        x_k_all = x_k_all[None, ...]
        w_k_all = w_k_all[None, ...]
        x_est_all = x_est_all[None, ...]
        S_all = S_all[None, ...]

    Xk = x_k_all[run_idx]        # (T_pf, S, N)
    Wk = w_k_all[run_idx]        # (T_pf, N)
    Xest = x_est_all[run_idx]    # (T_pf, S)
    Xtrue = S_all[run_idx]       # (T_sim, 6)

    T_pf, S_dim, N = Xk.shape
    T_sim = Xtrue.shape[0]

    # PF time axis
    t_pf = np.arange(T_pf, dtype=float) / float(imu_hz)

    # Align truth to PF timestamps
    sim_idx = np.clip(np.rint(t_pf * sim_hz).astype(int), 0, T_sim - 1)
    Xtrue_at_pf = Xtrue[sim_idx, :]   # (T_pf, 6)

    # x = state 0, y = state 1
    state_indices = [0, 1]
    labels = ['x', 'y']
    limits = [(X_LIM[0], X_LIM[1]), (Y_LIM[0], Y_LIM[1])]

    for s_idx, label, (vmin, vmax) in zip(state_indices, labels, limits):
        # Particles for this state over time: (T_pf, N)
        vals = Xk[:, s_idx, :]
        weights = Wk.copy()

        # Normalize weights per time step
        for t_idx in range(T_pf):
            w_t = weights[t_idx, :]
            s = w_t.sum()
            if (not np.isfinite(s)) or (s <= 0.0):
                weights[t_idx, :] = 1.0 / N
            else:
                weights[t_idx, :] = w_t / s

        # Flatten for scatter: (T_pf * N,)
        t_grid = np.repeat(t_pf, N)
        v_flat = vals.reshape(-1)
        w_flat = weights.reshape(-1)

        # Normalize weights globally for color [0,1]
        w_max = np.max(w_flat)
        if (not np.isfinite(w_max)) or (w_max <= 0.0):
            w_norm = np.zeros_like(w_flat)
        else:
            w_norm = w_flat / w_max

        plt.figure(figsize=(9, 5))

        # Scatter with brightness (grayscale) = weight
        sc = plt.scatter(
            t_grid,
            v_flat,
            s=15.0,                # fixed size
            c=w_norm,              # color encodes weight
            cmap='Greys',          # darker=low, brighter=high
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
        )

        # Overlay truth and estimate
        plt.plot(
            t_pf,
            Xtrue_at_pf[:, s_idx],
            linestyle='--',
            color='cyan',
            linewidth=1.2,
            label=f'{label} truth'
        )
        plt.plot(
            t_pf,
            Xest[:, s_idx],
            color='red',
            linewidth=1.0,
            label=f'{label} estimate'
        )

        plt.ylim(vmin, vmax)
        plt.xlabel('Time [s]')
        plt.ylabel(f'{label}(t)')
        plt.title(f'Particle weights vs time ({label}, run {run_idx})')
        plt.legend(loc='upper right')

        cbar = plt.colorbar(sc)
        cbar.set_label('normalized particle weight')

        plt.tight_layout()

        out_path = os.path.join(output_dir, f'state_{label}_pdf_run{run_idx}.jpg')
        plt.savefig(out_path, dpi=dpi)
        if show:
            plt.show()
        else:
            plt.close()

def weighted_percentiles(values, weights, percentiles=[0.025, 0.5, 0.975]):
    """
    Compute weighted percentiles of a 1D sample.

    Args:
        values:      array-like, shape (N,)
        weights:     array-like, shape (N,)  — weights must be >= 0
        percentiles: list of percentiles in [0,1], e.g. [0.025, 0.5, 0.975]

    Returns:
        A list of percentile values in the same order as `percentiles`.
    """
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Safety handling
    if len(values) == 0:
        return [np.nan for _ in percentiles]

    # Replace invalid weights (negative, NaN, inf)
    weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)

    # Normalize weights
    w_sum = weights.sum()
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / w_sum

    # Sort by value
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]

    # Cumulative distribution
    cw = np.cumsum(w)

    # Extract percentiles
    results = []
    for p in percentiles:
        # Ensure p is within valid bounds
        p = max(0.0, min(1.0, p))
        i = np.searchsorted(cw, p)
        i = min(max(i, 0), len(v) - 1)
        results.append(v[i])

    return results

def plot_pf_error_pdf_band_all_runs_about_estimate(
    monte_data,
    state_labels=None,
    output_dir="sim_results",
    dpi=150
):
    """
    One plot per state:

      - Shaded 3-sigma band for the *posterior error* (true - estimate),
        approximated by particles around the PF estimate:

            dev_p = x_particle - x_estimate

        The band is derived from the PDF of dev_p at each time using ALL runs.

      - All run errors (estimate - truth) plotted in black on top:

            err_run = x_estimate - x_true

      - No median line, just band + black error curves + zero line.

    Expects:
      monte_data["x_k"]        : (R, T_pf, S, Np)  particle states
      monte_data["w_k"]        : (R, T_pf, Np)     particle weights
      monte_data["x_estimate"] : (R, T_pf, S)      PF estimates
      monte_data["state_sum"]  : (R, T_sim, S)     true states
      monte_data["imu_hz"]     : scalar
      monte_data["sim_hz"]     : scalar
    """
    os.makedirs(output_dir, exist_ok=True)

    x_k_all    = np.asarray(monte_data["x_k"])        # (R, T_pf, S, Np)
    w_k_all    = np.asarray(monte_data["w_k"])        # (R, T_pf, Np)
    x_est_all  = np.asarray(monte_data["x_estimate"]) # (R, T_pf, S)
    x_true_all = np.asarray(monte_data["state_sum"])  # (R, T_sim, S)

    R, T_pf, S, Np = x_k_all.shape
    _, T_sim, _    = x_true_all.shape

    imu_hz = monte_data.get("imu_hz", 50.0)
    sim_hz = monte_data.get("sim_hz", 100.0)

    # PF time axis
    t_pf = np.arange(T_pf, dtype=float) / float(imu_hz)

    # Align truth to PF timestamps (for errors)
    sim_idx = np.clip(np.rint(t_pf * sim_hz).astype(int), 0, T_sim - 1)
    x_true_pf = x_true_all[:, sim_idx, :]   # (R, T_pf, S)

    # Labels
    if state_labels is None:
        state_labels = [f"s{i}" for i in range(S)]
    elif len(state_labels) != S:
        raise ValueError(f"state_labels length {len(state_labels)} != S {S}")

    # 3-sigma equivalent percentiles for (true - estimate)
    p_low  = 0.00135
    p_high = 1.0 - p_low    # 0.99865

    for s_idx in range(S):
        label = state_labels[s_idx]

        # ----------------------------
        # Build time-varying 3-sigma band for dev = (true - estimate)
        # approximated by particles around the estimate:
        #
        #   dev_p = x_particle - x_estimate
        #
        # using ALL runs at each time step.
        # ----------------------------
        low_band  = np.zeros(T_pf)
        high_band = np.zeros(T_pf)

        for t_idx in range(T_pf):
            devs_t_list = []
            w_t_list    = []

            for r in range(R):
                # Particle values and PF estimate at (r, t, s)
                vals = x_k_all[r, t_idx, s_idx, :]       # (Np,)
                est  = x_est_all[r, t_idx, s_idx]

                # Deviation of particle from estimate: dev ≈ (true - estimate)
                dev  = vals - est                        # (Np,)
                w    = w_k_all[r, t_idx, :]

                # Normalize weights for this (r, t)
                w_sum = np.sum(w)
                if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
                    w_norm = np.ones_like(w) / float(len(w))
                else:
                    w_norm = w / w_sum

                devs_t_list.append(dev)
                w_t_list.append(w_norm)

            # Combine across runs at this time
            devs_t = np.concatenate(devs_t_list)  # (R*Np,)
            w_t    = np.concatenate(w_t_list)
            w_t_sum = np.sum(w_t)
            if (not np.isfinite(w_t_sum)) or (w_t_sum <= 0.0):
                w_t = np.ones_like(w_t) / float(len(w_t))
            else:
                w_t = w_t / w_t_sum

            # Weighted percentiles for dev = (true - estimate)
            dev_lo, dev_hi = weighted_percentiles(
                devs_t, w_t, percentiles=[p_low, p_high]
            )
            low_band[t_idx]  = dev_lo
            high_band[t_idx] = dev_hi

        # ----------------------------
        # All run errors (estimate - truth) for plotting in black
        # ----------------------------
        err_runs = x_est_all[:, :, s_idx] - x_true_pf[:, :, s_idx]  # (R, T_pf)

        # y-limits from both band and actual errors
        all_vals = np.concatenate(
            [err_runs.ravel(), low_band, high_band]
        )
        e_min = np.quantile(all_vals, 0.01)
        e_max = np.quantile(all_vals, 0.99)
        pad = 0.1 * (e_max - e_min + 1e-12)

        # ----------------------------
        # Final plot
        # ----------------------------
        plt.figure(figsize=(10, 5))

        # Shaded 3-sigma band for (true - estimate) inferred from particles
        # (centered around 0 in "expected error" space)
        plt.fill_between(
            t_pf, low_band, high_band,
            color="C0", alpha=0.25,
            label="3-sigma band (PF posterior error about estimate)"
        )

        # All run errors (estimate - truth) in black
        for r in range(R):
            plt.plot(
                t_pf, err_runs[r, :],
                color="black", alpha=0.4, linewidth=0.8
            )

        # Zero-error line (where we *hope* errors lie)
        plt.axhline(
            0.0, color="k", linestyle="--", linewidth=1.0
        )

        plt.ylim(e_min - pad, e_max + pad)
        plt.xlabel("Time [s]")
        plt.ylabel(f"{label} error")
        plt.title(f"Error vs Time with PF PDF 3-sigma Band About Estimate — {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"state_{label}_error_band_about_est.jpg")
        plt.savefig(out_path, dpi=dpi)
        plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":

    monte_data = generate_trajectories()

    # ----- STANDARD PARTICLE FILTER ONLY -----
    monte_data = particle_filter.run_pf_for_all_runs(monte_data)
    # New “full PDF in error space” plots
    plot_pf_error_pdf_band_all_runs_about_estimate(
        monte_data,
        state_labels=["x", "y", "z", "vx", "vy", "vz"],  # or whatever S is
        output_dir="sim_results_error_band"
    )   
