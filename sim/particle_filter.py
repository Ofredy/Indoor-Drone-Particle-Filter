import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt

from monte import sim_hz
from system_model import *

# Particle Filter Constants
NUM_PARTICLES = 500
PERCENT_EFFECTIVE = 0.2
NUM_EFFECTIVE_THRESHOLD = int( NUM_PARTICLES * PERCENT_EFFECTIVE )

_rng = np.random.default_rng()
BETA = 0.2
REJUVENATION_VARIANCE = 1e-3   # variance for resampling jitter (position & velocity spread)
REJUVENATION_SCALE = 0.05

INCONSISTENCY_PROB = 0.995   # 99% chi-square gate
EXPANSION_POS_FRAC = 0.05   # 2% of room extent per axis as std for x,y,z
EXPANSION_VEL_STD  = 0.15   # std for vx, vy, vz during expansion

pos_0_std = 0.25
vel_0_std = 0.5


def pf_init():
    """
    Initialize PF particles uniformly within the room bounds.
    Ignores any measurement; use this when you want a pure room-uniform prior.

    Returns:
        x_0_all : (NUM_STATES, NUM_PARTICLES)
        w_0_all : (NUM_PARTICLES,)
    """
    # Positions: uniform across the room
    x = np.random.uniform(X_LIM[0], X_LIM[1], size=NUM_PARTICLES)
    y = np.random.uniform(Y_LIM[0], Y_LIM[1], size=NUM_PARTICLES)
    z = np.random.uniform(Z_LIM[0], Z_LIM[1], size=NUM_PARTICLES)
    pos = np.vstack([x, y, z])  # (3, N)

    # Velocities: small Gaussian around 0
    vx = np.random.normal(0.0, vel_0_std, size=NUM_PARTICLES)
    vy = np.random.normal(0.0, vel_0_std, size=NUM_PARTICLES)
    vz = np.random.normal(0.0, vel_0_std, size=NUM_PARTICLES)

    # State stack
    x_0_all = np.zeros((NUM_STATES, NUM_PARTICLES))
    x_0_all[0:3, :] = pos
    x_0_all[3,   :] = vx
    x_0_all[4,   :] = vy
    x_0_all[5,   :] = vz

    # Uniform weights
    w_0_all = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
    return x_0_all, w_0_all

# takes in particles & accelerametor measurement & gives back the new state of particles
def prediction_step(x_k, u_k):
    """
    x_k : (6, N) particle states [x, y, z, vx, vy, vz]
    u_k : (3,)   control acceleration input (e.g., commanded accel)

    Uses globals:
        A, B,
        pf_dt,              # time step of the PF dynamics [s]
        process_noise_std,  # interpreted as accel noise std [m/s^2]
        X_LIM, Y_LIM, Z_LIM,
        reinject_oob_from_valid_highweight
    """
    # -----------------------------
    # Deterministic motion update
    # -----------------------------
    x_pred = A @ x_k + (B @ u_k)[:, None]

    # -----------------------------
    # Process noise: random accel model
    # -----------------------------
    # Interpret process_noise_std as σ_a (accel noise std)
    sigma_a = float(process_noise_std)   # [m/s^2]

    # For each axis, with state [x, y, z, vx, vy, vz]:
    # diag(Qd) ≈ [ (dt^4/4)*σ_a^2, (dt^4/4)*σ_a^2, (dt^4/4)*σ_a^2,
    #              (dt^2)*σ_a^2,   (dt^2)*σ_a^2,   (dt^2)*σ_a^2 ]
    dt = pf_dt
    pos_std = 0.5 * (dt**2) * sigma_a      # sqrt(dt^4/4 * σ_a^2)
    vel_std = dt * sigma_a                 # sqrt(dt^2 * σ_a^2)

    # Build per-state std vector (6,)
    diag_std = np.array([
        pos_std, pos_std, pos_std,
        vel_std, vel_std, vel_std
    ], dtype=float)

    # Draw Gaussian noise per state dimension and particle
    noise = np.random.normal(
        loc=0.0,
        scale=diag_std[:, None],   # broadcast over particles
        size=x_pred.shape
    )
    x_pred = x_pred + noise

    # --------------------------------------------------------
    # Enforce bounds: clone valid high-weight parents for OOB
    # (here we don't have weights yet, so use uniform weights)
    # --------------------------------------------------------
    x_pred, w_pred = reinject_oob_from_valid_highweight(
        x_pred,
        np.ones(x_pred.shape[1], dtype=float) / x_pred.shape[1],
        bounds=((X_LIM[0], X_LIM[1]),
                (Y_LIM[0], Y_LIM[1]),
                (Z_LIM[0], Z_LIM[1]))
    )

    return x_pred, w_pred

def jitter_particles_diag(X, rejuvenation_variance=1e-4, kappa=0.25, rng=_rng):
    """
    Particle rejuvenation (a.k.a. jittering) to prevent sample impoverishment.

    Adds small Gaussian noise to each particle:
        X_new = X + N(0, kappa * diag(rejuvenation_variance))

    Args:
        X : (d, N)
            Particle matrix (d = state dimension, N = number of particles)
        rejuvenation_variance : float or (d,)
            Variance for each state dimension (separate from process noise).
            Controls how much diversity is reintroduced after resampling.
        kappa : float
            Scale factor for the noise intensity (0.1–0.3 is typical).
        rng : np.random.Generator
            Random number generator (default _rng).

    Returns:
        X_new : (d, N)
            Jittered particle set.
    """
    X = np.asarray(X, dtype=np.float64)
    d, N = X.shape

    # Ensure variance is per-dimension
    if np.isscalar(rejuvenation_variance):
        var_vec = float(rejuvenation_variance) * np.ones(d, dtype=np.float64)
    else:
        var_vec = np.asarray(rejuvenation_variance, dtype=np.float64)
        if var_vec.shape != (d,):
            raise ValueError(f"rejuvenation_variance must be scalar or shape ({d},), got {var_vec.shape}")

    # Compute per-state stddev with scaling
    std_vec = np.sqrt(np.maximum(0.0, kappa) * np.maximum(var_vec, 0.0))

    # Sample Gaussian noise for each particle
    noise = rng.standard_normal(size=(d, N)) * std_vec[:, None]

    return X + noise

def residual_resample(weights, rng=_rng):
    """
    Residual resampling (low variance):
    - Deterministically allocate floor(N * w_i) copies.
    - Multinomial draw the remaining R copies from residual probs.
    Returns: indices (N,)
    """
    w = np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        # fallback: uniform
        N = len(w)
        return rng.integers(low=0, high=N, size=N, endpoint=False)

    w = w / w_sum
    N = w.size

    Ns = np.floor(N * w).astype(int)          # integer copy counts
    R = N - Ns.sum()                           # how many left to draw
    idx = np.repeat(np.arange(N), Ns)          # deterministic part

    if R > 0:
        residual = N * w - Ns                  # fractional leftovers
        res_sum = residual.sum()
        if res_sum > 0 and np.isfinite(res_sum):
            p = residual / res_sum
        else:
            p = np.full(N, 1.0 / N)
        idx_res = rng.choice(N, size=R, replace=True, p=p)
        idx = np.concatenate([idx, idx_res])

    rng.shuffle(idx)                           # avoid ordering bias
    return idx

def update_step(sensor_measurement, x_k, w_k):
    """
    sensor_measurement : (M,)  vector of ranges (one per beacon)
    x_k                : (6, N) particles after prediction
    w_k                : (N,)   prior weights

    Uses globals:
      BEACONS, measurement_noise_variance,
      NUM_PARTICLES, NUM_EFFECTIVE_THRESHOLD,
      INCONSISTENCY_PROB,
      REJUVENATION_SCALE, REJUVENATION_VARIANCE,
      EXPANSION_POS_FRAC, EXPANSION_VEL_STD,
      X_LIM, Y_LIM, Z_LIM
    """
    # -------------------------
    # Likelihood / weight update
    # -------------------------
    # positions and predicted ranges to each beacon
    positions = x_k[0:3, :]                              # (3, N)
    diff = positions.T[None, :, :] - BEACONS[:, None, :] # (M, N, 3)
    d_hat = np.linalg.norm(diff, axis=2)                 # (M, N)

    # measurement noise: scalar or per-beacon
    sigma = np.sqrt(np.asarray(measurement_noise_variance, float))
    sigma = np.broadcast_to(sigma, sensor_measurement.shape)  # (M,)

    # per-beacon likelihoods -> (M, N)
    lk_per = norm.pdf(sensor_measurement[:, None], loc=d_hat, scale=sigma[:, None])

    # combine beacons: product across M -> (N,)
    lk = np.prod(np.maximum(lk_per, 1e-300), axis=0)

    # weight update (elementwise) + robust normalize
    w_k = w_k * (lk ** BETA)
    s = w_k.sum()
    if not np.isfinite(s) or s <= 0.0:
        w_k = np.full_like(w_k, 1.0 / w_k.size)
    else:
        w_k /= s

    # -------------------------------------------
    # Inconsistency check (chi-square on mean est)
    # -------------------------------------------
    # Weighted-mean state
    mu = estimate_state_from_particles(x_k, w_k)  # (6,)
    mu_pos = mu[0:3]                              # (3,)

    # Predicted ranges from mean position
    z_hat_mu = np.linalg.norm(BEACONS - mu_pos[None, :], axis=1)  # (M,)

    # Residual and chi-square statistic
    resid = sensor_measurement - z_hat_mu          # (M,)
    chi2_stat = np.sum((resid / sigma) ** 2)
    chi2_thr  = chi2.ppf(INCONSISTENCY_PROB, df=sensor_measurement.size)

    if chi2_stat > chi2_thr:
        # ======================================================
        # BIG mismatch: expand around *current mean* and reset
        # ======================================================
        N = x_k.shape[1]

        # Use room size only to set the scale of expansion
        room_extent = np.array(
            [X_LIM[1] - X_LIM[0],
             Y_LIM[1] - Y_LIM[0],
             Z_LIM[1] - Z_LIM[0]],
            dtype=np.float64
        )
        pos_std = EXPANSION_POS_FRAC * room_extent   # (3,)

        # Draw new particles around current mean state
        pos_noise = _rng.standard_normal(size=(3, N)) * pos_std[:, None]
        vel_noise = _rng.standard_normal(size=(3, N)) * EXPANSION_VEL_STD

        x_k = np.vstack([
            mu_pos[:, None]    + pos_noise,      # positions
            mu[3:6, None]      + vel_noise      # velocities
        ])

        # Clip positions back into bounds
        x_k[0, :] = np.clip(x_k[0, :], X_LIM[0], X_LIM[1])
        x_k[1, :] = np.clip(x_k[1, :], Y_LIM[0], Y_LIM[1])
        x_k[2, :] = np.clip(x_k[2, :], Z_LIM[0], Z_LIM[1])

        # Reset weights to uniform
        w_k = np.full(N, 1.0 / N, dtype=np.float64)

        # After a full "re-expand", neff is high (uniform), so
        # the degeneracy test below won't immediately resample.

    # ---------------------------------
    # Regular degeneracy-based resample
    # ---------------------------------
    eff_particles = 1.0 / np.sum(w_k**2)
    if eff_particles <= NUM_EFFECTIVE_THRESHOLD:
        # residual resampling
        idx = residual_resample(w_k)
        x_k = x_k[:, idx]

        # reset weights
        w_k = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES, dtype=np.float64)

        # mild jitter for diversity (uses your new constants)
        x_k = jitter_particles_diag(
            x_k,
            REJUVENATION_VARIANCE,
            kappa=REJUVENATION_SCALE
        )

    # ------------------------------
    # Ensure positions remain valid
    # ------------------------------
    x_k, w_k = reinject_oob_from_valid_highweight(
        x_k, w_k,
        bounds=((X_LIM[0], X_LIM[1]),
                (Y_LIM[0], Y_LIM[1]),
                (Z_LIM[0], Z_LIM[1]))
    )

    return x_k, w_k

def reinject_oob_from_valid_highweight(X, w_ref, bounds, rng=_rng, pos_jitter_std=None):
    """
    Enforce position-only bounds by replacing OOB particles with copies
    drawn from valid, high-weight particles (optionally with tiny pos jitter).

    Args:
        X : (d, N) particle states [x,y,z,vx,vy,vz]^T
        w_ref : (N,) reference weights to bias selection (use pre-update weights)
        bounds : ((x_lo,x_hi), (y_lo,y_hi), (z_lo,z_hi))
        pos_jitter_std : float or (3,), std dev for position-only jitter (optional)

    Returns:
        X_new : (d, N)
        w_new : (N,)
    """
    X = np.asarray(X, dtype=np.float64).copy()
    w_ref = np.asarray(w_ref, dtype=np.float64).copy()
    d, N = X.shape
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = bounds

    pos = X[0:3, :]  # (3, N)
    valid = (
        (pos[0] >= x_lo) & (pos[0] <= x_hi) &
        (pos[1] >= y_lo) & (pos[1] <= y_hi) &
        (pos[2] >= z_lo) & (pos[2] <= z_hi)
    )
    invalid_idx = np.where(~valid)[0]

    # 🔑 If nothing is out-of-bounds, don't touch weights or states
    if invalid_idx.size == 0:
        return X, w_ref

    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        # fallback: re-spawn positions uniformly within bounds, keep velocities as-is
        X[0, invalid_idx] = rng.uniform(x_lo, x_hi, size=invalid_idx.size)
        X[1, invalid_idx] = rng.uniform(y_lo, y_hi, size=invalid_idx.size)
        X[2, invalid_idx] = rng.uniform(z_lo, z_hi, size=invalid_idx.size)
        w_ref[:] = 1.0 / N
        return X, w_ref

    # sample parent particles among valid ones with prob ~ w_ref
    w = np.asarray(w_ref, dtype=np.float64)
    p = w[valid_idx].clip(min=0)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        p = np.full(valid_idx.size, 1.0 / valid_idx.size)
    else:
        p = p / s

    parents = rng.choice(valid_idx, size=invalid_idx.size, replace=True, p=p)
    X[:, invalid_idx] = X[:, parents]                 # copy state
    w_ref[invalid_idx] = w_ref[parents]               # copy weights

    # tiny optional position jitter to avoid exact duplicates
    if pos_jitter_std is None:
        xr, yr, zr = (x_hi - x_lo), (y_hi - y_lo), (z_hi - z_lo)
        pos_jitter_std = np.array([xr, yr, zr]) * 1e-3  # ~0.1% of room size
    pos_jitter_std = np.atleast_1d(pos_jitter_std).astype(float)
    if pos_jitter_std.size == 1:
        pos_jitter_std = np.repeat(pos_jitter_std[0], 3)

    jitter = rng.standard_normal(size=(3, invalid_idx.size)) * pos_jitter_std[:, None]
    X[0:3, invalid_idx] += jitter

    # ensure we’re back inside bounds after jitter
    X[0, :] = np.clip(X[0, :], x_lo, x_hi)
    X[1, :] = np.clip(X[1, :], y_lo, y_hi)
    X[2, :] = np.clip(X[2, :], z_lo, z_hi)

    # ✅ re-normalize weights only when we actually changed some of them
    w_sum = np.sum(w_ref)
    if w_sum <= 0 or not np.isfinite(w_sum):
        w_ref[:] = 1.0 / N
    else:
        w_ref /= w_sum

    return X, w_ref

def weighted_median(values, weights):
    """Compute the weighted median of a 1D array."""
    sorter = np.argsort(values)
    values, weights = values[sorter], weights[sorter]
    cdf = np.cumsum(weights) / np.sum(weights)
    return values[np.searchsorted(cdf, 0.5)]

def plot_pred_update_step(truth_pos, x_pred, w_pred, x_post, w_post, beacons=BEACONS, step_idx=None):
    """
    Debug one PF step in 3D: compare prediction vs update and show beacons.
      truth_pos : (3,) true [x,y,z] at this time
      x_pred    : (NUM_STATES, N) particles AFTER prediction (prior)
      w_pred    : (N,)           weights BEFORE update   (prior)
      x_post    : (NUM_STATES, N) particles AFTER update (posterior)
      w_post    : (N,)           weights AFTER update    (posterior)
    Uses global BEACONS = (M,3). Uses X_LIM/Y_LIM/Z_LIM if present.
    """
    def _norm(w):
        w = np.asarray(w).astype(float)
        s = np.sum(w)
        return (np.ones_like(w)/w.size) if (not np.isfinite(s) or s <= 0) else (w/s)

    def _sizes(w):
        w = w / (w.max() + 1e-12)
        return 6.0 + 120.0 * w

    def _set_bounds(ax):
        try:
            ax.set_xlim(X_LIM[0], X_LIM[1])
            ax.set_ylim(Y_LIM[0], Y_LIM[1])
            ax.set_zlim(Z_LIM[0], Z_LIM[1])
        except Exception:
            # fallback: auto data bounds with padding
            all_pts = np.column_stack([pos_pred, pos_post, truth_pos.reshape(3,1)])
            mn = all_pts.min(axis=1); mx = all_pts.max(axis=1)
            pad = 0.05 * (mx - mn + 1e-6)
            ax.set_xlim(mn[0]-pad[0], mx[0]+pad[0])
            ax.set_ylim(mn[1]-pad[1], mx[1]+pad[1])
            ax.set_zlim(mn[2]-pad[2], mx[2]+pad[2])

    w_pred = _norm(w_pred)
    w_post = _norm(w_post)

    pos_pred = x_pred[0:3, :]   # (3, N)
    pos_post = x_post[0:3, :]   # (3, N)

    # use the same estimator (mean vs median based on # beacons)
    mu_pred_full = estimate_state_from_particles(x_pred, w_pred)  # (6,)
    mu_post_full = estimate_state_from_particles(x_post, w_post)  # (6,)

    mu_pred = mu_pred_full[0:3]
    mu_post = mu_post_full[0:3]

    e_pred = np.linalg.norm(mu_pred - truth_pos)
    e_post = np.linalg.norm(mu_post - truth_pos)

    B = None
    if 'BEACONS' in globals():
        B = np.asarray(beacons)
        if B.ndim != 2 or B.shape[1] < 3:
            B = None

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # PRIOR (3D)
    ax1.scatter(pos_pred[0], pos_pred[1], pos_pred[2], s=_sizes(w_pred), alpha=0.28, label='particles')
    ax1.scatter([truth_pos[0]], [truth_pos[1]], [truth_pos[2]], marker='*', s=160, label='truth', zorder=5)
    ax1.scatter([mu_pred[0]], [mu_pred[1]], [mu_pred[2]], marker='o', s=80, label='mean', zorder=6)
    if B is not None:
        ax1.scatter(B[:,0], B[:,1], B[:,2], marker='X', s=100, label='beacons', zorder=7)
        for i, (bx, by, bz) in enumerate(B):
            ax1.text(bx, by, bz, f'B{i}', fontsize=8, ha='left', va='bottom')

    ax1.set_title(f'Prediction (prior){"" if step_idx is None else f" | step {step_idx}"}\n‖mean error‖={e_pred:.3g}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z'); ax1.grid(True, alpha=0.3); ax1.legend(loc='upper left')
    _set_bounds(ax1)

    # POSTERIOR (3D)
    ax2.scatter(pos_post[0], pos_post[1], pos_post[2], s=_sizes(w_post), alpha=0.28, label='particles')
    ax2.scatter([truth_pos[0]], [truth_pos[1]], [truth_pos[2]], marker='*', s=160, label='truth', zorder=5)
    ax2.scatter([mu_post[0]], [mu_post[1]], [mu_post[2]], marker='o', s=80, label='mean', zorder=6)
    if B is not None:
        ax2.scatter(B[:,0], B[:,1], B[:,2], marker='X', s=100, label='beacons', zorder=7)
        for i, (bx, by, bz) in enumerate(B):
            ax2.text(bx, by, bz, f'B{i}', fontsize=8, ha='left', va='bottom')

    ax2.set_title(f'Update (posterior){"" if step_idx is None else f" | step {step_idx}"}\n‖mean error‖={e_post:.3g}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z'); ax2.grid(True, alpha=0.3); ax2.legend(loc='upper left')
    _set_bounds(ax2)

    plt.show()

    return {
        'mean_prior':  mu_pred,
        'mean_post':   mu_post,
        'err_prior':   e_pred,
        'err_post':    e_post,
    }

def estimate_state_from_particles(x_k, w_k):
    """
    Adaptive state estimate:
      - If only 1 beacon: use weighted median per state dimension.
      - If >=2 beacons:   use weighted mean per state dimension.
    """
    x_k = np.asarray(x_k, dtype=float)
    w   = np.asarray(w_k, dtype=float)

    # Normalize weights safely
    s = w.sum()
    if not np.isfinite(s) or s <= 0.0:
        w = np.full_like(w, 1.0 / w.size)
    else:
        w = w / s

    # How many beacons?
    B = np.asarray(BEACONS)
    if B.ndim == 1:
        # e.g. BEACONS is (3,) -> treat as 1 beacon
        M = 1
    else:
        M = B.shape[0]

    d, N = x_k.shape
    est = np.empty(d, dtype=float)

    if M <= 1:
        # ----- 1 beacon: use weighted median per dimension -----
        for i in range(d):
            est[i] = weighted_median(x_k[i, :], w)
    else:
        # ----- >=2 beacons: standard weighted mean -----
        est = x_k @ w

    return est

def run_pf_for_all_runs(monte_data):
    """
    Expects:
      monte_data['state_sum'] : (R, T_sim, 6)
      monte_data['acc_sum']   : (R, T_sim, 3)

    Uses globals: NUM_STATES, NUM_PARTICLES, sim_hz, imu_hz,
                  CENTER, measurement_noise_variance,
                  pf_init(), prediction_step(), update_step()
    Adds to monte_data:
      'x_k'        : (R, T_pf, NUM_STATES, NUM_PARTICLES)
      'w_k'        : (R, T_pf, NUM_PARTICLES)
      'x_estimate' : (R, T_pf, NUM_STATES)
    """
    S_all = monte_data['state_sum']   # (R, T_sim, 6)
    A_all = monte_data['acc_sum']     # (R, T_sim, 3)
    Runs, T_sim, _ = S_all.shape

    # PF update every 'step_div' sim ticks
    step_div_imu = int(sim_hz // imu_hz)
    if step_div_imu < 1:
        raise ValueError("imu_hz must be <= sim_hz and yield an integer ratio for this path.")
    T_pf = 1 + (T_sim - 1) // step_div_imu   # include t=0

    # Allow ranging_hz < 1 (e.g. 0.5 Hz = every 2 seconds)
    if ranging_hz <= 0:
        raise ValueError("ranging_hz must be positive.")

    step_div_rng = int(round(sim_hz / ranging_hz))  # works for <1Hz too

    if not np.isclose(sim_hz / ranging_hz, step_div_rng, atol=1e-6):
        raise ValueError("ranging_hz must divide sim_hz evenly (can be fractional, e.g. 0.5Hz).")

    # Allocate
    x_k_all = np.zeros((Runs, T_pf, NUM_STATES, NUM_PARTICLES))
    w_k_all = np.full((Runs, T_pf, NUM_PARTICLES), 1.0 / NUM_PARTICLES)
    x_est_all = np.zeros((Runs, T_pf, NUM_STATES))

    for r in range(Runs):
        print("run: %d/%d" % (r+1, Runs))
        traj = S_all[r]     # (T_sim, 6)
        acc  = A_all[r]     # (T_sim, 3)

        # init (t = 0)
        x_k_all[r, 0], w_k_all[r, 0] = pf_init()
        x_est_all[r, 0] = estimate_state_from_particles(
                                x_k_all[r, 0], w_k_all[r, 0]
                            )

        pf_idx = 1
        for s in range(1, T_sim):
            # only update PF on IMU ticks
            if (s % step_div_imu) != 0:
                continue

            # Prediction with current acceleration sample
            x_k_all[r, pf_idx], w_k_all[r, pf_idx - 1] = prediction_step(x_k_all[r, pf_idx - 1], acc[s])
            x_pred_dbg = x_k_all[r, pf_idx].copy()
            w_pred_dbg = w_k_all[r, pf_idx - 1].copy()

            # Range sensor measurement (to CENTER) with noise
            z = np.linalg.norm(traj[s, :3] - BEACONS, axis=1) + np.random.normal(
                0.0, np.sqrt(measurement_noise_variance)
            )

            # Measurement update (your update_step returns (x_k_new, w_k_new))
            if ( s % step_div_rng ) == 0:
                x_k_all[r, pf_idx], w_k_all[r, pf_idx] = update_step(
                    z, x_k_all[r, pf_idx], w_k_all[r, pf_idx - 1])

                #_ = plot_pred_update_step(
                #                                  truth_pos=traj[s, :3],
                #                                  x_pred=x_pred_dbg, w_pred=w_pred_dbg,
                #                                  x_post=x_k_all[r, pf_idx], w_post=w_k_all[r, pf_idx],
                #                                  step_idx=pf_idx
                #                              )

            else:
                # no update this tick: carry weights forward
                w_k_all[r, pf_idx] = w_k_all[r, pf_idx - 1]

            # State estimate as weighted mean
            x_est_all[r, pf_idx] = estimate_state_from_particles(
                                            x_k_all[r, pf_idx], w_k_all[r, pf_idx]
                                        )
            pf_idx += 1

    # Stash results back
    monte_data['x_k'] = x_k_all
    monte_data['w_k'] = w_k_all
    monte_data['x_estimate'] = x_est_all

    return monte_data
