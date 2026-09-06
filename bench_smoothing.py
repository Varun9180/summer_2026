"""Benchmark alternative trajectory-smoothing methods against the current
per-track PINN, on the SAME raw (pre-smoothing) points already saved in
smoothed_trajectories_newest.csv (columns x_m/y_m = raw, x_m_smooth/y_m_smooth
= current PINN's output). Methods compared:
  - pinn      : the exact current per-track PINN (copied from run.py)
  - ca_rts    : constant-acceleration Kalman filter + RTS backward smoother
  - whittaker : closed-form penalized least-squares (discrete smoothing
                spline, 2nd-difference / acceleration-like penalty), solved
                as one sparse banded linear system, no iteration at all

Metrics, all against the RAW points:
  - residual_m      : mean per-point distance from smoothed to raw
  - path_retention  : smoothed path length / raw path length
  - p95_accel       : 95th-percentile |acceleration| implied by the smoothed
                       path (sanity check only -- not a reliable target on
                       its own at dt=0.05s)
  - sec_per_track   : wall-clock time per track
"""
import time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DT_DEFAULT = 0.05003
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 80
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11

df = pd.read_csv('smoothed_trajectories_newest.csv')
tracks = {tid: g.sort_values('time_s') for tid, g in df.groupby('track_id')}
lens = {tid: len(g) for tid, g in tracks.items() if len(g) >= 5}
rng = np.random.RandomState(SEED)
ids_all = np.array(list(lens.keys()))
sample_ids = rng.choice(ids_all, size=min(N_SAMPLE, len(ids_all)), replace=False)

# ── method 1: current PINN (copied verbatim from run.py) ───────────────────
class TrajectoryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, t):
        return self.net(t)

_dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_ACCEL = 5.0
ACCEL_TIME_BASE = 0.30
LAMBDA_PHYS = 0.05
W_SMOOTH = 1e-5
N_ITERS = 800

def smooth_pinn(xy_raw, times_array, dt_median):
    """Verbatim port of run.py's smooth_track_with_pinn (same loss, same
    iteration count, same ACCEL_TIME_BASE stride) -- verified against
    run.py:549-609 line by line."""
    t = torch.tensor(times_array, dtype=torch.float32, device=_dev).unsqueeze(1)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
    xy_mean = np.mean(xy_raw, axis=0); xy_std = np.std(xy_raw, axis=0) + 1e-6
    noisy = torch.tensor(xy_raw, dtype=torch.float32, device=_dev)
    mean_t = torch.tensor(xy_mean, dtype=torch.float32, device=_dev)
    std_t = torch.tensor(xy_std, dtype=torch.float32, device=_dev)
    model = TrajectoryPINN().to(_dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    _k = max(1, int(round(ACCEL_TIME_BASE / max(dt_median, 1e-6))))
    for _ in range(N_ITERS):
        opt.zero_grad()
        smooth_xy = model(t_norm) * std_t + mean_t
        loss = torch.nn.functional.mse_loss(smooth_xy, noisy)
        if smooth_xy.shape[0] > 3:
            acc_fine = (smooth_xy[2:] - 2 * smooth_xy[1:-1] + smooth_xy[:-2]) / (dt_median * dt_median)
            loss = loss + W_SMOOTH * (acc_fine ** 2).mean()
        if smooth_xy.shape[0] > 2 * _k:
            vel_c = (smooth_xy[_k:] - smooth_xy[:-_k]) / (_k * dt_median)
            acc_c = (vel_c[_k:] - vel_c[:-_k]) / (_k * dt_median)
            loss = loss + LAMBDA_PHYS * torch.relu(torch.norm(acc_c, dim=1) - MAX_ACCEL).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        out = (model(t_norm) * std_t + mean_t).cpu().numpy()
    return out

# ── method 2: constant-acceleration Kalman + RTS smoother ──────────────────
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def smooth_ca_rts(xy_raw, times_array, dt_median, q_var=2.0, r_std=0.15):
    n = len(xy_raw)
    kf = KalmanFilter(dim_x=6, dim_z=2)
    dt = dt_median
    kf.F = np.array([
        [1, dt, 0.5*dt*dt, 0, 0, 0],
        [0, 1, dt, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, 0.5*dt*dt],
        [0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 1],
    ])
    kf.H = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]])
    kf.R = np.eye(2) * (r_std ** 2)
    q1 = Q_discrete_white_noise(dim=3, dt=dt, var=q_var)
    kf.Q = np.zeros((6,6)); kf.Q[:3,:3] = q1; kf.Q[3:,3:] = q1
    kf.x = np.array([xy_raw[0,0], 0, 0, xy_raw[0,1], 0, 0.])
    kf.P *= 50.0

    mu, cov, F_hist, Q_hist = [], [], [], []
    for i in range(n):
        kf.predict()
        F_hist.append(kf.F.copy()); Q_hist.append(kf.Q.copy())
        kf.update(xy_raw[i])
        mu.append(kf.x.copy()); cov.append(kf.P.copy())
    mu, cov = np.array(mu), np.array(cov)
    # RTS backward smoother
    xs, ps = mu.copy(), cov.copy()
    for k in range(n-2, -1, -1):
        F = F_hist[k+1]
        P_pred = F @ cov[k] @ F.T + Q_hist[k+1]
        C = cov[k] @ F.T @ np.linalg.pinv(P_pred)
        xs[k] = mu[k] + C @ (xs[k+1] - F @ mu[k])
        ps[k] = cov[k] + C @ (ps[k+1] - P_pred) @ C.T
    return xs[:, [0, 3]]

# ── method 3: Whittaker / discrete smoothing-spline (closed-form) ──────────
from scipy import sparse
from scipy.sparse.linalg import spsolve

def smooth_whittaker(xy_raw, times_array, dt_median, lam=8000.0):
    n = len(xy_raw)
    if n < 4:
        return xy_raw.copy()
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).tocsc()
    A = sparse.eye(n, format='csc') + lam * (D.T @ D)
    out = np.zeros_like(xy_raw)
    for d in range(2):
        out[:, d] = spsolve(A, xy_raw[:, d])
    return out

def metrics(raw, smooth, dt_median, t_sec):
    resid = float(np.mean(np.hypot(raw[:,0]-smooth[:,0], raw[:,1]-smooth[:,1])))
    raw_len = float(np.hypot(np.diff(raw[:,0]), np.diff(raw[:,1])).sum())
    sm_len  = float(np.hypot(np.diff(smooth[:,0]), np.diff(smooth[:,1])).sum())
    retention = sm_len / raw_len if raw_len > 1e-6 else np.nan
    v = np.diff(smooth, axis=0) / dt_median
    a = np.diff(v, axis=0) / dt_median
    amag = np.hypot(a[:,0], a[:,1]) if len(a) else np.array([0.0])
    p95 = float(np.percentile(amag, 95)) if len(amag) else 0.0
    return dict(residual_m=resid, path_retention=retention, p95_accel=p95, sec=t_sec)

results = {m: [] for m in ('pinn','ca_rts','whittaker')}
print(f"Benchmarking {len(sample_ids)} tracks...")
for i, tid in enumerate(sample_ids):
    g = tracks[tid]
    xy = g[['x_m','y_m']].values
    times = g['time_s'].values
    dt_series = g['time_s'].diff().fillna(DT_DEFAULT)
    dt_med = dt_series.median() if not pd.isna(dt_series.median()) else DT_DEFAULT

    t0 = time.time(); out = smooth_pinn(xy, times, dt_med); t1 = time.time()
    results['pinn'].append(metrics(xy, out, dt_med, t1-t0))

    t0 = time.time(); out = smooth_ca_rts(xy, times, dt_med); t1 = time.time()
    results['ca_rts'].append(metrics(xy, out, dt_med, t1-t0))

    t0 = time.time(); out = smooth_whittaker(xy, times, dt_med); t1 = time.time()
    results['whittaker'].append(metrics(xy, out, dt_med, t1-t0))

    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(sample_ids)} tracks done")

print()
for m, rows in results.items():
    df_m = pd.DataFrame(rows)
    print(f"=== {m} ===")
    print(f"  residual_m       mean={df_m.residual_m.mean():.4f}  median={df_m.residual_m.median():.4f}")
    print(f"  path_retention   mean={df_m.path_retention.mean():.4f}  median={df_m.path_retention.median():.4f}")
    print(f"  p95_accel(m/s2)  mean={df_m.p95_accel.mean():.2f}  median={df_m.p95_accel.median():.2f}")
    print(f"  sec/track        mean={df_m.sec.mean():.4f}  -> est. total for 2898 tracks: {df_m.sec.mean()*2898/60:.1f} min")
    print()
