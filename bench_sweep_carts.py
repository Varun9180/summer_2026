"""Parameter sweep for the CA-RTS smoother's (q_var, r_std) to find an
operating point that matches or beats the current PINN's fidelity while
keeping the physical-plausibility and speed advantages already observed."""
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

DT_DEFAULT = 0.05003
N_SAMPLE = 60
SEED = 11

df = pd.read_csv('smoothed_trajectories_newest.csv')
tracks = {tid: g.sort_values('time_s') for tid, g in df.groupby('track_id')}
lens = {tid: len(g) for tid, g in tracks.items() if len(g) >= 5}
rng = np.random.RandomState(SEED)
ids_all = np.array(list(lens.keys()))
sample_ids = rng.choice(ids_all, size=min(N_SAMPLE, len(ids_all)), replace=False)

def smooth_ca_rts(xy_raw, times_array, dt_median, q_var, r_std):
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
    xs, ps = mu.copy(), cov.copy()
    for k in range(n-2, -1, -1):
        F = F_hist[k+1]
        P_pred = F @ cov[k] @ F.T + Q_hist[k+1]
        C = cov[k] @ F.T @ np.linalg.pinv(P_pred)
        xs[k] = mu[k] + C @ (xs[k+1] - F @ mu[k])
    return xs[:, [0, 3]]

def metrics(raw, smooth, dt_median):
    resid = float(np.mean(np.hypot(raw[:,0]-smooth[:,0], raw[:,1]-smooth[:,1])))
    raw_len = float(np.hypot(np.diff(raw[:,0]), np.diff(raw[:,1])).sum())
    sm_len  = float(np.hypot(np.diff(smooth[:,0]), np.diff(smooth[:,1])).sum())
    retention = sm_len / raw_len if raw_len > 1e-6 else np.nan
    v = np.diff(smooth, axis=0) / dt_median
    a = np.diff(v, axis=0) / dt_median
    amag = np.hypot(a[:,0], a[:,1]) if len(a) else np.array([0.0])
    p95 = float(np.percentile(amag, 95)) if len(amag) else 0.0
    return resid, retention, p95

configs = [
    (0.5, 0.15), (1.0, 0.15), (2.0, 0.15), (4.0, 0.15), (8.0, 0.15),
    (2.0, 0.05), (2.0, 0.10), (2.0, 0.25), (2.0, 0.35),
    (4.0, 0.10), (4.0, 0.20),
]
for q_var, r_std in configs:
    rows = []
    for tid in sample_ids:
        g = tracks[tid]
        xy = g[['x_m','y_m']].values
        times = g['time_s'].values
        dt_series = g['time_s'].diff().fillna(DT_DEFAULT)
        dt_med = dt_series.median() if not pd.isna(dt_series.median()) else DT_DEFAULT
        out = smooth_ca_rts(xy, times, dt_med, q_var, r_std)
        rows.append(metrics(xy, out, dt_med))
    arr = np.array(rows)
    print(f"q_var={q_var:5.2f} r_std={r_std:5.2f}  "
          f"residual_m mean={arr[:,0].mean():.4f}  "
          f"path_retention mean={arr[:,1].mean():.4f}  "
          f"p95_accel mean={arr[:,2].mean():.2f}")
