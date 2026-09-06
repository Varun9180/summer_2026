"""
benchmark_and_ablation.py — standalone, read-only evaluation companion.

Produces the quantitative evidence requested for (a) ablation against the
classical modelling assumptions, (b) degeneracy behaviour of classical TTC
versus the Continuous Criticality Index, and (c) computational cost.

Outputs
  ablation_table.csv        motion-model and agent-geometry ablations
  degeneracy_table.csv      TTC definedness vs speed / heading-angle regime
  complexity_table.csv      measured per-operation cost and concurrency
  sat_axis_check.csv        first-axis vs max-over-axes SAT comparison
  fig_degeneracy.png        definedness curves
  benchmark_summary.txt     human-readable digest

Nothing here modifies pipeline outputs.
"""
import os, sys, time, tracemalloc
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SM   = 'smoothed_trajectories_newest.csv'
IC   = 'interactions_classified.csv'
RNG  = np.random.RandomState(0)

VEHICLE_DIMS = {'pedestrian':(0.5,0.5),'motorcycle':(0.75,2.0),'auto_rickshaw':(1.3,2.65),
                'car':(1.5,3.5),'lcv':(1.8,4.5),'bus':(2.0,5.0),'truck':(2.0,5.0),
                'cyclist':(0.6,1.8)}
SWEEP_DT, SWEEP_HORIZON = 0.05, 10.0

# ───────────────────────── geometry helpers ────────────────────────────────
def corners(cx, cy, heading, cls):
    w, l = VEHICLE_DIMS.get(cls, (1.5, 3.5))
    hw, hl = w/2.0, l/2.0
    loc = np.array([[-hl,-hw],[hl,-hw],[hl,hw],[-hl,hw]])
    c, s = np.cos(heading), np.sin(heading)
    return loc @ np.array([[c,-s],[s,c]]).T + np.array([cx, cy])

def _axes(poly):
    out = []
    for i in range(len(poly)):
        e = poly[(i+1) % len(poly)] - poly[i]
        n = np.array([-e[1], e[0]]); nn = np.linalg.norm(n)
        if nn > 1e-12: out.append(n/nn)
    return out

def sat_max_axis(pa, pb):
    """Correct SAT: maximum projected gap across all axes."""
    mg, sep = 0.0, False
    for ax in _axes(pa) + _axes(pb):
        ap, bp = pa @ ax, pb @ ax
        g = max(bp.min()-ap.max(), ap.min()-bp.max())
        if g > 0:
            sep = True
            if g > mg: mg = g
    return mg if sep else 0.0

def sat_first_axis(pa, pb):
    """Legacy behaviour: return the gap on the FIRST separating axis found."""
    for ax in _axes(pa) + _axes(pb):
        ap, bp = pa @ ax, pb @ ax
        g = max(bp.min()-ap.max(), ap.min()-bp.max())
        if g > 0: return g
    return 0.0

def poly_ttc(a, b, use_accel=True, point_mass=False):
    """First sweep step at which footprints intersect while converging.
    point_mass=True collapses both agents to zero-size points (classical)."""
    (xa,ya,vxa,vya,aa,ha,ca) = a
    (xb,yb,vxb,vyb,ab,hb,cb) = b
    if not use_accel: aa = ab = 0.0
    axa, aya = (aa*np.cos(ha), aa*np.sin(ha))
    axb, ayb = (ab*np.cos(hb), ab*np.sin(hb))
    prev_d = np.hypot(xa-xb, ya-yb)
    t = SWEEP_DT
    while t <= SWEEP_HORIZON:
        pax = xa + vxa*t + 0.5*axa*t*t; pay = ya + vya*t + 0.5*aya*t*t
        pbx = xb + vxb*t + 0.5*axb*t*t; pby = yb + vyb*t + 0.5*ayb*t*t
        d = np.hypot(pax-pbx, pay-pby)
        if d <= prev_d:                      # still converging
            if point_mass:
                if d <= 0.0: return t
            else:
                if sat_max_axis(corners(pax,pay,ha,ca), corners(pbx,pby,hb,cb)) == 0.0:
                    return t
        prev_d = d
        t += SWEEP_DT
    return np.nan

# ───────────────────────── load ────────────────────────────────────────────
sm = pd.read_csv(SM)
ic = pd.read_csv(IC)
traj = {t: g.sort_values('frame').set_index('frame')
        for t, g in sm.groupby('track_id')}
cls_of = sm.groupby('track_id')['class_name'].first().to_dict()
lines = []
def say(s):
    print(s); lines.append(s)

say("="*74)
say("ABLATION / DEGENERACY / COMPLEXITY EVALUATION")
say("="*74)

# ───────────────── 1. ABLATION: motion model and agent geometry ────────────
say("\n[1] ABLATION AGAINST CLASSICAL MODELLING ASSUMPTIONS")
pairs = ic.sample(min(400, len(ic)), random_state=0)
rows = []
for _, r in pairs.iterrows():
    a, b = int(r['agent_a_id']), int(r['agent_b_id'])
    if a not in traj or b not in traj: continue
    ta, tb = traj[a], traj[b]
    common = ta.index.intersection(tb.index)
    if len(common) < 10: continue
    A = ta.loc[common, ['x_m_smooth','y_m_smooth','vx','vy','accel','heading']].values
    B = tb.loc[common, ['x_m_smooth','y_m_smooth','vx','vy','accel','heading']].values
    tv_all = ta.loc[common, 'time_s'].values
    # Drop frames with undefined velocity/heading. These occur once per track
    # (the first sample, where velocity is derived by differencing) and would
    # otherwise propagate NaN into the polygon corners, where every comparison
    # evaluates False and the SAT test would wrongly report zero separation.
    ok = ~(np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1))
    if ok.sum() < 10: continue
    A, B, tv_all = A[ok], B[ok], tv_all[ok]
    ca, cb = cls_of[a], cls_of[b]
    cd  = np.hypot(A[:,0]-B[:,0], A[:,1]-B[:,1])
    sep = np.array([sat_max_axis(corners(A[i,0],A[i,1],A[i,5],ca),
                                 corners(B[i,0],B[i,1],B[i,5],cb))
                    for i in range(len(A))])
    tv = tv_all
    # Forward sweeps launched from several frames across the encounter and
    # reduced by minimum, mirroring the pipeline definition of TTC_poly as the
    # earliest predicted footprint intersection over all observed frames
    # (sweeping only from the closest-approach frame would understate
    # definedness, since agents are typically already diverging by then).
    idxs = np.unique(np.linspace(0, len(A)-1, min(8, len(A))).astype(int))
    def sweep_min(use_accel, point_mass):
        vals = []
        for k in idxs:
            pa = (A[k,0],A[k,1],A[k,2],A[k,3],A[k,4],A[k,5],ca)
            pb = (B[k,0],B[k,1],B[k,2],B[k,3],B[k,4],B[k,5],cb)
            v = poly_ttc(pa, pb, use_accel, point_mass)
            if not np.isnan(v): vals.append(v)
        return min(vals) if vals else np.nan
    rows.append(dict(
        pair=f"{a}-{b}",
        min_centroid=cd.min(), min_footprint=sep.min(),
        t_centroid=tv[int(np.argmin(cd))], t_footprint=tv[int(np.argmin(sep))],
        ttc_accel_poly=sweep_min(True,  False),
        ttc_cv_poly   =sweep_min(False, False),
        ttc_accel_pt  =sweep_min(True,  True),
    ))
ab = pd.DataFrame(rows)
ab['clearance_overstatement'] = ab['min_centroid'] - ab['min_footprint']
ab['timing_lead'] = ab['t_centroid'] - ab['t_footprint']
ab.to_csv('ablation_table.csv', index=False)

n = len(ab)
say(f"  sample of {n} classified encounters")
say(f"  (a) AGENT GEOMETRY  point-mass centroid vs true rotated footprint")
say(f"      mean clearance overstatement by point-mass : {ab['clearance_overstatement'].mean():.2f} m")
say(f"      median                                     : {ab['clearance_overstatement'].median():.2f} m")
say(f"      encounters where point-mass 2.0 m alert never fires")
say(f"        but true footprint clearance < 1.5 m     : "
    f"{int(((ab.min_centroid>2.0)&(ab.min_footprint<1.5)).sum())}/{n} "
    f"({100*((ab.min_centroid>2.0)&(ab.min_footprint<1.5)).mean():.1f}%)  [missed by point-mass]")
say(f"      critical instant identified earlier by footprint model: "
    f"{int((ab.timing_lead>0).sum())}/{n} ({100*(ab.timing_lead>0).mean():.1f}%), "
    f"mean lead {ab.loc[ab.timing_lead>0,'timing_lead'].mean():.2f} s")
say(f"  (b) MOTION MODEL  constant-acceleration vs constant-velocity sweep")
d_acc = ab['ttc_accel_poly'].notna().mean(); d_cv = ab['ttc_cv_poly'].notna().mean()
say(f"      polygon-sweep TTC defined: accel {100*d_acc:.1f}%  vs  const-vel {100*d_cv:.1f}%")
both = ab.dropna(subset=['ttc_accel_poly','ttc_cv_poly'])
if len(both):
    say(f"      where both defined (n={len(both)}): mean |Δ TTC| = "
        f"{(both.ttc_accel_poly-both.ttc_cv_poly).abs().mean():.2f} s; "
        f"const-vel is later (more optimistic) in "
        f"{100*(both.ttc_cv_poly>both.ttc_accel_poly).mean():.1f}% of cases")
say(f"  (c) COLLISION TEST  polygon footprints vs point-mass")
say(f"      sweep predicts a collision: polygon {100*ab['ttc_accel_poly'].notna().mean():.1f}%  "
    f"vs point-mass {100*ab['ttc_accel_pt'].notna().mean():.1f}% "
    f"(point-mass requires exact co-location, so it almost never fires)")

# ───────────────── 2. SAT axis-selection check ─────────────────────────────
say("\n[2] SAT AXIS SELECTION: first-separating-axis vs max-over-axes")
chk = []
for _, r in pairs.head(150).iterrows():
    a, b = int(r['agent_a_id']), int(r['agent_b_id'])
    if a not in traj or b not in traj: continue
    ta, tb = traj[a], traj[b]
    common = ta.index.intersection(tb.index)
    if len(common) < 5: continue
    for i in list(range(0, len(common), max(1, len(common)//6)))[:6]:
        f = common[i]
        if pd.isna(ta.loc[f,'heading']) or pd.isna(tb.loc[f,'heading']): continue
        pa = corners(ta.loc[f,'x_m_smooth'], ta.loc[f,'y_m_smooth'], ta.loc[f,'heading'], cls_of[a])
        pb = corners(tb.loc[f,'x_m_smooth'], tb.loc[f,'y_m_smooth'], tb.loc[f,'heading'], cls_of[b])
        chk.append((sat_first_axis(pa, pb), sat_max_axis(pa, pb)))
chk = pd.DataFrame(chk, columns=['first_axis','max_axis'])
chk['understatement'] = chk['max_axis'] - chk['first_axis']
chk.to_csv('sat_axis_check.csv', index=False)
say(f"  {len(chk)} frame samples; first-axis rule understates true separation in "
    f"{100*(chk.understatement>1e-9).mean():.1f}% of samples")
say(f"  mean understatement {chk.understatement.mean():.2f} m, "
    f"max {chk.understatement.max():.2f} m")
_near = chk[chk.max_axis < 5.0]
say(f"  restricted to safety-relevant range (true separation < 5 m, n={len(_near)}): "
    f"understated in {100*(_near.understatement>1e-9).mean():.1f}% of samples, "
    f"mean {_near.understatement.mean():.2f} m")

# ───────────────── 3. DEGENERACY STRATIFICATION ────────────────────────────
say("\n[3] DEGENERACY: definedness of classical TTC vs CCI by regime")
recs = []
for _, r in ic.iterrows():
    a, b = int(r['agent_a_id']), int(r['agent_b_id'])
    if a not in traj or b not in traj: continue
    ta, tb = traj[a], traj[b]
    common = ta.index.intersection(tb.index)
    if len(common) < 5: continue
    sa = ta.loc[common,'speed'].values; sb = tb.loc[common,'speed'].values
    ha = ta.loc[common,'heading'].values; hb = tb.loc[common,'heading'].values
    dh = np.abs(np.arctan2(np.sin(ha-hb), np.cos(ha-hb)))
    recs.append(dict(min_speed=float(np.nanmin(np.minimum(sa, sb))),
                     head_diff_deg=float(np.degrees(np.nanmedian(dh))),
                     ttc_poly_def=pd.notna(r['TTC_poly_min']),
                     ttc_def=pd.notna(r['TTC_min']),
                     cci_def=pd.notna(r['Max_CCI'])))
dg = pd.DataFrame(recs)
sp_bins = [0, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf]
sp_lab  = ['<0.25','0.25-0.5','0.5-1','1-2','2-5','>5']
dg['speed_bin'] = pd.cut(dg['min_speed'], sp_bins, labels=sp_lab)
hd_bins = [0, 15, 30, 60, 120, 165, 180]
hd_lab  = ['0-15 (parallel)','15-30','30-60','60-120','120-165','165-180 (opposed)']
dg['head_bin'] = pd.cut(dg['head_diff_deg'], hd_bins, labels=hd_lab, include_lowest=True)

t1 = dg.groupby('speed_bin', observed=True).agg(
        n=('ttc_poly_def','size'),
        ttc_poly_defined_pct=('ttc_poly_def', lambda s: round(100*s.mean(),1)),
        cci_defined_pct=('cci_def', lambda s: round(100*s.mean(),1))).reset_index()
t2 = dg.groupby('head_bin', observed=True).agg(
        n=('ttc_poly_def','size'),
        ttc_poly_defined_pct=('ttc_poly_def', lambda s: round(100*s.mean(),1)),
        cci_defined_pct=('cci_def', lambda s: round(100*s.mean(),1))).reset_index()
t1.rename(columns={'speed_bin':'regime'}, inplace=True); t1.insert(0,'stratifier','min speed (m/s)')
t2.rename(columns={'head_bin':'regime'}, inplace=True);  t2.insert(0,'stratifier','heading difference (deg)')
deg_tab = pd.concat([t1, t2], ignore_index=True)
deg_tab.to_csv('degeneracy_table.csv', index=False)
say(deg_tab.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, tab, ttl in ((axes[0], t1, 'by minimum speed (m/s)'),
                     (axes[1], t2, 'by heading difference (deg)')):
    x = np.arange(len(tab))
    ax.plot(x, tab['ttc_poly_defined_pct'], 'o-', color='#C62828',
            label='predictive polygon TTC')
    ax.plot(x, tab['cci_defined_pct'], 's-', color='#1565C0', label='CCI (proposed)')
    ax.set_xticks(x); ax.set_xticklabels(tab['regime'], rotation=25, ha='right', fontsize=8)
    ax.set_ylim(-5, 105); ax.set_ylabel('% of encounters with a defined value')
    ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_degeneracy.png', dpi=170, bbox_inches='tight')

# ───────────────── 4. COMPLEXITY AND MEASURED COST ─────────────────────────
say("\n[4] COMPUTATIONAL COST (CPU, this machine)")
per_frame = sm.groupby('frame')['track_id'].nunique()
say(f"  concurrency: mean {per_frame.mean():.1f}, median {int(per_frame.median())}, "
    f"p95 {int(per_frame.quantile(.95))}, peak {int(per_frame.max())} agents/frame")
Npk = int(per_frame.max())
say(f"  worst-case pair evaluations per frame at peak: {Npk*(Npk-1)//2}")

pa = corners(0,0,0.3,'car'); pb = corners(4,1,1.1,'pedestrian')
N = 20000
t0=time.perf_counter()
for _ in range(N): sat_max_axis(pa, pb)
t_sat = (time.perf_counter()-t0)/N*1e6

def cci(x1,y1,c1,x2,y2,c2):
    w1,l1 = VEHICLE_DIMS[c1]; w2,l2 = VEHICLE_DIMS[c2]
    sx = max(w1/2,.5)**2 + max(w2/2,.5)**2; sy = max(l1/2,.5)**2 + max(l2/2,.5)**2
    return np.exp(-0.5*(((x1-x2)**2)/sx + ((y1-y2)**2)/sy))
t0=time.perf_counter()
for _ in range(N): cci(0,0,'car',4,1,'pedestrian')
t_cci = (time.perf_counter()-t0)/N*1e6

tracemalloc.start()
_ = {t: g[['x_m_smooth','y_m_smooth','vx','vy','heading']].values for t, g in sm.groupby('track_id')}
cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

say(f"  measured: SAT separation {t_sat:.1f} us/call, CCI {t_cci:.1f} us/call")
say(f"  per-frame risk-engine estimate at peak concurrency: "
    f"{Npk*(Npk-1)//2*(t_sat+t_cci)/1000:.2f} ms/frame "
    f"({1000.0/max(Npk*(Npk-1)//2*(t_sat+t_cci)/1000,1e-9):.0f} fps headroom, risk engine only)")
say(f"  trajectory state resident in memory: {peak/1e6:.1f} MB for "
    f"{sm['track_id'].nunique()} tracks / {len(sm)} samples")
pd.DataFrame([
    dict(quantity='SAT separation', value=round(t_sat,2), unit='us per agent pair'),
    dict(quantity='CCI evaluation', value=round(t_cci,2), unit='us per agent pair'),
    dict(quantity='mean concurrency', value=round(per_frame.mean(),1), unit='agents/frame'),
    dict(quantity='peak concurrency', value=Npk, unit='agents/frame'),
    dict(quantity='peak pair evaluations', value=Npk*(Npk-1)//2, unit='pairs/frame'),
    dict(quantity='risk engine at peak', value=round(Npk*(Npk-1)//2*(t_sat+t_cci)/1000,2), unit='ms/frame'),
    dict(quantity='trajectory memory', value=round(peak/1e6,1), unit='MB'),
]).to_csv('complexity_table.csv', index=False)

with open('benchmark_summary.txt','w', encoding='utf-8') as f:
    f.write("\n".join(lines) + "\n")
say("\nwrote ablation_table.csv, sat_axis_check.csv, degeneracy_table.csv, "
    "complexity_table.csv, fig_degeneracy.png, benchmark_summary.txt")
