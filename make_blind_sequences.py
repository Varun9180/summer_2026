"""Build 8-frame blind-verification sequences for randomly sampled encounters.

Each montage shows the two agents (A=red, B=blue) at eight instants evenly
spaced across their shared observation window, so motion and right-of-way
resolution can be judged by eye. The model's outcome/pattern is written ONLY
to manifest.csv, never onto the image, so judgement can be made blind.
"""
import cv2, os, sys
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/blind'
N_CASES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7
os.makedirs(OUT, exist_ok=True)

def find_input(f):
    for b in ('.', '..', '../..'):
        p = os.path.join(b, f)
        if os.path.exists(p): return p
    return f

ic = pd.read_csv('interactions_classified.csv')
sm = pd.read_csv('smoothed_trajectories_newest.csv')
H_OUT = np.load(find_input('H_out_v3.npy'))
H_INV = np.linalg.inv(H_OUT)
SCALE = 100.15 / 7.0
FPS = 19.99

traj = {t: g.sort_values('frame').set_index('frame') for t, g in sm.groupby('track_id')}

def shared(a, b):
    return sorted(set(traj[a].index) & set(traj[b].index))

# random sample of encounters long enough to judge (>= 1 s of shared view)
rng = np.random.RandomState(SEED)
elig = []
for i, r in ic.iterrows():
    a, b = int(r['agent_a_id']), int(r['agent_b_id'])
    if a in traj and b in traj and len(shared(a, b)) >= 20:
        elig.append(i)
pick = rng.choice(elig, size=min(N_CASES, len(elig)), replace=False)

def to_px(xm, ym):
    p = H_INV @ np.array([xm * SCALE, ym * SCALE, 1.0])
    return int(p[0] / p[2]), int(p[1] / p[2])

cap = cv2.VideoCapture(find_input('tracked_output_newest.mp4'))
W, H = 2592, 1520
PANEL_W = 520
rows_out = []

for n, idx in enumerate(pick):
    r = ic.loc[idx]
    a, b = int(r['agent_a_id']), int(r['agent_b_id'])
    fr = shared(a, b)
    ta, tb = traj[a], traj[b]
    sel = [fr[int(k)] for k in np.linspace(0, len(fr) - 1, 8)]
    panels = []
    for f in sel:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok: continue
        pa = to_px(ta.loc[f, 'x_m_smooth'], ta.loc[f, 'y_m_smooth'])
        pb = to_px(tb.loc[f, 'x_m_smooth'], tb.loc[f, 'y_m_smooth'])
        cv2.circle(img, pa, 30, (0, 0, 255), 4)
        cv2.putText(img, 'A', (pa[0]-14, pa[1]-38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
        cv2.circle(img, pb, 30, (255, 90, 0), 4)
        cv2.putText(img, 'B', (pb[0]-14, pb[1]-38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,90,0), 3)
        # generous crop around both agents so surroundings give context
        cxm, cym = (pa[0]+pb[0])//2, (pa[1]+pb[1])//2
        half = max(abs(pa[0]-pb[0]), abs(pa[1]-pb[1]))//2 + 260
        x0, x1 = max(cxm-half, 0), min(cxm+half, W)
        y0, y1 = max(cym-int(half*0.72), 0), min(cym+int(half*0.72), H)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0: continue
        s = PANEL_W / crop.shape[1]
        crop = cv2.resize(crop, (PANEL_W, max(1, int(crop.shape[0]*s))))
        bar = np.full((26, PANEL_W, 3), 20, np.uint8)
        cv2.putText(bar, f't={f/FPS:.1f}s', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        panels.append(np.vstack([bar, crop]))
    if len(panels) < 6: continue
    h = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.full((h-p.shape[0], PANEL_W, 3), 20, np.uint8)]) for p in panels]
    grid = np.vstack([np.hstack(panels[0:4]), np.hstack(panels[4:8])])
    head = np.full((44, grid.shape[1], 3), 45, np.uint8)
    cv2.putText(head, f'CASE {n:02d}   A(red)={r["agent_a_class"]} #{a}   B(blue)={r["agent_b_class"]} #{b}',
                (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
    cv2.imwrite(f'{OUT}/case_{n:02d}.jpg', np.vstack([head, grid]),
                [cv2.IMWRITE_JPEG_QUALITY, 86])
    d = np.hypot(ta.loc[fr,'x_m_smooth'].values - tb.loc[fr,'x_m_smooth'].values,
                 ta.loc[fr,'y_m_smooth'].values - tb.loc[fr,'y_m_smooth'].values)
    rows_out.append(dict(case=n, agent_a=a, agent_b=b,
                         cls_a=r['agent_a_class'], cls_b=r['agent_b_class'],
                         pair_type=r['pair_type'], model_outcome=r['outcome'],
                         model_pattern=r['pattern'], NM=r['NM'], SDD=r['SDD'],
                         min_dist_m=round(float(d.min()), 2),
                         dur_s=round(len(fr)/FPS, 1)))
pd.DataFrame(rows_out).to_csv(f'{OUT}/manifest.csv', index=False)
print(f'wrote {len(rows_out)} sequences to {OUT}')
