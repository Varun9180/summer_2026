"""Build blind-verification montages for a stratified sample of classified
interactions. For each sampled pair: extract start / closest-approach / end
frames from tracked_output_newest.mp4, mark agent A (red) and agent B (blue)
via inverse homography, and save one montage per interaction. The model's
outcome/pattern labels go only into manifest.csv (not onto the images), so the
visual judgment can be made blind."""
import cv2, os, sys
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/verify'
os.makedirs(OUT, exist_ok=True)

def find_input(fname):
    for base in ('.', '..', '../..'):
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    return fname

ic = pd.read_csv('interactions_classified.csv')
sm = pd.read_csv('smoothed_trajectories_newest.csv')
H_OUT = np.load(find_input('H_out_v3.npy'))
H_INV = np.linalg.inv(H_OUT)
SCALE = 100.15 / 7.0

rng = np.random.RandomState(42)
MIN_FRAMES = 20   # >= ~1 s of shared observation so the encounter is judgeable

traj = {tid: g.sort_values('frame').set_index('frame')
        for tid, g in sm.groupby('track_id')}

def shared_frames(a, b):
    return sorted(set(traj[a].index) & set(traj[b].index))

# stratified sample: up to 2 per pattern, judgeable windows only
sample = []
for pat, grp in ic.groupby('pattern'):
    ok = []
    for _, r in grp.iterrows():
        a, b = int(r['agent_a_id']), int(r['agent_b_id'])
        if a in traj and b in traj and len(shared_frames(a, b)) >= MIN_FRAMES:
            ok.append(r)
    if ok:
        idx = rng.choice(len(ok), size=min(2, len(ok)), replace=False)
        sample.extend([ok[i] for i in idx])

def to_px(xm, ym):
    p = H_INV @ np.array([xm * SCALE, ym * SCALE, 1.0])
    return int(p[0] / p[2]), int(p[1] / p[2])

cap = cv2.VideoCapture(find_input('tracked_output_newest.mp4'))
W, H = 2592, 1520
manifest = []
for k, r in enumerate(sample):
    a, b = int(r['agent_a_id']), int(r['agent_b_id'])
    fr = shared_frames(a, b)
    ta, tb = traj[a], traj[b]
    d = np.hypot(ta.loc[fr, 'x_m_smooth'].values - tb.loc[fr, 'x_m_smooth'].values,
                 ta.loc[fr, 'y_m_smooth'].values - tb.loc[fr, 'y_m_smooth'].values)
    f_sel = [fr[0], fr[int(np.argmin(d))], fr[-1]]
    labels = ['START', 'CLOSEST', 'END']
    panels = []
    for f, lab in zip(f_sel, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        okf, img = cap.read()
        if not okf:
            continue
        pa = to_px(ta.loc[f, 'x_m_smooth'], ta.loc[f, 'y_m_smooth'])
        pb = to_px(tb.loc[f, 'x_m_smooth'], tb.loc[f, 'y_m_smooth'])
        cv2.circle(img, pa, 34, (0, 0, 255), 5)      # A = red
        cv2.circle(img, pb, 34, (255, 80, 0), 5)     # B = blue
        x0 = max(min(pa[0], pb[0]) - 260, 0); x1 = min(max(pa[0], pb[0]) + 260, W)
        y0 = max(min(pa[1], pb[1]) - 200, 0); y1 = min(max(pa[1], pb[1]) + 200, H)
        crop = img[y0:y1, x0:x1]
        tw = 1150
        s = tw / crop.shape[1]
        crop = cv2.resize(crop, (tw, max(1, int(crop.shape[0] * s))))
        bar = np.full((44, tw, 3), 25, np.uint8)
        cv2.putText(bar, f'{lab}  frame {f}  t={f/19.99:.1f}s  dist={d[f_sel.index(f) if f in f_sel else 0]:.1f}m'
                    if False else f'{lab}  frame {f}  t={f/19.99:.1f}s',
                    (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        panels.append(np.vstack([bar, crop]))
    if not panels:
        continue
    head = np.full((52, 1150, 3), 45, np.uint8)
    cv2.putText(head, f'CASE {k:02d}:  A(red)=#{a} {r["agent_a_class"]}   B(blue)=#{b} {r["agent_b_class"]}',
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    mont = np.vstack([head] + panels)
    cv2.imwrite(f'{OUT}/case_{k:02d}.jpg', mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
    manifest.append({'case': k, 'agent_a': a, 'agent_b': b,
                     'cls_a': r['agent_a_class'], 'cls_b': r['agent_b_class'],
                     'pair_type': r['pair_type'], 'outcome': r['outcome'],
                     'pattern': r['pattern'], 'NM': r['NM'], 'SDD': r['SDD'],
                     'PET': r['PET'], 'TTC_min': r['TTC_min'], 'Max_CCI': r['Max_CCI'],
                     'frames': f'{fr[0]}-{fr[-1]}', 'min_dist_m': round(float(d.min()), 2)})
pd.DataFrame(manifest).to_csv(f'{OUT}/manifest.csv', index=False)
print(f'wrote {len(manifest)} montages to {OUT}')
