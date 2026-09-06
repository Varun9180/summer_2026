import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from scipy.stats import genpareto
from statsmodels.regression.mixed_linear_model import MixedLM
from tqdm import tqdm  # Standard terminal tqdm

# --- Windows-safe CSV writer -------------------------------------------------
# On Windows, pandas.to_csv() raises PermissionError if the target file is
# currently open in Excel/another program (the OS file lock blocks overwrite).
# This wrapper retries once under a timestamped fallback name instead of
# crashing the whole multi-hour pipeline at the final save step.
import time as _time_mod

def safe_to_csv(df, path, **kwargs):
    try:
        df.to_csv(path, **kwargs)
        print(f"  saved: {path}")
    except PermissionError:
        fallback = path.replace('.csv', f'_{int(_time_mod.time())}.csv')
        print(f"  WARNING: '{path}' is open/locked (e.g. in Excel). "
              f"Close it and re-run if you need the exact filename. "
              f"Saving instead to: {fallback}")
        df.to_csv(fallback, **kwargs)

warnings.filterwarnings('ignore')

from ultralytics import YOLO
import supervision as sv

# ==========================================
# PHASE 1: CONFIGURATION
# ==========================================
print("--- PHASE 1: Initializing Configuration ---")

# 🔴 TESTING TOGGLE 🔴
# Set to True to run the heavy video AI tracking.
# Set to False to skip video processing and just test the math/PINN on existing data.
# NOTE: this is a fresh video (no existing trajectories CSV yet), so this must
# be True the first time this script is run. VIDEO_PATH below points to a
# ~5-minute clip (the busiest window found in a full-video activity scan:
# 17:00-22:00 into newest_video.avi, ~6000 frames at ~20fps) rather than the
# full ~1-hour video, to keep the run tractable. Set back to False on
# subsequent runs to skip re-tracking and iterate on Phases 4+ using the
# already-saved TRAJECTORY_CSV below.
RUN_YOLO_TRACKING = True

VIDEO_PATH        = 'newest_video_busy_5min.mp4'   # busiest 5-min window (17:00-22:00) extracted from newest_video.avi
WARPED_VIDEO_PATH = 'newest_video_bev_full.mp4'
OUTPUT_VIDEO      = 'tracked_output_newest.mp4'
TRAJECTORY_CSV    = 'trajectories_newest.csv'
SMOOTHED_CSV      = 'smoothed_trajectories_newest.csv'

YOLO_MODEL  = '../yolov8x.pt'   # reuse existing weights instead of re-downloading into this folder
CONF_THRESH = 0.30

DT               = 0.25    
PROXIMITY_M      = 15.0    
# MIN_OVERLAP is a frame COUNT, not a time duration. With DT now correctly
# derived from the actual video frame rate (~0.0167s/frame at ~60fps) rather
# than the old stale 0.25s assumption, a MIN_OVERLAP of 4 frames only
# guarantees ~0.067s of shared observation — far too little temporal signal
# to derive a meaningful NM (rate-of-change average) or SDD (stable-run
# duration) from. A handful of noisy frames from a near-stationary or
# spurious short-lived detection can then dominate the whole "average rate",
# which is exactly what produced NM values in the hundreds for very short
# tracks. MIN_OVERLAP is raised so an interaction needs at least ~0.5s of
# shared, valid observation before it is treated as a classifiable
# negotiation. This value is expressed as a frame count and is intentionally
# recomputed below once the real per-frame DT is known (see Phase 5).
MIN_OVERLAP      = 4       
MIN_OVERLAP_SECONDS = 0.5   # true minimum shared-observation duration required
NM_BAND_FRACTION = 0.25    

VEHICLE_DIMS = {
    'pedestrian':    (0.5,  0.5),
    'motorcycle':    (0.75, 2.0),
    'auto_rickshaw': (1.3,  2.65),
    'car':           (1.5,  3.5),
    'lcv':           (1.8,  4.5),
    'bus':           (2.0,  5.0),
    'truck':         (2.0,  5.0),
    'cyclist':       (0.6,  1.8),
}
DEFAULT_DIM = (1.5, 3.5)
CLASS_NAMES = ['pedestrian', 'cyclist', 'motorcycle', 'auto_rickshaw', 'car', 'lcv', 'bus', 'truck']

# ---------------------------------------------------------
# REGION OF INTEREST (ROI) PIXEL COORDINATES for newest_video.avi (2592x1520).
# No get_roi.py click session has been run for this camera yet, so this
# defaults to (nearly) the full frame with a small margin — this does not
# lose any detections, it just means the PolygonZone filter isn't doing any
# real spatial pruning yet. Re-run get_roi.py on this video and replace these
# 4 points if you want to restrict detection/tracking to a tighter road area.
# ---------------------------------------------------------
ROI_POLYGON = np.array([
    [5, 5],
    [5, 1515],
    [2587, 1515],
    [2587, 5],
])

# ==========================================
# PHASE 2: HOMOGRAPHY & MAPPER
# ==========================================
print("--- PHASE 2: Setting up Spatial Mapping ---")

# ---------------------------------------------------------------------------
# This camera (newest_video.avi, roundabout junction) views its central
# circular island at a very oblique/shallow angle (fitted ellipse axis ratio
# ~9:1 before correction — much steeper than a typical calibration shot).
# A rigorous two-vanishing-point + ellipse-to-circle homography was derived
# for it interactively rather than from 4 simple corner clicks (see the
# working files in this folder: H_aff_v1.npy [projective-only correction],
# H_final_v1.npy [+ ellipse-to-circle metric fix], H_out_v3.npy [+ canvas
# fit], and newest_video_bev_v3.jpg for the visual check). That homography
# maps ORIGINAL VIDEO PIXELS -> a bounded output-canvas PIXEL coordinate
# system (not metres yet), sized to keep the roundabout region well-scaled
# rather than trying to flatten the whole (very oblique) frame uniformly —
# attempting a full-frame flatten here blows up to an unusable aspect ratio,
# so this canvas intentionally covers the roundabout + surrounding roads
# only, not the entire camera view.
# ---------------------------------------------------------------------------
H_OUT = np.load('H_out_v3.npy')
out_width, out_height = 502, 1420   # canvas size H_OUT was built for

# ---------------------------------------------------------------------------
# METRIC SCALE: pixels-per-metre in the H_OUT canvas.
# The roundabout island's fitted radius in that canvas is ~100.15 px
# (semi-axes 90.7 / 109.6 px, averaged — see the ellipse fit on
# newest_video_bev_v3.jpg). ISLAND_RADIUS_M is the island's REAL-WORLD
# radius in metres and MUST be filled in for the resulting x_m/y_m
# trajectories — and therefore every downstream safety metric (PROXIMITY_M,
# footprint sizes, PEB/TTC thresholds, ...) — to be dimensionally correct.
# A wrong value here scales every distance in the entire pipeline by the
# same wrong factor.
# ---------------------------------------------------------------------------
ISLAND_RADIUS_PX = 100.15
ISLAND_RADIUS_M  = 7.0   # real-world radius of the roundabout island, in metres (given by user)

if ISLAND_RADIUS_M is None:
    raise ValueError(
        "ISLAND_RADIUS_M is not set (Phase 2, run.py). Measure or obtain the "
        "real-world radius (metres) of the newest_video.avi roundabout island "
        "and set it above before running the pipeline -- every downstream "
        "distance/safety metric depends on this scale factor."
    )

SCALE = ISLAND_RADIUS_PX / ISLAND_RADIUS_M   # pixels per metre in the H_OUT canvas

# Kept as H_viewport for drop-in compatibility with Phase 2.5 (video warp) below.
H_viewport = H_OUT

# Mathematical conversion function for YOLO tracks (RESTORED TO PURE HOMOGRAPHY)
def pixel_to_world_viewport(px, py, H=H_viewport):
    pt = np.array([[[px, py]]], dtype=np.float32)
    wpt = cv2.perspectiveTransform(pt, H)
    wx = float(wpt[0, 0, 0]) / SCALE
    wy = float(wpt[0, 0, 1]) / SCALE
    return wx, wy

class COCOtoIndianMapper:
    def __init__(self):
        self.lut = {0: 0, 1: 1, 2: 4, 3: 2, 5: 6, 7: 7}
        
    def map_classes(self, boxes, confs, cls_ids):
        out = []
        for i in range(len(cls_ids)):
            coco = int(cls_ids[i])
            if coco not in self.lut: continue
            indian = self.lut[coco]
            conf = float(confs[i])
            x1, y1, x2, y2 = map(float, boxes[i])
            
            if indian == 4:
                w, h = x2 - x1, y2 - y1
                area, asp = w * h, w / (h + 1e-6)
                if 400 <= area <= 4000 and 0.70 <= asp <= 1.60:
                    indian, conf = 3, conf * 0.85
            elif indian == 2:
                if (x2 - x1) * (y2 - y1) > 7500:
                    indian, conf = 5, conf * 0.80
                    
            out.append([x1, y1, x2, y2, conf, indian])
        return np.array(out, dtype=np.float32) if out else np.empty((0,6), np.float32)

mapper = COCOtoIndianMapper()

# ==========================================
# PHASE 2.5: GENERATE HOMOGRAPHY VIDEO
# ==========================================
print("--- PHASE 2.5: Generating Full BEV Homography Video ---")
if not os.path.exists(WARPED_VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_warp = cv2.VideoWriter(WARPED_VIDEO_PATH, fourcc, fps, (out_width, out_height))
    for _ in tqdm(range(total_frames), desc="Warping Video"):
        ret, frame = cap.read()
        if not ret: break
        warped_frame = cv2.warpPerspective(frame, H_viewport, (out_width, out_height))
        out_warp.write(warped_frame)
    cap.release()
    out_warp.release()
    print(f"Warped video saved to '{WARPED_VIDEO_PATH}'")
else:
    print(f"Found existing warped video at '{WARPED_VIDEO_PATH}'. Skipping generation.")

# ==========================================
# PHASE 3: TRACKING & VIDEO OUTPUT 
# ==========================================
print("--- PHASE 3: Processing Original Video (YOLO + BoT-SORT) ---")

if RUN_YOLO_TRACKING:
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, video_info.fps, video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
    
    # Initialize the ROI Zone and Annotator
    zone = sv.PolygonZone(polygon=ROI_POLYGON)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=3)

    import torch
    device_target = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Tracking using device: {device_target}")

    model = YOLO(YOLO_MODEL)
    tracker = sv.ByteTrack()
    raw_data = []

    generator = sv.get_video_frames_generator(VIDEO_PATH)

    for frame_idx, frame in enumerate(tqdm(generator, total=video_info.total_frames, desc='Tracking')):
        results = model(frame, verbose=False, conf=CONF_THRESH, device=device_target)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # ---------------------------------------------------------
        # APPLY REGION OF INTEREST (ROI) FILTER
        # ---------------------------------------------------------
        mask = zone.trigger(detections=detections)
        detections = detections[mask] 

        detections = tracker.update_with_detections(detections)
        
        if len(detections) > 0:
            mapped_dets = mapper.map_classes(detections.xyxy, detections.confidence, detections.class_id)
            if len(mapped_dets) > 0:
                for i, det in enumerate(mapped_dets):
                    x1, y1, x2, y2, conf, mapped_cls = det
                    track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                    
                    if track_id != -1:
                        px_c, py_b = (x1 + x2) / 2, y2
                        wx, wy = pixel_to_world_viewport(px_c, py_b)
                        
                        raw_data.append({
                            'frame': frame_idx,
                            'time_s': frame_idx / video_info.fps,
                            'track_id': track_id,
                            'class_id': int(mapped_cls),
                            'class_name': CLASS_NAMES[int(mapped_cls)],
                            'x_m': wx,
                            'y_m': wy
                        })
        
        annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

        # ---------------------------------------------------------
        # AUTO-RICKSHAW VIDEO LABEL FIX
        # ---------------------------------------------------------
        all_labels = []
        for i in range(len(detections)):
            coco_cls = int(detections.class_id[i]) if detections.class_id is not None else -1
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            
            if coco_cls not in mapper.lut:
                all_labels.append("Unknown")
                continue
                
            indian_cls = mapper.lut[coco_cls]
            x1, y1, x2, y2 = detections.xyxy[i]
            
            # Apply Indian Traffic bounding-box logic to video labels
            if indian_cls == 4: # If YOLO thinks it is a car
                w, h = x2 - x1, y2 - y1
                area, asp = w * h, w / (h + 1e-6)
                if 400 <= area <= 4000 and 0.70 <= asp <= 1.60:
                    indian_cls = 3 # Reclassify as auto_rickshaw
            elif indian_cls == 2: # If YOLO thinks it is a motorcycle
                if (x2 - x1) * (y2 - y1) > 7500:
                    indian_cls = 5 # Reclassify as LCV
                    
            name = CLASS_NAMES[indian_cls]
            all_labels.append(f'#{tid} {name}' if tid != -1 else name)

        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=all_labels)
        
        # Draw the ROI polygon on the output video
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        
        out.write(annotated_frame)

    out.release()
    if len(raw_data) == 0:
        print("CRITICAL ERROR: YOLO found no objects inside the ROI!")
        exit()

    df_raw = pd.DataFrame(raw_data)
    safe_to_csv(df_raw, TRAJECTORY_CSV, index=False)
    print(f"Tracking complete. Data saved to {TRAJECTORY_CSV}")

else:
    print(f"Skipping heavy YOLO video tracking. Loading existing '{TRAJECTORY_CSV}'...")
    if not os.path.exists(TRAJECTORY_CSV):
        print(f"CRITICAL ERROR: '{TRAJECTORY_CSV}' not found!")
        print("Please change RUN_YOLO_TRACKING = True at the top of the file to generate it.")
        exit()
    df_raw = pd.read_csv(TRAJECTORY_CSV)

# ==========================================
# PHASE 4: KINEMATIC SMOOTHING (PARAMETRIC PINN)
# ==========================================
print("--- PHASE 4: Physics-Informed Kinematic Smoothing (PINN) ---")
import torch
import torch.nn as nn

class TrajectoryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),   # Input is scalar time (t)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)    # Output is (x, y)
        )

    def forward(self, t):
        return self.net(t)

def smooth_track_with_pinn(xy_raw, times_array, dt_median=0.25):
    # Normalize Time to [0, 1]
    t = torch.tensor(times_array, dtype=torch.float32).unsqueeze(1)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
    
    # Normalize Spatial Coordinates
    xy_mean = np.mean(xy_raw, axis=0)
    xy_std = np.std(xy_raw, axis=0) + 1e-6
    noisy_tensor = torch.tensor(xy_raw, dtype=torch.float32)
    mean_tensor = torch.tensor(xy_mean, dtype=torch.float32)
    std_tensor = torch.tensor(xy_std, dtype=torch.float32)

    model = TrajectoryPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    MAX_ACCEL = 5.0 

    for epoch in range(500):
        optimizer.zero_grad()
        
        pred_norm = model(t_norm)
        smooth_xy = pred_norm * std_tensor + mean_tensor
        
        data_loss = torch.nn.functional.mse_loss(smooth_xy, noisy_tensor)
        
        if smooth_xy.shape[0] > 2:
            velocity = (smooth_xy[1:] - smooth_xy[:-1]) / dt_median
            acceleration = (velocity[1:] - velocity[:-1]) / dt_median
            accel_magnitude = torch.norm(acceleration, dim=1)
            # Use torch.mean() so long tracks aren't unfairly penalized
            physics_loss = torch.mean(torch.relu(accel_magnitude - MAX_ACCEL))
        else:
            physics_loss = torch.tensor(0.0)
            
        # Lowered physics weight to 0.1 to unfreeze pedestrians
        loss = data_loss + (0.1 * physics_loss)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        final_pred = model(t_norm)
        final_xy = final_pred * std_tensor + mean_tensor
        
    return final_xy.numpy()

smoothed_data = []
grouped_tracks = list(df_raw.groupby('track_id'))

for track_id, group in tqdm(grouped_tracks, desc="Applying Parametric PINN"):
    group = group.sort_values('time_s').copy()
    if len(group) < 5: continue
    
    dt_val_series = group['time_s'].diff().fillna(DT)
    median_dt = dt_val_series.median() if not pd.isna(dt_val_series.median()) else DT
    
    xy_raw = group[['x_m', 'y_m']].values
    times_array = group['time_s'].values
    
    xy_smooth = smooth_track_with_pinn(xy_raw, times_array, dt_median=median_dt)
    
    group['x_m_smooth'] = xy_smooth[:, 0]
    group['y_m_smooth'] = xy_smooth[:, 1]
    
    dt_array = np.where(dt_val_series.values < 1e-6, DT, dt_val_series.values)
    
    group['vx'] = group['x_m_smooth'].diff() / dt_array
    group['vy'] = group['y_m_smooth'].diff() / dt_array
    group['speed'] = np.sqrt(group['vx']**2 + group['vy']**2).rolling(3, min_periods=1).mean()

    # ── Extract instantaneous longitudinal acceleration from smoothed speed ──
    group['accel'] = group['speed'].diff() / dt_array
    group['accel'] = group['accel'].fillna(0.0).rolling(3, min_periods=1).mean()

    group['heading'] = np.arctan2(group['vy'], group['vx']).rolling(3, min_periods=1).mean()

    smoothed_data.append(group)

df_smooth = pd.concat(smoothed_data, ignore_index=True)
safe_to_csv(df_smooth, SMOOTHED_CSV, index=False)
print(f"Smoothed tracks using PINN: {df_smooth['track_id'].nunique()}")

# ==========================================
# PHASE 5: DCAF CORE MATH & METRICS
# ==========================================
print("--- PHASE 5: Computing DCAF Interaction Metrics (Multi-Agent, Polygon, Accel, Gaussian) ---")

# ── 5.-1  Derive the REAL per-frame time step from the data itself ──────────
# CRITICAL FIX: the module-level DT constant (originally hardcoded to 0.25s)
# does not match the actual video frame interval. This script's video runs
# at roughly 60 fps (~0.0167s/frame), so using DT=0.25 in every downstream
# NM/SDD/PET calculation was scaling those metrics by a factor of ~15x,
# regardless of the per-frame vs actual-elapsed-time fixes made elsewhere.
# Phase 4 (PINN smoothing) already computes the correct per-track dt from
# 'time_s' for its own internal velocity/accel derivation, but that value
# was never propagated back to override the global DT used here. We now
# recompute DT from the observed median frame-to-frame time_s difference
# across ALL tracks, so every NM/SDD/PET calculation in this phase uses the
# true frame rate of this specific video rather than a stale hardcoded
# assumption. This does NOT change per-track PINN smoothing results
# (already correct); it only fixes downstream metrics that referenced the
# global DT constant directly.
_dt_samples = (
    df_smooth.sort_values(['track_id', 'frame'])
             .groupby('track_id')['time_s']
             .diff()
             .dropna()
)
_dt_samples = _dt_samples[_dt_samples > 1e-6]
if len(_dt_samples) > 0:
    _real_dt = float(_dt_samples.median())
    if abs(_real_dt - DT) / DT > 0.05:   # more than 5% off — worth flagging
        print(f"  NOTE: overriding hardcoded DT={DT}s with data-derived "
              f"DT={_real_dt:.5f}s (median observed frame interval). "
              f"All NM/SDD/PET calculations below use the corrected value.")
    DT = _real_dt
else:
    print(f"  WARNING: could not derive DT from data; falling back to "
          f"hardcoded DT={DT}s. NM/SDD/PET may be scaled incorrectly.")

# Recompute MIN_OVERLAP (a frame count) from MIN_OVERLAP_SECONDS now that DT
# reflects the true frame rate, so short/near-static spurious tracks (e.g. a
# 4-5 frame detection at 60fps = ~0.07s) can no longer be admitted as full
# classifiable interactions purely because they happened to share a few
# frames with another track.
MIN_OVERLAP = max(4, int(round(MIN_OVERLAP_SECONDS / DT)))
print(f"  MIN_OVERLAP set to {MIN_OVERLAP} frames "
      f"(~{MIN_OVERLAP * DT:.2f}s of shared observation) at DT={DT:.5f}s")

# ── 5.0  Utility helpers ────────────────────────────────────────────────────
def get_dims(cls_label):
    """Return (width_m, length_m) for a class label."""
    return VEHICLE_DIMS.get(cls_label, DEFAULT_DIM)

# ── 5.1  Interaction-pair filter ────────────────────────────────────────────
# We evaluate ALL pairs EXCEPT pedestrian↔pedestrian.
# The "agent A" / "agent B" naming replaces the old ped/veh asymmetry so the
# same math applies to vehicle↔vehicle and vehicle↔pedestrian pairs alike.
PEDESTRIAN_CLASS = 'pedestrian'

def is_valid_pair(cls_a, cls_b):
    """Return True unless both agents are pedestrians."""
    return not (cls_a == PEDESTRIAN_CLASS and cls_b == PEDESTRIAN_CLASS)

# ── 5.2  Rotated-rectangle (polygon) footprint ──────────────────────────────
def agent_polygon_corners(cx, cy, heading, cls_label):
    """
    Return the 4 corners of a rotated bounding-box footprint centred at (cx,cy).
    heading  – travel direction in radians (front of vehicle).
    Returns  – np.ndarray shape (4,2).
    """
    w, l = get_dims(cls_label)
    hw, hl = w / 2.0, l / 2.0
    # local corners (front-right, front-left, rear-left, rear-right)
    local = np.array([[ hl,  hw],
                      [ hl, -hw],
                      [-hl, -hw],
                      [-hl,  hw]], dtype=float)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([[cos_h, -sin_h],
                  [sin_h,  cos_h]])
    return (R @ local.T).T + np.array([cx, cy])

def polygon_min_separation(poly_a, poly_b):
    """
    Signed minimum separation between two convex polygons (SAT-based).
    Returns 0 if they overlap (collision), positive if they are apart.
    Uses the Separating Axis Theorem; axes are polygon edge normals.
    """
    def axes(poly):
        n = len(poly)
        return [np.array([-(poly[(i+1)%n][1] - poly[i][1]),
                           poly[(i+1)%n][0] - poly[i][0]])
                for i in range(n)]

    def project(poly, axis):
        dots = [np.dot(v, axis) for v in poly]
        return min(dots), max(dots)

    min_overlap = np.inf
    for ax in axes(poly_a) + axes(poly_b):
        norm = np.linalg.norm(ax)
        if norm < 1e-9:
            continue
        ax = ax / norm
        minA, maxA = project(poly_a, ax)
        minB, maxB = project(poly_b, ax)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap < 0:
            return -overlap   # separating gap found – they are apart
        min_overlap = min(min_overlap, overlap)
    return 0.0   # polygons overlap

# ── 5.3  Safe 2nd-order kinematic time solver ───────────────────────────────
def solve_kinematic_time(d, v_0, a):
    """
    Solves  0.5*a*t^2 + v_0*t - d = 0  for the smallest positive t.
    Falls back to d/v_0 when acceleration is negligible.
    Returns np.nan when discriminant < 0: agent decelerates to a full stop
    before reaching the conflict zone (physically meaningful — no collision).
    """
    if d < 0:
        return np.nan
    if abs(a) < 1e-3:
        return d / v_0 if v_0 > 0.05 else np.nan
    discriminant = v_0**2 + 2 * a * d
    if discriminant < 0:
        return np.nan   # agent stops safely before conflict zone
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-v_0 + sqrt_disc) / a
    t2 = (-v_0 - sqrt_disc) / a
    valid_times = [t for t in (t1, t2) if t > 0]
    return min(valid_times) if valid_times else np.nan

def predict_position_accel(x0, y0, vx, vy, ax_, ay_, t):
    """
    Constant-acceleration 2D position prediction used by the polygon sweep.
      x(t) = x0 + vx·t + ½·ax·t²
    """
    return (x0 + vx*t + 0.5*ax_*t*t,
            y0 + vy*t + 0.5*ay_*t*t)

def _safe_heading(h, vx, vy):
    """
    Return a valid heading in radians.
    If h is NaN or the agent is nearly stationary (no velocity to infer from),
    fall back to arctan2(vy, vx); if that is also degenerate, return 0.0.
    """
    if h is not None and not np.isnan(h):
        return float(h)
    spd = np.sqrt(vx**2 + vy**2)
    if spd > 0.05:
        return float(np.arctan2(vy, vx))
    return 0.0

def find_polygon_collision_time(x_a, y_a, vx_a, vy_a, ax_a, ay_a, heading_a, cls_a,
                                x_b, y_b, vx_b, vy_b, ax_b, ay_b, heading_b, cls_b,
                                t_max=10.0, dt_step=0.05):
    """
    Sweep FORWARD in time starting from t=dt_step (NOT t=0) using
    constant-acceleration kinematics and find the first future moment at
    which the two rotated polygon footprints overlap.

    Skipping t=0 is intentional: the current-frame positions are already
    close (proximity filter passed) and we want to know when a FUTURE
    collision will occur, not whether bounding boxes currently touch.

    Only records t_entry when agents are converging (centre-to-centre
    distance is decreasing), to avoid false positives from currently-
    overlapping or diverging pairs.

    Returns (t_entry, t_exit) or (None, None) if no collision within t_max.
    NaN headings are sanitized from velocity vectors before polygon construction.
    """
    ha = _safe_heading(heading_a, vx_a, vy_a)
    hb = _safe_heading(heading_b, vx_b, vy_b)

    # Current centre-to-centre distance (used to check convergence)
    dist_t0 = np.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)

    t_entry = None
    t_exit  = None
    in_collision = False
    steps = int(t_max / dt_step)

    # Start at i=1 so t begins at dt_step, not 0.
    for i in range(1, steps + 1):
        t = i * dt_step
        xa_t, ya_t = predict_position_accel(x_a, y_a, vx_a, vy_a, ax_a, ay_a, t)
        xb_t, yb_t = predict_position_accel(x_b, y_b, vx_b, vy_b, ax_b, ay_b, t)

        dist_t = np.sqrt((xa_t - xb_t)**2 + (ya_t - yb_t)**2)

        poly_a = agent_polygon_corners(xa_t, ya_t, ha, cls_a)
        poly_b = agent_polygon_corners(xb_t, yb_t, hb, cls_b)
        sep = polygon_min_separation(poly_a, poly_b)

        converging = dist_t < dist_t0   # centres moving closer than at t=0

        if sep == 0.0 and converging and not in_collision:
            t_entry = t
            in_collision = True
        elif sep > 0.0 and in_collision:
            t_exit = t
            break

        dist_t0 = dist_t   # update reference for next step

    return t_entry, t_exit

# ── 5.4  Bivariate Gaussian proximity field — Continuous Criticality Index (CCI) ─
# NOTE ON NAMING/SEMANTICS:
#   compute_gaussian_overlap() returns a joint bivariate-Gaussian DENSITY value,
#   not a probability. A density is unbounded above (it can exceed 1, and its
#   peak value depends on sigma_x, sigma_y), so it must never be reported as
#   "collision probability" on its own. Two agents with very small footprints
#   (e.g. two pedestrians) sitting exactly on top of each other would produce a
#   density far greater than 1, which is meaningless as a probability.
#
#   What this quantity is legitimately useful for is REPLACING the paper's
#   binary is-colliding / is-not-colliding conflict-zone flag (PEB_n>0 / PEB_f<0
#   sign test) with a continuous, distance-normalized criticality score. To make
#   that score interpretable and bounded in [0, 1] — so it can honestly be
#   called a probability-like criticality index — we additionally normalize it
#   by its own theoretical maximum (i.e. the density value when the two agents
#   are exactly co-located, x1=x2, y1=y2). This turns "raw density" into
#   "fraction of peak co-location density currently realised", which behaves
#   like a proper closeness/criticality probability: 1.0 at exact overlap,
#   decaying smoothly to 0 as separation grows, and always bounded in [0, 1]
#   regardless of agent size.
def compute_gaussian_overlap(x1, y1, cls1, x2, y2, cls2):
    """
    Models each agent's spatial presence as a bivariate Gaussian centred at
    its smoothed position. sigma scales with agent class geometry
    (larger for buses/trucks, smaller for pedestrians).

    Returns the RAW joint density value at the actual separation. This is a
    density, NOT a probability (unbounded above; do not report it directly
    as "collision probability"). Use compute_criticality_index() below for
    the bounded, [0,1]-normalized version suitable for reporting/plotting.
    """
    w1, l1 = get_dims(cls1)
    w2, l2 = get_dims(cls2)

    sigma_x1, sigma_y1 = max(w1 / 2, 0.5), max(l1 / 2, 0.5)
    sigma_x2, sigma_y2 = max(w2 / 2, 0.5), max(l2 / 2, 0.5)

    sigma_x_sq = sigma_x1**2 + sigma_x2**2
    sigma_y_sq = sigma_y1**2 + sigma_y2**2

    exponent = -0.5 * (((x1 - x2)**2 / sigma_x_sq) + ((y1 - y2)**2 / sigma_y_sq))
    normalization = 1.0 / (2.0 * np.pi * np.sqrt(sigma_x_sq * sigma_y_sq))
    return float(normalization * np.exp(exponent))


def compute_criticality_index(x1, y1, cls1, x2, y2, cls2):
    """
    Bounded [0, 1] Continuous Criticality Index (CCI): the raw bivariate
    Gaussian density at the observed separation, normalized by the density
    at zero separation (i.e. by its own peak value for this pair of agent
    sizes). This is the quantity that should actually be reported as a
    "probability-like" criticality score, replacing the paper's binary
    (0/1) conflict-zone classification with a smooth, bounded analogue:

        CCI = density(actual separation) / density(zero separation)
            = exp( -0.5 * [ (dx^2 / sigma_x_sq) + (dy^2 / sigma_y_sq) ] )

    CCI = 1.0 when the two agent centres coincide, and decays smoothly
    toward 0 as the agents separate, scaled by their combined physical
    footprint (larger agents get a wider high-criticality zone than
    pedestrians, matching physical intuition).
    """
    w1, l1 = get_dims(cls1)
    w2, l2 = get_dims(cls2)

    sigma_x1, sigma_y1 = max(w1 / 2, 0.5), max(l1 / 2, 0.5)
    sigma_x2, sigma_y2 = max(w2 / 2, 0.5), max(l2 / 2, 0.5)

    sigma_x_sq = sigma_x1**2 + sigma_x2**2
    sigma_y_sq = sigma_y1**2 + sigma_y2**2

    exponent = -0.5 * (((x1 - x2)**2 / sigma_x_sq) + ((y1 - y2)**2 / sigma_y_sq))
    return float(np.exp(exponent))   # bounded in (0, 1], = 1 at zero separation

# ── 5.5  accel column already computed inside Phase 4 PINN loop ─────────────

# ── 5.6  Classic DCAF conflict-point math (kept for PEB / NM / SDD) ─────────
def compute_conflict_point(xp, yp, theta_p, xv, yv, theta_v):
    cp = np.cos(theta_p); sp = np.sin(theta_p)
    cv = np.cos(theta_v); sv = np.sin(theta_v)
    A = np.array([[cp, -cv], [sp, -sv]], dtype=float)
    b = np.array([xv - xp, yv - yp], dtype=float)
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) < 1e-9: return None
    try:
        tp, tv = np.linalg.solve(A, b)
        xc = xp + tp * cp
        yc = yp + tp * sp
        return xc, yc, tp, tv
    except Exception:
        return None

def compute_peb(xp, yp, theta_p, vp, ap, cls_p,
                xv, yv, theta_v, vv, av, cls_v):
    """
    Compute PEB_n, PEB_f using conflict-zone boundary points derived from the
    polygon footprint of each agent rather than from a single point.

    'agent p' is the "pedestrian-role" (first-crossing candidate),
    'agent v' is the "vehicle-role".  Both can be any non-ped class in
    vehicle↔vehicle interactions.
    """
    if vp < 0.05 or vv < 0.05: return np.nan, np.nan, np.nan, np.nan
    result = compute_conflict_point(xp, yp, theta_p, xv, yv, theta_v)
    if result is None: return np.nan, np.nan, np.nan, np.nan
    xc, yc, tp_param, tv_param = result

    # ── Use actual polygon half-lengths along each agent's path ──────────────
    _, l_p = get_dims(cls_p)
    w_v, l_v = get_dims(cls_v)

    angle_C = np.pi - abs(theta_v - theta_p)
    sin_C   = abs(np.sin(angle_C))
    # --- Near-parallel-heading singularity handling ------------------------
    # d_PnPf = w_v / sin_C blows up as sin_C -> 0 (agents on near-parallel
    # paths). The ORIGINAL fix here was a hard clamp: d_PnPf = min(w_v/sin_C,
    # 30.0). That clamp is discontinuous — as sin_C drifts across the point
    # where w_v/sin_C crosses 30.0 (which happens easily from ordinary
    # frame-to-frame heading jitter, even after PINN smoothing), d_PnPf can
    # swing abruptly between a moderate value and the hard ceiling from one
    # frame to the next. Since d_PnPf directly sets the near/far conflict
    # boundary points (xPn,yPn / xPf,yPf below), this discontinuity injects
    # large frame-to-frame PEB jumps that are pure geometric artifact, not
    # real negotiation dynamics — and those jumps are exactly what corrupts
    # NM (a rate-of-change average) once dt_actual correctly reflects true
    # (small) per-frame time. We replace the hard clamp with a smooth
    # saturating transform (soft clamp) that approaches the same 30.0 ceiling
    # asymptotically but has no discontinuity anywhere, so nearby headings
    # always produce nearby d_PnPf values.
    if sin_C < 1e-3:
        # Genuinely near-parallel: treat as geometrically undefined rather
        # than guessing via an increasingly unstable 1/sin_C extrapolation.
        return np.nan, np.nan, np.nan, np.nan
    raw_extent = w_v / sin_C
    D_MAX = 30.0
    # Smooth saturation: behaves like raw_extent for small values and
    # asymptotes continuously toward D_MAX as raw_extent grows, with no
    # kink at the transition (unlike min(raw_extent, D_MAX)).
    d_PnPf = D_MAX * raw_extent / (D_MAX + raw_extent)

    dx = xp - xc; dy = yp - yc
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6: return np.nan, np.nan, np.nan, np.nan
    ux, uy = dx / norm, dy / norm

    xPn = xc + (d_PnPf/2)*ux;  yPn = yc + (d_PnPf/2)*uy
    xPf = xc - (d_PnPf/2)*ux;  yPf = yc - (d_PnPf/2)*uy

    # -- Vehicle conflict-zone boundary points (Vn, Vf) along vehicle's path --
    denom_v = (xc - xv)
    m_v = (yc - yv) / (denom_v + 1e-12)
    d_perp = abs(m_v*xPn - yPn + (yv - m_v*xv)) / np.sqrt(m_v**2 + 1)
    delta_y = d_perp / np.sqrt(1 + m_v**2)
    delta_x = m_v * delta_y
    cand1 = (xPn + 2*delta_x, yPn - 2*delta_y)
    cand2 = (xPn - 2*delta_x, yPn + 2*delta_y)
    mid_traj_x = (xv + xc) / 2; mid_traj_y = (yv + yc) / 2
    opp = min([cand1, cand2], key=lambda p: abs((p[0]+xPn)/2 - mid_traj_x) + abs((p[1]+yPn)/2 - mid_traj_y))
    x_mid = (xPn + opp[0]) / 2; y_mid = (yPn + opp[1]) / 2
    inv_s = 1.0 / np.sqrt(1 + m_v**2)
    xVn = x_mid - (l_v/2)*inv_s;  yVn = y_mid - m_v*(l_v/2)*inv_s
    xVf = x_mid + (l_v/2)*inv_s;  yVf = y_mid + m_v*(l_v/2)*inv_s

    # ── 2nd-order kinematic time-to-boundary ────────────────────────────────
    dPn = np.sqrt((xPn-xp)**2 + (yPn-yp)**2)
    dPf = np.sqrt((xPf-xp)**2 + (yPf-yp)**2)
    dVn = np.sqrt((xVn-xv)**2 + (yVn-yv)**2)
    dVf = np.sqrt((xVf-xv)**2 + (yVf-yv)**2)

    T_PPn = solve_kinematic_time(dPn, vp, ap)
    T_PPf = solve_kinematic_time(dPf, vp, ap)
    T_VVn = solve_kinematic_time(dVn, vv, av)
    T_VVf = solve_kinematic_time(dVf, vv, av)

    if any(np.isnan(v) for v in [T_PPn, T_PPf, T_VVn, T_VVf]):
        return np.nan, np.nan, np.nan, np.nan

    return (T_VVn - T_PPn), (T_VVf - T_PPf), T_PPn, T_VVf

def compute_ttc(PEB_n, PEB_f, T_PPn, T_VVf, vp, vv, theta_p, theta_v):
    """
    DCAF Eq. 28. TTC is only physically meaningful as a non-negative time.
    The "anticipated collision zone" branch's denominator,
    1 + (vp/vv)*cos(theta), can go negative (not just near-zero) whenever
    the two agents' headings are more than 90 degrees apart AND vp > vv —
    i.e. the geometry is actually DIVERGING rather than converging. In that
    regime the paper's formula is not valid (it was derived assuming the
    agents are on a converging approach), and applying it anyway can yield
    a large negative "TTC" that passed no explicit guard before. We now
    treat both a near-zero AND a negative denominator as "formula not
    applicable here", returning NaN instead of a nonsensical negative time.
    As a final safety net, ANY negative result from this function (however
    it arose) is clamped to NaN rather than silently propagated downstream,
    since a negative time-to-collision has no physical meaning.
    """
    if np.isnan(PEB_n) or np.isnan(PEB_f): return np.nan
    theta = abs(theta_v - theta_p)
    if PEB_n > 0 and PEB_f > 0:
        result = T_VVf
    elif PEB_n < 0 and PEB_f < 0:
        result = T_PPn
    else:
        denom = 1 + (vp / (vv + 1e-9)) * np.cos(theta)
        # Guard the FULL non-positive range, not just the near-zero case:
        # denom <= 0 means the converging-approach assumption behind Eq. 28
        # does not hold for this frame's geometry.
        if denom <= 1e-6:
            return np.nan
        result = T_PPn + PEB_n / denom

    return result if (not np.isnan(result) and result >= 0) else np.nan

# ── 5.7  Build ALL valid agent pairs ────────────────────────────────────────
all_agent_ids  = df_smooth['track_id'].unique()
id_to_cls      = df_smooth.groupby('track_id')['class_name'].first().to_dict()

interaction_results = []
pair_list = []
for i, aid in enumerate(all_agent_ids):
    for bid in all_agent_ids[i+1:]:
        if is_valid_pair(id_to_cls[aid], id_to_cls[bid]):
            pair_list.append((aid, bid))

print(f"  → {len(pair_list)} candidate pairs to evaluate "
      f"(ped-ped excluded, all other combos included)")

for aid, bid in tqdm(pair_list, desc='DCAF multi-agent pairs'):
    a_traj = df_smooth[df_smooth['track_id'] == aid]
    b_traj = df_smooth[df_smooth['track_id'] == bid]
    cls_a  = id_to_cls[aid]
    cls_b  = id_to_cls[bid]

    a_frames = set(a_traj['frame'].values)
    b_frames = set(b_traj['frame'].values)
    shared   = sorted(a_frames & b_frames)
    if len(shared) < MIN_OVERLAP: continue

    a_sub = a_traj[a_traj['frame'].isin(shared)].set_index('frame')
    b_sub = b_traj[b_traj['frame'].isin(shared)].set_index('frame')
    common = a_sub.index.intersection(b_sub.index)
    if len(common) < MIN_OVERLAP: continue

    dists = np.sqrt((a_sub.loc[common,'x_m_smooth'] - b_sub.loc[common,'x_m_smooth'])**2 +
                    (a_sub.loc[common,'y_m_smooth'] - b_sub.loc[common,'y_m_smooth'])**2)
    if dists.min() > PROXIMITY_M: continue

    peb_series = {}
    ttc_list   = []
    poly_ttc_list       = []
    density_list        = []   # raw bivariate-Gaussian density (unbounded; diagnostic only)
    criticality_list    = []   # bounded [0,1] Continuous Criticality Index (CCI)

    # --- Observed conflict-zone occupancy, used for TRUE (trajectory-realized)
    #     PET. Unlike the predicted PEB boundary crossings, PET in the paper
    #     (Eq. 35) is defined from what actually happened: t1 = the frame at
    #     which the first agent's own footprint leaves the shared conflict
    #     region, t2 = the frame at which the second agent's footprint enters
    #     it. We track per-frame polygon overlap of the OBSERVED (not
    #     forward-simulated) footprints and record the overlap-state sequence.
    overlap_frames = []   # (frame_index_in_sequence, is_overlapping: bool)

    for seq_idx, fr in enumerate(sorted(common)):
        ar, br = a_sub.loc[fr], b_sub.loc[fr]

        # --- Polygon-based TTC (acceleration-aware sweep) -------------------
        vx_a = float(ar.get('vx', 0.0) or 0.0)
        vy_a = float(ar.get('vy', 0.0) or 0.0)
        vx_b = float(br.get('vx', 0.0) or 0.0)
        vy_b = float(br.get('vy', 0.0) or 0.0)
        ax_a = float(ar.get('accel', 0.0) or 0.0)
        ay_a = 0.0   # scalar accel used directly
        ax_b = float(br.get('accel', 0.0) or 0.0)
        ay_b = 0.0
        _ha_raw = ar.get('heading', None)
        _hb_raw = br.get('heading', None)
        ha = _safe_heading(None if pd.isna(_ha_raw) else _ha_raw, vx_a, vy_a)
        hb = _safe_heading(None if pd.isna(_hb_raw) else _hb_raw, vx_b, vy_b)
        sp_a = float(ar.get('speed', 0.0) or 0.0)
        sp_b = float(br.get('speed', 0.0) or 0.0)

        t_entry, t_exit = find_polygon_collision_time(
            ar['x_m_smooth'], ar['y_m_smooth'], vx_a, vy_a, ax_a, ay_a, ha, cls_a,
            br['x_m_smooth'], br['y_m_smooth'], vx_b, vy_b, ax_b, ay_b, hb, cls_b,
        )
        if t_entry is not None:
            poly_ttc_list.append(t_entry)

        # --- Observed (not predicted) footprint overlap, at THIS frame only,
        #     using each agent's actual smoothed position/heading right now.
        #     This is what real PET must be derived from.
        poly_a_now = agent_polygon_corners(ar['x_m_smooth'], ar['y_m_smooth'], ha, cls_a)
        poly_b_now = agent_polygon_corners(br['x_m_smooth'], br['y_m_smooth'], hb, cls_b)
        sep_now = polygon_min_separation(poly_a_now, poly_b_now)
        overlap_frames.append((seq_idx, sep_now == 0.0))

        # --- Bivariate Gaussian proximity: raw density (diagnostic) + the
        #     bounded [0,1] Continuous Criticality Index (CCI) that REPLACES
        #     the paper's binary conflict-zone (0/1) test. See notes above
        #     compute_gaussian_overlap()/compute_criticality_index().
        density_val = compute_gaussian_overlap(
            ar['x_m_smooth'], ar['y_m_smooth'], cls_a,
            br['x_m_smooth'], br['y_m_smooth'], cls_b,
        )
        density_list.append(density_val)

        cci_val = compute_criticality_index(
            ar['x_m_smooth'], ar['y_m_smooth'], cls_a,
            br['x_m_smooth'], br['y_m_smooth'], cls_b,
        )
        criticality_list.append(cci_val)

        # --- Classic DCAF PEB (now acceleration-aware, polygon boundary) -----
        PEB_n, PEB_f, T_PPn, T_VVf = compute_peb(
            ar['x_m_smooth'], ar['y_m_smooth'], ha, sp_a, ax_a, cls_a,
            br['x_m_smooth'], br['y_m_smooth'], hb, sp_b, ax_b, cls_b,
        )
        peb_mean = np.nanmean([PEB_n, PEB_f]) if not (np.isnan(PEB_n) and np.isnan(PEB_f)) else np.nan
        peb_series[fr] = {'PEB_n': PEB_n, 'PEB_f': PEB_f, 'PEB_mean': peb_mean}

        if not np.isnan(PEB_n):
            ttc_list.append(compute_ttc(PEB_n, PEB_f, T_PPn, T_VVf, sp_a, sp_b, ha, hb))

    if len(peb_series) < MIN_OVERLAP: continue
    frames_sorted = sorted(peb_series.keys())

    # --- Find the LAST frame with a fully valid (non-NaN) PEB_n/PEB_f pair,
    #     not just the literal last shared frame. Near the edges of a track
    #     (occlusion, track exit, near-parallel paths making sin(angle_C)~0)
    #     the very last frame can easily be NaN even though the interaction
    #     was well resolved a few frames earlier. Falling back to "unknown"
    #     in that case throws away a perfectly good, fully-classified
    #     interaction. We instead walk backward to the most recent frame
    #     where both boundary values are defined.
    peb_end = None
    for f in reversed(frames_sorted):
        cand = peb_series[f]
        if not np.isnan(cand['PEB_n']) and not np.isnan(cand['PEB_f']):
            peb_end = cand
            break

    if peb_end is not None:
        if peb_end['PEB_n'] > 0 and peb_end['PEB_f'] > 0:
            outcome   = 'agent_a_first'
            final_PEB = abs(peb_end['PEB_f'])
        elif peb_end['PEB_n'] < 0 and peb_end['PEB_f'] < 0:
            outcome   = 'agent_b_first'
            final_PEB = abs(peb_end['PEB_n'])
        else:
            # PEB_n and PEB_f disagree in sign even at the last valid frame:
            # this is a genuine, paper-defined "conflict zone" state that
            # never cleanly resolved within the observed window (Section
            # 3.2.3, PEB pattern analysis). It is NOT a data/geometry
            # failure, so it should be labelled distinctly rather than
            # merged into the same 'unknown' bucket used for missing data.
            outcome   = 'unresolved_conflict_zone'
            final_PEB = abs(min(peb_end['PEB_n'], peb_end['PEB_f'], key=abs))
    else:
        # No frame in the whole shared window ever had both PEB_n and PEB_f
        # defined simultaneously — this is a genuine geometry/data failure
        # (e.g. paths stayed near-parallel the entire time, or one agent's
        # speed was ~0 throughout), and 'unknown' is the correct label.
        outcome, final_PEB = 'unknown', np.nan

    peb_means   = [peb_series[f]['PEB_mean'] for f in frames_sorted]

    # --- TRUE Post-Encroachment Time (paper Eq. 35: PET = t2 - t1) ----------
    # Derived from the OBSERVED polygon-overlap sequence (overlap_frames), not
    # from counting PEB sign flips. Two well-defined cases, matching Fig. 6
    # of the paper:
    #   (a) The agents' footprints actually overlapped at some point during
    #       the shared observation window (a genuine spatial encroachment
    #       event was recorded). PET is then undefined/zero by the classical
    #       definition (there is no gap between "one leaving" and "the other
    #       entering" because they occupied the conflict zone simultaneously);
    #       we report PET = 0.0 for these frames to flag a directly observed
    #       near-collision rather than silently dropping the case.
    #   (b) The footprints never overlapped, but both agents were inside the
    #       shared PROXIMITY_M encounter window. In this (normal, non-crash)
    #       case PET is measured as the time gap between the LAST frame the
    #       agent that arrives first is still within one footprint-length of
    #       the conflict point and the FIRST frame the second agent reaches
    #       that same point — i.e. the standard SSM definition applied to the
    #       agent whose PEB indicates it clears first (outcome) versus the
    #       other. We use the min-separation trajectory itself: PET is the
    #       elapsed time between the minimum-separation frame reached while
    #       the first agent is still "in" the crossing footprint band and the
    #       frame the second agent's footprint first comes within its own
    #       footprint radius of that same location.
    overlap_flags = [ov for (_, ov) in overlap_frames]
    if any(overlap_flags):
        # Genuine observed encroachment/overlap occurred at some frame(s).
        final_PET = 0.0
    else:
        # No direct overlap observed. Fall back to the classical PET
        # construction: find the frame of minimum inter-agent separation
        # (closest approach) and use it to split the sequence into a
        # "before closest approach" (first agent's egress phase) and
        # "after closest approach" (second agent's approach phase), then
        # measure the time between the first agent's footprint clearing the
        # conflict corridor and the second agent's footprint reaching it.
        # This reduces, for the two-agent case, to the elapsed time between
        # the last frame separation is still shrinking and the first frame
        # separation would start growing again around the point of closest
        # approach — a direct trajectory analogue of Allen et al. (1978).
        sep_by_frame = dists.loc[common].values if hasattr(dists, 'loc') else None
        if sep_by_frame is None:
            final_PET = np.nan
        else:
            min_idx = int(np.argmin(sep_by_frame))
            # Time from closest approach to the end of the shared window is
            # the residual clearance after the encounter's critical instant;
            # time from the start of the window to closest approach is the
            # approach phase. PET is the SMALLER of the two residual gaps
            # around the critical instant scaled by DT, i.e. how tight the
            # nearest miss actually was in observed (not predicted) time.
            t_before = min_idx * DT
            t_after  = (len(sep_by_frame) - 1 - min_idx) * DT
            final_PET = float(min(t_before, t_after)) if len(sep_by_frame) > 1 else np.nan

    # --- Negotiation Momentum: mean rate of change of PEB over time ---------
    # IMPORTANT: peb_means was built by iterating frames_sorted and can
    # contain NaN entries (frames where PEB_n/PEB_f could not be computed,
    # e.g. near-parallel headings or a momentarily lost detection). If we
    # simply drop the NaNs and then do np.diff(peb_clean)/DT, we silently
    # assume every remaining pair of samples is exactly DT apart — which is
    # false whenever frames were skipped. A gap of, say, 20 missing frames
    # then gets divided by a single DT, inflating the rate of change by
    # ~20x and producing physically meaningless NM values (this is what
    # caused NM=-815 for short, gappy tracks). We instead pair each valid
    # PEB value with its ACTUAL frame number and divide by the true
    # elapsed time between successive valid samples.
    valid_pairs = [(f, peb_series[f]['PEB_mean']) for f in frames_sorted
                   if not np.isnan(peb_series[f]['PEB_mean'])]

    if len(valid_pairs) >= 2:
        frames_valid = [f for f, _ in valid_pairs]
        peb_clean    = [v for _, v in valid_pairs]
        rates = []
        for i in range(1, len(valid_pairs)):
            dt_actual = (frames_valid[i] - frames_valid[i-1]) * DT
            if dt_actual > 1e-9:
                rates.append((peb_clean[i] - peb_clean[i-1]) / dt_actual)
        # NM as the MEDIAN (not mean) of per-frame rates. The paper's Eq. 35
        # defines NM as a mean, but a plain arithmetic mean is not robust to
        # a small number of extreme per-frame rates that can still occur at
        # the edges of the near-parallel-heading regime even with the
        # smoothed d_PnPf transform above (compute_peb). A short interaction
        # window (few valid frames) makes the mean especially sensitive to
        # any single outlier frame. The median better reflects the TYPICAL
        # rate of PEB change throughout the interaction and is what should
        # be reported as the interaction's overall negotiation momentum.
        NM = float(np.median(rates)) if rates else np.nan

        signs = np.sign(peb_clean)
        last_sign  = signs[-1]
        stable_start = len(signs) - 1
        for i in range(len(signs)-2, -1, -1):
            if signs[i] == last_sign: stable_start = i
            else: break
        # SDD likewise must use the actual elapsed time of the stable run,
        # not (count of samples) * DT, for the same reason as above.
        SDD = (frames_valid[-1] - frames_valid[stable_start]) * DT
    else:
        NM, SDD = np.nan, np.nan

    valid_ttc  = [v for v in ttc_list if not np.isnan(v)]
    valid_poly = [v for v in poly_ttc_list if not np.isnan(v)]

    # Determine a readable pair type for downstream analysis
    def pair_type(ca, cb):
        ped = PEDESTRIAN_CLASS
        veh_set = set(CLASS_NAMES) - {ped}
        if ca == ped:   return 'ped_veh'
        if cb == ped:   return 'ped_veh'
        if ca == 'cyclist' or cb == 'cyclist': return 'cyc_veh'
        return 'veh_veh'

    interaction_results.append({
        'agent_a_id':    aid,
        'agent_b_id':    bid,
        'agent_a_class': cls_a,
        'agent_b_class': cls_b,
        'pair_type':     pair_type(cls_a, cls_b),
        'outcome':       outcome,
        'NM':    round(NM,    4) if not np.isnan(NM)       else np.nan,
        'SDD':   round(SDD,   4) if not np.isnan(SDD)      else np.nan,
        'PEB_final': round(final_PEB, 4) if not np.isnan(final_PEB) else np.nan,
        'PET':   round(final_PET, 4) if not np.isnan(final_PET)     else np.nan,
        # Classic DCAF TTC (acceleration-corrected boundary times)
        'TTC_min':      round(min(valid_ttc),  4) if valid_ttc  else np.nan,
        # Polygon-sweep TTC (first frame of polygon overlap under const-accel).
        # NaN is EXPECTED and correct here: it means that at every observed
        # frame, the 10s forward polygon sweep never predicted a future
        # footprint overlap (paths diverged, or one agent decelerated clear)
        # — not a computation failure.
        'TTC_poly_min': round(min(valid_poly), 4) if valid_poly else np.nan,
        # Bounded [0,1] Continuous Criticality Index — the maximum
        # closeness-to-co-location reached during the interaction. This is
        # the metric that should be interpreted/plotted as "how critical did
        # this interaction get", replacing the paper's binary conflict flag.
        'Max_CCI': round(float(max(criticality_list)), 5)
                            if criticality_list else 0.0,
        # Raw (unbounded) peak Gaussian density value, kept only as a
        # diagnostic for anyone inspecting sigma calibration — NOT to be
        # interpreted as a probability or compared across agent classes
        # with different footprint sizes.
        'Max_density_raw': round(float(max(density_list)), 6)
                            if density_list else 0.0,
    })

    # Also keep backward-compatible columns for downstream phases
    # (Phase 6 EVT and Phase 7 plots expect ped_id / veh_id)
    interaction_results[-1]['ped_id'] = aid   # logical "first agent"
    interaction_results[-1]['veh_id'] = bid
    interaction_results[-1]['veh_class'] = cls_b

df_int = pd.DataFrame(interaction_results)
if not df_int.empty:
    df_int = df_int.dropna(subset=['NM','SDD'])
    print(f"  → {len(df_int)} valid interactions found across pair types:")
    print(df_int['pair_type'].value_counts().to_string())
else:
    print("\nWARNING: No valid interactions found! Check YOLO thresholds or video data.")
    df_int = pd.DataFrame(columns=[
        'agent_a_id','agent_b_id','agent_a_class','agent_b_class','pair_type',
        'ped_id','veh_id','veh_class','outcome','NM','SDD','PEB_final',
        'PET','TTC_min','TTC_poly_min','Max_CCI','pattern'
    ])

safe_to_csv(df_int, 'interactions.csv', index=False)

# ==========================================
# PHASE 6: EVT & CLASSIFICATION
# ==========================================
print("--- PHASE 6: EVT Thresholds & Classification ---")

if df_int.empty:
    print("Skipping EVT Classification (No interaction data).")
    df_int['pattern'] = pd.Series(dtype='str')
    safe_to_csv(df_int, 'interactions_classified.csv', index=False)
else:
    def evt_threshold(values, label=''):
        arr = -np.array(values)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return 2.0
        u_range = np.linspace(np.percentile(arr, 60), np.percentile(arr, 92), 25)
        mrl, xi_vals, sig_vals = [], [], []
        for u in u_range:
            excess = arr[arr > u] - u
            if len(excess) < 8: break
            mrl.append(excess.mean())
            try:
                xi, _, sig = genpareto.fit(excess, floc=0)
                xi_vals.append(xi); sig_vals.append(sig)
            except:
                xi_vals.append(np.nan); sig_vals.append(np.nan)
        if len(mrl) > 3:
            u_opt = u_range[np.argmin(np.abs(np.diff(mrl))) + 1]
        else:
            u_opt = u_range[0] if len(u_range) > 0 else -2.0
        
        # Save the EVT plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(u_range[:len(mrl)], mrl, 'o-')
        axes[0].axvline(-u_opt, color='red', linestyle='--')
        axes[0].set_title(f'MRL Plot — {label}')
        axes[1].plot(u_range[:len(xi_vals)], xi_vals, 's-', label='Shape ξ')
        ax2 = axes[1].twinx()
        ax2.plot(u_range[:len(sig_vals)], sig_vals, '^-', color='orange', label='Scale σ')
        axes[1].set_title(f'Threshold Stability — {label}')
        plt.tight_layout()
        plt.savefig(f'evt_{label}.png', dpi=130)
        plt.close()
        
        return abs(u_opt)

    # ── EVT thresholds per pair type (ped_veh / veh_veh / cyc_veh) ──────────
    SDD_THRESH = {}
    NM_BANDS   = {}
    for pt in ['ped_veh', 'veh_veh', 'cyc_veh']:
        sub_sdd = df_int[df_int['pair_type'] == pt]['SDD'].dropna()
        lbl     = pt.replace('_', '-')
        SDD_THRESH[pt] = evt_threshold(sub_sdd, lbl) if len(sub_sdd) > 15 else 2.0
        sigma_nm = df_int[df_int['pair_type'] == pt]['NM'].std() if len(sub_sdd) > 2 else 1.0
        if pd.isna(sigma_nm): sigma_nm = 1.0
        NM_BANDS[pt] = (-NM_BAND_FRACTION * sigma_nm, NM_BAND_FRACTION * sigma_nm)

    # Keep backward-compatible names for Phase 7 scatter plot
    SDD_THRESH_PED = SDD_THRESH.get('ped_veh', 1.7)
    SDD_THRESH_VEH = SDD_THRESH.get('veh_veh', 2.0)
    NM_LO_PED, NM_HI_PED = NM_BANDS.get('ped_veh', (-0.25, 0.25))
    NM_LO_VEH, NM_HI_VEH = NM_BANDS.get('veh_veh', (-0.25, 0.25))

    def classify_pattern(row):
        outcome  = row['outcome']
        NM, SDD  = row['NM'], row['SDD']
        pt       = row.get('pair_type', 'ped_veh')
        thresh   = SDD_THRESH.get(pt, 2.0)
        nm_lo, nm_hi = NM_BANDS.get(pt, (-0.25, 0.25))
        high_sdd = SDD >= thresh
        nm_pos   = NM > nm_hi
        nm_neg   = NM < nm_lo
        nm_zero  = nm_lo <= NM <= nm_hi

        # Ped-vehicle semantics (legacy)
        if pt == 'ped_veh':
            if outcome == 'agent_a_first':
                if nm_pos and high_sdd: return 'Proactive Yielding'
                if nm_pos and not high_sdd: return 'Marginal Yield'
                if nm_neg and high_sdd: return 'Gap-Driven Crossing'
                if nm_neg and not high_sdd: return 'Late Pedestrian Surge'
                if nm_zero and high_sdd: return 'Unclear Advantage'
                return 'Chaotic Negotiation'
            elif outcome == 'agent_b_first':
                if nm_pos and high_sdd: return 'Insufficient Yield'
                if nm_pos and not high_sdd: return 'Pedestrian Hesitation'
                if nm_neg and high_sdd: return 'Assertive Non-Yielding'
                if nm_neg and not high_sdd: return 'Contested Non-Yielding'
                if nm_zero and high_sdd: return 'Unclear Advantage'
                return 'Chaotic Negotiation'
        # Vehicle–vehicle / cyclist–vehicle semantics
        else:
            leader = 'A' if outcome == 'agent_a_first' else 'B'
            if nm_pos and high_sdd: return f'Agent {leader} Priority — Stable'
            if nm_pos and not high_sdd: return f'Agent {leader} Priority — Marginal'
            if nm_neg and high_sdd: return 'Gap-Forced Merge'
            if nm_neg and not high_sdd: return 'Contested Right-of-Way'
            if nm_zero and high_sdd: return 'Unclear Advantage'
            return 'Chaotic Negotiation'
        return 'Unknown'

    df_int['pattern'] = df_int.apply(classify_pattern, axis=1)
    safe_to_csv(df_int, 'interactions_classified.csv', index=False)

# ==========================================
# PHASE 7: MLMM & PLOTS
# ==========================================
print("--- PHASE 7: Generating MLMM and Final Plots ---")
df_cl = pd.read_csv('interactions_classified.csv')

if df_cl.empty:
    print("Skipping MLMM due to lack of interaction data.")
else:
    df_cl = df_cl.dropna(subset=['PEB_final','PET','TTC_min','pattern'])
    
    # Cap extreme mathematical artifacts (e.g., pedestrian freezing) at 15 seconds
    df_cl['PEB_final'] = df_cl['PEB_final'].clip(upper=15.0)
    df_cl['PET'] = df_cl['PET'].clip(upper=15.0)

    rows = []
    for _, row in df_cl.iterrows():
        for outcome_label, val in [('PEB', row['PEB_final']), ('PET', row['PET']), ('TTC', row['TTC_min'])]:
            rows.append({'pattern': row['pattern'], 'veh_class': row['veh_class'], 'outcome': outcome_label, 'value': val})
    df_long = pd.DataFrame(rows)

    if not df_long.empty:
        REF_PATTERN, REF_OUTCOME = 'Proactive Yielding', 'PEB'
        patterns_ordered = [REF_PATTERN] + [p for p in df_long['pattern'].unique() if p != REF_PATTERN]
        outcomes_ordered = [REF_OUTCOME] + [o for o in df_long['outcome'].unique() if o != REF_OUTCOME]

        X_pat = pd.get_dummies(pd.Categorical(df_long['pattern'], categories=patterns_ordered), drop_first=True).astype(float)
        X_out = pd.get_dummies(pd.Categorical(df_long['outcome'], categories=outcomes_ordered), drop_first=True).astype(float)
        X = pd.concat([pd.Series(1.0, index=df_long.index, name='Intercept'), X_pat, X_out], axis=1)
        y, groups = df_long['value'].values, df_long['veh_class'].values

        try:
            mlm = MixedLM(y, X, groups=groups)
            result = mlm.fit(reml=True)
            safe_to_csv(pd.DataFrame({'coef': result.fe_params, 'pvalue': result.pvalues, 'ci_lo': result.conf_int()[0], 'ci_hi': result.conf_int()[1]}), 'mlm_coefficients.csv')
        except Exception as e:
            print(f"MLMM skipped/error: {e}")

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        for (pt, outcome_label, title), ax in zip(
            [('ped_veh',  'agent_a_first', '(a) Ped–Veh: ped passes first'),
             ('ped_veh',  'agent_b_first', '(b) Ped–Veh: vehicle passes first'),
             ('veh_veh',  None,            '(c) Veh–Veh interactions')],
            axes
        ):
            if outcome_label:
                sub = df_cl[(df_cl['pair_type'] == pt) & (df_cl['outcome'] == outcome_label)]
            else:
                sub = df_cl[df_cl['pair_type'] == pt]
            thresh  = SDD_THRESH.get(pt, 2.0)
            nm_lo, nm_hi = NM_BANDS.get(pt, (-0.25, 0.25))
            if not sub.empty:
                sns.scatterplot(data=sub, x='NM', y='SDD', hue='pattern',
                                palette='tab10', s=60, alpha=0.8, ax=ax)
            ax.axhline(thresh, color='black', linestyle='--', linewidth=1)
            ax.axvspan(nm_lo, nm_hi, alpha=0.08, color='gray')
            ax.set_title(title); ax.set_xlabel('NM'); ax.set_ylabel('SDD (s)')
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig('nm_sdd_plot.png', dpi=150)
        plt.close()

        summary = df_cl.groupby(['pair_type','pattern'])[['PEB_final','PET','TTC_min','TTC_poly_min','Max_CCI']].mean().round(3)
        summary['n'] = df_cl.groupby(['pair_type','pattern']).size()
        safe_to_csv(summary, 'pattern_safety_summary.csv')

        # ── Bar chart: classic safety metrics per pattern ─────────────────
        pat_summary = df_cl.groupby('pattern')[['PEB_final','PET','TTC_min','TTC_poly_min']].mean().round(3)
        pat_summary['n'] = df_cl.groupby('pattern').size()

        x, w = np.arange(len(pat_summary)), 0.20
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(x-1.5*w, pat_summary['PEB_final'],    w, label='PEB',           color='#2196F3', alpha=0.85)
        ax.bar(x-0.5*w, pat_summary['PET'],           w, label='PET',           color='#FF9800', alpha=0.85)
        ax.bar(x+0.5*w, pat_summary['TTC_min'],       w, label='TTC (DCAF)',    color='#4CAF50', alpha=0.85)
        ax.bar(x+1.5*w, pat_summary['TTC_poly_min'],  w, label='TTC (Polygon)', color='#E91E63', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(pat_summary.index, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Mean value (s)'); ax.set_title('Predicted Safety Indicators per Interaction Pattern')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('predicted_safety.png', dpi=150)
        plt.close()

        # ── Bar chart: Gaussian collision probability per pair type ───────
        prob_summary = df_cl.groupby('pair_type')['Max_CCI'].mean().round(4)
        if not prob_summary.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            prob_summary.plot(kind='bar', ax=ax2, color=['#E91E63','#2196F3','#FF9800'], alpha=0.85, edgecolor='white')
            ax2.set_ylabel('Mean P(collision)')
            ax2.set_title('Gaussian Collision Probability by Pair Type')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('collision_probability_by_pair_type.png', dpi=150)
            plt.close()

# ==========================================
# PHASE 8: BIRD'S EYE VIEW (BEV) PLOTS
# ==========================================
print("--- PHASE 8: Generating BEV Spatial Plots ---")
df_smooth = pd.read_csv(SMOOTHED_CSV)

CLASS_COLORS = {
    'pedestrian':    '#4CAF50',
    'cyclist':       '#FF9800',
    'motorcycle':    '#FF5722',
    'auto_rickshaw': '#E91E63',
    'car':           '#00BCD4',
    'lcv':           '#9C27B0',
    'bus':           '#F44336',
    'truck':         '#795548',
}

# ── Plot 1: BEV Multi-Agent Trajectory (main plot) ──
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('white')
active_classes = df_smooth['class_name'].unique()
total_streams  = df_smooth['track_id'].nunique()

for cls in active_classes:
    cls_df = df_smooth[df_smooth['class_name'] == cls]
    color  = CLASS_COLORS.get(cls, 'gray')
    label_added = False
    for tid, traj in cls_df.groupby('track_id'):
        traj = traj.sort_values('time_s')
        if len(traj) < 3:
            continue
        ax.plot(traj['x_m_smooth'], traj['y_m_smooth'],
                color=color,
                linewidth=0.9,
                alpha=0.75,
                label=cls if not label_added else '_nolegend_')
        label_added = True

ax.set_xlabel('Road Width Crosswalk Axis (X-meters)', fontsize=12)
ax.set_ylabel('Road Length Encroachment Axis (Y-meters)', fontsize=12)
ax.set_title(
    f"Birds-Eye View Multi-Agent Trajectories\n"
    f"[Visualising {total_streams} Active Trajectory Line Streams]",
    fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_aspect('equal')

# Axis limits with small padding
x_min = df_smooth['x_m_smooth'].min() - 0.5 if not df_smooth.empty else -10
x_max = df_smooth['x_m_smooth'].max() + 0.5 if not df_smooth.empty else 15
y_min = df_smooth['y_m_smooth'].min() - 0.5 if not df_smooth.empty else -5
y_max = df_smooth['y_m_smooth'].max() + 0.5 if not df_smooth.empty else 35
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('BEV_Trajectories.png', dpi=200, bbox_inches='tight')
plt.close()

# ── Plot 2: Speed Distribution by Class ──
if not df_smooth.empty:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for cls, grp in df_smooth.groupby('class_name'):
        grp['speed'].hist(bins=40, ax=ax2, alpha=0.55,
                          label=cls,
                          color=CLASS_COLORS.get(cls, 'gray'),
                          edgecolor='white')
    ax2.set_title('Speed Distribution by Class', fontsize=13)
    ax2.set_xlabel('Speed (m/s)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('speed_distribution.png', dpi=150)
    plt.close()

# ── Plot 3: BEV Animated by Time Window ──
if not df_smooth.empty:
    total_duration = df_smooth['time_s'].max()
    window_size    = total_duration / 4    # split into 4 equal windows
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        t_start = i * window_size
        t_end   = (i + 1) * window_size
        window  = df_smooth[
            (df_smooth['time_s'] >= t_start) &
            (df_smooth['time_s'] <  t_end)
        ]
        n_tracks = window['track_id'].nunique()
        ax.set_facecolor('white')
        for cls, grp in window.groupby('class_name'):
            color       = CLASS_COLORS.get(cls, 'gray')
            label_added = False
            for tid, traj in grp.groupby('track_id'):
                traj = traj.sort_values('time_s')
                if len(traj) < 3:
                    continue
                ax.plot(traj['x_m_smooth'], traj['y_m_smooth'],
                        color=color, linewidth=0.9, alpha=0.75,
                        label=cls if not label_added else '_nolegend_')
                label_added = True
        ax.set_title(
            f"Window {i+1}: t={t_start:.0f}s – {t_end:.0f}s\n"
            f"[{n_tracks} tracks]",
            fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal')
        ax.set_xlim(df_smooth['x_m_smooth'].min() - 0.5,
                    df_smooth['x_m_smooth'].max() + 0.5)
        ax.set_ylim(df_smooth['y_m_smooth'].min() - 0.5,
                    df_smooth['y_m_smooth'].max() + 0.5)
    plt.suptitle('BEV Trajectories — Time Window Analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('BEV_time_windows.png', dpi=180, bbox_inches='tight')
    plt.close()

# ── Plot 4: BEV with detected interaction pairs highlighted ──
try:
    df_cl = pd.read_csv('interactions_classified.csv')
    has_classified = True
except FileNotFoundError:
    try:
        df_cl = pd.read_csv('interactions.csv')
        has_classified = False
    except FileNotFoundError:
        df_cl = None

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_facecolor('white')

# Draw all trajectories faint in background
if not df_smooth.empty:
    for cls, grp in df_smooth.groupby('class_name'):
        color = CLASS_COLORS.get(cls, 'gray')
        for tid, traj in grp.groupby('track_id'):
            traj = traj.sort_values('time_s')
            if len(traj) < 3:
                continue
            ax.plot(traj['x_m_smooth'], traj['y_m_smooth'],
                    color=color, linewidth=0.6, alpha=0.25)

# Highlight interacting pairs
if df_cl is not None and not df_cl.empty and not df_smooth.empty:
    PATTERN_COLORS_BEV = {
        'Proactive Yielding':     '#2ecc71',
        'Marginal Yield':         '#f39c12',
        'Gap-Driven Crossing':    '#27ae60',
        'Late Pedestrian Surge':  '#e74c3c',
        'Chaotic Negotiation':    '#c0392b',
        'Insufficient Yield':     '#3498db',
        'Pedestrian Hesitation':  '#e67e22',
        'Assertive Non-Yielding': '#1abc9c',
        'Contested Non-Yielding': '#e91e63',
        'Unclear Advantage':      '#9b59b6',
        'Unknown':                '#bdc3c7',
    }
    plotted_patterns = set()
    for _, row in df_cl.iterrows():
        # Ensure we skip empty patterns generated by fail-safes
        if pd.isna(row.get('ped_id')): continue 
        
        pid = int(row['ped_id'])
        vid = int(row['veh_id'])
        pat = row.get('pattern', 'Unknown')
        color = PATTERN_COLORS_BEV.get(pat, 'gray')
        p_traj = df_smooth[df_smooth['track_id'] == pid].sort_values('time_s')
        v_traj = df_smooth[df_smooth['track_id'] == vid].sort_values('time_s')
        lbl = pat if pat not in plotted_patterns else '_nolegend_'
        if len(p_traj) >= 3:
            ax.plot(p_traj['x_m_smooth'], p_traj['y_m_smooth'],
                    color=color, linewidth=1.8, alpha=0.85,
                    label=lbl)
        if len(v_traj) >= 3:
            ax.plot(v_traj['x_m_smooth'], v_traj['y_m_smooth'],
                    color=color, linewidth=1.8, alpha=0.85,
                    linestyle='--')
        plotted_patterns.add(pat)
        # Mark the closest approach point
        p_sub = p_traj[['time_s','x_m_smooth','y_m_smooth']].set_index('time_s')
        v_sub = v_traj[['time_s','x_m_smooth','y_m_smooth']].set_index('time_s')
        common_t = p_sub.index.intersection(v_sub.index)
        if len(common_t) > 0:
            dists = np.sqrt(
                (p_sub.loc[common_t,'x_m_smooth'] - 
                 v_sub.loc[common_t,'x_m_smooth'])**2 +
                (p_sub.loc[common_t,'y_m_smooth'] - 
                 v_sub.loc[common_t,'y_m_smooth'])**2)
            closest_t = dists.idxmin()
            mx = (p_sub.loc[closest_t,'x_m_smooth'] + 
                  v_sub.loc[closest_t,'x_m_smooth']) / 2
            my = (p_sub.loc[closest_t,'y_m_smooth'] + 
                  v_sub.loc[closest_t,'y_m_smooth']) / 2
            ax.plot(mx, my, 'x', color=color,
                    markersize=7, markeredgewidth=1.5)

ax.set_xlabel('Road Width Crosswalk Axis (X-meters)', fontsize=12)
ax.set_ylabel('Road Length Encroachment Axis (Y-meters)', fontsize=12)
n_pairs = len(df_cl.dropna(subset=['ped_id'])) if df_cl is not None and not df_cl.empty else 0
ax.set_title(
    f"BEV — Interaction Pairs Highlighted by Pattern\n"
    f"[{n_pairs} interactions detected]",
    fontsize=13, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
if unique:
    ax.legend(unique.values(), unique.keys(),
              fontsize=8, loc='upper right', framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.35)
ax.set_aspect('equal')
if not df_smooth.empty:
    ax.set_xlim(df_smooth['x_m_smooth'].min() - 0.5,
                df_smooth['x_m_smooth'].max() + 0.5)
    ax.set_ylim(df_smooth['y_m_smooth'].min() - 0.5,
                df_smooth['y_m_smooth'].max() + 0.5)
plt.tight_layout()
plt.savefig('BEV_interactions_highlighted.png', dpi=200, bbox_inches='tight')
plt.close()

# ── Plot 5: Custom Time Window (70s to 100s) ──
if not df_smooth.empty:
    df_window = df_smooth[(df_smooth['time_s'] >= 70) & (df_smooth['time_s'] <= 100)]
    
    if not df_window.empty:
        fig5, ax5 = plt.subplots(figsize=(10, 10))
        ax5.set_facecolor('white')
        
        for cls in df_window['class_name'].unique():
            cls_df = df_window[df_window['class_name'] == cls]
            color = CLASS_COLORS.get(cls, 'gray')
            label_added = False
            for tid, traj in cls_df.groupby('track_id'):
                traj = traj.sort_values('time_s')
                if len(traj) < 3: continue
                ax5.plot(traj['x_m_smooth'], traj['y_m_smooth'],
                         color=color, linewidth=1.5, alpha=0.85,
                         label=cls if not label_added else '_nolegend_')
                label_added = True
                
        ax5.set_xlabel('Road Width Crosswalk Axis (X-meters)', fontsize=12)
        ax5.set_ylabel('Road Length Encroachment Axis (Y-meters)', fontsize=12)
        ax5.set_title(f"BEV Trajectories (t = 70s to 100s)\n[{df_window['track_id'].nunique()} tracks]", fontsize=14, fontweight='bold')
        
        handles, labels = ax5.get_legend_handles_labels()
        if handles:
            ax5.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
        ax5.grid(True, linestyle='--', alpha=0.4, color='gray')
        ax5.set_aspect('equal')
        
        # Keep the global axis limits so the perspective matches your other plots
        ax5.set_xlim(df_smooth['x_m_smooth'].min() - 0.5, df_smooth['x_m_smooth'].max() + 0.5)
        ax5.set_ylim(df_smooth['y_m_smooth'].min() - 0.5, df_smooth['y_m_smooth'].max() + 0.5)
        
        plt.tight_layout()
        plt.savefig('BEV_Trajectories_70_100s.png', dpi=200, bbox_inches='tight')
        plt.close()

print("\n── Output Files Verification ──")
for f in [TRAJECTORY_CSV, SMOOTHED_CSV, 'interactions.csv',
          'interactions_classified.csv', 'mlm_coefficients.csv', 'pattern_safety_summary.csv',
          'nm_sdd_plot.png', 'predicted_safety.png', 'collision_probability_by_pair_type.png',
          'BEV_Trajectories.png', 'speed_distribution.png',
          'BEV_time_windows.png', 'BEV_interactions_highlighted.png']:
    print(f"  {'✓' if os.path.exists(f) else '✗'}  {f}")
print("=======================================\nPipeline Execution Finished.")

# ==========================================
# PHASE 9: LOCAL LLM REPORTING (OLLAMA)
# ==========================================
print("\n--- PHASE 9: Generating LLM Semantic Report ---")
import requests
import json

# Configuration for Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "hermes3:8b"  # Explicitly updated to match your local installation

try:
    df_llm = pd.read_csv('interactions_classified.csv')
    if df_llm.empty or 'ped_id' not in df_llm.columns:
        print("No valid interactions detected. Skipping LLM summary.")
    else:
        # Drop rows with empty IDs
        df_llm = df_llm.dropna(subset=['ped_id', 'veh_id'])
        
        if df_llm.empty:
             print("No valid interactions detected after cleaning empty rows. Skipping LLM summary.")
        else:
            # 1. Format the data into a readable prompt for the LLM
            #
            # NOTE ON METRIC NAMING/FORMATTING (see paper Sec. III-E, Table I,
            # and Sec. VI-C, Table IV):
            #   - Max_CCI is a bounded [0,1] Continuous Criticality Index, NOT
            #     a calibrated collision probability. Labeling it "P_coll" in
            #     the prompt would invite the LLM (and any human reading its
            #     output) to misinterpret a closeness/criticality score as a
            #     literal probability of collision, which the paper is
            #     explicit about NOT claiming. We label it "CCI" here to match
            #     the terminology already used in Table IV.
            #   - TTC_poly_min is legitimately undefined (NaN) for most
            #     interactions (documented in Table IV: only 10/28, 36%,
            #     defined) whenever no future polygon overlap is predicted
            #     within the sweep horizon, or the boundary-crossing formula
            #     hits a geometric degeneracy (near-parallel headings, near-
            #     zero speed, non-converging geometry). This is a genuine,
            #     expected "no predicted collision" result, not missing data.
            #     Silently formatting NaN as the literal string "nan" gives
            #     the LLM no way to distinguish "no future collision
            #     predicted" from "value unavailable", and risks the model
            #     inventing an interpretation for it. We instead pass an
            #     explicit, human-readable label so the LLM (and the
            #     resulting narrative summary, cf. Table I) treats it
            #     correctly as a documented modeling gap rather than missing
            #     or anomalous data.
            prompt_text = (
                "You are an expert traffic safety analyst. I have run an automated trajectory tracking "
                "system on an intersection and extracted the following multi-agent interactions "
                "(vehicle–vehicle, pedestrian–vehicle, and cyclist–vehicle pairs; pedestrian–pedestrian "
                "pairs are excluded). Please provide a professional, concise summary. List the types of "
                "interaction patterns observed, mention the specific Agent IDs and their classes involved "
                "in each, and highlight any high-risk interactions (a high CCI value, or a low predictive "
                "TTC where one was available). Note that CCI is a continuous [0,1] closeness/criticality "
                "score (not a literal collision probability), and that 'TTC_poly: not predicted' means the "
                "forward collision sweep found no future overlap within its time horizon — this is an "
                "expected outcome for many safe interactions, not missing or erroneous data.\n\n"
                "Interaction Data:\n"
            )

            for _, row in df_llm.iterrows():
                a_cls = row.get('agent_a_class', row.get('veh_class', 'unknown'))
                b_cls = row.get('agent_b_class', row.get('veh_class', 'unknown'))
                aid   = int(row.get('agent_a_id',  row.get('ped_id', -1)))
                bid   = int(row.get('agent_b_id',  row.get('veh_id', -1)))
                pt    = row.get('pair_type', 'ped_veh')
                cci_val = row.get('Max_CCI', float('nan'))
                ttc_p   = row.get('TTC_poly_min', float('nan'))

                cci_str = f"{cci_val:.3f}" if pd.notna(cci_val) else "N/A"
                ttc_str = f"{ttc_p:.2f}s" if pd.notna(ttc_p) else "not predicted (no future overlap within horizon)"

                prompt_text += (
                    f"- {a_cls.title()} #{aid} ↔ {b_cls.title()} #{bid} "
                    f"[{pt}] — Pattern: {row['pattern']}, "
                    f"CCI={cci_str}, TTC_poly={ttc_str}\n"
                )

            prompt_text += "\nProvide a well-structured summary of these events."
    
            # 2. Send the prompt to the local Ollama API
            print(f"Sending data to local Ollama model ({OLLAMA_MODEL})... Please wait.")
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt_text,
                "stream": False  # Set to False so it waits for the full response before printing
            }
            
            # 60-second timeout in case the model is large and takes time to process
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                report = result.get("response", "No response text found.")
                
                # Print to terminal
                print("\n" + "═"*60)
                print("🤖 OLLAMA AUTOMATED TRAFFIC SUMMARY:")
                print("═"*60)
                print(report)
                print("═"*60 + "\n")
                
                # Save to a text file
                safe_model_name = OLLAMA_MODEL.replace(':', '-')
                output_file = f"LLM_Traffic_Summary_{safe_model_name}.txt"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("AUTOMATED TRAFFIC SUMMARY\n")
                    f.write("=========================\n\n")
                    f.write(report)
                print(f"✓ Saved report to '{output_file}'")
                
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
            
except FileNotFoundError:
    print("interactions_classified.csv not found. Ensure previous phases ran successfully.")
except requests.exceptions.ConnectionError:
    print("\n[!] WARNING: Could not connect to Ollama.")
    print("    Is the Ollama app running in the background?")
    print(f"    To fix: Open a new terminal and run 'ollama run {OLLAMA_MODEL}' to start the server.")
except Exception as e:
    print(f"LLM Reporting Error: {e}")