"""
pick_roi.py — click the detection ROI polygon on a real video frame.

Run this on your own machine (it opens a window; it will not work over a
headless connection).

    python pick_roi.py                 # uses frame 2000 of the source video
    python pick_roi.py 4500            # use a different frame number

Controls
    left click    add a point
    u / backspace undo last point
    r             reset (clear all points)
    s / enter     save and exit
    q / esc       quit without saving

Saves `roi_polygon.npy` next to this script. `run.py` picks that file up
automatically on the next run with RUN_YOLO_TRACKING = True — no code edit
needed. The polygon is also printed so you can paste it into run.py if you
prefer to hard-code it.

Guidance for a good ROI: include the full carriageway and any footpath where
pedestrians wait at the kerb, but EXCLUDE the far background (the service
road behind the roundabout, distant parked traffic, foliage). Detections near
the horizon are projected by the homography to very large, unreliable ground
distances, which is what produces the stretched artefacts in the bird's-eye
view.
"""
import os
import sys
import cv2
import numpy as np

FRAME_NO = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def find_input(name):
    for base in ('.', '..', '../..'):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return name


VIDEO = find_input('newest_video_busy_5min.mp4')
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"could not open video: {VIDEO}")
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NO)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"could not read frame {FRAME_NO}")

H, W = frame.shape[:2]
print(f"frame {FRAME_NO} loaded: {W}x{H}")

# existing polygon (if any) shown faintly for reference
prev = None
if os.path.exists('roi_polygon.npy'):
    prev = np.load('roi_polygon.npy')
    print(f"existing roi_polygon.npy has {len(prev)} points (shown in grey)")

pts = []
WIN = 'Select ROI  |  click=add  u=undo  r=reset  s=save  q=quit'
# fit the window to the screen while keeping full-resolution coordinates
SCALE = min(1.0, 1600.0 / W)


def redraw():
    img = frame.copy()
    if prev is not None and len(prev) >= 3:
        cv2.polylines(img, [prev.astype(np.int32)], True, (160, 160, 160), 2)
    if len(pts) >= 3:
        ov = img.copy()
        cv2.fillPoly(ov, [np.array(pts, np.int32)], (0, 255, 0))
        img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 0, 255), 3)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, str(i), (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    bar = f"{len(pts)} points   click=add  u=undo  r=reset  s=save  q=quit"
    cv2.rectangle(img, (0, 0), (W, 44), (30, 30, 30), -1)
    cv2.putText(img, bar, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)
    disp = cv2.resize(img, (int(W * SCALE), int(H * SCALE))) if SCALE < 1 else img
    cv2.imshow(WIN, disp)


def on_mouse(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((int(x / SCALE), int(y / SCALE)))   # store full-res coords
        redraw()


cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WIN, on_mouse)
redraw()

while True:
    k = cv2.waitKey(20) & 0xFF
    if k in (ord('u'), 8) and pts:
        pts.pop(); redraw()
    elif k == ord('r'):
        pts.clear(); redraw()
    elif k in (ord('s'), 13):
        if len(pts) < 3:
            print("need at least 3 points to save")
            continue
        arr = np.array(pts, dtype=np.int32)
        np.save('roi_polygon.npy', arr)
        cv2.imwrite('roi_preview.jpg', (lambda i: i)(
            cv2.addWeighted(
                cv2.fillPoly(frame.copy(), [arr], (0, 255, 0)), 0.25,
                frame, 0.75, 0)))
        print(f"\nsaved roi_polygon.npy with {len(arr)} points")
        print("run.py will use it automatically on the next tracking run.\n")
        print("To hard-code it instead, paste this into run.py:\n")
        print("ROI_POLYGON = np.array([")
        for x, y in arr:
            print(f"    [{x}, {y}],")
        print("])")
        break
    elif k in (ord('q'), 27):
        print("quit without saving")
        break

cv2.destroyAllWindows()
