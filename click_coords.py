import cv2

VIDEO_PATH = '../newest_video.avi'
FRAME_PATH = 'newest_video_frame1.jpg'
OUT_TXT = 'clicked_points.txt'

window_name = "Click to get pixel coordinates - Press 'q' to quit"


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}]")
        with open(OUT_TXT, 'a') as f:
            f.write(f"{x}, {y}\n")
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img, f"{x},{y}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, f"{x},{y}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(window_name, img)


cap = cv2.VideoCapture(VIDEO_PATH)
ret, img = cap.read()
cap.release()
if not ret:
    img = cv2.imread(FRAME_PATH)
if img is None:
    print("ERROR: could not load video or frame image.")
    exit()

open(OUT_TXT, 'w').close()  # clear file at start of each run

print("Click anywhere on the image to print + save its pixel coordinates.")
print(f"Coordinates are appended to {OUT_TXT} as you click.")
print("Press 'q' to quit.")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1400, 820)
cv2.setMouseCallback(window_name, click_event)
cv2.imshow(window_name, img)

while True:
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Done. All clicked points saved in {OUT_TXT}")
