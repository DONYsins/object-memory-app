import cv2
import time
import requests

IPCAM_URL = "http://192.168.8.183:8080/video"
API_INGEST = "http://127.0.0.1:8000/ingest_frame"

FPS_SAMPLE = 5
CAMERA_FPS = 30
skip_interval = max(1, CAMERA_FPS // FPS_SAMPLE)

cap = cv2.VideoCapture(IPCAM_URL)
frame_count = 0

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_interval != 0:
        cv2.imshow("Ingest Preview (no inference here)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # learner note: send JPEG to backend
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        files = {"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")}
        try:
            r = requests.post(API_INGEST, files=files, timeout=2)
            # optional: print(r.json())
        except Exception as e:
            print("Ingest error:", e)

    cv2.imshow("Ingest Preview (no inference here)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
