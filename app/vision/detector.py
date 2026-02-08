from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def detect(self, frame_bgr):
        # Note: returns ultralytics Results list
        return self.model(frame_bgr, verbose=False)
