"""
detector.py
-----------
Unified detection interface.
Returns:
    list of Detection(x1, y1, x2, y2, cls_name, confidence)
"""

import cv2
import numpy as np
from collections import namedtuple

from .yolo_loader import YOLOLoader

Detection = namedtuple("Detection", ["x1", "y1", "x2", "y2", "cls", "conf"])


class Detector:
    def __init__(self, model_path_override=None, img_size_override=None):
        # Load YOLO
        self.yolo = YOLOLoader(model_path_override=model_path_override,
                               img_size_override=img_size_override)

        self.use_yolo = self.yolo.is_loaded()
        if self.use_yolo:
            print("[Detector] Using YOLO model for detection.")
        else:
            print("[Detector] YOLO unavailable -> Using fallback motion detector.")

        # Fallback motion detector
        self.bgsub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False)

    def detect(self, frame):
        if self.use_yolo:
            return self._detect_yolo(frame)
        else:
            return self._detect_motion(frame)

    # -------------------
    # YOLO DETECTION
    # -------------------
    def _detect_yolo(self, frame):
        model = self.yolo.get_model()
        img_size = self.yolo.get_img_size()
        conf_thres = self.yolo.get_conf_thres()
        classes = self.yolo.get_classes()

        results = model(frame, imgsz=img_size, conf=conf_thres, verbose=False)
        detections = []

        if len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())

            if cls_idx >= len(classes):
                cls_name = "other"
            else:
                cls_name = classes[cls_idx]

            x1, y1, x2, y2 = map(int, xyxy)
            detections.append(Detection(x1, y1, x2, y2, cls_name, conf))

        return detections

    # -------------------
    # MOTION-BASED DETECTION FALLBACK
    # -------------------
    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.bgsub.apply(gray)

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Basic heuristic classification
            cls_name = "other"
            if w > h * 1.2:
                cls_name = "car"
            elif h > w * 1.2:
                cls_name = "motorcycle"

            conf = min(1.0, area / 20000.0)
            detections.append(Detection(x, y, x + w, y + h, cls_name, conf))

        return detections
