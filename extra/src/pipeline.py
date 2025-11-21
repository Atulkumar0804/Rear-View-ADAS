"""
src/pipeline.py

Self-contained pipeline implementation for:
 - detection (YOLO if available else motion-based detector)
 - simple centroid tracking
 - ground projection and depth-from-bbox
 - per-track Kalman smoothing (simple or using filterpy)
 - constant-velocity trajectory prediction
 - TTC and warning decision

This file is intentionally designed as a working prototype. Replace
or extend the internal classes with your production modules when ready.
"""

import cv2
import os
import time
import yaml
import math
import numpy as np
from collections import deque, defaultdict, namedtuple

try:  # optional RealSense integration
    from src.utils.realsense import RealSenseStream
except Exception:  # pragma: no cover
    RealSenseStream = None

# Try optional libs
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except Exception:
    KalmanFilter = None
    FILTERPY_AVAILABLE = False

# ---------- simple utilities ----------
def load_yaml(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return default if default is not None else {}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Named structure for detection and track
Detection = namedtuple("Detection", ["x1", "y1", "x2", "y2", "cls", "conf"])
Track = namedtuple("Track", ["track_id", "bbox", "cls", "confidence", "age", "last_seen"])

# ---------- Default configurations ----------
DEFAULT_CAMERA_CFG = {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 640.0,
    "cy": 360.0,
    "mounting_height": 1.0,   # meters
    "pitch_deg": 0.0
}

DEFAULT_MODEL_CFG = {
    "model_path": "models/yolo/yolov8n_RearView.pt",
    "img_size": 640,
    "conf_thres": 0.35,
    "classes": ["car", "truck", "bus", "motorcycle", "other"]
}

DEFAULT_WARNING_CFG = {
    "critical_ttc": 1.5,
    "warn_ttc": 4.0,
    "critical_distance": 3.0,
    "warning_distance": 6.0,
    "lateral_overlap_threshold": 0.5
}

CLASS_HEIGHTS = {
    # Vehicles
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.0,
    "motorcycle": 1.1,
    "bicycle": 1.2,
    "train": 3.5,
    
    # Pedestrians & Animals
    "person": 1.7,
    "dog": 0.6,
    "cat": 0.3,
    "horse": 1.6,
    "cow": 1.5,
    "sheep": 1.0,
    "bird": 0.3,
    "elephant": 3.0,
    "bear": 1.5,
    "zebra": 1.5,
    "giraffe": 4.5,
    
    # Infrastructure
    "traffic_light": 0.5,
    "stop_sign": 0.8,
    "fire_hydrant": 0.7,
    "parking_meter": 1.5,
    
    # Default
    "other": 1.5
}

SEGMENTATION_COLORS = {
    # Vehicles
    "car": (0, 0, 255),
    "truck": (0, 100, 255),
    "bus": (0, 150, 255),
    "motorcycle": (0, 255, 255),
    "bicycle": (255, 255, 0),
    "train": (128, 0, 255),
    
    # Pedestrians & Animals
    "person": (255, 0, 0),
    "dog": (200, 100, 50),
    "cat": (150, 100, 50),
    "horse": (180, 120, 60),
    "cow": (160, 110, 55),
    "sheep": (140, 100, 50),
    "bird": (100, 200, 100),
    "elephant": (180, 140, 100),
    "bear": (120, 80, 40),
    "zebra": (200, 200, 200),
    "giraffe": (220, 180, 100),
    
    # Infrastructure
    "traffic_light": (0, 255, 0),
    "stop_sign": (0, 0, 200),
    "fire_hydrant": (0, 128, 255),
    "parking_meter": (128, 128, 128),
    
    # Default
    "other": (100, 255, 255),
    "lane": (255, 255, 0)
}


# ---------- Detector ----------
class Detector:
    """
    Tries to use YOLO if available and model exists; otherwise falls back to
    a simple motion-based detector (background subtraction).
    Returns list of Detection(x1,y1,x2,y2, cls, conf).
    """

    def __init__(self, config=None, model_path_override=None, img_size_override=None):
        cfg = load_yaml("config/model_config.yaml", DEFAULT_MODEL_CFG)
        if config:
            cfg.update(config)
        if model_path_override:
            cfg["model_path"] = model_path_override
        if img_size_override:
            cfg["img_size"] = img_size_override

        self.cfg = cfg
        self.use_yolo = False
        self.min_confidence = float(cfg.get("min_confidence", cfg.get("conf_thres", 0.35)))
        self.min_bbox_area = int(cfg.get("min_bbox_area", 1500))
        self.max_bbox_area = int(cfg.get("max_bbox_area", 999999))
        self.min_aspect_ratio = float(cfg.get("min_aspect_ratio", 0.2))
        self.max_aspect_ratio = float(cfg.get("max_aspect_ratio", 5.0))
        self.border_padding = int(cfg.get("border_padding", 0))
        self.motion_min_area = int(cfg.get("motion_min_area", 800))
        kernel_size = int(cfg.get("motion_morph_kernel", 5))
        if kernel_size % 2 == 0:
            kernel_size += 1  # need odd kernel for morphology
        self.motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        model_path = cfg.get("model_path")
        if ULTRALYTICS_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                print(f"[Detector] Loading YOLO model from {model_path}")
                self.model = YOLO(model_path)
                self.imgsz = cfg.get("img_size", 640)
                self.conf_thres = cfg.get("conf_thres", 0.35)
                self.class_names = cfg.get("classes", DEFAULT_MODEL_CFG["classes"])
                self.use_yolo = True
            except Exception as e:
                print("[Detector] YOLO init failed:", e)
                self.use_yolo = False

        if not self.use_yolo:
            print("[Detector] Using fallback motion detector (bgsegm).")
            self.bgsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    def detect(self, frame):
        h, w = frame.shape[:2]
        results = []
        if self.use_yolo:
            # ultralytics returns a results list
            out = self.model(frame, imgsz=self.imgsz, conf=self.conf_thres, verbose=False)
            # parse detections
            if len(out) > 0:
                boxes = out[0].boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()  # x1,y1,x2,y2
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else "other"
                        x1, y1, x2, y2 = map(int, xyxy)
                        if conf < self.min_confidence:
                            continue
                        results.append(Detection(x1, y1, x2, y2, cls_name, conf))
        else:
            # simple motion-based detection to get candidate boxes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = self.bgsub.apply(gray)
            # some morphology to clean up
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.motion_kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.motion_kernel)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.motion_min_area:
                    continue
                x,y,wc,hc = cv2.boundingRect(cnt)
                # classify by aspect/size heuristics
                cls = "other"
                if wc > 1.2*hc and wc*hc > 5000:
                    cls = "car"
                elif hc > 1.2*wc and hc > 120:
                    cls = "motorcycle"
                conf = clamp(min(1.0, area/20000.0), 0.2, 0.99)
                if conf < self.min_confidence:
                    continue
                results.append(Detection(x, y, x+wc, y+hc, cls, conf))
        return self._filter_detections(results, frame.shape)

    def _filter_detections(self, detections, frame_shape):
        if not detections:
            return []
        H, W = frame_shape[:2]
        filtered = []
        for det in detections:
            x1 = int(clamp(det.x1, 0, W-1))
            y1 = int(clamp(det.y1, 0, H-1))
            x2 = int(clamp(det.x2, 0, W-1))
            y2 = int(clamp(det.y2, 0, H-1))
            if x2 <= x1 or y2 <= y1:
                continue
            w = x2 - x1
            h = y2 - y1
            area = w * h
            if area < self.min_bbox_area or area > self.max_bbox_area:
                continue
            aspect = w / float(h)
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue
            if self.border_padding > 0:
                if (
                    x1 < self.border_padding
                    or y1 < self.border_padding
                    or x2 > W - self.border_padding
                    or y2 > H - self.border_padding
                ):
                    continue
            filtered.append(Detection(x1, y1, x2, y2, det.cls, det.conf))
        return filtered


# ---------- Simple centroid tracker ----------
class CentroidTracker:
    """
    Very simple tracker that assigns persistent IDs based on centroid distance.
    Meant as a lightweight placeholder for ByteTrack/DeepSORT.
    """
    def __init__(self, max_distance=80, max_lost=10, min_confidence=0.0, min_confirmed_age=1):
        self.next_id = 1
        self.tracks = {}  # id -> dict(bbox, centroid, lost, age, cls, conf)
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.min_confidence = float(min_confidence)
        self.min_confirmed_age = int(min_confirmed_age)

    @staticmethod
    def _centroid_from_bbox(bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)//2, (y1+y2)//2)

    @staticmethod
    def bbox_area(bbox):
        x1,y1,x2,y2 = bbox
        return max(0, x2-x1) * max(0, y2-y1)

    def update(self, detections):
        # Build list of current centroids
        det_centroids = []
        det_bboxes = []
        det_classes = []
        det_confs = []
        for d in detections:
            det_bboxes.append((d.x1, d.y1, d.x2, d.y2))
            det_centroids.append(self._centroid_from_bbox((d.x1,d.y1,d.x2,d.y2)))
            det_classes.append(d.cls)
            det_confs.append(d.conf)

        assigned = set()
        # match existing tracks to detections by nearest centroid
        if len(self.tracks) == 0:
            for i, bbox in enumerate(det_bboxes):
                self.tracks[self.next_id] = {
                    "bbox": bbox,
                    "centroid": det_centroids[i],
                    "lost": 0,
                    "age": 1,
                    "cls": det_classes[i],
                    "conf": det_confs[i],
                    "last_seen": time.time()
                }
                self.next_id += 1
        else:
            track_ids = list(self.tracks.keys())
            track_centroids = [self.tracks[t]["centroid"] for t in track_ids]
            if len(det_centroids) > 0 and len(track_centroids) > 0:
                D = np.linalg.norm(np.array(track_centroids)[:,None,:] - np.array(det_centroids)[None,:,:], axis=2)
                # greedy matching: for speed we do simple minimums
                for _ in range(min(D.shape[0], D.shape[1])):
                    t_idx, d_idx = np.unravel_index(np.argmin(D), D.shape)
                    dist = D[t_idx, d_idx]
                    if dist > self.max_distance:
                        break
                    tid = track_ids[t_idx]
                    if tid in assigned:
                        D[t_idx, :] = 1e9
                        continue
                    # assign
                    self.tracks[tid]["bbox"] = det_bboxes[d_idx]
                    self.tracks[tid]["centroid"] = det_centroids[d_idx]
                    self.tracks[tid]["lost"] = 0
                    self.tracks[tid]["age"] += 1
                    self.tracks[tid]["cls"] = det_classes[d_idx]
                    self.tracks[tid]["conf"] = det_confs[d_idx]
                    self.tracks[tid]["last_seen"] = time.time()
                    assigned.add(d_idx)
                    # invalidate this row & col
                    D[t_idx, :] = 1e9
                    D[:, d_idx] = 1e9
            # unmatched detections -> create new tracks
            for i in range(len(det_bboxes)):
                if i in assigned:
                    continue
                self.tracks[self.next_id] = {
                    "bbox": det_bboxes[i],
                    "centroid": det_centroids[i],
                    "lost": 0,
                    "age": 1,
                    "cls": det_classes[i],
                    "conf": det_confs[i],
                    "last_seen": time.time()
                }
                self.next_id += 1
            # increment lost for unmatched tracks
            to_delete = []
            for tid, t in list(self.tracks.items()):
                # find a match? if last seen was recent we keep; else increment lost
                if time.time() - t["last_seen"] > 0.05:
                    t["lost"] += 1
                if t["lost"] > self.max_lost:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]

        # return list of Track objects
        out = []
        for tid, t in self.tracks.items():
            if t["conf"] < self.min_confidence:
                continue
            if t["age"] < self.min_confirmed_age:
                continue
            out.append(Track(tid, t["bbox"], t["cls"], t["conf"], t["age"], t["last_seen"]))
        return out


# ---------- Ground projection & depth estimation ----------
class GroundProjector:
    """
    Project bottom-center pixel to ground coordinates (X lateral, Z forward)
    using camera intrinsics and mounting height (assumes flat ground).
    """

    def __init__(self, cfg=None):
        self.cfg = load_yaml("config/camera_config.yaml", DEFAULT_CAMERA_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.fx = float(self.cfg.get("fx", 1000.0))
        self.fy = float(self.cfg.get("fy", 1000.0))
        self.cx = float(self.cfg.get("cx", 640.0))
        self.cy = float(self.cfg.get("cy", 360.0))
        self.h = float(self.cfg.get("mounting_height", 1.0))
        # pitch is not fully used in this simple model; it's recommended to compensate pitch via IMU or extrinsics
        self.pitch = math.radians(float(self.cfg.get("pitch_deg", 0.0)))

    def bottom_to_ground(self, bbox):
        # bbox is (x1,y1,x2,y2)
        x1,y1,x2,y2 = bbox
        u = (x1 + x2) / 2.0
        v = y2  # bottom pixel
        # simple pinhole ground-plane formula:
        denom = (v - self.cy)
        if denom == 0:
            denom = 1e-6
        Z = (self.fy * (-self.h)) / denom   # careful with sign
        X = (u - self.cx) * Z / self.fx
        # correct sign: if cy below bottom, Z negative; but typically v > cy and h>0 gives negative Z -> take abs
        Z = abs(Z)
        return float(X), float(Z)


class DepthEstimator:
    """
    Depth from bbox-height: Z = f * H_real / h_img
    """
    def __init__(self, cfg=None):
        cam_cfg = load_yaml("config/camera_config.yaml", DEFAULT_CAMERA_CFG)
        self.f = float(cam_cfg.get("fy", 1000.0))
        self.class_heights = CLASS_HEIGHTS

    def bbox_depth(self, bbox, cls):
        x1,y1,x2,y2 = bbox
        h_img = max(2, (y2 - y1))
        H_real = self.class_heights.get(cls, 1.5)
        Z = (self.f * H_real) / float(h_img)
        return float(Z)


# ---------- Kalman wrapper ----------
class SimpleKalman:
    """
    4-D constant velocity Kalman for [X, Z, vx, vz].
    If filterpy is available, it uses it; otherwise uses a very small
    custom discrete-time KF implementation (approx).
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        self.filters = {}  # track_id -> filter state dict
        self.process_q = 1.0

    def create_filter(self, track_id, init_x, init_z):
        if FILTERPY_AVAILABLE:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.F = np.array([[1,0,self.dt,0],
                             [0,1,0,self.dt],
                             [0,0,1,0],
                             [0,0,0,1]], dtype=float)
            kf.H = np.array([[1,0,0,0],
                             [0,1,0,0]], dtype=float)
            kf.x = np.array([init_x, init_z, 0.0, 0.0], dtype=float)
            kf.P *= 1.0
            q = self.process_q
            kf.Q = q * np.eye(4)
            kf.R = np.diag([0.5, 0.8])  # measurement noise
            self.filters[track_id] = kf
        else:
            # store a simple dict with mean and cov (very rough)
            self.filters[track_id] = {
                "x": float(init_x),
                "z": float(init_z),
                "vx": 0.0,
                "vz": 0.0,
                "last_t": time.time()
            }

    def predict_and_update(self, track_id, meas_x, meas_z, dt=None):
        if track_id not in self.filters:
            self.create_filter(track_id, meas_x, meas_z)
        if FILTERPY_AVAILABLE:
            kf = self.filters[track_id]
            if dt is None:
                dt = self.dt
            # update F if dt changed
            kf.F[0,2] = dt
            kf.F[1,3] = dt
            kf.predict()
            kf.update(np.array([meas_x, meas_z], dtype=float))
            x = float(kf.x[0]); z = float(kf.x[1]); vx = float(kf.x[2]); vz = float(kf.x[3])
            return x, z, vx, vz
        else:
            f = self.filters[track_id]
            now = time.time()
            if dt is None:
                dt = now - f["last_t"] if f["last_t"] is not None else self.dt
            # predict
            f["x"] += f["vx"] * dt
            f["z"] += f["vz"] * dt
            # innovation (simple PD)
            alpha = 0.6
            vx_new = (meas_x - f["x"]) / max(dt, 1e-3)
            vz_new = (meas_z - f["z"]) / max(dt, 1e-3)
            # exponential blend
            f["vx"] = 0.6 * f["vx"] + 0.4 * vx_new
            f["vz"] = 0.6 * f["vz"] + 0.4 * vz_new
            # update positions to measured (smoothed)
            f["x"] = (1-alpha) * f["x"] + alpha * meas_x
            f["z"] = (1-alpha) * f["z"] + alpha * meas_z
            f["last_t"] = now
            return float(f["x"]), float(f["z"]), float(f["vx"]), float(f["vz"])


# ---------- Trajectory prediction ----------
class TrajectoryPredictor:
    """
    Very simple constant-velocity predictor that returns a list of (t, X, Z)
    for t in [0 .. horizon] with step.
    """
    def __init__(self, horizon=5.0, step=0.2):
        self.horizon = horizon
        self.step = step

    def predict(self, X, Z, vx, vz):
        points = []
        t = 0.0
        while t <= self.horizon:
            Xp = X + vx * t
            Zp = Z + vz * t
            points.append((t, Xp, Zp))
            t += self.step
        return points


# ---------- Warning logic ----------
class WarningSystem:
    def __init__(self, cfg=None):
        self.cfg = load_yaml("config/warning_config.yaml", DEFAULT_WARNING_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.crit_ttc = float(self.cfg.get("critical_ttc", 1.5))
        self.warn_ttc = float(self.cfg.get("warn_ttc", 4.0))
        self.crit_dist = float(self.cfg.get("critical_distance", 3.0))
        self.warn_dist = float(self.cfg.get("warning_distance", 6.0))
        self.lat_thresh = float(self.cfg.get("lateral_overlap_threshold", 0.5))

    def decide(self, X, Z, vx, vz, predictions):
        # simple TTC
        v_rel = vz  # in ground frame we take forward velocity positive towards rear-camera origin
        if v_rel > 0.01:
            ttc = Z / v_rel
        else:
            ttc = float("inf")

        # check predicted lateral overlap with ego corridor (ego assumed at X=0, width +-0.6m)
        ego_left = -0.6
        ego_right = 0.6
        predicted_collision = False
        for t, xp, zp in predictions:
            # collision when longitudinal close and lateral overlap
            if 0.0 < zp < 2.0:
                # object assumed width ~1.6m for car, use simpler overlap test
                obj_left = xp - 0.8
                obj_right = xp + 0.8
                overlap = max(0.0, min(obj_right, ego_right) - max(obj_left, ego_left))
                if overlap > self.lat_thresh:
                    predicted_collision = True
                    break

        # Decision logic
        if ttc < self.crit_ttc or predicted_collision or Z < self.crit_dist:
            return "CRITICAL", ttc
        if ttc < self.warn_ttc or Z < self.warn_dist:
            return "WARN", ttc
        return "NONE", ttc


# ---------- Visualization ----------
def draw_track_overlay(frame, track_id, bbox, cls, conf, state, pred_points, warning_level, ttc):
    x1,y1,x2,y2 = bbox
    color = (0,255,0)
    if warning_level == "WARN":
        color = (0,165,255)
    elif warning_level == "CRITICAL":
        color = (0,0,255)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    label = f"ID:{track_id} {cls} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # state: X,Z,vx,vz
    if state is not None:
        X,Z,vx,vz = state
        s = f"Z:{Z:.1f}m v:{vz:.1f}m/s TTC:{'inf' if ttc==float('inf') else f'{ttc:.2f}s'}"
        cv2.putText(frame, s, (x1, y2+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # draw predicted path (project back to image roughly)
    H, W = frame.shape[:2]
    for t, xp, zp in pred_points[:30]:
        # project to image using simple pinhole with fx,fy,cx,cy from config
        cam_cfg = load_yaml("config/camera_config.yaml", DEFAULT_CAMERA_CFG)
        fx = cam_cfg.get("fx", 1000.0)
        fy = cam_cfg.get("fy", 1000.0)
        cx = cam_cfg.get("cx", W/2)
        cy = cam_cfg.get("cy", H/2)
        if zp <= 0.1: continue
        u = int((xp * fx) / zp + cx)
        v = int(( -cam_cfg.get("mounting_height",1.0) * fy) / zp + cy)
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(frame, (u,v), 2, color, -1)


# ---------- Main pipeline class ----------
class RearADASPipeline:
    def __init__(self, model_path=None, img_size=None, show_fps=False):
        # Load configs
        self.cam_cfg = load_yaml("config/camera_config.yaml", DEFAULT_CAMERA_CFG)
        self.model_cfg = load_yaml("config/model_config.yaml", DEFAULT_MODEL_CFG)
        if model_path:
            self.model_cfg["model_path"] = model_path
        if img_size:
            self.model_cfg["img_size"] = img_size
        self.warn_cfg = load_yaml("config/warning_config.yaml", DEFAULT_WARNING_CFG)

        # Subsystems
        self.detector = Detector(config=self.model_cfg, model_path_override=self.model_cfg.get("model_path"))
        self.tracker_cfg = load_yaml("config/tracker_config.yaml", {})
        self.tracker = CentroidTracker(
            max_distance=int(self.tracker_cfg.get("max_track_distance", 100)),
            max_lost=int(self.tracker_cfg.get("max_lost_frames", 10)),
            min_confidence=float(self.tracker_cfg.get("min_detection_confidence", 0.25)),
            min_confirmed_age=int(self.tracker_cfg.get("min_confirmed_age", 2))
        )
        self.projector = GroundProjector(cfg=self.cam_cfg)
        self.depther = DepthEstimator()
        self.kf = SimpleKalman(dt=1/30.0)
        self.predictor = TrajectoryPredictor(horizon=5.0, step=0.2)
        self.warnsys = WarningSystem(cfg=self.warn_cfg)

        # tracking state store (to map track ID -> kalman)
        self.track_state = {}  # track_id -> (X,Z,vx,vz)
        self.show_fps = show_fps
        # allowed detection classes (only these will be processed)
        allowed = self.model_cfg.get("allowed_detection_classes", None)
        if allowed is None:
            # default to vehicle classes
            allowed = ["car", "truck", "bus", "motorcycle", "other"]
        self.allowed_detection_classes = set(allowed)
        # segmentation overlay settings
        self.enable_segmentation = bool(self.model_cfg.get("enable_segmentation_overlay", True))
        self.segmentation_alpha = float(self.model_cfg.get("segmentation_alpha", 0.35))
        config_colors = self.model_cfg.get("segmentation_colors", {}) or {}
        self.segmentation_colors = SEGMENTATION_COLORS.copy()
        for cls_name, rgb in config_colors.items():
            if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                self.segmentation_colors[cls_name] = tuple(int(c) for c in rgb)

    # --- simple lane detection ---
    def detect_lanes(self, frame):
        H, W = frame.shape[:2]
        # Focus on bottom 60% where lanes are typically visible
        roi_y = int(H * 0.35)  # Start higher to capture more lane area
        roi = frame[roi_y:, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for better lane detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply CLAHE for better contrast in various lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        
        # Adaptive thresholding to handle varying lighting
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Canny edge detection with tuned parameters
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Define region of interest mask (trapezoid shape)
        mask = np.zeros_like(edges)
        roi_h, roi_w = edges.shape
        # Trapezoid vertices
        vertices = np.array([[
            (int(roi_w * 0.1), roi_h),           # bottom left
            (int(roi_w * 0.4), int(roi_h * 0.3)), # top left
            (int(roi_w * 0.6), int(roi_h * 0.3)), # top right
            (int(roi_w * 0.9), roi_h)             # bottom right
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough Transform with improved parameters
        lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, 
                                threshold=30, minLineLength=40, maxLineGap=150)
        
        left_lines = []
        right_lines = []
        
        if lines is None:
            return []
        
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # Convert to full frame coordinates
            y1_full = y1 + roi_y
            y2_full = y2 + roi_y
            
            # Calculate slope
            if x2 == x1:
                continue  # Skip vertical lines
            
            slope = (y2 - y1) / float(x2 - x1)
            
            # Filter out near-horizontal lines (not lanes)
            if abs(slope) < 0.5 or abs(slope) > 3.0:  # Stricter slope threshold
                continue
            
            # Calculate line length for weighting
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Classify left/right lanes by slope and position
            center_x = (x1 + x2) / 2
            
            if slope < -0.5 and center_x < W * 0.55:  # Left lane: negative slope, left side
                left_lines.append((x1, y1_full, x2, y2_full, abs(slope), length))
            elif slope > 0.5 and center_x > W * 0.45:  # Right lane: positive slope, right side
                right_lines.append((x1, y1_full, x2, y2_full, slope, length))
        
        # Average and extrapolate lane lines with weighted fitting
        def extrapolate_lane(lines, side='left'):
            if not lines:
                return None
            
            # Weight by line length
            points = []
            weights = []
            for (x1, y1, x2, y2, slope, length) in lines:
                points.append([x1, y1])
                points.append([x2, y2])
                weights.append(length)
                weights.append(length)
            
            if len(points) < 4:
                return None
            
            points = np.array(points, dtype=np.float32)
            weights = np.array(weights, dtype=np.float32)
            
            # Fit line using least squares with weights
            try:
                # Calculate weighted centroid
                w_sum = np.sum(weights)
                x_mean = np.sum(points[:, 0] * weights) / w_sum
                y_mean = np.sum(points[:, 1] * weights) / w_sum
                
                # Calculate weighted covariance
                dx = points[:, 0] - x_mean
                dy = points[:, 1] - y_mean
                
                cov_xx = np.sum(weights * dx * dx) / w_sum
                cov_yy = np.sum(weights * dy * dy) / w_sum
                cov_xy = np.sum(weights * dx * dy) / w_sum
                
                # Calculate line direction (eigenvector of covariance matrix)
                lambda_val = (cov_xx + cov_yy + np.sqrt((cov_xx - cov_yy)**2 + 4*cov_xy**2)) / 2
                
                if abs(cov_xy) > 1e-6:
                    vx = lambda_val - cov_yy
                    vy = cov_xy
                else:
                    vx = 1.0
                    vy = 0.0
                
                # Normalize direction
                norm = np.sqrt(vx**2 + vy**2)
                vx /= norm
                vy /= norm
                
                # Extrapolate to bottom and middle of frame
                y_bottom = H
                y_top = int(H * 0.4)
                
                if abs(vy) < 0.01:  # Nearly horizontal
                    return None
                
                # Calculate x coordinates
                t_bottom = (y_bottom - y_mean) / vy
                t_top = (y_top - y_mean) / vy
                
                x_bottom = int(x_mean + t_bottom * vx)
                x_top = int(x_mean + t_top * vx)
                
                # Sanity check - ensure points are within frame bounds with margin
                if x_bottom < -50 or x_bottom > W + 50 or x_top < -50 or x_top > W + 50:
                    return None
                
                # Clamp to frame bounds
                x_bottom = max(0, min(W - 1, x_bottom))
                x_top = max(0, min(W - 1, x_top))
                
                return (x_top, y_top, x_bottom, y_bottom)
            except:
                return None
        
        out = []
        L = extrapolate_lane(left_lines, 'left')
        R = extrapolate_lane(right_lines, 'right')
        
        if L is not None:
            out.append(L)
        if R is not None:
            out.append(R)
        
        return out

    def build_segmentation_mask(self, frame_shape, tracks, traffic_lights, lanes):
        mask = np.zeros(frame_shape, dtype=np.uint8)
        # draw tracked objects
        for tr in tracks:
            color = self.segmentation_colors.get(tr.cls, (200, 200, 200))
            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        # draw extra traffic lights detected via color heuristic
        for tl in traffic_lights:
            x1, y1, x2, y2 = tl["bbox"]
            color = self.segmentation_colors.get("traffic_light", (0, 255, 0))
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        # draw lane lines
        lane_color = self.segmentation_colors.get("lane", (255, 255, 0))
        for lane in lanes:
            x1, y1, x2, y2 = lane
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), lane_color, 12)
        return mask

    # --- simple traffic light color detection ---
    def detect_traffic_lights(self, frame):
        H, W = frame.shape[:2]
        # focus on upper portion where traffic lights usually appear
        roi = frame[0:int(H*0.5), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # red can appear in two hue ranges
        red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        yellow = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        green = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))
        masks = {"RED": red, "YELLOW": yellow, "GREEN": green}
        detections = []
        for color_name, mask in masks.items():
            # morphological cleanup
            km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, km)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 40:
                    continue
                x,y,w,h = cv2.boundingRect(cnt)
                # convert to full-frame coords
                detections.append({"bbox": (x, y, x+w, y+h), "color": color_name})
        # merge overlapping detections and prefer the color with largest area
        # For simplicity, return raw detections
        return detections

    def run(self, video_source=0, use_realsense=False, realsense_cfg=None):
        # open capture
        cap = None
        rs_stream = None
        if use_realsense:
            if RealSenseStream is None:
                raise RuntimeError(
                    "RealSense support requested but pyrealsense2 is not installed. "
                    "Install it via `pip install pyrealsense2` and try again."
                )
            try:
                rs_stream = RealSenseStream(**(realsense_cfg or {}))
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize RealSense camera: {exc}") from exc
            fps = rs_stream.fps
        else:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source {video_source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"[Pipeline] Video FPS estimated: {fps:.2f}")
        prev_time = time.time()
        window_name = "Rear View ADAS Prototype"
        # main loop
        while True:
            if rs_stream is not None:
                ret, frame = rs_stream.read()
            else:
                ret, frame = cap.read()
            if not ret:
                print("[Pipeline] End of stream or cannot read frame.")
                break
            t0 = time.time()

            detections = self.detector.detect(frame)
            # filter to allowed classes
            detections = [d for d in detections if d.cls in self.allowed_detection_classes]
            tracks = self.tracker.update(detections)

            # lane detection with improved algorithm - draw filled polygon
            lanes = self.detect_lanes(frame)
            if len(lanes) >= 2:
                # Create filled polygon between two lanes
                left_lane = lanes[0]
                right_lane = lanes[1]
                
                # Create polygon points (left bottom, left top, right top, right bottom)
                polygon_pts = np.array([
                    [left_lane[2], left_lane[3]],    # left bottom
                    [left_lane[0], left_lane[1]],    # left top
                    [right_lane[0], right_lane[1]],  # right top
                    [right_lane[2], right_lane[3]]   # right bottom
                ], dtype=np.int32)
                
                # Create overlay for transparency
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_pts], (255, 100, 0))  # Blue color (BGR)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # 30% transparent
                
                # Draw lane boundaries
                cv2.line(frame, (int(left_lane[0]), int(left_lane[1])), 
                        (int(left_lane[2]), int(left_lane[3])), (0, 255, 255), 3)
                cv2.line(frame, (int(right_lane[0]), int(right_lane[1])), 
                        (int(right_lane[2]), int(right_lane[3])), (0, 255, 255), 3)
            elif len(lanes) == 1:
                # Draw single lane line
                l = lanes[0]
                cv2.line(frame, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 255), 3)

            # traffic light detection DISABLED for rear-view ADAS (not relevant)
            # tls = self.detect_traffic_lights(frame)
            # for t in tls:
            #     x1,y1,x2,y2 = t["bbox"]
            #     color = (0,255,0) if t["color"]=="GREEN" else ((0,255,255) if t["color"]=="YELLOW" else (0,0,255))
            #     cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            #     cv2.putText(frame, f"TL:{t['color']}", (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            tls = []  # Empty list for segmentation mask

            # segmentation-style overlay
            if self.enable_segmentation:
                seg_mask = self.build_segmentation_mask(frame.shape, tracks, tls, lanes)
                frame = cv2.addWeighted(frame, 1.0 - self.segmentation_alpha, seg_mask, self.segmentation_alpha, 0)

            # process tracks
            for tr in tracks:
                tid = tr.track_id
                bbox = tr.bbox
                cls = tr.cls
                # Track objects from different tracker implementations may
                # expose confidence as `conf` or `confidence`. Handle both.
                conf = getattr(tr, 'conf', None)
                if conf is None:
                    conf = getattr(tr, 'confidence', 1.0)

                # 1) two depth estimates
                Xg, Zg_proj = self.projector.bottom_to_ground(bbox)
                Zb = self.depther.bbox_depth(bbox, cls)
                # fuse simply: weighted by confidence and heuristic (give more weight to ground projection for near-field)
                w_proj = 0.6
                w_bbox = 0.4
                Zf = w_proj * Zg_proj + w_bbox * Zb

                # 2) Kalman smooth (X,Z)
                x, z, vx, vz = self.kf.predict_and_update(tid, Xg, Zf)
                self.track_state[tid] = (x, z, vx, vz)

                # 3) predict forward & compute warnings
                preds = self.predictor.predict(x, z, vx, vz)
                level, ttc = self.warnsys.decide(x, z, vx, vz, preds)

                # 4) draw overlay
                draw_track_overlay(frame, tid, bbox, cls, conf, (x,z,vx,vz), preds, level, ttc)

                # debug print
                # print(f"Track {tid} cls={cls} Z={z:.2f}m vz={vz:.2f} m/s ttc={ttc if ttc==float('inf') else f'{ttc:.2f}'} lvl={level}")

            # show fps
            if self.show_fps:
                now = time.time()
                fps_show = 1.0 / max(1e-6, now - prev_time)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps_show:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

        if cap is not None:
            cap.release()
        if rs_stream is not None:
            rs_stream.release()
        cv2.destroyAllWindows()
