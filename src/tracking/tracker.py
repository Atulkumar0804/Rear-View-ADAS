"""
tracker.py

Tracking wrapper. Provides:
 - Tracker.update(detections) -> list of Track namedtuples

Behavior:
 - If an external ByteTrack implementation is installed and desired, the wrapper
   will attempt to use it (user must provide that dependency).
 - Otherwise the module uses a robust internal IOU+Kalman tracker.

Detection input:
 - A list of Detection-like objects with attributes (x1,y1,x2,y2, cls, conf)
   (That matches the Detection namedtuple used elsewhere in the repo.)

Returned tracks:
 - A list of Track(track_id, bbox, cls, confidence, age, last_seen)
   where bbox is (x1,y1,x2,y2) ints.

This file depends on byte_tracker_utils for bbox/Iou helpers.
"""

import time
from typing import List, Tuple
from collections import namedtuple, OrderedDict
import numpy as np

from .byte_tracker_utils import xyxy_to_xywh, xywh_to_xyxy, iou_matrix

# Optional imports
try:
    # many ByteTrack wrappers are named 'bytetrack' or similar; handle gracefully
    import bytetrack   # this is optional and may not exist
    BYTETRACK_AVAILABLE = True
except Exception:
    BYTETRACK_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except Exception:
    KalmanFilter = None
    FILTERPY_AVAILABLE = False

# Track namedtuple to return (compatible with pipeline Track expectations)
Track = namedtuple("Track", ["track_id", "bbox", "cls", "confidence", "age", "last_seen"])

# Internal TrackRecord used inside tracker
class TrackRecord:
    def __init__(self, track_id: int, bbox: Tuple[int,int,int,int], cls: str, conf: float):
        self.id = track_id
        self.bbox = bbox  # (x1,y1,x2,y2)
        self.cls = cls
        self.conf = conf
        self.age = 1
        self.last_seen = time.time()
        self.lost = 0

        # state for kalman: we use center x,y and w,h and velocity on x,y
        cx, cy, w, h = xyxy_to_xywh(bbox)
        if FILTERPY_AVAILABLE:
            kf = KalmanFilter(dim_x=7, dim_z=4)
            # state: [cx, cy, w, h, vx, vy, vw] (we keep simple)
            dt = 1.0/30.0
            kf.F = np.eye(7)
            kf.F[0,4] = dt
            kf.F[1,5] = dt
            kf.F[2,6] = dt
            # measurement maps cx,cy,w,h
            kf.H = np.zeros((4,7))
            kf.H[0,0] = 1
            kf.H[1,1] = 1
            kf.H[2,2] = 1
            kf.H[3,3] = 1
            kf.x = np.array([cx, cy, w, h, 0., 0., 0.])
            kf.P *= 10.0
            kf.R = np.diag([1.0, 1.0, 4.0, 4.0])
            kf.Q = np.eye(7) * 0.01
            self.kf = kf
        else:
            # fallback simple smoothing state
            self.state = {
                "cx": cx, "cy": cy, "w": w, "h": h,
                "vx": 0.0, "vy": 0.0, "vw": 0.0
            }

    def predict(self, dt=1/30.0):
        if FILTERPY_AVAILABLE:
            # update F with dt
            self.kf.F[0,4] = dt
            self.kf.F[1,5] = dt
            self.kf.F[2,6] = dt
            self.kf.predict()
        else:
            self.state["cx"] += self.state["vx"] * dt
            self.state["cy"] += self.state["vy"] * dt
            self.state["w"]  += self.state["vw"] * dt

    def update(self, bbox: Tuple[int,int,int,int], conf: float):
        cx, cy, w, h = xyxy_to_xywh(bbox)
        if FILTERPY_AVAILABLE:
            z = np.array([cx, cy, w, h])
            self.kf.update(z)
            cxp = float(self.kf.x[0]); cyp = float(self.kf.x[1]); wp = float(self.kf.x[2]); hp = float(self.kf.x[3])
            self.bbox = xywh_to_xyxy((cxp, cyp, wp, hp))
        else:
            # compute simple velocity estimates via difference
            alpha = 0.6
            vx_new = (cx - self.state["cx"])
            vy_new = (cy - self.state["cy"])
            vw_new = (w - self.state["w"])
            # exponential smoothing
            self.state["vx"] = 0.6 * self.state["vx"] + 0.4 * vx_new
            self.state["vy"] = 0.6 * self.state["vy"] + 0.4 * vy_new
            self.state["vw"] = 0.6 * self.state["vw"] + 0.4 * vw_new
            self.state["cx"] = (1-alpha) * self.state["cx"] + alpha * cx
            self.state["cy"] = (1-alpha) * self.state["cy"] + alpha * cy
            self.state["w"]  = (1-alpha) * self.state["w"]  + alpha * w
            self.state["h"]  = (1-alpha) * self.state["h"]  + alpha * h
            self.bbox = xywh_to_xyxy((self.state["cx"], self.state["cy"], self.state["w"], self.state["h"]))
        self.conf = conf
        self.age += 1
        self.last_seen = time.time()
        self.lost = 0

    def mark_missed(self):
        self.lost += 1

    def to_track(self):
        return Track(self.id, self.bbox, self.cls, float(self.conf), int(self.age), float(self.last_seen))

# --------------------------
# IOU + Kalman fallback tracker
# --------------------------
class IOUTracker:
    def __init__(self,
                 max_lost=10,
                 iou_threshold=0.3,
                 max_distance_pixels=100):
        """
        max_lost: frames to keep 'lost' tracks before deletion
        iou_threshold: minimum IoU to consider a match
        """
        self.next_id = 1
        self.tracks = OrderedDict()  # track_id -> TrackRecord
        self.max_lost = int(max_lost)
        self.iou_thresh = float(iou_threshold)
        self.max_distance = max_distance_pixels

    def _associate(self, detections: List[Tuple[int,int,int,int]]):
        """
        Associate current detections to existing tracks via IoU.
        Returns lists: matches (track_idx, det_idx), unmatched_tracks, unmatched_detections
        """
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[t].bbox for t in track_ids]
        if len(track_boxes) == 0:
            return [], track_ids, list(range(len(detections)))

        iou_mat = iou_matrix(track_boxes, detections)  # shape (T, D)

        if iou_mat.size == 0:
            return [], track_ids, list(range(len(detections)))

        matches = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_dets = set(range(len(detections)))

        if SCIPY_AVAILABLE:
            # Hungarian assignment on negative IoU (maximize IoU -> minimize -IoU)
            cost = -iou_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thresh:
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)
            unmatched_tracks = [track_ids[i] for i in unmatched_tracks]
            unmatched_dets = list(unmatched_dets)
            # convert matched row indices to track ids
            matches = [(track_ids[r], c) for (r, c) in matches]
            return matches, unmatched_tracks, unmatched_dets
        else:
            # greedy matching fallback
            iou_copy = iou_mat.copy()
            while True:
                idx = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
                maxval = iou_copy[idx]
                if maxval < self.iou_thresh:
                    break
                t_idx, d_idx = idx
                matches.append((track_ids[t_idx], d_idx))
                iou_copy[t_idx, :] = -1.0
                iou_copy[:, d_idx] = -1.0
            matched_track_idxs = [list(self.tracks.keys()).index(tid) for (tid, _) in matches] if matches else []
            unmatched_tracks = [tid for i, tid in enumerate(track_ids) if i not in matched_track_idxs]
            unmatched_dets = [i for i in range(len(detections)) if i not in [d for (_, d) in matches]]
            return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Tuple[int,int,int,int]], classes: List[str], confs: List[float]) -> List[Track]:
        """
        detections: list of (x1,y1,x2,y2)
        classes: list of class strings (same length)
        confs: list of confidences (same length)

        returns list of Track namedtuples
        """
        now = time.time()
        # Predict step for all tracks
        for tr in list(self.tracks.values()):
            tr.predict()

        matches, unmatched_track_ids, unmatched_det_idxs = self._associate(detections)

        # Update matched tracks
        for track_id, det_idx in matches:
            self.tracks[track_id].update(detections[det_idx], confs[det_idx])
            self.tracks[track_id].cls = classes[det_idx]

        # Mark unmatched tracks as missed
        for track_id in unmatched_track_ids:
            if track_id in self.tracks:
                self.tracks[track_id].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_det_idxs:
            box = detections[det_idx]
            cls = classes[det_idx]
            conf = confs[det_idx]
            tr = TrackRecord(self.next_id, box, cls, conf)
            self.tracks[self.next_id] = tr
            self.next_id += 1

        # Remove old tracks
        to_delete = []
        for tid, tr in list(self.tracks.items()):
            if tr.lost > self.max_lost:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Prepare output list
        out = []
        for tid, tr in self.tracks.items():
            out.append(tr.to_track())
        return out

# --------------------------
# Top-level Tracker wrapper
# --------------------------
class Tracker:
    """
    Top-level tracker. Public API:
        t = Tracker(use_bytetrack=False, **kwargs)
        tracks = t.update(detections)  # detections are list of Detection-like objects

    If a bytetrack implementation exists and use_bytetrack=True, the wrapper will try
    to instantiate and call it. Otherwise uses IOUTracker fallback.
    """
    def __init__(self,
                 use_bytetrack: bool = False,
                 max_lost: int = 10,
                 iou_threshold: float = 0.3,
                 max_distance_pixels: int = 100):
        self.use_bytetrack = use_bytetrack and BYTETRACK_AVAILABLE
        if self.use_bytetrack:
            # instantiate / configure external bytetrack as needed
            try:
                # This is a generic placeholder; actual usage depends on the external package API.
                self.bt = bytetrack.ByteTrack()  # may fail if interface different
                print("[Tracker] ByteTrack detected and will be used.")
            except Exception:
                print("[Tracker] ByteTrack import succeeded but instantiation failed; falling back.")
                self.use_bytetrack = False

        # fallback
        self.iou_tracker = IOUTracker(max_lost=max_lost, iou_threshold=iou_threshold,
                                      max_distance_pixels=max_distance_pixels)

    def update(self, detections: List[object]) -> List[Track]:
        """
        detections: list of objects with attributes x1,y1,x2,y2, cls, conf (or tuple/list)
        returns list of Track namedtuples
        """
        if self.use_bytetrack:
            # Convert to the form expected by bytetrack and call it.
            # NOTE: This is a placeholder. Replace with the actual bytetrack call if you add the dependency.
            try:
                bt_inputs = []
                for d in detections:
                    x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                    conf = float(d.conf)
                    cls = d.cls if hasattr(d, "cls") else 0
                    bt_inputs.append([x1, y1, x2, y2, conf, cls])
                results = self.bt.update(bt_inputs)
                # Convert results to Track namedtuples
                out = []
                for res in results:
                    # expected res format: [x1,y1,x2,y2, track_id, cls, conf]
                    x1,y1,x2,y2, tid, cls, conf = res
                    out.append(Track(int(tid), (int(x1),int(y1),int(x2),int(y2)), cls, float(conf), 1, time.time()))
                return out
            except Exception as e:
                # fallback to internal tracker
                print("[Tracker] ByteTrack usage failed at runtime:", e)
                self.use_bytetrack = False

        # Prepare lists for IOU tracker
        boxes = []
        classes = []
        confs = []
        for d in detections:
            # accept either namedtuple or simple tuple/list
            try:
                x1 = int(d.x1); y1 = int(d.y1); x2 = int(d.x2); y2 = int(d.y2)
                cls = d.cls if hasattr(d, "cls") else "other"
                conf = float(d.conf) if hasattr(d, "conf") else 1.0
            except Exception:
                # maybe it's a plain tuple like (x1,y1,x2,y2)
                x1,y1,x2,y2 = d[0], d[1], d[2], d[3]
                cls = "other"
                conf = 1.0
            boxes.append((x1,y1,x2,y2))
            classes.append(cls)
            confs.append(conf)

        return self.iou_tracker.update(boxes, classes, confs)


# If module executed directly, simple smoke test
if __name__ == "__main__":
    # small manual test: create a tracker and feed a few detections
    dets = [
        type("D",(object,),{"x1":100,"y1":200,"x2":200,"y2":350,"cls":"car","conf":0.9})(),
        type("D",(object,),{"x1":400,"y1":210,"x2":520,"y2":360,"cls":"motorcycle","conf":0.85})()
    ]
    tr = Tracker()
    out = tr.update(dets)
    print("Initial tracks:", out)
    # simulate next frame with slight movement
    dets2 = [
        type("D",(object,),{"x1":105,"y1":205,"x2":205,"y2":355,"cls":"car","conf":0.92})(),
        type("D",(object,),{"x1":405,"y1":220,"x2":525,"y2":370,"cls":"motorcycle","conf":0.80})()
    ]
    out2 = tr.update(dets2)
    print("Updated tracks:", out2)
