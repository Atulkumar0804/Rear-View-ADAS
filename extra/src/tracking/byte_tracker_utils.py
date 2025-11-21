"""
byte_tracker_utils.py

Utility functions for tracking:
 - bbox formats conversion
 - IoU computation
 - matching helpers

This file is used by src/tracking/tracker.py
"""

from typing import Tuple, List
import numpy as np

def xyxy_to_xywh(bbox: Tuple[int,int,int,int]) -> Tuple[float,float,float,float]:
    """
    Convert bbox (x1,y1,x2,y2) -> (cx, cy, w, h)
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0
    return (cx, cy, w, h)

def xywh_to_xyxy(bbox: Tuple[float,float,float,float]) -> Tuple[int,int,int,int]:
    """
    Convert bbox (cx, cy, w, h) -> (x1,y1,x2,y2) as ints
    """
    cx, cy, w, h = bbox
    x1 = int(round(cx - w/2.0))
    y1 = int(round(cy - h/2.0))
    x2 = int(round(cx + w/2.0))
    y2 = int(round(cy + h/2.0))
    return (x1, y1, x2, y2)

def iou_xyxy(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
    """
    Compute IoU between two boxes in xyxy format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / float(union)

def iou_matrix(boxesA: List[Tuple[int,int,int,int]], boxesB: List[Tuple[int,int,int,int]]) -> np.ndarray:
    """
    Returns IoU matrix (len(A) x len(B))
    """
    if len(boxesA) == 0 or len(boxesB) == 0:
        return np.zeros((len(boxesA), len(boxesB)), dtype=float)
    A = np.array(boxesA)
    B = np.array(boxesB)
    mat = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            mat[i, j] = iou_xyxy(tuple(A[i]), tuple(B[j]))
    return mat
