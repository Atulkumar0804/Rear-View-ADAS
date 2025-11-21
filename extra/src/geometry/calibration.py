"""
calibration.py
--------------
Helper functions for camera intrinsic calibration using OpenCV.
Use with checkerboard images stored in data/calibration/checkerboard_images/

This module:
 - detects corners on checkerboard
 - runs cv2.calibrateCamera
 - prints fx, fy, cx, cy
 - writes results into config/camera_config.yaml

This file does not depend on pipeline directly.
"""

import cv2
import numpy as np
import glob
import yaml
import os


def save_intrinsics(out_path, fx, fy, cx, cy, mounting_height=None, pitch_deg=None):
    cfg = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy)
    }
    if mounting_height is not None:
        cfg["mounting_height"] = float(mounting_height)
    if pitch_deg is not None:
        cfg["pitch_deg"] = float(pitch_deg)

    with open(out_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"[Calibration] Saved camera intrinsics to {out_path}")


def calibrate_checkerboard(img_dir="data/calibration/checkerboard_images/",
                           pattern_size=(9, 6),
                           square_size=0.025,
                           output_path="config/camera_config.yaml"):
    """
    Calibrate camera intrinsics using checkerboard images.

    Args:
        img_dir: folder containing checkerboard images.
        pattern_size: (corners_x, corners_y) on board.
        square_size: size of 1 checkerboard square in meters.
        output_path: where to save final yaml file.
    """

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare 3D obj points like (0,0,0), (1,0,0) ... scaled by square size
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

    if len(images) == 0:
        print("[Calibration] No checkerboard images found!")
        return None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    if len(objpoints) == 0:
        print("[Calibration] No corners detected!")
        return None

    print("[Calibration] Running calibrateCamera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    fx = mtx[0,0]
    fy = mtx[1,1]
    cx = mtx[0,2]
    cy = mtx[1,2]

    print(f"[Calibration] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    save_intrinsics(output_path, fx, fy, cx, cy)

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion": dist,
        "mtx": mtx
    }
