"""
yolo_loader.py
----------------
Responsible for loading a YOLO model from the given path and preparing
the predictor for inference.
"""

import os
import yaml

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False


def load_yaml(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return default if default is not None else {}


class YOLOLoader:
    def __init__(self, config_path="config/model_config.yaml",
                 model_path_override=None, img_size_override=None):
        
        # Load model config
        self.cfg = load_yaml(config_path, default={
            "model_path": "models/yolo/yolov8n_RearView.pt",
            "img_size": 640,
            "conf_thres": 0.40,
            "classes": ["car", "truck", "bus", "motorcycle", "other"]
        })

        # Apply overrides (from pipeline)
        if model_path_override:
            self.cfg["model_path"] = model_path_override
        if img_size_override:
            self.cfg["img_size"] = img_size_override

        self.model_path = self.cfg["model_path"]
        self.img_size = self.cfg["img_size"]
        self.conf_thres = self.cfg["conf_thres"]
        self.class_names = self.cfg["classes"]

        self.model = None
        self.loaded = False

        # Try loading YOLO
        self._load_model()

    def _load_model(self):
        if not ULTRALYTICS_AVAILABLE:
            print("[YOLOLoader] ultralytics not installed. YOLO disabled.")
            return

        if not os.path.exists(self.model_path):
            print(f"[YOLOLoader] Model file not found: {self.model_path}")
            return

        try:
            print(f"[YOLOLoader] Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.loaded = True
        except Exception as e:
            print(f"[YOLOLoader] Failed to load YOLO: {e}")
            self.loaded = False

    def is_loaded(self):
        return self.loaded

    def get_model(self):
        return self.model

    def get_classes(self):
        return self.class_names

    def get_img_size(self):
        return self.img_size

    def get_conf_thres(self):
        return self.conf_thres
