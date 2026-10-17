"""Cheap quality checks on detector crops."""
import cv2
import numpy as np

from . import config


def check(crop, det_conf: float):
    """Returns (usable, reason, metrics). reason is None when usable.
    metrics are reported back per image so thresholds can be tuned on real photos."""
    metrics = {"detection_confidence": round(float(det_conf), 4)}
    if crop is None:
        return False, "no_turtle_detected", metrics
    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    metrics["crop_size"] = [crop.width, crop.height]
    metrics["sharpness"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
    if det_conf < config.DET_MIN_CONF:
        return False, "low_detection_confidence", metrics
    if min(crop.size) < config.MIN_CROP_SIDE:
        return False, "turtle_too_small", metrics
    if metrics["sharpness"] < config.BLUR_MIN_VAR:
        return False, "too_blurry", metrics
    return True, None, metrics