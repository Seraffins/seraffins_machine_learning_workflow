"""Cheap quality checks on detector crops."""
import cv2
import numpy as np

from . import config


def check(crop, det_conf: float):
    """Returns (usable, reason). reason is None when usable."""
    if crop is None:
        return False, "no_turtle_detected"
    if det_conf < config.DET_MIN_CONF:
        return False, "low_detection_confidence"
    if min(crop.size) < config.MIN_CROP_SIDE:
        return False, "turtle_too_small"
    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < config.BLUR_MIN_VAR:
        return False, "too_blurry"
    return True, None
