"""ALIKED + LightGlue local feature matching -> geometric inlier count."""
import math

import cv2
import numpy as np
import torch

from . import config

MAX_SIDE = 512


class Matcher:
    def __init__(self, device: str):
        from lightglue import ALIKED, LightGlue

        self.device = device
        self.extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
        self.lg = LightGlue(features="aliked").eval().to(device)

    def _tensor(self, img):
        s = MAX_SIDE / max(img.size)
        if s < 1:
            img = img.resize((int(img.width * s), int(img.height * s)))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).to(self.device)

    @torch.no_grad()
    def features(self, img):
        return self.extractor.extract(self._tensor(img))

    @torch.no_grad()
    def inliers(self, f0, f1) -> int:
        from lightglue.utils import rbd

        out = self.lg({"image0": f0, "image1": f1})
        a, b, out = rbd(f0), rbd(f1), rbd(out)
        m = out["matches"]
        if len(m) < 8:
            return 0
        p0 = a["keypoints"][m[:, 0]].cpu().numpy()
        p1 = b["keypoints"][m[:, 1]].cpu().numpy()
        # Fundamental matrix: tolerant of the head being a curved (non-planar) surface
        _, mask = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, 3.0, 0.999, 10000)
        return int(mask.sum()) if mask is not None else 0


def inliers_to_confidence(n: int) -> float:
    """Logistic map from inlier count to 0..1. Calibrate CONF_MID/CONF_SCALE on known pairs."""
    return round(1.0 / (1.0 + math.exp(-(n - config.CONF_MID) / config.CONF_SCALE)), 4)
