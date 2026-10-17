"""RF-DETR turtle detector -> best box + padded crop."""
from PIL import Image

from . import config


class Detector:
    def __init__(self, weights_path: str):
        from rfdetr import RFDETRLarge  # same variant you trained

        self.model = RFDETRLarge(pretrain_weights=weights_path)
        try:
            self.model.optimize_for_inference()
        except Exception:
            pass  # optional speed-up; safe to skip

    def detect(self, img: Image.Image):
        """Returns (crop, box, confidence) for the most confident turtle, or (None, None, conf)."""
        dets = self.model.predict(img, threshold=0.1)
        if len(dets) == 0:
            return None, None, 0.0
        i = int(dets.confidence.argmax())
        conf = float(dets.confidence[i])
        x1, y1, x2, y2 = [float(v) for v in dets.xyxy[i]]
        # 5% padding so the head/flippers are not clipped
        pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        box = (
            max(0, int(x1 - pw)), max(0, int(y1 - ph)),
            min(img.width, int(x2 + pw)), min(img.height, int(y2 + ph)),
        )
        return img.crop(box), box, conf
