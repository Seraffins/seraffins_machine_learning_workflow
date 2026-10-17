"""Scute segmentation -> masked, tight scute crop.

Model: YOLO26s-seg (ultralytics >= 8.4), your fine-tuned scute checkpoint uploaded as
/models/segmenter.pt. All predicted scute masks are merged into one region.
To swap models later, only `_polygons()` needs to return polygons in original image pixels.

Licence note: ultralytics is AGPL-3.0. For commercial closed-source use you need their
enterprise licence, or swap in a permissively licensed segmenter.
"""
import numpy as np
from PIL import Image, ImageDraw

from . import config


class Segmenter:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)

    def _polygons(self, img: Image.Image):
        r = self.model.predict(img, conf=0.25, verbose=False)[0]
        if r.masks is None:
            return []
        return [p for p in r.masks.xy if len(p) >= 3]

    def scute_crop(self, img: Image.Image):
        """Returns a crop of the scute region with background blacked out, or None."""
        polys = self._polygons(img)
        if not polys:
            return None
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        for p in polys:  # union of all scute masks
            draw.polygon([tuple(map(float, pt)) for pt in p], fill=255)
        m = np.asarray(mask) > 0
        if m.mean() < config.MIN_MASK_AREA_FRAC:
            return None
        ys, xs = np.where(m)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        black = Image.new("RGB", img.size, (0, 0, 0))
        masked = Image.composite(img, black, mask)
        return masked.crop(box)
