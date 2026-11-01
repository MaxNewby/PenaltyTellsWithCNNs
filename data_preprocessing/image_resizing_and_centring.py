import cv2
import numpy as np
from pathlib import Path
from config import ROOT


class ResizeCentre:
    """Crops each foreground PNG tight to the player, centres it on a square
    transparent canvas, and resizes. Keeps the alpha channel.
    Writes preprocessing/resized/<name>.png beside 'raw'.
    """

    def __init__(self, margin=0.05):
        self.margin = margin
        self.base = Path(ROOT) / 'image_data' / 'Outfield_Players'

    def _square(self, im, size):
        """Pad to a square (player centred, no stretching), then resize."""
        h, w = im.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        im = cv2.copyMakeBorder(im, top, side - h - top, left, side - w - left,
                                cv2.BORDER_CONSTANT, value=0)   # (0,0,0,0) = transparent
        return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)

    def process_image(self, path, size=224):
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
            print(f'No alpha channel in {path}')
            return False

        alpha = rgba[..., 3]
        mask = np.where(alpha > 127, 255, 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            print(f'Empty mask in {path.name}')
            return False

        # tight box around the player, plus a small margin
        x, y, w, h = cv2.boundingRect(mask)
        pad = int(self.margin * max(w, h))
        H, W = mask.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)

        # zero the hidden colours under transparent pixels so they can't blend into the edges
        rgba = rgba.copy()
        rgba[alpha == 0, :3] = 0

        out_im = self._square(rgba[y1:y2, x1:x2], size)

        out = path.parent.parent / 'resized'
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / path.name), out_im)
        return True

    def process_all(self, size=224):
        ok = failed = 0
        for path in sorted(self.base.glob('*/*/preprocessing/foreground/*.png')):
            if self.process_image(path, size):
                ok += 1
            else:
                failed += 1
        print(f'Resized and centred: {ok}, failed: {failed}')
        return ok, failed