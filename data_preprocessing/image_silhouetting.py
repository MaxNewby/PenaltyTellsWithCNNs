import cv2
import numpy as np
from pathlib import Path
from config import ROOT


class ImageSilhouette:
    """Reads preprocessing/resized/*.png and writes, beside 'raw':

        preprocessing/cutout/       player on black background
        preprocessing/silhouette/   white player on black
    """

    def __init__(self, grey=False, save_to_preprocessed=False):
        self.grey = grey
        self.base = Path(ROOT) / 'image_data' / 'Outfield_Players'
        self.save_to_preprocessed = save_to_preprocessed

    def process_image(self, path):
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
            print(f'No alpha channel in {path}')
            return False

        mask = np.where(rgba[..., 3] > 127, 255, 0).astype(np.uint8)  # hard edge

        bgr = rgba[..., :3]
        cutout = cv2.bitwise_and(bgr, bgr, mask=mask)  # black background
        if self.grey:
            cutout = cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)

        player_path = path.parent.parent.parent  # .../Marquinhos
        preprocessing = path.parent.parent  # .../preprocessing

        # stage folders inside preprocessing/ (unchanged)
        for stage, im in (('cutout', cutout), ('silhouette', mask)):  # mask = white player
            out = preprocessing / stage
            out.mkdir(exist_ok=True)
            cv2.imwrite(str(out / path.name), im)

        # optional: also save the silhouette to preprocessed/, beside raw/
        if self.save_to_preprocessed:
            out = player_path / 'preprocessed'
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / path.name), mask)

        return True

    def process_all(self):
        ok = failed = 0
        for path in sorted(self.base.glob('*/*/preprocessing/resized/*.png')):
            if self.process_image(path):
                ok += 1
            else:
                failed += 1
        print(f'Background filled: {ok}, failed: {failed}')
        return ok, failed