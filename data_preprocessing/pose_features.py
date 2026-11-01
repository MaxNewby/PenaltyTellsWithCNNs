import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from config import ROOT


class PoseFeatures:
    """Extracts body-pose features from each raw frame and saves them to
    image_data/pose_features.csv. Also writes a skeleton overlay to
    <nation>/<player>/preprocessing/pose_overlay/ for checking.

    'L' and 'R' mean left and right AS SEEN IN THE IMAGE. From a rear camera
    that is the player's own left and right.
    """

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
    # COCO keypoint ids
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
    HEAD = (0, 1, 2, 3, 4)
    SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

    def __init__(self, weights='yolov8m-pose.pt', min_conf=0.3, imgsz=1280):
        self.model = YOLO(weights)
        self.min_conf = min_conf
        self.imgsz = imgsz
        self.base = Path(ROOT) / 'image_data' / 'Outfield_Players'
        self.csv_path = Path(ROOT) / 'image_data' / 'pose_features.csv'

    # ---------- detection ----------
    def find_taker(self, img):
        res = self.model(img, imgsz=self.imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, None, None
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(areas.argmax())                       # largest person = the taker
        kp = res.keypoints.xy[i].cpu().numpy()
        conf = res.keypoints.conf[i].cpu().numpy()
        height = float(boxes[i][3] - boxes[i][1])
        return kp, conf, height

    # ---------- helpers ----------
    def _pair(self, kp, conf, a, b):
        """Return (left, right) as seen in the image, or (None, None) if unreliable.
        Sorting by x means a left/right mix-up by the model can't flip the signs."""
        if conf[a] < self.min_conf or conf[b] < self.min_conf:
            return None, None
        p, q = kp[a], kp[b]
        return (p, q) if p[0] <= q[0] else (q, p)

    @staticmethod
    def _angle(a, b):
        """Angle of the line a->b from horizontal, in degrees (image y points down)."""
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    @staticmethod
    def _mid(a, b):
        return (a + b) / 2

    # ---------- features ----------
    def features(self, kp, conf, height):
        f = {}
        sl, sr = self._pair(kp, conf, self.L_SH, self.R_SH)
        hl, hr = self._pair(kp, conf, self.L_HIP, self.R_HIP)
        el, er = self._pair(kp, conf, self.L_EL, self.R_EL)
        wl, wr = self._pair(kp, conf, self.L_WR, self.R_WR)
        kl, kr = self._pair(kp, conf, self.L_KNEE, self.R_KNEE)
        al, ar = self._pair(kp, conf, self.L_ANK, self.R_ANK)

        if sl is not None:
            f['shoulder_width'] = (sr[0] - sl[0]) / height
            f['shoulder_tilt'] = self._angle(sl, sr)      # + = right shoulder lower
        if hl is not None:
            f['hip_width'] = (hr[0] - hl[0]) / height
            f['hip_tilt'] = self._angle(hl, hr)
        if sl is not None and hl is not None:
            f['shoulder_hip_width_ratio'] = (sr[0] - sl[0]) / max(hr[0] - hl[0], 1e-6)
            f['tilt_diff'] = f['shoulder_tilt'] - f['hip_tilt']
            ms, mh = self._mid(sl, sr), self._mid(hl, hr)
            f['torso_lean'] = math.degrees(math.atan2(ms[0] - mh[0], mh[1] - ms[1]))  # + = shoulders right of hips

        # head: how far the head sits left/right of the shoulder midpoint (yaw proxy)
        head = [kp[i] for i in self.HEAD if conf[i] >= self.min_conf]
        f['head_points_visible'] = len(head)
        if head and sl is not None:
            f['head_offset'] = (np.mean([p[0] for p in head]) - self._mid(sl, sr)[0]) / max(sr[0] - sl[0], 1e-6)

        # arms (arms-on-hips shows up as elbows wider than shoulders)
        if el is not None:
            f['elbow_width'] = (er[0] - el[0]) / height
            if sl is not None:
                f['elbow_shoulder_ratio'] = (er[0] - el[0]) / max(sr[0] - sl[0], 1e-6)
        if wl is not None and hl is not None:
            mh = self._mid(hl, hr)
            f['wrist_l_dx'], f['wrist_r_dx'] = (wl[0] - mh[0]) / height, (wr[0] - mh[0]) / height
            f['wrist_l_dy'], f['wrist_r_dy'] = (wl[1] - mh[1]) / height, (wr[1] - mh[1]) / height

        # legs
        if kl is not None:
            f['knee_width'] = (kr[0] - kl[0]) / height
            f['knee_dy'] = (kr[1] - kl[1]) / height
        if al is not None:
            f['stance_width'] = (ar[0] - al[0]) / height
            f['ankle_dy'] = (ar[1] - al[1]) / height      # + = right foot lower in the image
            if hl is not None:
                mh = self._mid(hl, hr)
                f['ankle_l_dx'], f['ankle_r_dx'] = (al[0] - mh[0]) / height, (ar[0] - mh[0]) / height

        core = [self.L_SH, self.R_SH, self.L_HIP, self.R_HIP, self.L_ANK, self.R_ANK]
        f['core_conf_min'] = float(min(conf[j] for j in core))
        return f

    # ---------- QA overlay ----------
    def save_overlay(self, img, kp, conf, path):
        out = img.copy()
        for a, b in self.SKELETON:
            if conf[a] >= self.min_conf and conf[b] >= self.min_conf:
                cv2.line(out, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), (255, 255, 0), 2)
        for i, (x, y) in enumerate(kp):
            if conf[i] >= self.min_conf:
                cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), -1)
        d = path.parent.parent / 'preprocessing' / 'pose_overlay'
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / path.name), out)

    # ---------- main entry points ----------
    def process_image(self, path):
        nation, player = path.parts[-4], path.parts[-3]
        row = {'nation': nation, 'player': player, 'file': path.name, 'pose_found': False}
        img = cv2.imread(str(path))
        if img is None:
            print(f'Could not read {path}')
            return row
        kp, conf, height = self.find_taker(img)
        if kp is None:
            print(f'No person found in {path.name}')
            return row
        row['pose_found'] = True
        row.update(self.features(kp, conf, height))
        self.save_overlay(img, kp, conf, path)
        return row

    def process_all(self):
        rows = [self.process_image(p) for p in sorted(self.base.glob('*/*/raw/*'))
                if p.suffix.lower() in self.IMG_EXTS]
        df = pd.DataFrame(rows)          # features missing for an image become NaN
        df.to_csv(self.csv_path, index=False)
        print(f"Pose found in {df['pose_found'].sum()} of {len(df)} images")
        print('Share of missing values per column:')
        print(df.isna().mean().sort_values(ascending=False).head(8).round(2))
        return df