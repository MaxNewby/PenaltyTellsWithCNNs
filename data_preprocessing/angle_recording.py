import math
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from config import ROOT


class GeometryExtractor:
    """Records player-to-ball angle and distance for each penalty image,
    and saves QA overlays and cropped player images beside each 'raw' folder:

        <nation>/<player>/raw/
        <nation>/<player>/preprocessing/overlay/
        <nation>/<player>/preprocessing/crop/
    """

    SPORTS_BALL = 32            # COCO class id
    L_ANKLE, R_ANKLE = 15, 16   # COCO keypoint ids
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

    def __init__(self, pose_weights='yolov8m-pose.pt', ball_weights='yolov8m.pt', grey=False):
        self.base = Path(ROOT) / 'image_data' / 'Outfield_Players'
        self.csv_path = Path(ROOT) / 'image_data' / 'geometry_features.csv'
        self.grey = grey
        self.pose_model = YOLO(pose_weights)
        self.ball_model = YOLO(ball_weights)

    # ---------- folders ----------
    @staticmethod
    def stage_dir(raw_path, stage):
        """Folder for a preprocessing stage, sitting beside 'raw'."""
        out = raw_path.parent.parent / 'preprocessing' / stage
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---------- detection ----------
    def find_taker(self, img):
        """Return (foot_xy, bbox_height, bbox_xyxy) for the largest person."""
        res = self.pose_model(img, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, None, None
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(areas.argmax())
        kp = res.keypoints.xy[i].cpu().numpy()
        conf = res.keypoints.conf[i].cpu().numpy()
        ankles = [kp[j] for j in (self.L_ANKLE, self.R_ANKLE) if conf[j] > 0.4]
        if not ankles:
            return None, None, None
        foot = (sum(a[0] for a in ankles) / len(ankles),
                sum(a[1] for a in ankles) / len(ankles))
        return foot, float(boxes[i][3] - boxes[i][1]), boxes[i]

    def find_ball(self, img):
        res = self.ball_model(img, imgsz=1280, classes=[self.SPORTS_BALL],
                              conf=0.1, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        best = res.boxes.conf.argmax()
        x, y, _, _ = res.boxes.xywh[best].cpu().numpy()
        return float(x), float(y)

    # ---------- manual fallback ----------
    @staticmethod
    def click_point(img, title):
        """Click the point yourself. Press any key to skip."""
        pt = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pt.append((x, y))

        cv2.imshow(title, img)
        cv2.setMouseCallback(title, on_click)
        while not pt:
            if cv2.waitKey(50) != -1:
                break
        cv2.destroyAllWindows()
        return pt[0] if pt else None

    # ---------- maths ----------
    @staticmethod
    def measure(foot, ball, height):
        """Angle from straight up (0 = ball ahead, + = ball to the right) and distance."""
        dx = ball[0] - foot[0]
        dy = foot[1] - ball[1]  # image y points down, so flip it
        dist = math.hypot(dx, dy)
        return {
            'angle_deg': math.degrees(math.atan2(dx, dy)),
            'dist_px': dist,
            'dist_norm': dist / height,
        }

    # ---------- outputs ----------
    def save_overlay(self, img, row, path):
        """QA image with markers drawn on. Don't train on this."""
        foot = (int(row['foot_x']), int(row['foot_y']))
        ball = (int(row['ball_x']), int(row['ball_y']))
        cv2.circle(img, foot, 8, (0, 255, 0), -1)
        cv2.circle(img, ball, 8, (0, 0, 255), -1)
        cv2.line(img, foot, ball, (255, 255, 0), 2)
        cv2.putText(img, f"{row['angle_deg']:.1f} deg", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(self.stage_dir(path, 'overlay') / path.name), img)

    def save_crop(self, img, bbox, path, margin=0.1, size=224):
        """Crop around the player, pad to square (no stretching), resize, save."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        pad = margin * (y2 - y1)
        x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        x2, y2 = min(w, int(x2 + pad)), min(h, int(y2 + pad))
        crop = img[y1:y2, x1:x2]

        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        top = (side - ch) // 2
        left = (side - cw) // 2
        crop = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

        if self.grey:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(str(self.stage_dir(path, 'crop') / path.name), crop)

    # ---------- main entry points ----------
    def process_image(self, path):
        nation, player = path.parts[-4], path.parts[-3]
        img = cv2.imread(str(path))
        if img is None:
            print(f'Could not read {path}')
            return None

        foot, height, bbox = self.find_taker(img)
        foot_src = 'auto'
        if foot is None:
            foot = self.click_point(img, f'Click player feet: {path.name}')
            height = img.shape[0] * 0.5  # rough placeholder; fix manually later
            foot_src = 'manual'

        ball = self.find_ball(img)
        ball_src = 'auto'
        if ball is None:
            ball = self.click_point(img, f'Click ball: {path.name}')
            ball_src = 'manual'

        if foot is None or ball is None:
            print(f'Skipped {path}')
            return None

        row = {'nation': nation, 'player': player, 'file': path.name,
               'foot_x': foot[0], 'foot_y': foot[1],
               'ball_x': ball[0], 'ball_y': ball[1],
               'player_height_px': height,
               'foot_source': foot_src, 'ball_source': ball_src,
               **self.measure(foot, ball, height)}

        self.save_overlay(img.copy(), row, path)

        if bbox is not None:
            self.save_crop(img, bbox, path)
        else:
            print(f'No bounding box for {path.name}, no crop saved')

        return row

    def process_all(self):
        rows = []
        for path in sorted(self.base.glob('*/*/raw/*')):
            if path.suffix.lower() not in self.IMG_EXTS:
                continue
            row = self.process_image(path)
            if row is not None:
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
        return df