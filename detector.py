# detector.py
import cv2
import numpy as np
from ultralytics import YOLO

CONFUSED_WITH_GUN = {65, 67, 74, 78, 79}
CONFUSED_NAMES = {'cell phone', 'remote', 'toothbrush', 'hair drier', 'mouse'}

class Detector:
    def __init__(self,
                 model_path: str = 'models/yolov8n.pt',
                 weapon_model_path: str = 'models/weapon_yolov8.pt',
                 conf: float = 0.35,
                 weapon_conf: float = 0.40):
        self.conf = conf
        self.weapon_conf = weapon_conf
        self.model = YOLO(model_path)

        self.weapon_model = None
        try:
            import os
            if os.path.exists(weapon_model_path):
                self.weapon_model = YOLO(weapon_model_path)
                print(f"[Detector] Loaded weapon model: {weapon_model_path}")
            else:
                print("[Detector] No dedicated weapon model found – using general model only.")
                print("[Detector] TIP: Download a weapon-trained model and place it at:")
                print(f"           {weapon_model_path}")
        except Exception as e:
            print(f"[Detector] Could not load weapon model: {e}")

        self.weapon_classes = {
            'knife', 'gun', 'pistol', 'rifle', 'weapon',
            'sword', 'scissors', 'handgun', 'firearm', 'shotgun'
        }
        self.person_class_names = {'person'}

    def _is_weapon_name(self, name: str) -> bool:
        n = name.lower()
        return any(w in n for w in self.weapon_classes)

    def _aspect_ratio(self, bbox) -> float:
        x1, y1, x2, y2 = bbox
        return max(x2 - x1, 1) / max(y2 - y1, 1)

    def _phone_not_gun(self, cname: str, conf: float, bbox, cls_idx: int) -> bool:
        name_lc = cname.lower()
        if name_lc in CONFUSED_NAMES and conf >= 0.55:
            return True
        if cls_idx in CONFUSED_WITH_GUN and conf < 0.70:
            if self._aspect_ratio(bbox) < 0.55:
                return True
        return False

    def _parse_boxes(self, res, model_fallback=None):
        try:
            names = res.names
        except Exception:
            try:
                names = model_fallback.names if model_fallback else {}
            except Exception:
                names = {}

        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], [], [], names

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_idxs = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        return xyxy, confs, cls_idxs, names

    @staticmethod
    def _iou(a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aA = (a[2] - a[0]) * (a[3] - a[1])
        aB = (b[2] - b[0]) * (b[3] - b[1])
        denom = aA + aB - inter
        return inter / denom if denom > 0 else 0.0

    def detect(self, frame: np.ndarray) -> list:
        detections = []
        if not isinstance(frame, np.ndarray):
            return detections
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # Smaller inference size greatly reduces CPU inference time on Render.
        # YOLO rescales returned boxes back to the original frame coordinates.
        results = self.model(frame, conf=self.conf, imgsz=640, max_det=30, verbose=False)
        if results:
            xyxy, confs, cls_idxs, names = self._parse_boxes(results[0], self.model)
            for i in range(len(xyxy)):
                bbox = [float(v) for v in xyxy[i]]
                conf = float(confs[i])
                cls_idx = int(cls_idxs[i])
                cname = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                is_weapon = self._is_weapon_name(cname)
                if is_weapon and self._phone_not_gun(cname, conf, bbox, cls_idx):
                    cname = f"[?phone/gun] {cname}"
                    is_weapon = False
                detections.append({
                    'class': cname,
                    'conf': conf,
                    'bbox': bbox,
                    'is_weapon': is_weapon,
                    'is_person': cname.lower() in self.person_class_names,
                    'source': 'general'
                })

        if self.weapon_model is not None:
            w_results = self.weapon_model(frame, conf=self.weapon_conf, imgsz=640, max_det=20, verbose=False)
            if w_results:
                xyxy, confs, cls_idxs, names = self._parse_boxes(w_results[0], self.weapon_model)
                for i in range(len(xyxy)):
                    bbox = [float(v) for v in xyxy[i]]
                    conf = float(confs[i])
                    cls_idx = int(cls_idxs[i])
                    cname = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                    wd = {'class': cname, 'conf': conf, 'bbox': bbox,
                          'is_weapon': True, 'is_person': False, 'source': 'weapon_model'}
                    merged = False
                    for d in detections:
                        if self._iou(bbox, d['bbox']) > 0.4:
                            d['is_weapon'] = True
                            d['class'] = cname
                            d['source'] = 'weapon_model'
                            merged = True
                            break
                    if not merged:
                        detections.append(wd)

        return detections

    def compute_motion_intensity(self, prev_frame, frame) -> float:
        if prev_frame is None:
            return 0.0
        # Motion calculation on a small grayscale image is much cheaper than
        # calculating the diff at the full uploaded-video resolution.
        gray1 = cv2.cvtColor(cv2.resize(prev_frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        total = thresh.shape[0] * thresh.shape[1]
        return (cv2.countNonZero(thresh) / total) * 100.0 if total else 0.0
