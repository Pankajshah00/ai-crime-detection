# detector.py
import cv2
import numpy as np
from ultralytics import YOLO

# ── COCO class IDs that look visually similar to guns ───────────────────────
# 65=remote, 67=cell phone, 78=hair drier, 79=toothbrush, 74=mouse
CONFUSED_WITH_GUN = {65, 67, 74, 78, 79}
CONFUSED_NAMES    = {'cell phone', 'remote', 'toothbrush', 'hair drier', 'mouse'}


class Detector:
    def __init__(self,
                 model_path: str        = 'models/yolov8n.pt',
                 weapon_model_path: str = 'models/weapon_yolov8.pt',
                 conf: float            = 0.35,
                 weapon_conf: float     = 0.40):
        self.conf        = conf
        self.weapon_conf = weapon_conf
        self.model       = YOLO(model_path)

        # Optional dedicated weapon model
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

        self.weapon_classes     = {
            'knife', 'gun', 'pistol', 'rifle', 'weapon',
            'sword', 'scissors', 'handgun', 'firearm', 'shotgun'
        }
        self.person_class_names = {'person'}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_weapon_name(self, name: str) -> bool:
        n = name.lower()
        return any(w in n for w in self.weapon_classes)

    def _aspect_ratio(self, bbox) -> float:
        """width / height."""
        x1, y1, x2, y2 = bbox
        return max(x2 - x1, 1) / max(y2 - y1, 1)

    def _could_be_gun(self, cname: str, conf: float,
                      bbox, cls_idx: int) -> bool:
        """
        Return True when a 'cell phone / remote' detection might actually be a gun.

        A real handgun held horizontally is WIDER than it is tall (aspect ratio > 1.0).
        A phone held upright is taller than wide (aspect ratio < 0.6).

        Rules:
          - COCO class is in the confusion set
          - Confidence is LOW  (model is not sure)  -> more likely misclassified
          - Aspect ratio is LANDSCAPE (>= 0.8)      -> gun shape, not portrait phone
        """
        if cls_idx not in CONFUSED_WITH_GUN:
            return False
        ar = self._aspect_ratio(bbox)
        # Low confidence + landscape shape = suspicious
        if conf < 0.65 and ar >= 0.8:
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

        xyxy     = boxes.xyxy.cpu().numpy()  if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs    = boxes.conf.cpu().numpy()  if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_idxs = boxes.cls.cpu().numpy()   if hasattr(boxes.cls,  "cpu") else np.array(boxes.cls)
        return xyxy, confs, cls_idxs, names

    @staticmethod
    def _iou(a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter  = max(0, x2 - x1) * max(0, y2 - y1)
        aA     = (a[2] - a[0]) * (a[3] - a[1])
        aB     = (b[2] - b[0]) * (b[3] - b[1])
        denom  = aA + aB - inter
        return inter / denom if denom > 0 else 0.0

    # ── public ───────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        """
        Full detection pipeline.

        Steps:
          1. Run general COCO model
          2. Apply gun-confusion upgrade: phone/remote detections that look like
             a gun (low conf + landscape shape) are re-labelled '[?gun]' and
             flagged is_weapon=True so they surface as alerts.
          3. Run optional weapon model and merge results
          4. Return unified detection list

        Each dict: {class, conf, bbox:[x1,y1,x2,y2], is_weapon, is_person, source}
        """
        detections = []

        # ── Step 1 & 2: general model + confusion upgrade ─────────────────
        results = self.model(frame, conf=self.conf, verbose=False)
        if results:
            xyxy, confs, cls_idxs, names = self._parse_boxes(results[0], self.model)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                conf    = float(confs[i])
                cls_idx = int(cls_idxs[i])
                cname   = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                bbox    = [x1, y1, x2, y2]

                is_weapon = self._is_weapon_name(cname)

                # Upgrade ambiguous phone/remote → possible gun
                if not is_weapon and self._could_be_gun(cname, conf, bbox, cls_idx):
                    cname     = f"[?gun] {cname}"
                    is_weapon = True   # fire the weapon alert

                detections.append({
                    'class':     cname,
                    'conf':      conf,
                    'bbox':      bbox,
                    'is_weapon': is_weapon,
                    'is_person': cname.lower() in self.person_class_names,
                    'source':    'general'
                })

        # ── Step 3: dedicated weapon model (optional) ─────────────────────
        if self.weapon_model is not None:
            w_results = self.weapon_model(frame, conf=self.weapon_conf, verbose=False)
            if w_results:
                xyxy, confs, cls_idxs, names = self._parse_boxes(w_results[0], self.weapon_model)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    conf    = float(confs[i])
                    cls_idx = int(cls_idxs[i])
                    cname   = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                    bbox    = [x1, y1, x2, y2]

                    wd = {
                        'class':     cname,
                        'conf':      conf,
                        'bbox':      bbox,
                        'is_weapon': True,
                        'is_person': False,
                        'source':    'weapon_model'
                    }

                    # If this box overlaps a general-model box, upgrade it instead of duplicating
                    merged = False
                    for d in detections:
                        if self._iou(bbox, d['bbox']) > 0.4:
                            d['is_weapon'] = True
                            d['class']     = cname
                            d['source']    = 'weapon_model'
                            merged = True
                            break
                    if not merged:
                        detections.append(wd)

        return detections

    # ── motion ───────────────────────────────────────────────────────────────

    def compute_motion_intensity(self, prev_frame, frame) -> float:
        """Frame-diff motion as % of changed pixels."""
        if prev_frame is None:
            return 0.0
        # Resize prev_frame if source switched and resolution changed
        if prev_frame.shape != frame.shape:
            prev_frame = cv2.resize(prev_frame, (frame.shape[1], frame.shape[0]))
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY)
        diff  = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        total = frame.shape[0] * frame.shape[1]
        return (cv2.countNonZero(thresh) / total) * 100.0 if total else 0.0