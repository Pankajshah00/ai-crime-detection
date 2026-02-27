# detector.py
import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='models/yolov8n.pt', conf=0.3):
        # load YOLO model
        self.model = YOLO(model_path)
        self.conf = conf

        # weapon and violence related classes (expand as needed)
        self.weapon_classes = {'knife', 'gun', 'pistol', 'rifle', 'weapon', 'sword', 'scissors', 'handgun'}
        self.person_class_names = {'person'}

    def _is_weapon_name(self, name):
        # normalize and check if class name likely indicates a weapon
        n = name.lower()
        for w in self.weapon_classes:
            if w in n:
                return True
        return False

    def detect(self, frame):
        """
        Run YOLO on a single BGR frame and return list of detections.
        Each detection: {'class': class_name, 'conf': float, 'bbox': [x1,y1,x2,y2]}
        """
        detections = []
        # ultralytics YOLO accepts BGR frames
        results = self.model(frame, conf=self.conf, verbose=False)
        if not results:
            return detections

        # results is a list-like; take first result
        res = results[0]

        # names mapping (class index -> name)
        try:
            names = res.names
        except Exception:
            # fallback to model.names
            try:
                names = self.model.names
            except Exception:
                names = {}

        # iterate boxes
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return detections

        # boxes.xyxy, boxes.conf, boxes.cls are torch tensors or numpy arrays
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_idxs = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            conf = float(confs[i])
            cls_idx = int(cls_idxs[i])
            class_name = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)

            detections.append({
                'class': class_name,
                'conf': conf,
                'bbox': [x1, y1, x2, y2],
                'is_weapon': self._is_weapon_name(class_name),
                'is_person': class_name.lower() in self.person_class_names
            })

        return detections

    def compute_motion_intensity(self, prev_frame, frame):
        """
        Simple frame-diff based motion intensity (percentage of changed pixels).
        """
        if prev_frame is None:
            return 0.0

        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = frame.shape[0] * frame.shape[1]
        if total_pixels == 0:
            return 0.0
        motion_intensity = (motion_pixels / total_pixels) * 100.0
        return motion_intensity
