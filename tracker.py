import cv2
import time

class Track:
    def __init__(self, tid, bbox, class_name):
        self.tid = tid
        self.bbox = bbox
        self.class_name = class_name
        self.last_seen = 0
        self.hist_positions = []
        self.is_weapon = False

class Tracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
    
    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        
        if areaA+areaB-inter == 0:
            return 0
        
        return inter / (areaA+areaB-inter)
    
    def update(self, detections, frame=None):
        assigned = set()
        new_tracks = {}
        
        for det in detections:
            bbox = det['bbox']
            cls_name = det['name']
            is_weapon = det.get('is_weapon', False)
            
            best_id = None
            best_iou = 0.2
            
            for tid, tr in self.tracks.items():
                i = self.iou(bbox, tr.bbox)
                if i > best_iou:
                    best_iou = i
                    best_id = tid
            
            if best_id is not None:
                tr = self.tracks[best_id]
                tr.bbox = bbox
                tr.last_seen = 0
                tr.is_weapon = is_weapon
                tr.hist_positions.append(((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2, time.time()))
                new_tracks[best_id] = tr
                assigned.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                tr = Track(tid, bbox, cls_name)
                tr.last_seen = 0
                tr.is_weapon = is_weapon
                tr.hist_positions = [((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2, time.time())]
                new_tracks[tid] = tr
        
        for tid, tr in self.tracks.items():
            if tid not in assigned:
                tr.last_seen += 1
                if tr.last_seen < 30:
                    new_tracks[tid] = tr
        
        self.tracks = new_tracks
        return list(self.tracks.values())
    
    def draw_tracks(self, frame, tracks):
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.bbox)
            
            # Red box for weapons, green for others
            color = (0, 0, 255) if tr.is_weapon else (0, 255, 0)
            thickness = 3 if tr.is_weapon else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"ID:{tr.tid} {tr.class_name}"
            if tr.is_weapon:
                label += " ⚠️ WEAPON"
            
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
