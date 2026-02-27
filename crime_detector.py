# crime_detector.py
import time

class CrimeDetector:
    def __init__(self):
        self.motion_history = []
        self.max_history = 30
        self.prev_frame = None
        # threshold (percent of changed pixels) to consider violent motion high
        self.violence_threshold = 6.0  # tune this (6% changed pixels is an example)

        # keep counters of close encounters per pair (simple approach)
        self.close_frames = {}  # key: (id1,id2) -> consecutive frames they were close

    def detect_fighting(self, tracks, motion_intensity):
        """
        tracks: list of objects that at least contain .tid and .bbox (x1,y1,x2,y2)
        motion_intensity: float percent of changed pixels
        returns list of events dict
        """
        events = []
        now = time.time()

        # 1) Weapon detection in tracks (if trackers set is_weapon flag)
        for tr in tracks:
            if getattr(tr, "is_weapon", False):
                events.append({
                    'type': 'weapon_detected',
                    'track_id': tr.tid,
                    'class': tr.class_name,
                    'bbox': tr.bbox,
                    'timestamp': now
                })

        # 2) Fighting detection: if motion intensity is high and two persons are repeatedly close
        fighting_distance = 120  # pixels, tune for your resolution
        required_close_frames = 4  # frames in a row to consider fight

        # consider pairs
        for i in range(len(tracks)):
            for j in range(i+1, len(tracks)):
                t1 = tracks[i]; t2 = tracks[j]
                # only consider pairs of people (class_name 'person')
                if t1.class_name.lower() != 'person' or t2.class_name.lower() != 'person':
                    continue
                d = self._distance_between(t1.bbox, t2.bbox)
                key = tuple(sorted((t1.tid, t2.tid)))
                if d < fighting_distance and motion_intensity >= self.violence_threshold:
                    self.close_frames[key] = self.close_frames.get(key, 0) + 1
                else:
                    # reset counter if not close or not much motion
                    self.close_frames[key] = 0

                if self.close_frames.get(key, 0) >= required_close_frames:
                    events.append({
                        'type': 'fighting',
                        'pair': key,
                        'distance': d,
                        'motion_intensity': motion_intensity,
                        'timestamp': now
                    })
                    # reset to avoid repeated spam
                    self.close_frames[key] = 0

        # 3) maintain motion history
        self.update_motion_history(motion_intensity)
        return events

    def _distance_between(self, bbox1, bbox2):
        # bbox format: [x1,y1,x2,y2]
        x1_center = (bbox1[0] + bbox1[2]) / 2.0
        y1_center = (bbox1[1] + bbox1[3]) / 2.0
        x2_center = (bbox2[0] + bbox2[2]) / 2.0
        y2_center = (bbox2[1] + bbox2[3]) / 2.0
        return ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5

    def update_motion_history(self, motion_intensity):
        self.motion_history.append(motion_intensity)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)

    def get_average_motion(self):
        if not self.motion_history:
            return 0.0
        return sum(self.motion_history) / len(self.motion_history)
