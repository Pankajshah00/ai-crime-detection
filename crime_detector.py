# crime_detector.py
import time
import numpy as np

class CrimeDetector:
    def __init__(self):
        self.motion_history   = []
        self.max_history      = 45          # ~1.5 s at 30 fps
        self.close_frames     = {}          # (id1,id2) → consecutive close frames

        # ── Tunable thresholds ────────────────────────────────────────────
        self.violence_threshold   = 5.0     # % changed pixels = vigorous motion
        self.fighting_distance    = 130     # px – centre-to-centre proximity
        self.required_close_frames= 4       # frames in a row to declare fight

        # Robbery / flash-mob: many people enter quickly
        self.robbery_person_threshold = 4   # ≥ N persons in frame
        self.robbery_motion_threshold = 8.0 # AND high motion
        self._robbery_frames          = 0
        self.robbery_required_frames  = 3   # must sustain N frames

        # Sudden crowd surge: person count jumps by ≥ K in one window
        self._person_history = []           # rolling person-count buffer
        self._surge_window   = 10           # frames to look back
        self.surge_jump      = 3            # person-count increase to flag

        # Cooldowns (seconds) – prevent duplicate alerts
        self._last = {}
        self.cooldowns = {
            'fighting':   8,
            'robbery':   15,
            'crowd_surge':12,
        }

    # ── public entry point ────────────────────────────────────────────────────
    def detect_fighting(self, tracks, motion_intensity: float) -> list:
        """
        Master event detector.  Returns a list of event dicts covering:
          - fighting
          - robbery / flash-mob
          - crowd_surge
        Weapon events are handled upstream in detector.py / loitering.py.
        """
        events = []
        now    = time.time()

        persons = [t for t in tracks if t.class_name.lower() == 'person']
        n_persons = len(persons)

        # ── 1. Fighting ───────────────────────────────────────────────────
        if motion_intensity >= self.violence_threshold:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    t1, t2 = persons[i], persons[j]
                    dist = self._centroid_dist(t1.bbox, t2.bbox)
                    key  = tuple(sorted((t1.tid, t2.tid)))

                    if dist < self.fighting_distance:
                        self.close_frames[key] = self.close_frames.get(key, 0) + 1
                    else:
                        self.close_frames[key] = 0

                    if self.close_frames.get(key, 0) >= self.required_close_frames:
                        if self._can_alert('fighting', now):
                            events.append({
                                'type':             'fighting',
                                'pair':             key,
                                'distance':         round(dist, 1),
                                'motion_intensity': round(motion_intensity, 1),
                                'timestamp':        now,
                            })
                        self.close_frames[key] = 0   # reset to avoid re-spam

        # ── 2. Robbery / flash-mob ────────────────────────────────────────
        # Many people + very high motion = smash-and-grab / mob rush
        if (n_persons >= self.robbery_person_threshold and
                motion_intensity >= self.robbery_motion_threshold):
            self._robbery_frames += 1
        else:
            self._robbery_frames = max(0, self._robbery_frames - 1)

        if self._robbery_frames >= self.robbery_required_frames:
            if self._can_alert('robbery', now):
                events.append({
                    'type':        'robbery',
                    'person_count': n_persons,
                    'motion':      round(motion_intensity, 1),
                    'timestamp':   now,
                })

        # ── 3. Crowd surge ────────────────────────────────────────────────
        self._person_history.append(n_persons)
        if len(self._person_history) > self._surge_window:
            self._person_history.pop(0)

        if len(self._person_history) == self._surge_window:
            baseline = min(self._person_history[:self._surge_window // 2])
            current  = max(self._person_history[self._surge_window // 2:])
            if current - baseline >= self.surge_jump:
                if self._can_alert('crowd_surge', now):
                    events.append({
                        'type':      'crowd_surge',
                        'from':      baseline,
                        'to':        current,
                        'timestamp': now,
                    })

        self._update_motion_history(motion_intensity)
        return events

    # ── helpers ───────────────────────────────────────────────────────────────
    def _can_alert(self, kind: str, now: float) -> bool:
        last = self._last.get(kind, 0)
        if now - last >= self.cooldowns.get(kind, 10):
            self._last[kind] = now
            return True
        return False

    @staticmethod
    def _centroid_dist(b1, b2) -> float:
        cx1 = (b1[0] + b1[2]) / 2.0;  cy1 = (b1[1] + b1[3]) / 2.0
        cx2 = (b2[0] + b2[2]) / 2.0;  cy2 = (b2[1] + b2[3]) / 2.0
        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

    def _update_motion_history(self, v: float):
        self.motion_history.append(v)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)

    def get_average_motion(self) -> float:
        return sum(self.motion_history) / len(self.motion_history) if self.motion_history else 0.0