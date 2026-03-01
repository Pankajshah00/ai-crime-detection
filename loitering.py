# loitering.py
import time

class LoiterMonitor:
    def __init__(self, stationary_threshold_seconds: float = 30,
                 movement_threshold_px: float = 10):
        self.stationary_threshold = stationary_threshold_seconds
        self.movement_threshold   = movement_threshold_px

        # Weapon class names that should trigger a weapon alert here
        # (complements detector.py upstream detections)
        self._weapon_names = {
            'gun', 'pistol', 'rifle', 'firearm', 'handgun',
            'shotgun', 'knife', 'weapon', 'sword',
        }

        # Cooldown state per track / global
        self._loiter_alerted: dict = {}   # tid → last alert time
        self._weapon_alerted: dict = {}   # tid → last alert time
        self._crowd_alerted_at = 0.0
        self._run_alerted_at   = 0.0

        # Cooldown durations (seconds)
        self.loiter_cooldown = 30
        self.weapon_cooldown = 8
        self.crowd_cooldown  = 20
        self.run_cooldown    = 10

        # Crowd threshold
        self.crowd_threshold = 8

        # Running detection: speed (px/s) above which a person is "running"
        self.run_speed_threshold = 180   # tune per camera resolution

        # How many consecutive running tracks needed to alert
        self.run_count_threshold = 2

    def update(self, tracks, frame=None) -> list:
        events = []
        now    = time.time()
        active_ids = {tr.tid for tr in tracks}

        for tr in tracks:
            hist     = getattr(tr, 'hist_positions', [])
            cls_low  = tr.class_name.lower()

            # ── Loitering ────────────────────────────────────────────────
            if len(hist) >= 2:
                x0, y0, t0 = hist[0]
                x1, y1, t1 = hist[-1]
                dist     = ((x1-x0)**2 + (y1-y0)**2) ** 0.5
                duration = max(t1 - t0, 0.001)
                speed    = dist / duration

                if speed < 1.0 and duration >= self.stationary_threshold:
                    last = self._loiter_alerted.get(tr.tid, 0)
                    if now - last >= self.loiter_cooldown:
                        self._loiter_alerted[tr.tid] = now
                        events.append({
                            'type':      'loitering',
                            'track_id':  tr.tid,
                            'class':     tr.class_name,
                            'bbox':      tr.bbox,
                            'duration':  round(duration, 1),
                            'timestamp': now,
                        })

            # ── Weapon from tracker flag ──────────────────────────────────
            weapon_by_name = any(w in cls_low for w in self._weapon_names)
            weapon_by_flag = getattr(tr, 'is_weapon', False)
            if weapon_by_name or weapon_by_flag:
                last = self._weapon_alerted.get(tr.tid, 0)
                if now - last >= self.weapon_cooldown:
                    self._weapon_alerted[tr.tid] = now
                    events.append({
                        'type':      'weapon_detected',
                        'track_id':  tr.tid,
                        'class':     tr.class_name,
                        'bbox':      tr.bbox,
                        'timestamp': now,
                    })

        # ── Running / panic detection ─────────────────────────────────────
        running_count = 0
        for tr in tracks:
            hist = getattr(tr, 'hist_positions', [])
            if len(hist) >= 2 and tr.class_name.lower() == 'person':
                x0, y0, t0 = hist[-min(len(hist), 6)]   # last ~6 frames
                x1, y1, t1 = hist[-1]
                duration = max(t1 - t0, 0.001)
                speed = ((x1-x0)**2 + (y1-y0)**2)**0.5 / duration
                if speed >= self.run_speed_threshold:
                    running_count += 1

        if running_count >= self.run_count_threshold:
            if now - self._run_alerted_at >= self.run_cooldown:
                self._run_alerted_at = now
                events.append({
                    'type':          'running_detected',
                    'running_count': running_count,
                    'timestamp':     now,
                })

        # ── Crowd ─────────────────────────────────────────────────────────
        if len(tracks) > self.crowd_threshold:
            if now - self._crowd_alerted_at >= self.crowd_cooldown:
                self._crowd_alerted_at = now
                events.append({
                    'type':      'crowd',
                    'count':     len(tracks),
                    'timestamp': now,
                })

        # Clean stale cooldown entries
        self._loiter_alerted = {k: v for k, v in self._loiter_alerted.items() if k in active_ids}
        self._weapon_alerted = {k: v for k, v in self._weapon_alerted.items() if k in active_ids}

        return events