import time

class LoiterMonitor:
    def __init__(self, stationary_threshold_seconds=30, movement_threshold_px=10):
        self.stationary_threshold = stationary_threshold_seconds
        self.movement_threshold = movement_threshold_px
    
    def update(self, tracks, frame=None):
        events = []
        now = time.time()
        
        for tr in tracks:
            hist = getattr(tr, 'hist_positions', [])
            if len(hist) < 2:
                continue
            
            x0, y0, t0 = hist[0]
            x1, y1, t1 = hist[-1]
            
            dist = ((x1-x0)**2 + (y1-y0)**2)**0.5
            duration = t1 - t0 if (t1 - t0) > 0 else 0.001
            speed = dist / duration
            
            if speed < 1.0 and duration >= self.stationary_threshold:
                events.append({
                    'type': 'loitering',
                    'track_id': tr.tid,
                    'class': tr.class_name,
                    'bbox': tr.bbox,
                    'timestamp': now
                })
            
            if 'knife' in tr.class_name.lower() or 'gun' in tr.class_name.lower():
                events.append({
                    'type': 'weapon',
                    'track_id': tr.tid,
                    'class': tr.class_name,
                    'bbox': tr.bbox,
                    'timestamp': now
                })
        
        if len(tracks) > 8:
            events.append({
                'type': 'crowd',
                'count': len(tracks),
                'timestamp': now
            })
        
        return events
