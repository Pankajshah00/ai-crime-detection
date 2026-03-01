# alert.py
import os
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_ENABLED = True
except ImportError:
    TWILIO_ENABLED = False


# ── Priority & labels for every event type ────────────────────────────────────
EVENT_META = {
    'weapon_detected': {'label': '🔫 Weapon Detected',   'priority': 'CRITICAL'},
    'fighting':        {'label': '⚡ Fighting Detected',  'priority': 'HIGH'},
    'robbery':         {'label': '🚨 Robbery / Mob',      'priority': 'CRITICAL'},
    'crowd_surge':     {'label': '📈 Crowd Surge',        'priority': 'MEDIUM'},
    'crowd':           {'label': '👥 Crowd Detected',     'priority': 'LOW'},
    'loitering':       {'label': '⏱ Loitering',          'priority': 'LOW'},
    'running_detected':{'label': '🏃 Running / Panic',    'priority': 'MEDIUM'},
    'high_motion':     {'label': '⚡ High Motion',        'priority': 'MEDIUM'},
}

# Only email for these priorities  (LOW events are logged only)
EMAIL_PRIORITIES = {'CRITICAL', 'HIGH', 'MEDIUM'}

# Per-event-type email cooldown (seconds) — avoid inbox flood
EMAIL_COOLDOWNS = {
    'weapon_detected':  30,
    'fighting':         45,
    'robbery':          60,
    'crowd_surge':      60,
    'running_detected': 60,
    'high_motion':      90,
    'crowd':           120,
    'loitering':       120,
}


class AlertManager:
    def __init__(self):
        # ── Email config ─────────────────────────────────────────────────
        self.email_from  = (os.getenv('ALERT_EMAIL_FROM') or '').strip()
        self.email_to    = (os.getenv('ALERT_EMAIL_TO') or '').strip()
        self.smtp_server = (os.getenv('SMTP_SERVER') or 'smtp.gmail.com').strip()
        self.smtp_port   = int((os.getenv('SMTP_PORT') or '587').strip())
        self.smtp_user   = (os.getenv('SMTP_USER') or '').strip()
        self.smtp_pass   = (os.getenv('SMTP_PASS') or '').strip().strip('"').strip("'")

        self._email_ready = all([
            self.email_from, self.email_to,
            self.smtp_server, self.smtp_user, self.smtp_pass
        ])
        if self._email_ready:
            print(f"[AlertManager] Email alerts → {self.email_to}")
        else:
            print("[AlertManager] Email not configured — set SMTP vars in .env")

        # ── Twilio SMS config ─────────────────────────────────────────────
        self.twilio_client = None
        self.twilio_from   = None
        self.alert_sms_to  = None
        if TWILIO_ENABLED:
            sid   = os.getenv('TWILIO_SID')
            token = os.getenv('TWILIO_TOKEN')
            self.twilio_from  = os.getenv('TWILIO_FROM')
            self.alert_sms_to = os.getenv('ALERT_SMS_TO')
            if sid and token:
                self.twilio_client = TwilioClient(sid, token)
                print(f"[AlertManager] SMS alerts → {self.alert_sms_to}")

        # ── Cooldown tracking ─────────────────────────────────────────────
        self._last_email: dict = {}   # event_type → last sent timestamp (epoch)
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def handle_event(self, event: dict, frame=None):
        """
        Called by app.py for every detected event.
        Sends email (and optionally SMS) in a background thread so the
        video pipeline is never blocked by network I/O.
        """
        etype    = event.get('type', 'unknown')
        meta     = EVENT_META.get(etype, {'label': etype.upper(), 'priority': 'LOW'})
        priority = meta['priority']
        label    = meta['label']
        ts       = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"[AlertManager] [{priority}] {label} at {ts}")

        if priority not in EMAIL_PRIORITIES:
            return   # skip LOW priority emails

        if not self._cooldown_ok(etype):
            return   # within cooldown window, skip

        # Encode frame snapshot if provided
        frame_bytes = None
        if frame is not None:
            try:
                import cv2
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buf.tobytes()
            except Exception:
                pass

        # Fire email in background thread — never blocks the video loop
        threading.Thread(
            target=self._send_email_safe,
            args=(event, label, priority, ts, frame_bytes),
            daemon=True
        ).start()

        # SMS for CRITICAL only
        if priority == 'CRITICAL' and self.twilio_client:
            threading.Thread(
                target=self._send_sms_safe,
                args=(label, event, ts),
                daemon=True
            ).start()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cooldown_ok(self, etype: str) -> bool:
        """Return True and update timestamp if cooldown has elapsed."""
        import time
        cooldown = EMAIL_COOLDOWNS.get(etype, 60)
        with self._lock:
            last = self._last_email.get(etype, 0)
            if time.time() - last >= cooldown:
                self._last_email[etype] = time.time()
                return True
        return False

    def _send_email_safe(self, event: dict, label: str,
                         priority: str, ts: str, frame_bytes=None):
        """Build and send a rich HTML email with optional snapshot attachment."""
        if not self._email_ready:
            return
        try:
            subject = f"[{priority}] {label} — AI Security Alert"
            html    = self._build_html(event, label, priority, ts, frame_bytes)
            self._send_email(subject, html, frame_bytes)
            print(f"[AlertManager] Email sent: {subject}")
        except Exception as e:
            print(f"[AlertManager] Email error: {e}")

    def _build_html(self, event: dict, label: str,
                    priority: str, ts: str, frame_bytes=None) -> str:
        priority_color = {
            'CRITICAL': '#ff3b3b',
            'HIGH':     '#ff6b00',
            'MEDIUM':   '#ffaa00',
            'LOW':      '#00b894',
        }.get(priority, '#888')

        # Build a detail table from whatever fields the event has
        skip = {'type', 'timestamp'}
        rows = ''
        for k, v in event.items():
            if k in skip:
                continue
            rows += f'<tr><td style="padding:4px 12px 4px 0;color:#888;font-size:13px;">{k}</td>'
            rows += f'<td style="padding:4px 0;font-size:13px;color:#eee;">{v}</td></tr>'

        return f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#0a0c10;font-family:'Segoe UI',Arial,sans-serif;">
<div style="max-width:560px;margin:32px auto;background:#111318;border-radius:10px;
            overflow:hidden;border:1px solid #1e2230;">

  <!-- Header bar -->
  <div style="background:{priority_color};padding:18px 24px;">
    <span style="font-size:22px;font-weight:700;color:#fff;letter-spacing:.03em;">
      {label}
    </span>
    <span style="float:right;background:rgba(0,0,0,.3);color:#fff;
                 font-size:11px;padding:4px 10px;border-radius:20px;margin-top:4px;">
      {priority}
    </span>
  </div>

  <!-- Body -->
  <div style="padding:24px;">
    <p style="color:#aaa;font-size:13px;margin:0 0 20px;">
      🕐 Detected at <strong style="color:#fff;">{ts}</strong>
    </p>

    <table style="width:100%;border-collapse:collapse;">
      {rows}
    </table>

    {"<p style='color:#aaa;font-size:12px;margin:20px 0 0;'>📸 Frame snapshot attached.</p>" if frame_bytes else ""}
  </div>

  <!-- Footer -->
  <div style="padding:14px 24px;border-top:1px solid #1e2230;
              font-size:11px;color:#444;text-align:center;">
    AI Suspicious Activity Detection System — automated alert
  </div>
</div>
</body>
</html>"""

    def _send_email(self, subject: str, html: str, frame_bytes=None):
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From']    = self.email_from
        msg['To']      = self.email_to

        alt = MIMEMultipart('alternative')
        alt.attach(MIMEText(html, 'html'))
        msg.attach(alt)

        if frame_bytes:
            img = MIMEImage(frame_bytes, _subtype='jpeg')
            img.add_header('Content-Disposition', 'attachment',
                           filename='alert_snapshot.jpg')
            msg.attach(img)

        with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()  # required after starttls for Gmail
            s.login(self.smtp_user, self.smtp_pass)
            s.send_message(msg)

    def _send_sms_safe(self, label: str, event: dict, ts: str):
        if not self.twilio_client:
            return
        try:
            body = f"ALERT: {label} at {ts}. Type: {event.get('type')}."
            self.twilio_client.messages.create(
                body=body,
                from_=self.twilio_from,
                to=self.alert_sms_to
            )
            print(f"[AlertManager] SMS sent to {self.alert_sms_to}")
        except Exception as e:
            print(f"[AlertManager] SMS error: {e}")