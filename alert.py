import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

try:
    from twilio.rest import Client
    TWILIO_ENABLED = True
except ImportError:
    TWILIO_ENABLED = False

class AlertManager:
    def __init__(self):
        self.email_from = os.getenv('ALERT_EMAIL_FROM')
        self.email_to = os.getenv('ALERT_EMAIL_TO')
        self.smtp_server = os.getenv('SMTP_SERVER')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_user = os.getenv('SMTP_USER')
        self.smtp_pass = os.getenv('SMTP_PASS')
        
        if TWILIO_ENABLED:
            self.twilio_sid = os.getenv('TWILIO_SID')
            self.twilio_token = os.getenv('TWILIO_TOKEN')
            self.twilio_from = os.getenv('TWILIO_FROM')
            self.twilio_client = Client(self.twilio_sid, self.twilio_token) if self.twilio_sid and self.twilio_token else None
    
    def send_email(self, subject, body, image_bytes=None):
        if not self.smtp_server:
            print('SMTP not configured, skipping email')
            return
        
        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg.set_content(body)
            
            if image_bytes:
                msg.add_attachment(image_bytes, maintype='image', subtype='jpeg', filename='event.jpg')
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as s:
                s.starttls()
                s.login(self.smtp_user, self.smtp_pass)
                s.send_message(msg)
        except Exception as e:
            print(f'Email send error: {e}')
    
    def send_sms(self, body):
        if not TWILIO_ENABLED or not getattr(self, 'twilio_client', None):
            print('Twilio not configured, skipping SMS')
            return
        
        try:
            to_number = os.getenv('ALERT_SMS_TO')
            self.twilio_client.messages.create(body=body, from_=self.twilio_from, to=to_number)
        except Exception as e:
            print(f'SMS send error: {e}')
    
    def handle_event(self, event, frame=None):
        ts = datetime.utcnow().isoformat()
        t = event.get('type')
        body = f"Event: {t} at {ts} - {event}"
        print('ALERT:', body)
