import json
import os
from datetime import datetime

class EventLogger:
    def __init__(self, log_path="logs/events.json"):
        self.log_path = log_path
        self.events = []

        # Create logs folder if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_event(self, event_type, confidence):
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "confidence": round(confidence, 2)
        }
        self.events.append(event)
        self._save()
        print(f"Event logged: {event}")

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.events, f, indent=4)