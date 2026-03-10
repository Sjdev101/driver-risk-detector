from ultralytics import YOLO

class PhoneDetector:
    def __init__(self):
        # Load pretrained YOLOv8 model
        self.model = YOLO("yolov8n.pt")  # n = nano, smallest and fastest
        self.phone_class_id = 67  # YOLO's class ID for cell phone

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        phones_detected = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id == self.phone_class_id and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    phones_detected.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence
                    })

        return phones_detected