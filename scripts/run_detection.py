import cv2
from src.phone_detection import PhoneDetector
from src.detection import detect_face, get_eye_landmarks, calculate_ear, calibrate_ear
from src.analyzer import DrowsinessAnalyzer,PhoneAnalyzer
from src.logger import EventLogger

FRAME_THRESHOLD = 20

cap = cv2.VideoCapture(0)

# Calibrate to your face first
EAR_THRESHOLD = calibrate_ear(cap, detect_face, get_eye_landmarks, calculate_ear)

# Initialize the decision engine
analyzer = DrowsinessAnalyzer(ear_threshold=EAR_THRESHOLD, frame_threshold=FRAME_THRESHOLD)

phone_analyzer = PhoneAnalyzer()

#Initialize the event logger
logger = EventLogger()

#Initialize the phone detector
phone_detector = PhoneDetector()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    results = detect_face(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape)

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Update decision engine
            analyzer.update(avg_ear)
            status = analyzer.get_status()

            # Draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Display EAR and status
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if analyzer.is_new_event():
                 logger.log_event("DROWSY", avg_ear)

            color = (0, 0, 255) if status == "DROWSY" else (0, 255, 0)
            cv2.putText(frame, status, (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            

            # Phone detection
            phones = phone_detector.detect(frame)
            for phone in phones:
                x1, y1, x2, y2 = phone["bbox"]
                conf = phone["confidence"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if phone_analyzer.is_new_event(phones):
                    logger.log_event("PHONE_DETECTED", phones[0]["confidence"])
    

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
