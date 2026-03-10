import cv2
from src.detection import detect_face, get_eye_landmarks, calculate_ear, calibrate_ear

FRAME_THRESHOLD = 20

cap = cv2.VideoCapture(0)
drowsy_counter = 0

# Calibrate to your face first
EAR_THRESHOLD = calibrate_ear(cap, detect_face, get_eye_landmarks, calculate_ear)

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

            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {EAR_THRESHOLD:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if avg_ear < EAR_THRESHOLD:
                drowsy_counter += 1
                if drowsy_counter >= FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSY!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                drowsy_counter = 0

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
