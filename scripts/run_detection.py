import cv2
from src.detection import detect_face, get_eye_landmarks, calculate_ear

# Thresholds
EAR_THRESHOLD = 0.45
FRAME_THRESHOLD = 20  # number of consecutive frames before alert

cap = cv2.VideoCapture(0)
drowsy_counter = 0

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

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw eye landmarks
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Show EAR value on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Drowsiness check
            if avg_ear < EAR_THRESHOLD:
                drowsy_counter += 1
                if drowsy_counter >= FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSY!", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                drowsy_counter = 0

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()