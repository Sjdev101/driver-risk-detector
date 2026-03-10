import cv2
import mediapipe as mp
from src.detection import detect_face, get_eye_landmarks

mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

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

            # Draw eye landmarks as green dots
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
