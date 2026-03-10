import cv2
import mediapipe as mp
from scipy.spatial import distance

mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

# 6 points per eye: [left corner, top-left, top-right, right corner, bottom-right, bottom-left]
LEFT_EYE  = [362, 386, 387, 263, 374, 385]
RIGHT_EYE = [33, 159, 160, 133, 145, 158]

def detect_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb_frame)
    return results

def get_eye_landmarks(face_landmarks, frame_shape):
    h, w = frame_shape[:2]
    left_eye = []
    right_eye = []

    for idx in LEFT_EYE:
        lm = face_landmarks.landmark[idx]
        left_eye.append((int(lm.x * w), int(lm.y * h)))

    for idx in RIGHT_EYE:
        lm = face_landmarks.landmark[idx]
        right_eye.append((int(lm.x * w), int(lm.y * h)))

    return left_eye, right_eye

def calculate_ear(eye_landmarks):
    # Vertical distances (2 pairs)
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])

    ear = (A + B) / (2.0 * C)
    return ear

def calibrate_ear(cap, detect_face, get_eye_landmarks, calculate_ear):
    print("Calibrating... Look at the camera and keep eyes open for 3 seconds")
    ear_values = []
    calibration_frames = 90  # 3 seconds at 30fps

    while len(ear_values) < calibration_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        results = detect_face(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape)
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                ear_values.append(avg_ear)

        # Show countdown on screen
        remaining = calibration_frames - len(ear_values)
        cv2.putText(frame, f"Calibrating... {remaining} frames left", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Driver Monitor", frame)
        cv2.waitKey(1)

    baseline_ear = sum(ear_values) / len(ear_values)
    threshold = baseline_ear * 0.75
    print(f"Calibration done. Baseline EAR: {baseline_ear:.2f} Threshold: {threshold:.2f}")
    return threshold
