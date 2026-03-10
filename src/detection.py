import cv2
import mediapipe as mp
from scipy.spatial import distance

mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

# 6 points per eye: [left corner, top-left, top-right, right corner, bottom-right, bottom-left]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

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
