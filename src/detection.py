import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

# MediaPipe landmark indices for each eye
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

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