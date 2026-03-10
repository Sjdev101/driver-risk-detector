import cv2
import mediapipe as mp

# Initialize MediaPipe face detection
mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

def detect_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb_frame)
    return results
