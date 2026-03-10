import time
import cv2
import torch
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np

def benchmark_pytorch(frame, runs=100):
    model = YOLO("yolov8n.pt")
    
    # Warmup
    model(frame, verbose=False)
    
    start = time.time()
    for _ in range(runs):
        model(frame, verbose=False)
    end = time.time()
    
    fps = runs / (end - start)
    print(f"PyTorch FPS: {fps:.2f}")
    return fps

def benchmark_onnx(frame, runs=100):
    session = ort.InferenceSession("yolov8n.onnx")
    
    # Prepare input
    img = cv2.resize(frame, (640, 640))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    input_name = session.get_inputs()[0].name
    
    # Warmup
    session.run(None, {input_name: img})
    
    start = time.time()
    for _ in range(runs):
        session.run(None, {input_name: img})
    end = time.time()
    
    fps = runs / (end - start)
    print(f"ONNX Runtime FPS: {fps:.2f}")
    return fps

if __name__ == "__main__":
    # Use a sample frame from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not grab frame")
        exit()

    print("Benchmarking 100 inference runs each...\n")
    pytorch_fps = benchmark_pytorch(frame)
    onnx_fps = benchmark_onnx(frame)

    improvement = ((onnx_fps - pytorch_fps) / pytorch_fps) * 100
    print(f"\nONNX is {improvement:.1f}% faster than PyTorch")