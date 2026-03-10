from ultralytics import YOLO

def export_to_onnx():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    print("Exporting to ONNX...")
    model.export(format="onnx")
    print("Export complete! yolov8n.onnx created.")

if __name__ == "__main__":
    export_to_onnx()
