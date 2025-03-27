from ultralytics import YOLO

# Load YOLOv10 model
model = YOLO('yolov10n.pt')

# Train the model
model.train(data=r'/content/drive/MyDrive/bone-fracture-detection/data.yaml', epochs=100, imgsz=640, batch=16, device="cuda")