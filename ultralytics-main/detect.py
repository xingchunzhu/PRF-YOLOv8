from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/train2/weights/best.pt')
# model = YOLO('yolov8s.pt')

# Run inference on the source
model.predict('ultralytics/data/ImageSets/test/images', save=True, conf=0.5, imgsz=640, line_width=1, visualize=False)