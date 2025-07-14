from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8-prf.yaml')  # build a new mod el from YAML
    # model = YOLO('yolov8-p2-cut-rfe.yaml')
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Use the model0
    model.train(data='ultralytics/data/ore.yaml', epochs=200, imgsz=640, batch=8, device=0)  # 训练模型
