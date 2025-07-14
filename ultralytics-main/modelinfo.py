from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8-p2-cut-rfe.yaml')  # build a new model from YAML
    model = YOLO('yolov8-prc.yaml')
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Use the model
    # results = model.train(data='ultralytics/data/ore.yaml', epochs=200, imgsz=1280, batch=16)  # 训练模型
    print(model.info(detailed=True))