
import sys
import os
import cv2
from ultralytics import YOLO

def test_img():
	# 训练好的模型权重路径
    model = YOLO(r'G:\study\study\目标检测\代码\yolov8-main\ultralytics-main\runs\detect\train\weights\best.pt')
    # 测试图片的路径
    img = cv2.imread(r'G:\study\study\目标检测\代码\yolov8-main\ultralytics-main\ultralytics\data\ImageSets\test\images')
    res = model(img)
    ann = res[0].plot()
    while True:
        cv2.imshow("yolo", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 设置保存图片的路径
    cur_path = sys.path[0]
    print(cur_path, sys.path)

    if os.path.exists(cur_path):
        cv2.imwrite(cur_path + "out.jpg", ann)
    else:
        os.mkdir(cur_path)
        cv2.imwrite(cur_path + "out.jpg", ann)

test_img()
