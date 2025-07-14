'''
将json文件转为yolo所需要的txt文件。将未转换的标注放入labels文件夹中，图片放入images文件夹中
json中[x1,y1,x2,y2]，（x1,y1)表示目标左上角坐标，（x2,y2)表示目标右下角坐标，图片左上角坐标为（0，0）
yolo的txt中[class,x_center,y_center,width,height](需要根据图片宽高进行归一化处理）
yolo（类别 中心点的x 中心点的y 宽度w 高度h）
'''

import json
import os
from PIL import Image


def convert(img_size, box):  # 坐标转换
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    # 左上到右下，否则负数
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return x, y, w, h


def decode_json(json_floder_path, json_name):
    txt_name = r'G:ultralytics\data\Annotations\txt/' + json_name[0:-5] + '.txt'  # 生成txt文件存放的路径
    txt_file = open(txt_name, 'w')
    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='utf-8'))

    image_path = r'G:\study\study\目标检测\代码\yolov8-main\ultralytics-main\ultralytics\data\images/' + json_name[0:-5] + '.tif'  # 图片存放路径

    # 使用pillow读取图片，获取图片的宽和高
    img_pillow = Image.open(image_path)
    img_w = img_pillow.width  # 图片宽度
    img_h = img_pillow.height  # 图片高度

    # 'break', 'errorbreak', 'strip', 'whitebreak'
    for i in data['shapes']:

        if i['label'] == 'mine':  # 目标的类别
            x1, y1= i['points'][0]
            x2, y2= i['points'][1]

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write('0' + " " + " ".join([str(a) for a in bbox]) + '\n')  # 此处将该目标类别记为“0”

        if i['label'] == 'error_break':  # 目标的类别
            x1, y1 = i['points'][0]
            x2, y2 = i['points'][1]

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write('1' + " " + " ".join([str(a) for a in bbox]) + '\n')  # 此处将该目标类别记为“1”

        if i['label'] == 'strip':  # 目标的类别
            x1, y1 = i['points'][0]
            x2, y2 = i['points'][1]

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write('2' + " " + " ".join([str(a) for a in bbox]) + '\n')  # 此处将该目标类别记为“2”

        if i['label'] == 'white_break':  # 目标的类别
            x1, y1 = i['points'][0]
            x2, y2 = i['points'][1]

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write('3' + " " + " ".join([str(a) for a in bbox]) + '\n')  # 此处将该目标类别记为“3”


if __name__ == "__main__":

    json_floder_path = 'G:ultralytics\data\Annotations/'  # json文件的路径
    json_names = os.listdir(json_floder_path)
    for json_name in json_names:
        decode_json(json_floder_path, json_name)
