import json
import os
from xml.dom.minidom import parseString
import xml.dom.minidom
import cv2
import numpy as np
from tqdm import tqdm

json_path = r"G:\study\study\目标检测\代码\yolov8-main\ultralytics-main\ultralytics\data\Annotations\0.json"
xml_path = r"G:\study\study\目标检测\代码\yolov8-main\ultralytics-main\ultralytics\data\Annotations"
jpgs_path = r"G:\study\study\目标检测\代码\yolov8-main\ultralytics-main\ultralytics\data\Annotations"
jdata = json.load(open(json_path))["labels"]

_ROOT_NODE = 'annotation'
_FOLDER_NODE = 'voc2007'


# 封装创建节点的过程
def createElementNode(doc, tag, attr):  # 创建一个元素节点
    element_node = doc.createElement(tag)
    # 创建一个文本节点
    text_node = doc.createTextNode(attr)
    # 将文本节点作为元素节点的子节点
    element_node.appendChild(text_node)
    return element_node


def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)
    parent_node.appendChild(child_node)


if __name__ == '__main__':
    for s_jd in tqdm(jdata):
        xml_filename = os.path.join(xml_path, s_jd["filename"].split(".")[0] + ".xml")
        img_path = os.path.join("\\".join(json_path.split("\\")[:-1]), s_jd["filename"])
        im = cv2.imdecode(np.fromfile(img_path), 1)
        if im is None:
            im = cv2.imread(img_path, 1)
        cv2.imwrite(os.path.join(jpgs_path, s_jd["filename"]), im)
        height, width, channel = im.shape

        """创建xml框架"""
        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None, _ROOT_NODE, None)
        # 获得根节点
        root_node = doc.documentElement
        # folder节点
        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)
        # filename节点
        createChildNode(doc, 'filename', s_jd["filename"], root_node)
        print("正在创建各个结点中...")
        # size节点
        size_node = doc.createElement('size')
        # size子节点
        createChildNode(doc, 'width', str(width), size_node)
        createChildNode(doc, 'height', str(height), size_node)
        createChildNode(doc, 'depth', str(channel), size_node)
        root_node.appendChild(size_node)

        labels = []
        xmin, ymin, xmax, ymax = [], [], [], []
        for i in range(len(s_jd["annotations"])):
            if "x" not in s_jd["annotations"][i] or "y" not in s_jd["annotations"][i] or "width" not in \
                    s_jd["annotations"][i] or "height" not in s_jd["annotations"][i]:
                labels.append(0), xmin.append(0), ymin.append(0), xmax.append(0), ymax.append(0)
                continue
            xmin.append(s_jd["annotations"][i]["x"])
            ymin.append(s_jd["annotations"][i]["y"])
            xmax.append(s_jd["annotations"][i]["x"] + s_jd["annotations"][i]["width"])
            ymax.append(s_jd["annotations"][i]["y"] + s_jd["annotations"][i]["height"])
            labels.append(s_jd["annotations"][i]["class"])

            # object节点
            object_node = doc.createElement('object')
            # object的子节点
            createChildNode(doc, 'name', str(labels[i]), object_node)
            createChildNode(doc, 'pose', 'undefine', object_node)
            createChildNode(doc, 'truncated', 'undefine', object_node)
            createChildNode(doc, 'difficute', str(0), object_node)
            # 创建object子节点bndbox的子节点
            bndbox_node = doc.createElement('bndbox')
            createChildNode(doc, 'xmin', str(xmin[i]), bndbox_node)
            createChildNode(doc, 'ymin', str(ymin[i]), bndbox_node)
            createChildNode(doc, 'xmax', str(xmax[i]), bndbox_node)
            createChildNode(doc, 'ymax', str(ymax[i]), bndbox_node)
            object_node.appendChild(bndbox_node)
            root_node.appendChild(object_node)

        with open('tmp.xml', 'w', encoding='utf-8') as f:
            doc.writexml(f, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        f.close()
        fin = open('tmp.xml')
        fout = open(xml_filename, 'w')
        lines = fin.readlines()
        for line in lines[1:]:
            if line.split():
                fout.writelines(line)
        fin.close()
        fout.close()

