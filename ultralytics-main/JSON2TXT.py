import json
import os

def convert_json_to_yolo(json_file_path, output_dir, class_mapping):
    """
    将单个JSON文件转换为YOLO格式的TXT文件
    :param json_file_path: JSON文件路径
    :param output_dir: 输出目录
    :param class_mapping: 类别名称到类别索引的映射
    """
    with open(json_file_path, 'r', encoding='utf-8') as json_file:  # 显式指定文件编码为utf-8
        data = json.load(json_file)

    # 获取图片宽度和高度
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 创建输出文件名
    txt_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.txt'
    txt_file_path = os.path.join(output_dir, txt_file_name)

    with open(txt_file_path, 'w') as txt_file:
        for shape in data['shapes']:
            # 获取类别名称
            class_name = shape['label']
            if class_name not in class_mapping:
                raise ValueError(f"类别 '{class_name}' 未在类别映射中找到。请检查类别映射是否正确。")

            class_id = class_mapping[class_name]

            # 获取标注框坐标
            points = shape['points']
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])

            # 转换为YOLO格式
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 写入TXT文件
            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
def main():
    # 输入和输出目录
    # input_dir = r"D:\picture\CI1\labels1\val"
    # output_dir = r"D:\picture\CI1\labels\val"
    input_dir = r"G:ultralytics\data\Annotations/"
    output_dir = r"G:ultralytics\data\labels/"
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 类别映射（根据你的数据集修改）
    class_mapping = {
        # "mine": 0
        "coal": 0,
        "gangue": 1,
        # 添加更多类别
    }

    # 遍历输入目录中的所有JSON文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_dir, filename)
            convert_json_to_yolo(json_file_path, output_dir, class_mapping)
            print(f"已转换 {filename} 到 {output_dir}")

if __name__ == "__main__":
    main()

