import os
import cv2
import json
import numpy as np

def json_to_label_image(json_file, output_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取图像的尺寸信息
    image_filename = data['imagePath']
    # 使用 os.path.join 构建完整的图片路径
    image_path = os.path.join(os.path.dirname(json_file), image_filename)

    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"图片文件 {image_path} 不存在，请检查路径。")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片文件 {image_path}，请检查文件完整性。")
        return

    height, width = image.shape[:2]

    # 创建空白的标签图片
    label_image = np.zeros((height, width), dtype=np.uint8)

    # 遍历所有的目标标注
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            xmin = int(min(points[0][0], points[1][0]))
            ymin = int(min(points[0][1], points[1][1]))
            xmax = int(max(points[0][0], points[1][0]))
            ymax = int(max(points[0][1], points[1][1]))

            # 在标签图片上绘制矩形框
            label_image[ymin:ymax, xmin:xmax] = 255

    # 保存标签图片
    base_name = os.path.basename(json_file).replace('.json', '')
    label_filename = f"label_{base_name}.png"
    label_path = os.path.join(output_folder, label_filename)
    cv2.imwrite(label_path, label_image)

def convert_jsons_to_label_images(json_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            json_to_label_image(json_path, output_folder)

json_folder = "annotations"  # 保存 JSON 标注文件的文件夹
output_folder = "labels"
convert_jsons_to_label_images(json_folder, output_folder)