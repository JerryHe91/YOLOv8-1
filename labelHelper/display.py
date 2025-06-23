import cv2
import numpy as np
import os

# 为不同的类别分配不同的颜色
def get_class_color(class_id):
    color_map = {
        0: (0, 255, 0),   # 类别 0，绿色
        1: (0, 0, 255),   # 类别 1，红色
        2: (255, 0, 0),   # 类别 2，蓝色
        3: (0, 255, 255), # 类别 3，黄色
        4: (255, 0, 255), # 类别 4，品红色
        5: (255, 255, 0), # 类别 5，青色
    }
    return color_map.get(class_id, (255, 255, 255))  # 默认白色

def __rectangle_points_to_polygon(points):
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    if points[0][0] > points[1][0]:
        xmax = points[0][0]
        ymax = points[0][1]
        xmin = points[1][0]
        ymin = points[1][1]
    else:
        xmax = points[1][0]
        ymax = points[1][1]
        xmin = points[0][0]
        ymin = points[0][1]
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

def __points_to_bbox(points):
    points = np.array(points, dtype=np.float32)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    y_max = np.max(points[:, 1])
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h

# 解析 YOLO-seg 的 TXT 文件
def parse_yolov5seg_txt(txt_file, image_width, image_height):
    annotations = []
    
    # 读取txt文件中的标注
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip().split()
        
        # 解析每一行数据
        class_id = int(line[0])
        # x_center = float(line[1])
        # y_center = float(line[2])
        # width = float(line[3])
        # height = float(line[4])
        
        # 获取多边形点坐标
        polygon_points = []
        for i in range(1, len(line), 2):
            x = float(line[i]) * image_width
            y = float(line[i+1]) * image_height
            polygon_points.append((int(x), int(y)))
        
        p = __rectangle_points_to_polygon(polygon_points)
        x_center, y_center, width, height = __points_to_bbox(polygon_points)
        # 保存解析后的数据
        annotations.append((class_id, x_center, y_center, width, height, polygon_points))
    
    return annotations

# 在图像上绘制目标框和分割多边形，并保存可视化图像
def visualize_annotations(image_path, txt_file, output_folder):
    # 读取图像
    img = cv2.imread(image_path)
    image_height, image_width, _ = img.shape
    
    # 解析YOLOv5-seg的标注文件
    annotations = parse_yolov5seg_txt(txt_file, image_width, image_height)
    
    # 绘制每一个标注
    for annotation in annotations:
        class_id, x_center, y_center, width, height, polygon_points = annotation
        
        # 获取类别对应的颜色
        color = get_class_color(class_id)
        
        # 计算目标框的左上角和右下角
        # x1 = int((x_center - width / 2) * image_width)
        # y1 = int((y_center - height / 2) * image_height)
        # x2 = int((x_center + width / 2) * image_width)
        # y2 = int((y_center + height / 2) * image_height)
        
        x1 = int((x_center - width / 2) )
        y1 = int((y_center - height / 2))
        x2 = int((x_center + width / 2) )
        y2 = int((y_center + height / 2) )

        # 绘制边界框（使用类别的颜色）（可以对这一小部分进行注即不显示框）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制多边形（使用类别的颜色）
        polygon_points = np.array(polygon_points, np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        cv2.polylines(img, [polygon_points], isClosed=True, color=color, thickness=2)
        
        # 在目标框中心添加类别标签（颜色与目标框相同）（可以对这一小部分进行注即不显示标签类型）
        label = f"Class {class_id}" 
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 构建保存路径
    output_filename = os.path.join(output_folder, os.path.basename(image_path))
    
    # 保存可视化图像到文件夹
    cv2.imwrite(output_filename, img)

# 批量可视化图像和标注文件，并保存结果
def visualize_folder(image_folder, annotation_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".bmp"):
            image_path = os.path.join(image_folder, image_file)
            txt_file = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + ".txt")
            
            if os.path.exists(txt_file):
                visualize_annotations(image_path, txt_file, output_folder)
            else:
                print(f"Warning: Annotation file for {image_file} not found.")

# 设置图像、标注文件和输出文件夹路径
image_folder = "/home/ndvision/dl/ultralytics/datasets/p0304/images/train"  # 替换为你的图像文件夹路径
annotation_folder = "/home/ndvision/dl/ultralytics/datasets/p0304/labels/train"  # 替换为你存储txt标注文件的文件夹路径
output_folder = "/home/ndvision/dl/ultralytics/datasets/p0304/show"  # 替换为你希望保存可视化图像的输出文件夹

image_folder = "/home/ndvision/dl/datasets/chip/images/train"  # 替换为你的图像文件夹路径
annotation_folder = "/home/ndvision/dl/ultralytics/datasets/tablet/labels"  # 替换为你存储txt标注文件的文件夹路径
output_folder = "/home/ndvision/dl/ultralytics/datasets/tablet/show"  # 替换为你希望保存可视化图像的输出文件夹
#路径


# 可视化图像和标注，并保存结果
visualize_folder(image_folder, annotation_folder, output_folder)
