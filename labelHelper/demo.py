import os
import cv2
from ultralytics import YOLO

def visualize_folder_model(image_folder, modelPath, output_folder):
    model = YOLO(modelPath)
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png") or image_file.endswith(".bmp"):
            image_path = os.path.join(image_folder, image_file)
            img = model(image_path)  # 对图像进行预测

                # 构建保存路径
            output_filename = os.path.join(output_folder, os.path.basename(image_path))
            
            # 保存可视化图像到文件夹
            cv2.imwrite(output_filename, img)


image_folder = r'/home/ndvision/dl/datasets/chip/train/images'  # 替换为你的图像文件夹路径
modelPath ="/home/ndvision/dl/ultralytics/runs/segment/train2/weights/best.pt"  # 替换为你存储txt标注文件的文件夹路径
output_folder = r'/home/ndvision/dl/datasets/chip/display'   # 替换为你希望保存可视化图像的输出文件夹
visualize_folder_model(image_folder ,modelPath, output_folder)