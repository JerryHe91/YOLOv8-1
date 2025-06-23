
from ultralytics import YOLO

struct_file = "/home/ndvision/dl/ultralytics/train/segment/chip/yolov8-seg.yaml"
# 加载一个预训练的 YOLO11n 模型
model = YOLO(struct_file).load('yolo11n-seg.pt')
# model = YOLO("/home/ndvision/dl/ultralytics/runs/segment/train17/weights/best.pt")

# 在 COCO8 数据集上训练模型 100 个周期
train_results = model.train(
    data="/home/ndvision/dl/ultralytics/train/segment/chip/dataset.yaml",  # 数据集配置文件路径
    batch = 1,
    epochs=50,  # 训练周期数
    imgsz=640*2,  # 训练图像尺寸
    device="0",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
    mosaic= 0,
    close_mosaic=0,
)

# 评估模型在验证集上的性能
metrics = model.val()

# # 对图像执行目标检测
# results = model("path/to/image.jpg")  # 对图像进行预测
# results[0].show()  # 显示结果

# 将模型导出为 ONNX 格式以进行部署
path = model.export(format="onnx")  # 返回导出模型的路径