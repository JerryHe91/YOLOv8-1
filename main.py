from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import  cv2
# https://blog.csdn.net/qq_37553692/article/details/130898732
if __name__ == '__main__':
    # Create a new YOLO model from scratch
    # model = YOLO('yolov8n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('yolov8n-seg.pt')
    #
    # # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # results = model.train(data='coco128.yaml', epochs=3,batch= 1,workers= 0)

    base_folder = str(Path(__file__).parent.resolve())
    train_folder = base_folder + '/datasets/tablet/'
    build_file = train_folder + 'yolov8-seg-tablet.yaml'
    yaml_file = train_folder+ 'tablet-seg.yaml'
    test_img = train_folder+'/images/1.png'


    pre_model = 'yolov8n-seg.pt'
    # pre_model = base_folder+'/runs/train/exp3/weights/best.pt'

    # 加载模型
    # model = YOLO(r'yolov8-seg.yaml')  # 不使用预训练权重训练
    # model = YOLO(build_file).load("yolov8n-seg.pt")  # 使用预训练权重训练
    # model = YOLO(build_file).load(pre_model)  # 使用预训练权重训练
    # model = YOLO(pre_model)  # 使用预训练权重训练
    model = YOLO(build_file)
    # model._load(pre_model)

    # results = model.train(data=r'F:/work/Python/YOLOV8/datasets/tablet/tablet-seg.yaml',imgsz=(256,1024), epochs=3,batch= 1,workers= 0)

    # https://docs.ultralytics.com/modes/predict/#inference-arguments
    # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(
        data=yaml_file,
        epochs=1,  # (int) 训练的周期数
        patience=50,  # (int) 等待无明显改善以进行早期停止的周期数
        batch=1,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=1024,  # (int) 输入图像的大小，整数或h，w
        save=True,  # (bool) 保存训练检查点和预测结果
        # save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
        cache=False,  # (bool) True/ram、磁盘或False。使用缓存加载数据
        device=[0, ],  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=0,  # (int) 数据加载的工作线程数（每个DDP进程）
        project='runs/train',  # (str, optional) 项目名称
        name='exp',  # (str, optional) 实验名称，结果保存在'project/name'目录下
        exist_ok=False,  # (bool) 是否覆盖现有实验
        pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
        optimizer='Adam',  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        verbose=True,  # (bool) 是否打印详细输出
        seed=0,  # (int) 用于可重复性的随机种子
        deterministic=True,  # (bool) 是否启用确定性模式
        single_cls=False,  # (bool) 将多类数据训练为单类
        rect=False,  # (bool) 如果mode='train'，则进行矩形训练，如果mode='val'，则进行矩形验证
        cos_lr=False,  # (bool) 使用余弦学习率调度器
        close_mosaic=0,  # (int) 在最后几个周期禁用马赛克增强
        resume=True,  # (bool) 从上一个检查点恢复训练
        # amp=False,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
        # fraction=1.0,  # (float) 要训练的数据集分数（默认为1.0，训练集中的所有图像）
        # profile=False,  # (bool) 在训练期间为记录器启用ONNX和TensorRT速度
        # freeze=None,  # (int | list, 可选) 在训练期间冻结前 n 层，或冻结层索引列表。
        # 分割
        overlap_mask=True,  # (bool) 训练期间是否应重叠掩码（仅适用于分割训练）
        mask_ratio=4,  # (int) 掩码降采样比例（仅适用于分割训练）
        # 分类
        dropout=0.0,  # (float) 使用丢弃正则化（仅适用于分类训练）
        # 超参数 ----------------------------------------------------------------------------------------------
        lr0=0.01,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
        lrf=0.01,  # (float) 最终学习率（lr0 * lrf）
        momentum=0.937,  # (float) SGD动量/Adam beta1
        weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
        warmup_epochs=3.0,  # (float) 预热周期（分数可用）
        warmup_momentum=0.8,  # (float) 预热初始动量
        warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
        box=7.5,  # (float) 盒损失增益
        cls=0.5,  # (float) 类别损失增益（与像素比例）
        dfl=1.5,  # (float) dfl损失增益
        # pose=12.0,  # (float) 姿势损失增益
        # kobj=1.0,  # (float) 关键点对象损失增益
        label_smoothing=0.0,  # (float) 标签平滑（分数）
        nbs=64,  # (int) 名义批量大小
        hsv_h=0.005,  # (float) 图像HSV-Hue增强（分数）
        hsv_s=0.3,  # (float) 图像HSV-Saturation增强（分数）
        hsv_v=0.2,  # (float) 图像HSV-Value增强（分数）
        degrees=0.0,  # (float) 图像旋转（+/- deg）
        translate=0.1,  # (float) 图像平移（+/- 分数）
        scale=0.1,  # (float) 图像缩放（+/- 增益）
        shear=0.0,  # (float) 图像剪切（+/- deg）
        perspective=0.0,  # (float) 图像透视（+/- 分数），范围为0-0.001
        flipud=0.5,  # (float) 图像上下翻转（概率）
        fliplr=0.5,  # (float) 图像左右翻转（概率）
        mosaic=0.0,  # (float) 图像马赛克（概率）
        mixup=0.0,  # (float) 图像混合（概率）
        copy_paste=0.0,  # (float) 分割复制-粘贴（概率）
    )

    model.export(format='onnx')
    
    # Evaluate the model's performance on the validation set
    # results = # 验证模型
    #     model.val(
    #         val=True,  # (bool) 在训练期间进行验证/测试
    #         data=r'coco128.yaml',
    #         split='val',  # (str) 用于验证的数据集拆分，例如'val'、'test'或'train'
    #         batch=1,  # (int) 每批的图像数量（-1 为自动批处理）
    #         imgsz=640,  # 输入图像的大小，可以是整数或w，h
    #         device='',  # 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
    #         workers=8,  # 数据加载的工作线程数（每个DDP进程）
    #         save_json=False,  # 保存结果到JSON文件
    #         save_hybrid=False,  # 保存标签的混合版本（标签 + 额外的预测）
    #         conf=0.001,  # 检测的目标置信度阈值（默认为0.25用于预测，0.001用于验证）
    #         iou=0.6,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
    #         project='runs/val',  # 项目名称（可选）
    #         name='exp',  # 实验名称，结果保存在'project/name'目录下（可选）
    #         max_det=300,  # 每张图像的最大检测数
    #         half=False,  # 使用半精度 (FP16)
    #         dnn=False,  # 使用OpenCV DNN进行ONNX推断
    #         plots=True,  # 在训练/验证期间保存图像
    #     )

    # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')


    results = model(test_img, save=True)

    # masks = results[0].masks.cpu().numpy().data if results[0].masks is not None else None
    # [cv2.imwrite(disk+':\\work\\Python\\YOLOV8\\runs\\{}.bmp'.format(x),masks[x]) for x in range(masks.shape[0])]
    # t = sum([masks[x].astype(np.uint8)*x for x in range(masks.shape[0])])
    # cv2.imwrite(disk+':\\work\\Python\\YOLOV8\\runs\\1.bmp',t.astype(np.uint8))

    # orig_img = results[0].orig_img
    # orig_masks = cv2.resize(masks[0], orig_img.shape[0:-1])
    # plt.figure(figsize=(12, 8))
    # plt.imshow(results)
    # plt.axis('off')
    # plt.show()

    # Export the model to ONNX format
    # success = model.export(format='onnx')