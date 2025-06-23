import copy
import cv2
import os
import shutil
import numpy as np
import torch

# by https://blog.csdn.net/weixin_43694096
def restore_masks_to_image(mask_path, image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path)
    file = open(mask_path, "r")
    mask_data = file.readlines()
    file.close()
    # 将掩码数据还原到图像上
    for mask in mask_data:
        values = list(map(float, mask.split()))
        class_id = int(values[0])
        mask_values = values[1:]

        # 将掩码数据转换为NumPy数组
        mask_array = np.array(mask_values, dtype=np.float32).reshape((int(len(mask_values) / 2), 2))

        # 将相对于图像大小的百分比转换为具体坐标值
        mask_array[:, 0] *= img.shape[1]  # 宽度
        mask_array[:, 1] *= img.shape[0]  # 高度

        # 将坐标值转换为整数
        mask_array = mask_array.astype(np.int32)

        # 在图像上绘制掩码
        cv2.polylines(img, [mask_array], isClosed=True, color=(0, 255, 0), thickness=2)
        # cv2.imwrite('F:/work/Python/YOLOV8/datasets/tablet/2.bmp',img)
        # # 在图像上绘制每个坐标点
        # for point in mask_array:
        #     cv2.circle(img, tuple(point), 3, (255, 0, 0), -1)  # -1 表示填充圆
        print(" ")
    # 保存带有掩码和坐标点的图像
    cv2.imwrite(output_path, img)

def convert_masks_to_txt(path,dest_path):
    # path = "你的mask路径  /Dataset/mask"
    # dest_path = 标签保存路径 Dataset/labels
    files = os.listdir(path)
    for file in files:
        name = file.split('.')[0]
        file_path = os.path.join(path,name+'.png')
        img = cv2.imread(file_path)
        # img = cv2.imread(path)
        H,W=img.shape[0:2]
        print(H,W)

        #img1 = cv2.imread("F:/Deep_Learning/Model/YOLOv8_Seg/Dataset/images/20160222_080933_361_1.jpg")

        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,bin_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnt,hit = cv2.findContours(bin_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

        #cv2.drawContours(img1,cnt,-1,(0,255,0),5)

        cnt = list(cnt)
        filename = "{}/{}.txt".format(dest_path, file.split(".")[0])
        if os.path.exists(filename):
            os.remove(filename)
        f = open(filename, "a+")
        for j in cnt:
            result = []
            pre = j[0]
            for i in j:
                if abs(i[0][0] - pre[0][0]) > 1 or abs(i[0][1] - pre[0][1]) > 1:# 在这里可以调整间隔点，我设置为1
                    pre = i
                    temp = list(i[0])
                    temp[0] /= W
                    temp[1] /= H
                    result.append(temp)
                    if  temp[0]==0 or temp[1]==0:
                        print("")
                    #cv2.circle(img1,i[0],1,(0,0,255),2)

            print(result)
            print(len(result))

            # if len(result) != 0:

            if len(result) != 0:
                f.write("0 ")
                liness =""
                for line in result:
                    line = str(line)[1:-1].replace(",","")
                    f.write(line+" ")
                    liness +=(line+" ")
                print(liness)
                f.write("\n")
        f.close()

        #cv2.imshow("test",img1)
        # while True:
        #     key = cv2.waitKey(1)  # 等待 1 毫秒，返回键盘按键的 ASCII 值
        #     if key == ord('q'):  # 如果按下 'q' 键，退出循环
        #         break
        #
        # cv2.destroyAllWindows()  # 关闭窗口




def mask_to_rle(tensor: torch.Tensor) :
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    h, w, b = tensor.shape  #需要根据tensor的shape修改
    tensor = tensor.permute(2, 1, 0).flatten(1)  #需要根据tensor的shape修改

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1] #要求值是int型
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out





if __name__ == '__main__':
    # path= "E:\\Study\\seg\\unet-pytorch-main\\unet_datasets\\Labels"
    # dest_path = "E:\\Study\\seg\\unet-pytorch-main\\unet_datasets\\labels_txt"
    # if not os.path.exists(dest_path):
    #     os.makedirs(dest_path)
    # convert_masks_to_txt(path,dest_path)


    # restore_masks_to_image('F:\\work\\Python\\YOLOV8\\datasets\\tablet\\labels\\3.txt',
    #                         'F:\\work\\Python\\YOLOV8\\datasets\\tablet\\images\\3.png', 
    #                         'F:\\work\\Python\\YOLOV8\\datasets\\tablet\\1.bmp')
    restore_masks_to_image('F:\\work\\Python\\YOLOV8\\datasets\\coco128-seg\\labels\\train2017\\000000000459.txt',
                            'F:\\work\\Python\\YOLOV8\\datasets\\coco128-seg\\images\\train2017\\000000000459.jpg', 
                            'F:\\work\\Python\\YOLOV8\\datasets\\tablet\\1.bmp')
