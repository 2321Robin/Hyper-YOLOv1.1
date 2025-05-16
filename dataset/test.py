import os

import cv2
import random

# 随机可视化5个样本
label_dir = "PSD/labels/val"
image_dir = "PSD/images/val"

for _ in range(5):
    # 随机选择文件
    txt_file = random.choice([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    img_file = txt_file.replace(".txt", ".jpg")

    # 加载数据
    img = cv2.imread(os.path.join(image_dir, img_file))
    h, w = img.shape[:2]

    # 绘制标注
    with open(os.path.join(label_dir, txt_file)) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id, xc, yc, bw, bh = parts[:5]

            # 计算实际坐标
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Check", img)
    cv2.waitKey(0)
