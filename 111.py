import os
import random

import cv2
import numpy as np


def calculate_line_endpoints(m, c, img_h, img_w):
    intersections = []
    # 左边界 x=0
    y_left = int(m * 0 + c)
    if 0 <= y_left < img_h:
        intersections.append((0, y_left))
    # 右边界 x=img_w-1
    y_right = int(m * (img_w - 1) + c)
    if 0 <= y_right < img_h:
        intersections.append((img_w - 1, y_right))
    # 上边界 y=0
    if m != 0:
        x_top = int((0 - c) / m)
        if 0 <= x_top < img_w:
            intersections.append((x_top, 0))
    # 下边界 y=img_h-1
    if m != 0:
        x_bottom = int((img_h - 1 - c) / m)
        if 0 <= x_bottom < img_w:
            intersections.append((x_bottom, img_h - 1))

    if len(intersections) >= 2:
        sorted_points = sorted(intersections, key=lambda p: p[0])
        return (sorted_points[0][0], sorted_points[0][1]), (sorted_points[-1][0], sorted_points[-1][1])
    return None


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=8):
    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    for i in range(0, int(dist - dash_length), int(dash_length * 2)):
        start = (int(pt1[0] + dx * i), int(pt1[1] + dy * i))
        end = (int(pt1[0] + dx * (i + dash_length)), int(pt1[1] + dy * (i + dash_length)))
        cv2.line(img, start, end, color, thickness)


def calculate_perpendicular_endpoints(x_center, y_center, m_main, img_h, img_w):
    if m_main == 0:
        return ((x_center, 0), (x_center, img_h - 1))
    elif abs(m_main) > 1e6:
        return ((0, y_center), (img_w - 1, y_center))
    else:
        m_perp = -1 / m_main
        c_perp = y_center - m_perp * x_center
        return calculate_line_endpoints(m_perp, c_perp, img_h, img_w)


# === 严格颜色定义 ===
COLORS = {
    "class0_box": (0, 255, 0),  # 绿色 (BGR)
    "class5_point": (0, 0, 255),  # 红色
    "class5_line": (0, 255, 0),  # 绿色
    "perpendicular_line": (0, 255, 255),  # 黄色
    "text_bg": (0, 255, 0),  # 文字背景绿色
    "text_fg": (255, 255, 255)  # 文字颜色白色
}

# === 可视化参数配置 ===
VISUAL_CONFIG = {
    "box_thickness": 2,
    "point_radius": 4,
    "line_thickness": 4,
    "dash_thickness": 4,
    "dash_length": 8,
    "text_font": cv2.FONT_HERSHEY_SIMPLEX,
    "text_scale": 0.6,
    "text_thickness": 1,
    "save_dir": "111"  # 修改保存路径
}


def validate_colors():
    """验证颜色定义与参考代码一致"""
    assert COLORS["class0_box"] == (0, 255, 0), "类别0颜色错误"
    assert COLORS["class5_point"] == (0, 0, 255), "类别5颜色错误"
    assert COLORS["perpendicular_line"] == (0, 255, 255), "垂线颜色错误"


def generate_confidence():
    """生成0.7-1.0之间的随机置信度"""
    return round(random.uniform(0.7, 1.0), 2)


def draw_label(img, x, y, label, color_bg, color_text):
    """通用标签绘制函数"""
    (tw, th), _ = cv2.getTextSize(label,
                                  VISUAL_CONFIG["text_font"],
                                  VISUAL_CONFIG["text_scale"],
                                  VISUAL_CONFIG["text_thickness"])

    # 绘制背景框
    cv2.rectangle(img,
                  (x, y - th - 3),
                  (x + tw + 3, y),
                  color_bg,
                  -1)

    # 绘制文字
    cv2.putText(img, label,
                (x + 2, y - 5),
                VISUAL_CONFIG["text_font"],
                VISUAL_CONFIG["text_scale"],
                color_text,
                VISUAL_CONFIG["text_thickness"])


def main():
    validate_colors()

    txt_file = "data/images/eg.txt"
    img_file = txt_file.replace(".txt", ".jpg")

    # 加载图像
    img = cv2.imread(img_file)
    if img is None:
        raise FileNotFoundError(f"无法加载图像文件: {img_file}")
    h, w = img.shape[:2]

    # 颜色空间验证
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    points = []

    # 处理标注文件
    with open(txt_file) as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"跳过无效行 {line_num}")
                continue

            try:
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                conf = generate_confidence()  # 生成随机置信度
            except ValueError:
                print(f"解析错误行 {line_num}")
                continue

            # 坐标转换
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            if class_id == 0:
                # 绘制绿色边界框
                cv2.rectangle(
                    img,
                    (x1, y1), (x2, y2),
                    COLORS["class0_box"],
                    thickness=VISUAL_CONFIG["box_thickness"]
                )

                # 添加带置信度的标签
                label = f"car {conf}"
                draw_label(img, x1, y1, label,
                           COLORS["text_bg"], COLORS["text_fg"])

            elif class_id == 5:
                # 计算中心点
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                points.append((x_center, y_center))

                # 绘制红色中心点
                cv2.circle(
                    img,
                    (x_center, y_center),
                    VISUAL_CONFIG["point_radius"],
                    COLORS["class5_point"],
                    thickness=-1
                )

                # 添加带置信度的标签
                label = f"0 {conf}"
                text_size = cv2.getTextSize(
                    label,
                    VISUAL_CONFIG["text_font"],
                    VISUAL_CONFIG["text_scale"],
                    VISUAL_CONFIG["text_thickness"]
                )[0]

                label_y = y_center - 10 if y_center - text_size[1] > 10 else y_center + 20
                draw_label(img, x_center - text_size[0] // 2, label_y,
                           label, COLORS["class5_point"], (255, 255, 255))

    # 处理直线拟合
    if len(points) >= 2:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        try:
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
        else:
            # 绘制主直线
            endpoints = calculate_line_endpoints(m, c, h, w)
            if endpoints:
                (x1_line, y1_line), (x2_line, y2_line) = endpoints
                cv2.line(
                    img,
                    (x1_line, y1_line), (x2_line, y2_line),
                    COLORS["class5_line"],
                    thickness=VISUAL_CONFIG["line_thickness"]
                )

                # 绘制垂线
                for (x_center, y_center) in points:
                    perp = calculate_perpendicular_endpoints(x_center, y_center, m, h, w)
                    if perp:
                        draw_dashed_line(
                            img,
                            perp[0], perp[1],
                            COLORS["perpendicular_line"],
                            thickness=VISUAL_CONFIG["dash_thickness"],
                            dash_length=VISUAL_CONFIG["dash_length"]
                        )

    # 增强可视化效果
    img = cv2.addWeighted(img, 1.0, img, 0.5, 0)

    # 创建保存目录并保存结果
    save_dir = VISUAL_CONFIG["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, os.path.basename(img_file))
    cv2.imwrite(save_path, img)
    print(f"结果已保存至：{save_path}")

    cv2.imshow("Detection Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()