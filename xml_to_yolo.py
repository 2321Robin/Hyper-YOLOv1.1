import xml.etree.ElementTree as ET
import os


def convert_bbox(size, box):
    """将矩形框坐标转换为YOLO格式"""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0 * dw
    y_center = (box[2] + box[3]) / 2.0 * dh
    w = (box[1] - box[0]) * dw
    h = (box[3] - box[2]) * dh
    return (x_center, y_center, w, h)


def parse_polygon(polygon):
    """解析多边形标注并返回最小外接矩形"""
    coords = {'x': [], 'y': []}

    # 提取所有顶点坐标（支持4-8个顶点）
    for i in range(1, 9):  # 尝试读取最多8个顶点
        x_elem = polygon.find(f'x{i}')
        y_elem = polygon.find(f'y{i}')
        if x_elem is None or y_elem is None:
            break

        try:
            coords['x'].append(float(x_elem.text))
            coords['y'].append(float(y_elem.text))
        except (ValueError, TypeError):
            return None

    # 至少需要3个有效顶点才能形成有效区域
    if len(coords['x']) < 3 or len(coords['y']) < 3:
        return None

    # 计算最小外接矩形
    xmin, xmax = min(coords['x']), max(coords['x'])
    ymin, ymax = min(coords['y']), max(coords['y'])
    return (xmin, xmax, ymin, ymax)


def xml_to_yolo(xml_path, output_dir, classes, create_empty=False):
    """核心转换函数"""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"[ERROR] 解析失败 {xml_path}: {str(e)}")
        return False

    root = tree.getroot()

    print(f"\nProcessing: {os.path.basename(xml_path)}")

    # 验证图像尺寸
    size_info = root.find('size')
    if not size_info:
        print(f"[WARNING] {xml_path} 缺少尺寸信息")
        return False

    try:
        width = int(size_info.find('width').text)
        height = int(size_info.find('height').text)
        if width <= 0 or height <= 0:
            print(f"[ERROR] {xml_path} 无效尺寸 {width}x{height}")
            return False
    except (ValueError, AttributeError):
        print(f"[ERROR] {xml_path} 尺寸格式错误")
        return False

    txt_lines = []

    # 遍历所有目标对象
    for obj_idx, obj in enumerate(root.findall('object'), 1):
        # 类别验证
        name_elem = obj.find('name')
        if name_elem == 'feright car':
            name_elem = 'feright_car'

        # 情况1：完全缺失name标签
        if name_elem is None:
            print(f"[WARNING] 对象#{obj_idx} ❌ 缺失<name>标签 | 完整标签结构：")
            print(ET.tostring(obj, encoding='unicode').strip())
            continue

        # 情况2：name标签内容为空
        raw_name = name_elem.text or ""
        class_name = raw_name.strip()
        if not class_name:
            print(f"[WARNING] 对象#{obj_idx} ❌ 空<name>内容 | 原始内容：{repr(raw_name)}")
            print("上下文标签结构：")
            print(ET.tostring(obj, encoding='unicode').strip())
            continue

        # 情况3：类别未在列表注册
        if class_name not in classes:
            print(f"[WARNING] 对象#{obj_idx} ❌ 未注册类别 '{class_name}'")
            print("支持类别列表：", classes)
            continue

        # 多边形解析
        polygon = obj.find('polygon')
        if not polygon:
            print(f"[WARNING] 对象#{obj_idx} 缺少多边形标注")
            continue

        bbox = parse_polygon(polygon)
        if not bbox:
            print(f"[WARNING] 对象#{obj_idx} 无效多边形坐标")
            continue

        # 坐标合理性检查
        xmin, xmax, ymin, ymax = bbox
        if xmin >= xmax or ymin >= ymax:
            print(f"[WARNING] 对象#{obj_idx} 无效边界框 {bbox}")
            continue

        # 转换为YOLO格式
        try:
            yolo_coords = convert_bbox((width, height), (xmin, xmax, ymin, ymax))
            cls_id = classes.index(class_name)
            line = f"{cls_id} {' '.join(f'{x:.6f}' for x in yolo_coords)}"
            txt_lines.append(line)
        except Exception as e:
            print(f"[ERROR] 对象#{obj_idx} 转换失败: {str(e)}")
            continue

    # 输出文件处理
    txt_filename = os.path.basename(xml_path).replace('.xml', '.txt')
    txt_path = os.path.join(output_dir, txt_filename)

    if txt_lines:
        with open(txt_path, 'w') as f:
            f.write('\n'.join(txt_lines))
        return True
    elif create_empty:
        open(txt_path, 'a').close()  # 创建空文件
        return True
    else:
        print(f"[INFO] {xml_path} 无有效对象，已跳过")
        return False


def batch_convert(xml_dir, output_dir, classes, create_empty=False):
    """批量转换入口函数"""
    os.makedirs(output_dir, exist_ok=True)
    total_files = 0
    success_files = 0

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        if xml_to_yolo(xml_path, output_dir, classes, create_empty):
            success_files += 1
        total_files += 1

    print(f"\n转换完成: 成功 {success_files}/{total_files} 个文件")
    if success_files < total_files:
        print("提示：请检查警告信息定位问题文件")


# 示例调用
if __name__ == "__main__":
    # ================= 配置区 =================
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'feright_car', 'van']  # 必须与XML中name标签完全一致
    XML_DIR = 'dataset/PSD/label/val'
    OUTPUT_DIR = 'dataset/PSD/labels/val'
    CREATE_EMPTY = True  # 是否生成空txt文件（无标注时）
    # ==========================================

    batch_convert(
        xml_dir=XML_DIR,
        output_dir=OUTPUT_DIR,
        classes=CLASSES,
        create_empty=CREATE_EMPTY
    )
