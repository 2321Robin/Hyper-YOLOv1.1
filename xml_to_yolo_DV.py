import xml.etree.ElementTree as ET
import os
from collections import defaultdict


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

    for i in range(1, 9):
        x_elem = polygon.find(f'x{i}')
        y_elem = polygon.find(f'y{i}')
        if x_elem is None or y_elem is None:
            break

        try:
            coords['x'].append(float(x_elem.text))
            coords['y'].append(float(y_elem.text))
        except (ValueError, TypeError):
            return None

    if len(coords['x']) < 3 or len(coords['y']) < 3:
        return None

    xmin, xmax = min(coords['x']), max(coords['x'])
    ymin, ymax = min(coords['y']), max(coords['y'])
    return (xmin, xmax, ymin, ymax)


def parse_bndbox(bndbox):
    """解析标准边界框格式"""
    try:
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)

        # 确保坐标有效性
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        return (xmin, xmax, ymin, ymax)
    except Exception as e:
        print(f"[ERROR] 解析边界框失败: {str(e)}")
        return None


def xml_to_yolo(xml_path, output_dir, classes, create_empty=False, existing_classes=None):
    """核心转换函数（添加了类别统计功能）"""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"[ERROR] 解析失败 {xml_path}: {str(e)}")
        return False

    root = tree.getroot()
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

    for obj_idx, obj in enumerate(root.findall('object'), 1):
        name_elem = obj.find('name')

        # 处理没有name标签的情况
        if name_elem is None:
            print(f"[WARNING] 对象#{obj_idx} ❌ 缺失<name>标签")
            continue

        # 获取原始名称并处理特殊情况
        raw_name = name_elem.text or ""
        class_name = raw_name.strip()

        # 特定名称修正
        if class_name == 'feright car' or class_name == 'feright_car':
            class_name = 'freight_car'

        # 统计所有遇到的类别（无论是否有效）
        if existing_classes is not None:
            existing_classes[class_name] += 1

        # 有效性检查
        if not class_name:
            print(f"[WARNING] 对象#{obj_idx} ❌ 空<name>内容")
            continue

        if class_name not in classes:
            print(f"[WARNING] 对象#{obj_idx} ❌ 未注册类别 '{class_name}'")
            continue

        # 多边形解析
        polygon = obj.find('polygon')
        bndbox = obj.find('bndbox')

        if not polygon and not bndbox:
            print(f"[WARNING] 文件 {os.path.basename(xml_path)} 对象#{obj_idx} 缺少标注（无polygon/bndbox）")
            continue

            # 优先处理多边形标注
        if polygon:
            bbox = parse_polygon(polygon)
        elif bndbox:
            bbox = parse_bndbox(bndbox)

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
    """批量转换入口函数（添加统计输出功能）"""
    os.makedirs(output_dir, exist_ok=True)
    total_files = 0
    success_files = 0

    # 初始化类别统计字典
    existing_classes = defaultdict(int)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        if xml_to_yolo(xml_path, output_dir, classes, create_empty, existing_classes):
            success_files += 1
        total_files += 1

    # 打印统计结果
    print("\n实际存在的类别统计:")
    print("====================")
    print("类别名称\t出现次数")
    for cls, count in sorted(existing_classes.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}\t{count}")

    print(f"\n转换完成: 成功 {success_files}/{total_files} 个文件")


# 示例调用
if __name__ == "__main__":
    # ================= 配置区 =================
    CLASSES = ['car', 'truck', 'bus', 'freight_car', 'van']  # 必须与XML中name标签完全一致
    XML_DIR = 'dataset/1/labels/train'
    OUTPUT_DIR = 'dataset/1/label/train'
    CREATE_EMPTY = True  # 是否生成空txt文件（无标注时）
    # ==========================================

    batch_convert(
        xml_dir=XML_DIR,
        output_dir=OUTPUT_DIR,
        classes=CLASSES,
        create_empty=CREATE_EMPTY
    )
