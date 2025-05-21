# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 处理操作系统相关功能
import platform  # 获取平台信息
import sys  # 系统相关参数和函数
from pathlib import Path  # 处理文件路径

import torch  # PyTorch深度学习框架
import numpy as np

# 获取当前文件的绝对路径并解析项目根目录
FILE = Path(__file__).resolve()  # 解析当前文件的绝对路径
ROOT = FILE.parents[0]  # 获取根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到Python路径以便模块导入
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 将ROOT转换为相对路径（相对于当前工作目录）

# 导入自定义模块
from models.common import DetectMultiBackend  # 多后端模型加载类
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # 数据加载工具
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer,
                           xyxy2xywh)  # 通用工具函数
from utils.plots import Annotator, colors, save_one_box  # 绘图工具
from utils.torch_utils import select_device, smart_inference_mode  # PyTorch工具函数


# === 修改后的直线端点计算 ===
def calculate_line_endpoints(m, c, img_h, img_w):
    """
    计算直线在图像边界内的两个端点
    """
    # 定义图像边界：x=0, x=img_w-1, y=0, y=img_h-1
    intersections = []

    # 计算直线与四条边界的交点
    # 1. 左边界 (x=0)
    y_left = int(m * 0 + c)
    if 0 <= y_left < img_h:
        intersections.append((0, y_left))

    # 2. 右边界 (x=img_w-1)
    y_right = int(m * (img_w - 1) + c)
    if 0 <= y_right < img_h:
        intersections.append((img_w - 1, y_right))

    # 3. 上边界 (y=0)
    if m != 0:  # 避免除以零
        x_top = int((0 - c) / m)
        if 0 <= x_top < img_w:
            intersections.append((x_top, 0))

    # 4. 下边界 (y=img_h-1)
    if m != 0:
        x_bottom = int((img_h - 1 - c) / m)
        if 0 <= x_bottom < img_w:
            intersections.append((x_bottom, img_h - 1))

    # 至少需要两个交点才能绘制直线
    if len(intersections) >= 2:
        # 选择距离最远的两个点（确保直线贯穿图像）
        sorted_points = sorted(intersections, key=lambda p: p[0])
        return (sorted_points[0][0], sorted_points[0][1]), (sorted_points[-1][0], sorted_points[-1][1])
    else:
        return None


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """
    绘制虚线
    - img: 图像矩阵
    - pt1: 起点 (x1, y1)
    - pt2: 终点 (x2, y2)
    - color: 颜色 (BGR)
    - thickness: 线宽
    - dash_length: 虚线段长度（像素）
    """
    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    for i in range(0, int(dist - dash_length), int(dash_length * 2)):
        start = (int(pt1[0] + dx * i), int(pt1[1] + dy * i))
        end = (int(pt1[0] + dx * (i + dash_length)), int(pt1[1] + dy * (i + dash_length)))
        cv2.line(img, start, end, color, thickness)

def calculate_perpendicular_endpoints(x_center, y_center, m_main, img_h, img_w):
    """
    计算垂线在图像边界内的端点
    - m_main: 主直线的斜率
    - x_center, y_center: 垂线经过的点
    返回：两个端点坐标的元组 ((x1, y1), (x2, y2)) 或 None
    """
    if m_main == 0:  # 主直线水平，垂线垂直
        return ((x_center, 0), (x_center, img_h-1))
    elif abs(m_main) > 1e6:  # 主直线垂直，垂线水平
        return ((0, y_center), (img_w-1, y_center))
    else:
        # 垂线斜率为 -1/m_main
        m_perpendicular = -1 / m_main
        c_perpendicular = y_center - m_perpendicular * x_center
        return calculate_line_endpoints(m_perpendicular, c_perpendicular, img_h, img_w)

# 使用智能推理模式装饰器（自动选择最优推理设置）
@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # 模型路径或Triton服务器URL，默认yolo.pt
        source=ROOT / 'data/images',  # 输入源：文件/目录/URL/摄像头/屏幕
        data=ROOT / 'data/coco.yaml',  # 数据集配置文件路径
        imgsz=(640, 640),  # 推理尺寸（高度，宽度）
        conf_thres=0.25,  # 置信度阈值（滤除低置信度检测）
        iou_thres=0.45,  # NMS的IoU阈值
        max_det=1000,  # 每张图像最大检测数量
        device='',  # 设备选择，如cuda 0或cpu
        view_img=True,  # 是否实时显示结果
        save_txt=True,  # 是否保存结果为.txt文件
        save_conf=True,  # 是否在.txt中保存置信度
        save_crop=True,  # 是否保存裁剪的检测框
        nosave=False,  # 是否不保存图像/视频
        classes=None,  # 按类别过滤，如--class 0 或 0 2 3
        agnostic_nms=False,  # 是否使用类别无关的NMS
        augment=False,  # 是否使用增强推理
        visualize=False,  # 是否可视化特征图
        project=ROOT / 'runs/detect',  # 结果保存目录
        name='exp',  # 实验名称
        exist_ok=False,  # 是否允许覆盖已有项目
        line_thickness=3,  # 边界框线条粗细
        hide_labels=False,  # 是否隐藏标签
        hide_conf=False,  # 是否隐藏置信度
        half=False,  # 是否使用FP16半精度推理
        dnn=False,  # 是否使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧采样步长（处理每隔几帧）
):
    # 处理输入源路径
    source = str(source)
    # 确定是否需要保存结果（当不是.txt输入且未设置nosave时保存）
    save_img = not nosave and not source.endswith('.txt')

    # 创建保存结果的目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 自动递增路径防止覆盖
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建标签目录（如果需要）

    # 创建调试目录（添加到 run 函数开头）
    debug_dir = save_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = select_device(device)  # 选择设备（GPU/CPU）
    models = [DetectMultiBackend(w, device=device, dnn=dnn, data=data, fp16=half) for w in weights]  # 初始化多后端模型

    model1 = models[0]
    model2 = models[1]
    stride, names1, pt = model1.stride, model1.names, model1.pt  # 获取模型的步长、类别名称和是否PyTorch模型
    stride, names2, pt = model2.stride, model2.names, model2.pt  # 获取模型的步长、类别名称和是否PyTorch模型
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸是否是stride的倍数

    # 初始化数据加载器
    bs = 1  # 批处理大小

    # 图像/视频文件模式
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 文件加载器
    vid_path, vid_writer = [None] * bs, [None] * bs  # 视频路径和写入器初始化

    # 模型预热（用虚拟数据运行一次推理）
    model1.warmup(imgsz=(1 if pt or model1.triton else bs, 3, *imgsz))  # 根据后端类型调整输入形状
    model2.warmup(imgsz=(1 if pt or model2.triton else bs, 3, *imgsz))  # 根据后端类型调整输入形状
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # 初始化计数器、窗口和时间分析器

    # 遍历数据集的每一帧
    for path, im, im0s, vid_cap, s in dataset:
        im2 = im
        with dt[0]:  # 预处理时间统计

            im = torch.from_numpy(im).to(model1.device)  # 将图像转换为PyTorch张量并送到设备
            im = im.half() if model1.fp16 else im.float()  # 转换为半精度或单精度
            im /= 255  # 归一化像素值到0-1
            if len(im.shape) == 3:
                im = im[None]  # 添加批次维度（如果输入是单张图像）

            im2 = torch.from_numpy(im2).to(model2.device)  # 将图像转换为PyTorch张量并送到设备
            im2 = im2.half() if model2.fp16 else im2.float()  # 转换为半精度或单精度
            im2 /= 255  # 归一化像素值到0-1
            if len(im2.shape) == 3:
                im2 = im2[None]  # 添加批次维度（如果输入是单张图像）

        # 推理
        with dt[1]:  # 推理时间统计
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # 可视化路径
            pred1 = model1(im, augment=augment, visualize=visualize)  # 执行模型推理
            pred2 = model2(im2, augment=augment, visualize=visualize)  # 执行模型推理

        # 非极大值抑制（NMS）
        with dt[2]:  # NMS时间统计
            pred1 = non_max_suppression(pred1, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 执行NMS
            pred2 = non_max_suppression(pred2, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 执行NMS

            # --- 初始化共享变量 ---
            seen += 1  # 每张图只计数一次
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  # 唯一图像副本
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]  # 显示第一个模型的输入尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化系数
            annotator = Annotator(im0, line_width=line_thickness, example=str(names1))  # 共享标注器

            # --- 处理第一个模型的结果 ---
            for i, det in enumerate(pred1):
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    # 统计类别
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names1[int(c)]}{'s' * (n > 1)}, "
                    # 绘制和保存
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or save_crop or view_img:
                            c = int(cls)
                            label = None if hide_labels else (names1[c] if hide_conf else f'{names1[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))  # 默认颜色
                        if save_crop:
                            save_one_box(xyxy, im0.copy(), file=save_dir / 'crops' / names1[c] / f'{p.stem}.jpg',
                                         BGR=True)

            # --- 处理第二个模型的结果 ---
            for i, det in enumerate(pred2):
                if len(det):
                    det[:, :4] = scale_boxes(im2.shape[2:], det[:, :4], im0.shape).round()
                    points = []  # 存储所有中心点坐标
                    # 统计类别（可选）
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names2[int(c)]}{'s' * (n > 1)}, "
                    # 处理每个检测结果
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            # 保存原始框的坐标（如需保存点坐标，需修改此处）
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                            with open(f'{txt_path}_model2.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or save_crop or view_img:
                            c = int(cls)
                            label = None if hide_labels else (names2[c] if hide_conf else f'{names2[c]} {conf:.2f}')

                            # ====== 关键修改：绘制中心点 ======
                            x1, y1, x2, y2 = map(int, xyxy)
                            x_center = (x1 + x2) // 2
                            y_center = (y1 + y2) // 2
                            points.append((x_center, y_center))  # 添加到列表

                            # 绘制红色实心圆点
                            cv2.circle(im0, (x_center, y_center), 5, (0, 0, 255), -1)

                            # 绘制标签（在点上方）
                            if not hide_labels:
                                label_y = y_center - 10 if y_center - 10 > 10 else y_center + 20
                                cv2.putText(
                                    im0,
                                    label,
                                    (x_center, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),  # 红色文字
                                    2
                                )
                        if save_crop:
                            pass  # 跳过保存
                            # === 新增：拟合直线并绘制 ===
                            # === 拟合并绘制直线 ===

                        debug_img_points = im0.copy()
                        for (x, y) in points:
                            cv2.circle(debug_img_points, (x, y), 5, (255, 0, 0), -1)
                        cv2.imwrite(str(debug_dir / f"{p.stem}_raw_points.jpg"), debug_img_points)

                        if len(points) >= 2:
                            x = np.array([p[0] for p in points])
                            y = np.array([p[1] for p in points])

                            # 初次拟合
                            try:
                                A = np.vstack([x, np.ones(len(x))]).T
                                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                            except np.linalg.LinAlgError:
                                continue  # 初次拟合失败则跳过

                            # 动态阈值剔除离群点
                            residuals = np.abs(m * x - y + c) / np.sqrt(m ** 2 + 1)
                            median_residual = np.median(residuals)
                            std_residual = np.std(residuals)
                            threshold = median_residual + 3 * std_residual
                            mask = residuals < threshold
                            x_filtered = x[mask]
                            y_filtered = y[mask]

                            # === 可视化筛选后的点 + 初次拟合直线 ===
                            debug_img_filtered = im0.copy()
                            for x, y in zip(x_filtered, y_filtered):
                                cv2.circle(debug_img_filtered, (int(x), int(y)), 5, (0, 255, 0), -1)
                            # 计算初次拟合直线的端点
                            x1_line_init = 0
                            y1_line_init = int(m * x1_line_init + c)
                            x2_line_init = im0.shape[1]
                            y2_line_init = int(m * x2_line_init + c)
                            cv2.line(debug_img_filtered, (x1_line_init, y1_line_init), (x2_line_init, y2_line_init),
                                     (0, 0, 255), 2)
                            cv2.imwrite(str(debug_dir / f"{p.stem}_filtered_points.jpg"), debug_img_filtered)

                            # 重新拟合（仅在筛选后点数足够时执行）
                            if len(x_filtered) >= 2:
                                try:
                                    A_filtered = np.vstack([x_filtered, np.ones(len(x_filtered))]).T
                                    m_final, c_final = np.linalg.lstsq(A_filtered, y_filtered, rcond=None)[0]
                                except np.linalg.LinAlgError:
                                    continue  # 重新拟合失败则跳过

                                # === 修改后的代码 ===
                                # --- 所有使用 m_final/c_final 的代码必须在此处 ---
                                # 计算直线端点
                                h, w = im0.shape[:2]
                                endpoints = calculate_line_endpoints(m_final, c_final, h, w)  # 调用稳健的端点计算函数

                                if endpoints:  # 确保存在有效端点
                                    (x1_line, y1_line), (x2_line, y2_line) = endpoints
                                    # 绘制直线（必须在此条件块内）
                                    cv2.line(im0, (x1_line, y1_line), (x2_line, y2_line), (0, 255, 0), 2)

                                    # === 沿直线绘制垂线（延伸至边界）===
                                    max_distance = 20  # 点与直线的最大允许距离（像素）
                                    h, w = im0.shape[:2]

                                    for (x_center, y_center) in points:
                                        # 计算点到主直线的距离
                                        distance = abs(m_final * x_center - y_center + c_final) / np.sqrt(
                                            m_final ** 2 + 1)
                                        if distance <= max_distance:
                                            # 计算垂线端点
                                            endpoints_perp = calculate_perpendicular_endpoints(x_center, y_center,
                                                                                               m_final, h, w)
                                            if endpoints_perp:
                                                (px1, py1), (px2, py2) = endpoints_perp
                                                # 绘制虚线（黄色）
                                                draw_dashed_line(im0, (px1, py1), (px2, py2), (0, 255, 255),
                                                                 thickness=1, dash_length=5)

            # --- 统一保存和显示 ---
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印处理速度
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印总处理速度
    t = tuple(x.t / seen * 1E3 for x in dt)  # 计算每张图像的平均时间
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # 保存结果统计
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加所有命令行参数
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='模型路径或Triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='输入源路径')
    parser.add_argument('--data', nargs='+', type=str, default=ROOT / 'data/coco128.yaml', help='数据集配置文件')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='推理尺寸[h, w]')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='每图最大检测数')
    parser.add_argument('--device', default='', help='计算设备（如cuda:0或cpu）')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='保存结果为.txt')
    parser.add_argument('--save-conf', action='store_true', help='在.txt中保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的检测框')
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
    parser.add_argument('--classes', nargs='+', type=int, help='过滤指定类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--visualize', action='store_true', help='可视化特征')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='保存结果的项目路径')
    parser.add_argument('--name', default='exp', help='实验结果名称')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有项目')
    parser.add_argument('--line-thickness', default=3, type=int, help='边界框线条粗细')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN进行ONNX推理')
    parser.add_argument('--vid-stride', type=int, default=1, help='视频帧采样步长')

    opt = parser.parse_args()  # 解析参数
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 扩展尺寸参数（如[640]变为[640,640]）
    print_args(vars(opt))  # 打印参数
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))  # 检查依赖（此处被注释）
    run(**vars(opt))  # 运行主函数


if __name__ == "__main__":
    opt = parse_opt()  # 解析命令行参数
    main(opt)  # 执行主程序
