import warnings
from pathlib import Path
import cv2
import json
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from post_processor import filter_anomalies
import numpy as np

# 忽略运行过程中可能出现的警告
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # --- 配置 ---
    # 模型文件路径
    MODEL_PATH = r"../atom_yolo11/s3_recall_finetune/weights/best.pt"
    # 原始图片所在的目录
    SOURCE_DIR = r"../r_dataset/warn"
    # 项目测试/输出的根目录
    PROJECT_DIR = Path(r"../test/stage1")
    # 本次实验的名称（用于创建子目录）
    EXPERIMENT_NAME = "v2"

    # --- 输出目录设置 ---
    # 中间JSON文件的输出目录（存储检测和后处理的结果）
    INTERMEDIATE_JSON_DIR = PROJECT_DIR / EXPERIMENT_NAME / "intermediate_json"
    INTERMEDIATE_JSON_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    # 中间可视化图片的输出目录（存储绘制了检测结果的图片）
    INTERMEDIATE_IMAGE_DIR = PROJECT_DIR / EXPERIMENT_NAME / "intermediate_images"
    INTERMEDIATE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    # (模型和检测参数)
    # SAHI 切片检测的参数
    SLICE_HEIGHT, SLICE_WIDTH = 1024, 1024
    OVERLAP_HEIGHT_RATIO, OVERLAP_WIDTH_RATIO = 0.25, 0.25
    # 模型置信度阈值和IOU（交并比）阈值
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD = 0.25, 0.01
    # 运行设备（"cuda:0" 表示使用第一块GPU，"cpu" 表示使用CPU）
    DEVICE = "cuda:0"
    # 可视化时，内点（inlier）的颜色 (B, G, R)
    INLIER_COLOR = (0, 0, 255)  # 红色
    # 是否启用后处理（RANSAC晶格滤波）
    ENABLE_POST_PROCESSING = True

    # --- RANSAC后处理参数 ---
    # 为较小图片（如1024x1024）准备的参数
    POST_PROCESS_PARAMS_SMALL = {
        "num_iterations": 100, "inlier_threshold": 5, "min_inliers_ratio": 0.7,
        "k_neighbors_for_basis": 10, "basis_angle_tolerance_deg": 30, "min_basis_len_override": 20,
        "max_basis_len_ratio": 1.5, "adaptive_min_basis_len_neighbors": 10, "adaptive_min_basis_len_percentile": 16,
        "cross_size": 4, "cross_thickness": 1,  # 可视化参数
    }
    # 为较大图片（如4096x4096）准备的参数（主要是阈值等比例放大）
    POST_PROCESS_PARAMS_LARGE = {
        "num_iterations": 100, "inlier_threshold": 40, "min_inliers_ratio": 0.7,
        "k_neighbors_for_basis": 10, "basis_angle_tolerance_deg": 30, "min_basis_len_override": 100,
        "max_basis_len_ratio": 1.5, "adaptive_min_basis_len_neighbors": 10, "adaptive_min_basis_len_percentile": 16,
        "cross_size": 15, "cross_thickness": 2,  # 可视化参数
    }
    # 用于区分大小图片的阈值（取短边）
    SIZE_THRESHOLD = 1030

    # --- 模型加载 ---
    print("正在加载模型...")
    # 使用 SAHI 的 AutoDetectionModel 加载 YOLOv11(或兼容)模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolo11",  # 指定模型类型
        model_path=MODEL_PATH,  # 模型权重文件
        confidence_threshold=CONFIDENCE_THRESHOLD,  # 检测置信度
        device=DEVICE  # 运行设备
    )
    print("模型加载完毕。")

    # --- 图像获取 ---
    source_path_obj = Path(SOURCE_DIR)
    # 搜索支持的图片格式
    image_paths = sorted(
        [str(p) for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"] for p in source_path_obj.glob(ext)])

    if not image_paths:
        print(f"未找到图片，路径: {source_path_obj.resolve()}");
        exit()

    print(f"找到 {len(image_paths)} 张图片，开始进行检测和后处理...")

    # --- 主循环 ---
    # 使用 tqdm 显示进度条
    for image_path in tqdm(image_paths, desc="步骤1: 检测与后处理"):
        img_temp = cv2.imread(image_path)
        if img_temp is None: continue
        height, width, _ = img_temp.shape

        # 根据图片短边选择合适的后处理参数
        current_post_process_params = POST_PROCESS_PARAMS_SMALL if min(height,
                                                                       width) < SIZE_THRESHOLD else POST_PROCESS_PARAMS_LARGE

        # --- SAHI 切片预测 ---
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            perform_standard_pred=True,  # 如果图片小于切片大小，则执行标准预测
            postprocess_type="NMS",  # 切片合并后的后处理类型
            postprocess_match_metric="IOU",
            postprocess_match_threshold=IOU_THRESHOLD,  # 合并时的IOU阈值
        )

        # --- 数据格式化 ---
        # 将检测结果从 SAHI 格式转换为项目所需的 "midpoint"（中心点）格式
        image_targets_data = [
            {
                "id": f"{Path(image_path).stem}_{i:04d}",  # 为每个点创建唯一ID
                "midpoint": {
                    "x": (int(p.bbox.to_xyxy()[0]) + int(p.bbox.to_xyxy()[2])) // 2,  # x = (x_min + x_max) / 2
                    "y": (int(p.bbox.to_xyxy()[1]) + int(p.bbox.to_xyxy()[3])) // 2  # y = (y_min + y_max) / 2
                }
            }
            for i, p in enumerate(prediction_result.object_prediction_list)
        ]

        # --- 后处理 (RANSAC) ---
        final_data, _, debug_info = [], [], {}
        if ENABLE_POST_PROCESSING:
            # 准备RANSAC参数（过滤掉可视化参数）
            ransac_params = {k: v for k, v in current_post_process_params.items() if
                             k not in ["cross_size", "cross_thickness"]}

            # 调用 post_processor.py 中的 filter_anomalies 函数进行RANSAC滤波
            final_data, _, debug_info = filter_anomalies(image_targets_data, ransac_params)

            # 鲁棒性检查：如果RANSAC过滤掉了所有点，但原本有检测结果，则退回使用原始检测结果
            if not final_data and image_targets_data:
                final_data = image_targets_data
        else:
            # 如果不启用后处理，则直接使用原始检测结果
            final_data = image_targets_data

        # --- 可视化中间结果 ---
        image_to_draw = cv2.imread(image_path)
        # 获取可视化参数
        cross_size = current_post_process_params.get("cross_size", 5)
        cross_thickness = current_post_process_params.get("cross_thickness", 1)

        # 在图上绘制所有内点（final_data）的中心十字
        for t in final_data:
            cx, cy = t["midpoint"]["x"], t["midpoint"]["y"]
            cv2.line(image_to_draw, (cx - cross_size, cy), (cx + cross_size, cy), INLIER_COLOR, cross_thickness)
            cv2.line(image_to_draw, (cx, cy - cross_size), (cx, cy + cross_size), INLIER_COLOR, cross_thickness)

        # 保存中间可视化图片
        out_img_path = INTERMEDIATE_IMAGE_DIR / (Path(image_path).stem + ".jpg")
        cv2.imwrite(str(out_img_path), image_to_draw)

        # --- 保存中间JSON文件 ---
        # 准备输出到JSON的数据
        intermediate_output = {
            "source_image_path": image_path,
            "image_height": height,
            "image_width": width,
            "filtered_detections": final_data,  # 经RANSAC过滤后的点
            "ransac_debug_info": debug_info,  # RANSAC调试信息
            "post_process_params_used": current_post_process_params  # 本次运行使用的参数
        }

        # 定义输出JSON路径
        out_json_path = INTERMEDIATE_JSON_DIR / (Path(image_path).stem + ".json")
        # 写入JSON文件
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(intermediate_output, f, indent=4, ensure_ascii=False)

    print("\n步骤1完成！")
    print(f"所有中间JSON文件已保存在: {INTERMEDIATE_JSON_DIR}")
    print(f"所有中间可视化图片已保存在: {INTERMEDIATE_IMAGE_DIR}")