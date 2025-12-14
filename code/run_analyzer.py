import warnings
from pathlib import Path
import cv2
import json
from tqdm import tqdm
import numpy as np

# 从 local_lattice_analyzer.py 导入核心的晶格分析函数
from local_lattice_analyzer import analyze_central_structure

# 忽略警告
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # --- 配置 ---
    PROJECT_DIR = Path(r"../test/stage1")
    EXPERIMENT_NAME = "v2"

    # --- 输入/输出目录设置 ---
    # 输入目录：步骤1 (run_detection.py) 生成的中间JSON文件
    INTERMEDIATE_DIR = PROJECT_DIR / EXPERIMENT_NAME / "intermediate_json"
    # 输出目录：最终的可视化图片
    FINAL_IMAGE_DIR = PROJECT_DIR / EXPERIMENT_NAME / "final_images"
    # 输出目录：最终的JSON文件（包含检测和分析结果）
    FINAL_JSON_DIR = PROJECT_DIR / EXPERIMENT_NAME / "final_json"
    # 输出目录：几何特征文件（用于后续数据分析）
    GEOMETRIC_FEATURES_DIR = PROJECT_DIR / EXPERIMENT_NAME / "geometric_features"

    # 确保所有输出目录都存在
    FINAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_JSON_DIR.mkdir(parents=True, exist_ok=True)
    GEOMETRIC_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 定义可视化颜色 (B, G, R)
    INLIER_COLOR = (0, 0, 255)  # 红色 (用于所有内点)
    DEBUG_TEXT_COLOR = (0, 255, 0)  # 绿色 (用于绘制标签)

    # --- 获取所有中间JSON文件 ---
    json_paths = sorted(list(INTERMEDIATE_DIR.glob("*.json")))
    if not json_paths:
        print(f"未在 {INTERMEDIATE_DIR} 中找到中间JSON文件。请先运行 run_detection.py。");
        exit()
    print(f"找到 {len(json_paths)} 个中间文件，开始进行晶格分析和特征计算...")

    # --- 主循环 ---
    for json_path in tqdm(json_paths, desc="步骤2: 分析与计算特征"):
        # 读取步骤1生成的JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取需要的信息
        source_image_path = data["source_image_path"]
        height = data["image_height"]
        width = data["image_width"]
        final_data = data["filtered_detections"]  # RANSAC过滤后的内点
        params = data["post_process_params_used"]  # 步骤1中使用的参数

        # --- 核心分析 ---
        # 调用 local_lattice_analyzer.py 中的函数进行局部晶格分析
        # (前提是 final_data 不为空)
        local_lattice_info = analyze_central_structure(final_data, height, width) if final_data else {}

        # --- 保存几何特征 ---
        # 如果分析成功并返回了 'ordered_points'（中心原子+邻居）
        if local_lattice_info and local_lattice_info.get('ordered_points'):
            # 准备一个专门用于存储几何特征的字典
            geometric_data = {
                "source_image": Path(source_image_path).name,
                "lattice_type": local_lattice_info.get('lattice_type'),  # 晶格类型
                "center_atom": local_lattice_info['ordered_points'][0],  # 中心原子
                "neighbor_atoms": local_lattice_info['ordered_points'][1:],  # 邻居原子
                "r_distances_center_to_neighbor": local_lattice_info.get("r_distances_center_to_neighbor", {}),
                # r值 (中心到邻居)
                "d_distances_neighbor_to_neighbor": local_lattice_info.get("d_distances_neighbor_to_neighbor", {}),
                # d值 (邻居到邻居)
                "a_center_angles_degrees": local_lattice_info.get("a_center_angles_degrees", {})  # a值 (中心角)
            }
            # 定义几何特征JSON文件的保存路径
            geo_json_path = GEOMETRIC_FEATURES_DIR / (Path(source_image_path).stem + "_features.json")
            # 保存文件
            with open(geo_json_path, 'w', encoding='utf-8') as f:
                json.dump(geometric_data, f, indent=4, ensure_ascii=False)

        # --- 绘制最终可视化图片 ---
        image_to_draw = cv2.imread(source_image_path)  # 加载原始图片
        if image_to_draw is None: continue

        # 如果分析成功
        if local_lattice_info and local_lattice_info.get('ordered_points'):
            ordered_points = local_lattice_info['ordered_points']
            final_neighbors = ordered_points[1:]  # 邻居列表

            # 绘制邻居原子构成的多边形
            if final_neighbors:
                overlay = image_to_draw.copy()  # 创建一个覆盖层
                # 获取邻居的坐标点
                polygon_points = np.array([[p["midpoint"]["x"], p["midpoint"]["y"]] for p in final_neighbors],
                                          dtype=np.int32)
                # 绘制填充的多边形
                cv2.fillPoly(overlay, [polygon_points], color=(255, 0, 0), lineType=cv2.LINE_AA)  # 蓝色
                alpha = 0.4  # 透明度
                # 将覆盖层与原始图片混合
                image_to_draw = cv2.addWeighted(overlay, alpha, image_to_draw, 1 - alpha, 0)

            # 设置字体大小（根据图片宽度自适应）
            font_scale = 2.0 if width > 1024 else 0.8
            font_thickness, font = 2, cv2.FONT_HERSHEY_SIMPLEX

            # 在图上标记中心原子(1)和邻居(2, 3, ...)
            for i, point in enumerate(ordered_points):
                label = str(i + 1)  # 标签 (1, 2, 3...)
                cx, cy = point["midpoint"]["x"], point["midpoint"]["y"]
                text_position = (cx + 10, cy - 10)  # 标签位置
                cv2.putText(image_to_draw, label, text_position, font, font_scale, DEBUG_TEXT_COLOR, font_thickness,
                            cv2.LINE_AA)

        # --- 绘制所有内点 ---
        # (这些是步骤1 RANSAC 后的所有点，而不仅仅是局部晶格分析的点)
        cross_size = params.get("cross_size", 5)
        cross_thickness = params.get("cross_thickness", 1)
        for t in final_data:
            cx, cy = t["midpoint"]["x"], t["midpoint"]["y"]
            cv2.line(image_to_draw, (cx - cross_size, cy), (cx + cross_size, cy), INLIER_COLOR, cross_thickness)
            cv2.line(image_to_draw, (cx, cy - cross_size), (cx, cy + cross_size), INLIER_COLOR, cross_thickness)

        # 保存最终的可视化图片
        out_img_path = FINAL_IMAGE_DIR / (Path(source_image_path).stem + ".jpg")
        cv2.imwrite(str(out_img_path), image_to_draw)

        # --- 保存最终JSON文件 ---
        # 将分析结果 (local_lattice_info) 添加回从步骤1加载的数据 (data) 中
        data["local_lattice_analysis"] = local_lattice_info
        # 定义最终JSON的保存路径
        out_json_path = FINAL_JSON_DIR / (Path(source_image_path).stem + ".json")
        # 保存包含“检测+后处理+分析”所有信息的完整JSON
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n步骤2完成！")
    print(f"所有最终可视化图片已保存在: {FINAL_IMAGE_DIR}")
    print(f"所有最终JSON文件已保存在: {FINAL_JSON_DIR}")
    print(f"所有几何特征JSON文件已保存在: {GEOMETRIC_FEATURES_DIR}")