"""
后端处理器 (Backend Processor)

这个文件将 run_detection.py 和 run_analyzer.py 中的核心逻辑封装成一个
可供 GUI 调用的类 (LatticeProcessor)。

主要职责:
1.  加载和管理 SAHI/YOLO 模型。
2.  提供一个单一的、可在工作线程中运行的处理函数 (run_full_process)，
    该函数执行完整的 "检测 -> RANSAC -> 局部晶格分析" 流程。
"""

import warnings
from pathlib import Path
import cv2
import json
import numpy as np

# --- 核心依赖 ---
# 依赖 SAHI 进行模型加载和切片预测
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    print("错误: 无法导入 'sahi' 库。请确保已安装: pip install sahi")


    # 如果 sahi 无法导入，定义一个假的 AutoDetectionModel 以便代码能被解析，但在运行时会失败
    class AutoDetectionModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("SAHI 库未安装或导入失败。")

# --- 导入你项目中的算法文件 ---
# 依赖 post_processor.py 进行 RANSAC 滤波
try:
    from post_processor import filter_anomalies
except ImportError:
    print("错误: 无法从 'post_processor.py' 导入 'filter_anomalies'。")


    # 定义一个假的 filter_anomalies
    def filter_anomalies(data, params):
        print("警告: 'filter_anomalies' 未能正确导入。")
        return data, [], {"message": "filter_anomalies 未加载"}

# 依赖 local_lattice_analyzer.py 进行局部晶格分析
try:
    from local_lattice_analyzer import analyze_central_structure
except ImportError:
    print("错误: 无法从 'local_lattice_analyzer.py' 导入 'analyze_central_structure'。")


    # 定义一个假的 analyze_central_structure
    def analyze_central_structure(data, h, w):
        print("警告: 'analyze_central_structure' 未能正确导入。")
        return {"message": "analyze_central_structure 未加载"}

# 忽略来自 SAHI 或其他库的运行时警告
warnings.filterwarnings("ignore")


def _generate_heatmap_data(points, height, width, blur_ksize=21, blur_sigma=10):
    """
    根据点列表生成一个 4通道 (BGRA) 的密度热力图 numpy 数组。
    """
    if not points:
        return None

    try:
        # 1. 创建一个空白的 2D 数组
        heatmap_data = np.zeros((height, width), dtype=np.float32)

        # 2. 在点的位置上“点亮”像素
        for p in points:
            x = p["midpoint"]["x"]
            y = p["midpoint"]["y"]
            if 0 <= y < height and 0 <= x < width:
                heatmap_data[y, x] = 255.0

        # 3. 应用高斯模糊以创建“密度”效果
        if blur_ksize > 0 and blur_sigma > 0:
            # 确保 ksize 是奇数
            ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            heatmap_data = cv2.GaussianBlur(heatmap_data, (ksize, ksize), blur_sigma)

        # 4. 归一化到 0-255
        min_val, max_val = np.min(heatmap_data), np.max(heatmap_data)
        if max_val > min_val:
            heatmap_data = ((heatmap_data - min_val) / (max_val - min_val)) * 255

        heatmap_data_uint8 = heatmap_data.astype(np.uint8)

        # 5. 应用颜色映射 (生成 BGR 图像)
        heatmap_bgr = cv2.applyColorMap(heatmap_data_uint8, cv2.COLORMAP_JET)

        # 6. 创建 Alpha 通道 (将低密度区域设为透明)
        alpha_channel = np.ones(heatmap_data_uint8.shape, dtype=np.uint8) * 255
        # 阈值可以调整，这里设为 5 (在0-255范围)
        alpha_channel[heatmap_data_uint8 < 5] = 0

        # 7. 合并为 BGRA
        heatmap_bgra = cv2.merge((heatmap_bgr, alpha_channel))

        return heatmap_bgra  # 返回 4 通道 BGRA 数组

    except Exception as e:
        print(f"生成热力图时出错: {e}")
        return None


def generate_intensity_heatmap_from_file(image_path):
    """
    (新功能) 读取图像文件，根据像素强度生成 Jet 伪彩色图像。
    对应用户提供的截图效果。
    """
    try:
        # 1. 以灰度模式读取图像
        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return None

        # 2. 归一化图像 (增强对比度，确保范围覆盖 0-255)
        img_float = img_gray.astype(np.float32)
        min_val, max_val = np.min(img_float), np.max(img_float)

        if max_val > min_val:
            img_norm = ((img_float - min_val) / (max_val - min_val)) * 255
        else:
            img_norm = img_float

        img_uint8 = img_norm.astype(np.uint8)

        # 3. 应用颜色映射 (COLORMAP_JET)
        heatmap_bgr = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)

        # 4. 转换为 BGRA (全不透明)
        alpha_channel = np.ones(img_uint8.shape, dtype=np.uint8) * 255
        heatmap_bgra = cv2.merge((heatmap_bgr, alpha_channel))

        return heatmap_bgra

    except Exception as e:
        print(f"生成强度热力图失败: {e}")
        return None


class LatticeProcessor:
    """
    封装了所有检测和分析逻辑的后端处理器类。
    """

    def __init__(self):
        """
        初始化后端处理器。
        """
        self.model = None  # 用于存储已加载的 SAHI/YOLO 模型实例
        self.device = None  # 存储运行设备 (e.g., "cuda:0" or "cpu")

    def load_model(self, model_path, confidence_threshold=0.25, device="cuda:0"):
        """
        加载 SAHI AutoDetectionModel 模型。
        """
        print(f"正在加载模型: {model_path} ...")
        try:
            self.device = device
            self.model = AutoDetectionModel.from_pretrained(
                model_type="yolo11",  # 或者你实际使用的类型
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=self.device
            )
            print("模型加载成功。")
            return True, "模型加载成功。"
        except Exception as e:
            self.model = None
            import traceback
            print(f"模型加载失败:\n{traceback.format_exc()}")
            return False, f"模型加载失败: {e}"

    def run_full_process(self, image_path, sahi_params, ransac_params, enable_post_processing=True):
        """
        对单个图像执行完整的“检测 -> RANSAC -> 晶格分析”流程。
        """
        if not self.model:
            return {"success": False, "error": "模型未加载。"}

        try:
            # --- 步骤 1: 检测与 RANSAC ---

            # 加载图像并获取尺寸
            img_temp = cv2.imread(str(image_path))
            if img_temp is None:
                return {"success": False, "error": f"无法读取图像: {image_path}"}
            height, width, _ = img_temp.shape

            # --- [修改点] 确定实际使用的 RANSAC 参数 ---
            actual_ransac_params = {}

            # 检查是否是来自 GUI 的自适应配置字典
            if isinstance(ransac_params, dict) and ransac_params.get("is_adaptive", False):
                threshold = ransac_params.get("size_threshold", 1030)
                # 根据短边长度判断
                if min(height, width) < threshold:
                    actual_ransac_params = ransac_params.get("params_small", {})
                else:
                    actual_ransac_params = ransac_params.get("params_large", {})
            else:
                # 兼容脚本或其他直接传递单一字典的调用方式
                actual_ransac_params = ransac_params

            # 从 GUI 参数中获取 SAHI 配置
            slice_height = sahi_params.get("slice_height", 1024)
            slice_width = sahi_params.get("slice_width", 1024)
            overlap_h = sahi_params.get("overlap_height_ratio", 0.25)
            overlap_w = sahi_params.get("overlap_width_ratio", 0.25)
            iou_threshold = sahi_params.get("iou_threshold", 0.01)

            self.model.confidence_threshold = sahi_params.get("confidence_threshold", self.model.confidence_threshold)

            # 运行 SAHI 切片预测
            prediction_result = get_sliced_prediction(
                image=image_path,
                detection_model=self.model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_h,
                overlap_width_ratio=overlap_w,
                perform_standard_pred=True,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=iou_threshold,
            )

            # 格式化数据为 "midpoint" 格式
            image_targets_data = [
                {
                    "id": f"{Path(image_path).stem}_{i:04d}",
                    "midpoint": {
                        "x": (int(p.bbox.to_xyxy()[0]) + int(p.bbox.to_xyxy()[2])) // 2,
                        "y": (int(p.bbox.to_xyxy()[1]) + int(p.bbox.to_xyxy()[3])) // 2
                    }
                }
                for i, p in enumerate(prediction_result.object_prediction_list)
            ]

            # --- RANSAC 后处理 ---
            final_data = image_targets_data
            lattice_nodes_viz = []
            ransac_debug_info = {}

            if enable_post_processing:
                # 调用 post_processor.py 中的 filter_anomalies
                # 注意：这里传入的是根据尺寸选择后的 actual_ransac_params
                final_data, lattice_nodes_viz, ransac_debug_info = filter_anomalies(
                    image_targets_data,
                    actual_ransac_params
                )

                # 鲁棒性检查：如果 RANSAC 过滤掉了所有点，则退回使用原始检测结果
                if not final_data and image_targets_data:
                    final_data = image_targets_data
                    ransac_debug_info["message"] = "RANSAC 过滤失败，已退回使用原始检测点。"
            else:
                ransac_debug_info["message"] = "RANSAC 已被禁用。"

            # --- 步骤 2: 局部晶格分析 ---

            local_lattice_info = {}
            if final_data:  # 仅当有内点时才进行分析
                local_lattice_info = analyze_central_structure(
                    final_data,
                    height,
                    width
                )

            # 生成检测点密度热力图 (旧功能)
            heatmap_data = _generate_heatmap_data(final_data, height, width)

            # --- 准备最终结果 ---
            return {
                "success": True,
                "source_image_path": str(image_path),
                "image_height": height,
                "image_width": width,
                "original_detections_count": len(image_targets_data),
                "filtered_detections": final_data,
                "lattice_nodes_for_viz": lattice_nodes_viz,
                "ransac_debug_info": ransac_debug_info,
                "local_lattice_analysis": local_lattice_info,
                "heatmap_data": heatmap_data
            }

        except Exception as e:
            import traceback
            error_msg = f"处理图像 {image_path} 时发生错误: {e}\n{traceback.format_exc()}"
            print(error_msg)
            return {"success": False, "error": error_msg}