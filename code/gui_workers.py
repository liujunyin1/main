import traceback
import time
from pathlib import Path
import os
import numpy as np

# 修复 1: QPointF 应该从 QtCore 导入
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, Qt, QPointF
# 修复 1: 从 QtGui 导入其余绘图类 (移除了 QPointF)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFontDatabase, QPolygonF
)

# 导入分析器，因为 DataEditWorker 需要它
try:
    from local_lattice_analyzer import analyze_central_structure
except ImportError:
    print("CRITICAL (gui_workers): local_lattice_analyzer.py 未找到。")

    def analyze_central_structure(data, h, w, forced_center_id=None):
        return {"message": "local_lattice_analyzer 未加载"}


# <--- (新) 导入热力图生成器 ---
try:
    # 假设 _generate_heatmap_data 在 backend_processor.py 中定义
    from backend_processor import _generate_heatmap_data
except ImportError:
    print("警告 (gui_workers): 无法导入热力图函数。热力图将不会更新。")
    def _generate_heatmap_data(points, h, w, **kwargs):
        return None
# --- (新) 结束 ---


# =============================================================================
# 步骤 1: 工作线程信号
# =============================================================================
class WorkerSignals(QObject):
    '''
    定义一个 QObject 来承载 QRunnable 的信号。
    '''
    finished = pyqtSignal(bool, str)  # (success, message) - 用于模型加载
    logMessage = pyqtSignal(str)


# =============================================================================
# 步骤 1: 模型加载工作线程 (使用 QRunnable)
# =============================================================================
class ModelLoaderWorker(QRunnable):
    """
    在单独的线程 (来自 QThreadPool) 中加载模型。
    """

    def __init__(self, processor, model_path, confidence, device):
        super(ModelLoaderWorker, self).__init__()
        self.processor = processor
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """工作线程的执行体"""
        self.signals.logMessage.emit(f"工作线程(QRunnable)：正在调用 load_model (device={self.device})...")
        try:
            success, message = self.processor.load_model(
                self.model_path,
                self.confidence,
                self.device
            )
            self.signals.finished.emit(success, message)
        except Exception as e:
            error_msg = f"ModelLoaderWorker 异常: {e}\n{traceback.format_exc()}"
            self.signals.logMessage.emit(error_msg)
            self.signals.finished.emit(False, error_msg)


# =============================================================================
# (步骤 5: 重构步骤 4 的 RecalcSignals)
# =============================================================================
class DataEditSignals(QObject):
    finished = pyqtSignal(dict)  # (new_result_dict)
    logMessage = pyqtSignal(str)


# =============================================================================
# (步骤 5: 重构步骤 4 的 RecalculateWorker)
# =============================================================================
class DataEditWorker(QRunnable):
    """
    (步骤 5)
    在单独的线程 (来自 QThreadPool) 中处理数据编辑（添加或删除）。
    然后重新计算局部晶格分析。
    """

    # <--- (新) 修改 __init__ 签名 (应用崩溃修复) --->
    # [修改] 增加 forced_center_id 参数
    def __init__(self, current_result, signals, ids_to_remove=None, point_to_add=None, point_to_modify=None, forced_center_id=None):
        super(DataEditWorker, self).__init__()

        # (修改 1) 不再创建新信号，而是存储传入的信号
        self.signals = signals

        # (其余代码不变)
        self.current_result = current_result
        self.ids_to_remove = ids_to_remove
        self.point_to_add = point_to_add
        self.point_to_modify = point_to_modify  # <--- (新) 存储拖动数据
        self.forced_center_id = forced_center_id # [新增] 存储强制中心ID

    @pyqtSlot()
    def run(self):
        try:
            # 1. 复制当前内点数据
            new_inlier_data = self.current_result['filtered_detections'].copy()

            # 2. (步骤 5) 处理删除
            if self.ids_to_remove:
                self.signals.logMessage.emit(f"正在删除 {len(self.ids_to_remove)} 个点...")
                new_inlier_data = [
                    p for p in new_inlier_data
                    if p['id'] not in self.ids_to_remove
                ]

            # 3. (步骤 5) 处理添加
            if self.point_to_add:
                self.signals.logMessage.emit(f"正在添加点 {self.point_to_add['id']}...")
                new_inlier_data.append(self.point_to_add)

            # <--- (新) 处理修改/拖动 --->
            if self.point_to_modify:
                self.signals.logMessage.emit(f"正在移动点 {self.point_to_modify['id']}...")
                found = False
                for i, p in enumerate(new_inlier_data):
                    if p['id'] == self.point_to_modify['id']:
                        new_inlier_data[i] = self.point_to_modify  # 替换
                        found = True
                        break
                if not found:  # 如果因为某些原因找不到，就当成添加
                    self.signals.logMessage.emit(
                        f"警告: 移动的点 {self.point_to_modify['id']} 未在列表中找到，将其作为新点添加。")
                    new_inlier_data.append(self.point_to_modify)
            # <--- (新) 拖动逻辑结束 --->

            self.signals.logMessage.emit("正在重新计算局部晶格...")

            h = self.current_result['image_height']
            w = self.current_result['image_width']

            # 4. 调用分析函数 (这是唯一的耗时操作)
            # [修改] 传入 forced_center_id
            new_analysis_data = analyze_central_structure(
                new_inlier_data,
                h,
                w,
                forced_center_id=self.forced_center_id
            )

            # <--- (新) 4b. 重新生成密度热力图 ---
            heatmap_data = _generate_heatmap_data(new_inlier_data, h, w)
            # --- (新) 结束 ---

            # 5. 准备一个新的结果字典
            new_result = self.current_result.copy()
            new_result['filtered_detections'] = new_inlier_data
            new_result['local_lattice_analysis'] = new_analysis_data
            new_result['heatmap_data'] = heatmap_data # <--- (新) 添加

            # 6. (重要) 更新调试信息
            if 'ransac_debug_info' in new_result:
                new_result['ransac_debug_info']['final_inliers_count'] = len(new_inlier_data)
                total = new_result['ransac_debug_info'].get('total_points',
                                                            new_result.get('original_detections_count', 0))
                if total > 0:
                    new_result['ransac_debug_info']['inlier_ratio'] = len(new_inlier_data) / total
                else:
                    new_result['ransac_debug_info']['inlier_ratio'] = 0

            self.signals.logMessage.emit("局部晶格重算完成。")
            self.signals.finished.emit(new_result)

        except Exception as e:
            error_msg = f"DataEditWorker 异常: {e}\n{traceback.format_exc()}"
            self.signals.logMessage.emit(error_msg)
            # 发生错误时，也发回一个 "空" 结果，避免UI卡住
            self.signals.finished.emit(self.current_result)


# =============================================================================
# 步骤 1: 图像处理工作线程 (使用 QThread)
# =============================================================================
class ProcessingWorker(QObject):
    """
    在单独的线程中批量处理所有图像。
    (使用 QObject + moveToThread 以便发送进度信号)
    """
    finished = pyqtSignal(list)
    progressUpdated = pyqtSignal(int, int, str)
    logMessage = pyqtSignal(str)

    def __init__(self, processor, image_list, sahi_params, ransac_params, enable_post_proc):
        super().__init__()
        self.processor = processor
        self.image_list = image_list
        self.sahi_params = sahi_params
        self.ransac_params = ransac_params
        self.enable_post_proc = enable_post_proc
        self._is_running = True

    @pyqtSlot()
    def run(self):
        results = []
        total = len(self.image_list)
        if total == 0:
            self.logMessage.emit("错误: 在指定目录中未找到图片。")
            self.finished.emit([])
            return

        for i, image_path in enumerate(self.image_list):
            if not self._is_running:
                break

            filename = Path(image_path).name
            self.progressUpdated.emit(i + 1, total, filename)

            result = self.processor.run_full_process(
                image_path,
                self.sahi_params,
                self.ransac_params,
                self.enable_post_proc
            )

            if not result["success"]:
                self.logMessage.emit(f"处理失败: {filename}. 错误: {result.get('error', '未知错误')}")

            results.append(result)

        self.logMessage.emit("--- 批量处理完成 ---")
        self.finished.emit(results)

    def stop(self):
        self._is_running = False


# =============================================================================
# (新) 批量保存工作线程
# =============================================================================
class SaveAllSignals(QObject):
    finished = pyqtSignal(int, str)  # (count, output_dir_or_error)
    progress = pyqtSignal(int, int, str)  # (current, total, filename)
    logMessage = pyqtSignal(str)


class SaveAllWorker(QRunnable):
    """
    (新) 在后台线程中渲染并保存所有处理过的图像。
    """

    def __init__(self, results_list, view_settings, output_dir):
        super().__init__()
        self.signals = SaveAllSignals()
        self.results_list = results_list
        self.view_settings = view_settings
        self.output_dir = output_dir

    @pyqtSlot()
    def run(self):
        try:
            total = len(self.results_list)
            count = 0
            for i, result in enumerate(self.results_list):
                if not result.get("success"):
                    continue

                filename = Path(result["source_image_path"]).stem
                save_path = os.path.join(self.output_dir, f"{filename}_annotated.png")
                self.signals.progress.emit(i + 1, total, f"{filename}_annotated.png")

                # --- 修复 2: 使用 QImage 而不是 QPixmap (QPixmap 在非GUI线程不安全) ---

                # 1. 直接加载到 QImage
                image = QImage(result["source_image_path"])
                if image.isNull():
                    self.signals.logMessage.emit(f"警告: 无法加载基础图像: {filename}")
                    continue

                # 2. 确保格式适合绘制 (支持透明度和高质量渲染)
                image = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)

                # 3. 创建 Painter
                painter = QPainter(image)
                painter.setRenderHint(QPainter.Antialiasing)

                # --- 4. 绘制标注 ---
                self._draw_annotations_on_painter(painter, result, self.view_settings)

                painter.end()

                # --- 5. 保存 ---
                if not image.save(save_path):
                    self.signals.logMessage.emit(f"错误: 保存图像失败: {save_path}")
                else:
                    count += 1

            self.signals.finished.emit(count, self.output_dir)
        except Exception as e:
            error_msg = f"SaveAllWorker 异常: {e}\n{traceback.format_exc()}"
            self.signals.logMessage.emit(error_msg)
            self.signals.finished.emit(0, str(e))

    def _draw_annotations_on_painter(self, painter, result, view_settings):
        """
        (新) 在 QPainter 上绘制所有标注。
        这是从 ImageViewer.set_image_data 复制和改编的离线版本。
        """
        inlier_data = result.get("filtered_detections", [])
        analysis_data = result.get("local_lattice_analysis", {})
        show_analysis = view_settings.get("show_analysis", False)
        sphere_radius = view_settings.get("sphere_radius", 8)

        cross_half_size = sphere_radius
        cross_thickness = max(1, sphere_radius // 4)

        # --- 1. 绘制内点 (红十字) ---
        pen_inlier = QPen(QColor(Qt.red), cross_thickness)
        painter.setPen(pen_inlier)
        for p in inlier_data:
            x, y = p["midpoint"]["x"], p["midpoint"]["y"]
            # 使用 drawLine 绘制十字
            painter.drawLine(QPointF(x - cross_half_size, y), QPointF(x + cross_half_size, y))
            painter.drawLine(QPointF(x, y - cross_half_size), QPointF(x, y + cross_half_size))

        # --- 2. 绘制分析覆盖层 ---
        if show_analysis and analysis_data and analysis_data.get('ordered_points'):
            ordered_points = analysis_data['ordered_points']
            final_neighbors = ordered_points[1:]

            # 绘制多边形
            if final_neighbors:
                polygon = QPolygonF()
                for p in final_neighbors:
                    polygon.append(QPointF(p["midpoint"]["x"], p["midpoint"]["y"]))

                painter.setPen(QPen(QColor(0, 0, 255), 2))
                painter.setBrush(QBrush(QColor(0, 0, 255, 40)))
                painter.drawPolygon(polygon)

            # 绘制数字
            font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            font.setPointSize(12)  # 固定 12pt
            painter.setFont(font)
            painter.setPen(QColor(Qt.yellow))  # 固定黄色

            offset = cross_half_size + 4

            for i, point in enumerate(ordered_points):
                label = str(i + 1)
                cx, cy = point["midpoint"]["x"], point["midpoint"]["y"]

                text_rect = painter.fontMetrics().boundingRect(label)

                if label == "1":
                    # 右下角
                    painter.drawText(QPointF(cx + offset, cy + offset + text_rect.height()), label)
                else:
                    # 右上角 (注意: drawText 的y坐标是基线)
                    painter.drawText(QPointF(cx + offset, cy - offset), label)