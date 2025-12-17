import numpy as np  # <--- (新) 需要 numpy 来处理 .data 和 .shape
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItemGroup, QGraphicsLineItem,
    QGraphicsTextItem, QGraphicsRectItem, QGraphicsPolygonItem, QMenu,
    QGraphicsItem
)
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QBrush, QFontDatabase, QPolygonF,
    QImage, QPixmap  # <--- (新) 添加 QImage 和 QPixmap
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QPointF, QRectF
)


# =============================================================================
# 步骤 3, 4 & 5: 更新 ImageViewer (添加缩放、平移和选择)
# =============================================================================
class ImageViewer(QGraphicsView):
    # --- (步骤 4 & 5: 新增信号) ---
    selectionChanged = pyqtSignal(set)
    deleteRequested = pyqtSignal(set)
    addPointRequested = pyqtSignal(QPointF)  # (步骤 5: 新增信号)
    pointMoved = pyqtSignal(str, QPointF)  # <--- (新) 添加拖动信号

    # [新增] 定义设置中心点的信号，传递点ID
    centerPointSet = pyqtSignal(str)
    resetCenterRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = None
        self._heatmap_item = None  # <--- (新) 添加热力图层
        self._point_items = {}
        self._analysis_overlay_items = []

        self._drag_start_pos = None
        self._drag_item = None  # <--- (新) 添加拖动项状态

        # --- (新) 添加框选所需的状态变量 ---
        self._box_drag_start_pos = None  # (新) 框选的起始位置
        self._selection_rect_item = None  # (新) 用于在场景中显示选框的 QGraphicsRectItem
        # --- (新) 结束 ---

        self._pan_start_pos_view = None
        self._current_zoom = 1.0
        self._selected_ids = set()

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self.is_edit_mode = False  # (用户请求: 添加编辑模式状态)

        # -----------------------------------------------------------------
        # (新) 将全局变量移入类中，以修复Bug
        self.cross_half_size = 8
        self.cross_thickness = 2
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # (新) 修复：设置变换锚点为鼠标光标，实现"缩放到光标"
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # -----------------------------------------------------------------

    # <--- (新) 修改函数签名 ---
    def set_image_data(self, pixmap, inliers_data, lattice_nodes, analysis_data,
                       heatmap_data_bgra, heatmap_type, heatmap_opacity,  # <--- 新参数
                       selected_ids=None, sphere_radius=8,
                       show_analysis=False):

        self.scene().clear()
        self._point_items.clear()
        self._analysis_overlay_items = []
        self._heatmap_item = None  # <--- (新) 重置
        self._selected_ids = set(selected_ids) if selected_ids else set()

        if pixmap and not pixmap.isNull():
            self._pixmap_item = self.scene().addPixmap(pixmap)
            self._pixmap_item.setZValue(0)  # <--- (新) ZValue = 0 (最底层)
            self.setSceneRect(self._pixmap_item.boundingRect())
        else:
            self._pixmap_item = None
            self.setSceneRect(QRectF(0, 0, 100, 100))  # Default rect

        # <--- (新) 2. 添加热力图层 ---
        if heatmap_type == "检测点密度" and heatmap_data_bgra is not None:
            try:
                h, w, ch = heatmap_data_bgra.shape
                if ch == 4:
                    # 从 BGRA numpy 数组创建 QImage
                    # 注意：bytesPerLine = width * 4 (4 通道)
                    q_image = QImage(heatmap_data_bgra.data, w, h, w * 4, QImage.Format_ARGB32)

                    # QImage 共享 numpy 数组的内存。
                    # 创建 QPixmap 会执行深拷贝，这是安全的。
                    heatmap_pixmap = QPixmap.fromImage(q_image)

                    self._heatmap_item = self.scene().addPixmap(heatmap_pixmap)
                    self._heatmap_item.setOpacity(heatmap_opacity)
                    self._heatmap_item.setZValue(1)  # (新) ZValue = 1 (在图像之上)
            except Exception as e:
                # 添加一个打印，以便在热力图显示失败时调试
                print(f"未能显示热力图: {e}")
        # --- (新) 结束 ---

        # -----------------------------------------------------------------
        # (新) 移除 global 关键字，设置成员变量
        # global cross_half_size, cross_thickness
        self.cross_half_size = sphere_radius
        self.cross_thickness = max(1, sphere_radius // 4)
        # -----------------------------------------------------------------

        # 步骤 2 & 4: 绘制RANSAC过滤后的内点 (红十字 / 选中时黄十字)
        for p in inliers_data:
            point_id = p["id"];
            x, y = p["midpoint"]["x"], p["midpoint"]["y"]
            is_selected = point_id in self._selected_ids
            color = Qt.yellow if is_selected else Qt.red

            cross_group = QGraphicsItemGroup()
            h_line = QGraphicsLineItem(x - self.cross_half_size, y, x + self.cross_half_size, y)
            v_line = QGraphicsLineItem(x, y - self.cross_half_size, x, y + self.cross_half_size)

            # (新) 使用 self.cross_thickness
            pen = QPen(QColor(color), self.cross_thickness)

            h_line.setPen(pen)
            v_line.setPen(pen)
            cross_group.addToGroup(h_line)
            cross_group.addToGroup(v_line)
            cross_group.setData(0, point_id)
            self.scene().addItem(cross_group)

            cross_group.setZValue(2)  # <--- (新) ZValue = 2 (在热力图之上)

            self._point_items[point_id] = cross_group

            if is_selected:
                self._add_point_id_text(cross_group, point_id)

        # 步骤 3: 绘制局部晶格分析覆盖层
        if show_analysis and analysis_data and analysis_data.get('ordered_points'):
            # (新) 将ZValue设置传递给辅助函数
            self._draw_analysis_overlay(analysis_data['ordered_points'], z_value=3)

        if self.transform().m11() == 1.0 and self._pixmap_item:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._current_zoom = self.transform().m11()

    # -----------------------------------------------------------------
    # (新) 修复：为蓝色ID标签添加固定大小逻辑
    # -----------------------------------------------------------------
    def _add_point_id_text(self, item_group, point_id):
        existing_text = None
        for child in item_group.childItems():
            if isinstance(child, QGraphicsTextItem):
                existing_text = child
                break
        if existing_text:
            self.scene().removeItem(existing_text)

        text_item = QGraphicsTextItem(point_id)
        text_item.setDefaultTextColor(Qt.blue)

        # (新) 设置固定字体
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(10)  # 稍小一点的 10pt 字体
        text_item.setFont(font)

        # (新) 设置标志以忽略缩放
        text_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        center_pos = item_group.boundingRect().center()
        scene_pos = item_group.mapToScene(center_pos)

        # (新) 使用 self.cross_half_size
        text_rect = text_item.boundingRect()
        text_item.setPos(scene_pos.x() + self.cross_half_size + 2,
                         scene_pos.y() - self.cross_half_size - text_rect.height())

        # (新) 移除 setScale
        # text_item.setScale(1 / self._current_zoom if self._current_zoom != 0 else 1)

        self.scene().addItem(text_item)
        item_group.setData(1, text_item)

    def _remove_point_id_text(self, item_group):
        """ (步骤 4) 移除与点关联的ID文本 """
        text_item = item_group.data(1)
        if text_item and isinstance(text_item, QGraphicsTextItem):
            self.scene().removeItem(text_item)
            item_group.setData(1, None)

    # -----------------------------------------------------------------
    # (新) 步骤 3: 辅助函数 (已按建议修改并修复)
    # -----------------------------------------------------------------
    # <--- (新) 修改函数签名 ---
    def _draw_analysis_overlay(self, ordered_points, z_value=3):
        """步骤 3: 绘制蓝色多边形和数字"""
        if not ordered_points:
            return

        final_neighbors = ordered_points[1:]

        if final_neighbors:
            polygon = QPolygonF()
            for p in final_neighbors:
                polygon.append(QPointF(p["midpoint"]["x"], p["midpoint"]["y"]))

            poly_item = QGraphicsPolygonItem(polygon)
            poly_item.setPen(QPen(QColor(0, 0, 255), 2))
            poly_item.setBrush(QBrush(QColor(0, 0, 255, 40)))
            self.scene().addItem(poly_item)
            poly_item.setZValue(z_value)  # <--- (新) 设置 ZValue
            self._analysis_overlay_items.append(poly_item)

        # --- (建议 1) 设置一个固定的字体大小 ---
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(12)  # 设为固定的 12 磅

        for i, point in enumerate(ordered_points):
            label = str(i + 1)  # 标签 (1, 2, 3...)
            cx, cy = point["midpoint"]["x"], point["midpoint"]["y"]

            text_item = QGraphicsTextItem(label)

            # --- (建议 3) 使用对比度更高的颜色 ---
            text_item.setDefaultTextColor(Qt.yellow)

            text_item.setFont(font)

            # --- (建议 1) 移除 setScale() 和动态 setPointSize() ---

            # --- (建议 1) 添加此标志，使文本大小固定，不随缩放变化 ---
            text_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

            # --- (建议 2 + 修复)
            # (新) 使用 self.cross_half_size 作为偏移基准
            offset = self.cross_half_size + 4  # 4px 留白

            # (新) 特殊处理中心点 "1"，将其放在右下方，避免遮挡
            if label == "1":
                text_item.setPos(cx + offset, cy + offset)
            else:
                # 其他点放在右上角
                # (减去boundingRect().height()使其锚点在文本底部)
                text_rect = text_item.boundingRect()
                text_item.setPos(cx + offset, cy - offset - text_rect.height())

            self.scene().addItem(text_item)
            text_item.setZValue(z_value + 1)  # <--- (新) 确保文本在多边形之上
            self._analysis_overlay_items.append(text_item)

    # -----------------------------------------------------------------
    # (结束)
    # -----------------------------------------------------------------

    # =============================================================================
    # (用户请求: "画笔" 模式)
    # =============================================================================
    def set_edit_mode(self, enabled):
        """ (用户请求) 由 MainWindow 设置，用于控制上下文菜单。"""
        self.is_edit_mode = enabled

    # =============================================================================
    # (步骤 4: 新增) 视图更新辅助函数
    # =============================================================================
    def _update_point_selection(self, new_selected_ids):
        """
        (步骤 4) 根据传入的 ID 集合，动态更新视图中点的视觉状态 (颜色)。
        """
        if new_selected_ids == self._selected_ids:
            return

        # -----------------------------------------------------------------
        # (新) 移除 global 关键字，使用成员变量
        # global cross_half_size, cross_thickness
        pen_thickness = self.cross_thickness
        # -----------------------------------------------------------------

        for point_id, cross_group in self._point_items.items():
            is_selected = point_id in new_selected_ids
            was_selected = point_id in self._selected_ids

            if is_selected and not was_selected:
                color = Qt.yellow
                pen = QPen(QColor(color), pen_thickness)
                for line in cross_group.childItems():
                    if isinstance(line, QGraphicsLineItem):
                        line.setPen(pen)
                self._add_point_id_text(cross_group, point_id)

            elif not is_selected and was_selected:
                color = Qt.red
                pen = QPen(QColor(color), pen_thickness)
                for line in cross_group.childItems():
                    if isinstance(line, QGraphicsLineItem):
                        line.setPen(pen)
                self._remove_point_id_text(cross_group)

        self._selected_ids = new_selected_ids.copy()
        self.selectionChanged.emit(self._selected_ids)

    # =============================================================================
    # 步骤 3 & 4: 激活事件处理器
    # =============================================================================

    # -----------------------------------------------------------------
    # (新) 步骤 3: 缩放 (已修复锚点)
    # -----------------------------------------------------------------
    def wheelEvent(self, event):
        """
        处理鼠标滚轮事件：
        1. Ctrl + 滚轮: 以鼠标为中心缩放 (放大/缩小)
        2. Shift + 滚轮: 左右水平滚动
        3. 无修饰键: 上下垂直滚动 (调用父类默认行为)
        """
        # --- 场景 1: Ctrl + 滚轮 -> 缩放 ---
        if event.modifiers() == Qt.ControlModifier:
            # 定义缩放因子
            zoom_in_factor = 1.15
            zoom_out_factor = 1 / zoom_in_factor

            # 判断滚轮方向 (y > 0 表示向上滚动/推远)
            if event.angleDelta().y() > 0:
                scale_factor = zoom_in_factor
            else:
                scale_factor = zoom_out_factor

            # 更新内部记录的缩放级别
            self._current_zoom *= scale_factor

            # 执行缩放
            # 注意：在 __init__ 中已设置 self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            # 所以调用 scale() 时会自动以鼠标光标位置为中心进行缩放，无需额外计算平移。
            self.scale(scale_factor, scale_factor)

        # --- 场景 2: Shift + 滚轮 -> 横向滚动 ---
        elif event.modifiers() == Qt.ShiftModifier:
            # 获取水平滚动条和滚轮增量
            h_bar = self.horizontalScrollBar()
            delta = event.angleDelta().y()

            # 调整滚动值。
            # 通常 delta 为 120 (一格)。直接减去 delta 可以实现自然的滚动方向：
            # 滚轮向上 -> 视图向左移 (滚动条值减小)
            # 滚轮向下 -> 视图向右移 (滚动条值增加)
            h_bar.setValue(h_bar.value() - delta)

        # --- 场景 3: 无修饰键 -> 纵向滚动 ---
        else:
            # 调用父类 QGraphicsView 的默认行为，它会自动处理垂直滚动条
            super().wheelEvent(event)

    # -----------------------------------------------------------------
    # (结束)
    # -----------------------------------------------------------------

    # =============================================================================
    # (!!!) MODIFIED FUNCTION
    # =============================================================================
    # --- (新) 步骤 3 & 4 & 修复: 鼠标按下 (平移 / 点选 / 框选) ---
    def mousePressEvent(self, event):
        pos_view = event.pos()
        pos_scene = self.mapToScene(pos_view)

        # --- 步骤 3: 平移 (中键) ---
        if event.button() == Qt.MiddleButton:
            self._pan_start_pos_view = pos_view
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # --- 步骤 4: 框选与点选 (左键) ---
        if event.button() == Qt.LeftButton:

            # <--- (新) 核心修改 --->
            # 如果不是编辑模式，则忽略所有左键操作
            if not self.is_edit_mode:
                event.accept()  # 接受事件，防止传播
                return  # 不做任何事
            # <--- (新) 修改结束 --->

            # --- 从这里开始，所有代码都只在 is_edit_mode == True 时运行 ---

            item = self.itemAt(event.pos())
            clicked_group = None
            if item:
                if isinstance(item, QGraphicsItemGroup):
                    clicked_group = item
                elif item.group():
                    clicked_group = item.group()

            if clicked_group and clicked_group.data(0) is not None:
                # --- 1. 点击了红十字 (点选) ---
                point_id = clicked_group.data(0)
                new_selection = self._selected_ids.copy()

                if event.modifiers() == Qt.ControlModifier:
                    if point_id in new_selection:
                        new_selection.remove(point_id)
                    else:
                        new_selection.add(point_id)
                else:
                    if len(new_selection) == 1 and point_id in new_selection:
                        new_selection.clear()
                    else:
                        new_selection = {point_id}

                self._update_point_selection(new_selection)
                event.accept()

                # <--- (拖动逻辑开始) --->
                self._drag_item = clicked_group
                self._drag_start_pos = pos_scene
                self.setCursor(Qt.OpenHandCursor)
                # <--- (拖动逻辑结束) --->

            else:
                # --- 2. (修改) 点击了空白区域 ---
                if event.modifiers() != Qt.ControlModifier:
                    self._update_point_selection(set())

                # (新) 开始框选
                self._box_drag_start_pos = pos_scene
                # (新) 创建一个可视的选框
                self._selection_rect_item = QGraphicsRectItem(QRectF(pos_scene, pos_scene))
                pen = QPen(Qt.DashLine)
                pen.setColor(QColor(255, 255, 255, 150))  # 白色虚线
                self._selection_rect_item.setPen(pen)
                self._selection_rect_item.setBrush(QBrush(QColor(255, 255, 255, 50)))  # 半透明白色填充
                self.scene().addItem(self._selection_rect_item)

                event.accept()
                # (重要) 不再 return

    # --- 步骤 3 & 4: 鼠标移动 (平移 / 框选) ---
    def mouseMoveEvent(self, event):
        pos_view = event.pos()
        pos_scene = self.mapToScene(pos_view)

        # --- 步骤 3: 平移 (中键拖动) ---
        if self._pan_start_pos_view is not None:
            delta_view = pos_view - self._pan_start_pos_view
            delta_scene_x = delta_view.x() / self.transform().m11()
            delta_scene_y = delta_view.y() / self.transform().m22()
            self.translate(-delta_scene_x, -delta_scene_y)
            self._pan_start_pos_view = pos_view
            event.accept()
            return

        # <--- (新) 拖动点逻辑 --->
        if self._drag_item is not None and self._drag_start_pos is not None and self.is_edit_mode:
            delta = pos_scene - self._drag_start_pos

            # 移动十字的所有子项 (线)
            for item in self._drag_item.childItems():
                if isinstance(item, QGraphicsLineItem):
                    item.moveBy(delta.x(), delta.y())

            # 移动关联的文本标签 (data(1) 中存储了它)
            text_item = self._drag_item.data(1)
            if text_item and isinstance(text_item, QGraphicsTextItem):
                text_item.moveBy(delta.x(), delta.y())

            self._drag_start_pos = pos_scene
            self.setCursor(Qt.ClosedHandCursor)  # 拖动时变为“抓紧”
            event.accept()
            return
        # <--- (新) 拖动逻辑结束 --->

        # --- (新) 添加框选拖动逻辑 ---
        if self._box_drag_start_pos is not None and self.is_edit_mode:
            # 更新选框的矩形区域
            rect = QRectF(self._box_drag_start_pos, pos_scene).normalized()
            self._selection_rect_item.setRect(rect)
            event.accept()
            return
        # --- (新) 结束 ---

        super().mouseMoveEvent(event)

    # --- 步骤 3 & 4: 鼠标释放 (平移 / 框选) ---
    def mouseReleaseEvent(self, event):
        # --- 步骤 3: 结束平移 ---
        if event.button() == Qt.MiddleButton and self._pan_start_pos_view is not None:
            self._pan_start_pos_view = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        # <--- (新) 拖动释放逻辑 --->
        if event.button() == Qt.LeftButton and self._drag_item is not None and self.is_edit_mode:
            # 查找新坐标
            h_line = None
            for item in self._drag_item.childItems():
                if isinstance(item, QGraphicsLineItem) and abs(item.line().dx()) > 1e-6:
                    h_line = item  # 找到水平线
                    break

            if h_line:
                # h_line.line().p1() 返回的是该 item 的局部坐标
                # 我们需要将其映射回场景坐标
                scene_p1 = h_line.mapToScene(h_line.line().p1())

                # new_x 和 new_y 是十字中心的新场景坐标
                new_x = int(scene_p1.x() + self.cross_half_size)
                new_y = int(scene_p1.y())

                point_id = self._drag_item.data(0)
                self.pointMoved.emit(point_id, QPointF(new_x, new_y))

            # 重置状态
            self._drag_item = None
            self._drag_start_pos = None
            self.setCursor(Qt.ArrowCursor)  # 恢复光标
            event.accept()
            return
        # <--- (新) 拖动逻辑结束 --->

        # --- (新) 添加框选释放逻辑 ---
        if event.button() == Qt.LeftButton and self._box_drag_start_pos is not None and self.is_edit_mode:
            # 1. 获取选框的最终矩形
            selection_rect = self._selection_rect_item.rect()

            # 2. 清理框选状态和
            self.scene().removeItem(self._selection_rect_item)
            self._selection_rect_item = None
            self._box_drag_start_pos = None

            # 3. 判断是“新增”选择还是“覆盖”选择
            # (如果按住了Ctrl，则在现有选择上添加，否则创建新选择)
            new_selection = self._selected_ids.copy() if event.modifiers() == Qt.ControlModifier else set()

            # 4. 查找所有与选框相交的点
            for point_id, item_group in self._point_items.items():
                # 获取点在场景中的边界框
                item_scene_rect = item_group.sceneBoundingRect()

                # 检查点是否与选框相交
                if selection_rect.intersects(item_scene_rect):
                    new_selection.add(point_id)

            # 5. 更新选择
            self._update_point_selection(new_selection)
            event.accept()
            return
        # --- (新) 结束 ---

        super().mouseReleaseEvent(event)

    # --- (步骤 4 & 5: 新增) 右键菜单 (已修改) ---
    def contextMenuEvent(self, event):
        if not self.is_edit_mode:
            event.accept()  # 接受事件，防止传播
            return
        """ (步骤 5 / 用户请求: 已修改) 右键菜单 """
        scene_pos = self.mapToScene(event.pos())
        menu = QMenu(self)

        # --- (新功能) 1. 获取当前鼠标位置下的点（如果有的话） ---
        item = self.itemAt(event.pos())
        clicked_group = None
        if item:
            if isinstance(item, QGraphicsItemGroup):
                clicked_group = item
            elif item.group():
                clicked_group = item.group()

        point_id_under_mouse = None
        if clicked_group:
            point_id_under_mouse = clicked_group.data(0)

        # --- 菜单项构造 ---

        # 选项 A: 针对选中的点或鼠标下的点操作 (新功能)
        if point_id_under_mouse:
            # [新增] 设为中心点动作
            set_center_action = menu.addAction(f"设为中心点 (点1): {point_id_under_mouse}")
            set_center_action.triggered.connect(lambda: self.centerPointSet.emit(point_id_under_mouse))

            menu.addSeparator()

            # [新增] --- 选项 B: 全局操作 ---
            # 无论鼠标下有没有点，都允许重置回自动模式
            reset_action = menu.addAction("重置为自动中心点")
            reset_action.triggered.connect(lambda: self.resetCenterRequested.emit())
            menu.addSeparator()
        # (步骤 5: 启用 "添加新点")
        add_action = menu.addAction("在此处添加新点")

        # (用户请求: 检查 self.is_edit_mode)
        add_action.setEnabled(self._pixmap_item is not None and self.is_edit_mode)
        add_action.triggered.connect(lambda: self.addPointRequested.emit(scene_pos))

        # (用户请求: 检查 self.is_edit_mode)
        if self._selected_ids and self.is_edit_mode:
            menu.addSeparator()
            delete_action = menu.addAction(f"删除 {len(self._selected_ids)} 个选中点")
            delete_action.triggered.connect(lambda: self.deleteRequested.emit(self._selected_ids.copy()))

        menu.exec_(event.globalPos())
        event.accept()

    # =============================================================================
    # (新) 修复双击崩溃问题，并禁用双击添加功能
    # =============================================================================
    def mouseDoubleClickEvent(self, event):
        """
        处理鼠标双击事件。
        1. 修复：捕获双击事件，防止基类 QGraphicsView 默认调用 mousePressEvent()，
           这可能与我们的单击/拖拽逻辑冲突导致不稳定或崩溃。
        2. 功能：(已移除) 不再通过双击添加点。
        """
        if event.button() == Qt.LeftButton:
            # (新) 无论是否在编辑模式，我们都只接受事件，
            # 不执行任何操作 (如添加点)。
            # 这可以防止双击事件传播并调用 mousePressEvent，
            # 从而修复了之前的崩溃问题，
            # 同时也满足了用户“禁用双击添加”的新要求。
            event.accept()
        else:
            # 其他按钮（如中键）的双击，保留默认行为
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """ (新) 捕获键盘事件以实现批量删除 """

        # 1. 检查是否按下了 Delete 键
        if event.key() != Qt.Key_Delete:
            # 如果不是，则调用父类的默认行为
            super().keyPressEvent(event)
            return

        # 2. 检查是否在编辑模式，并且有选中的点
        if self.is_edit_mode and self._selected_ids:
            # 3. 发出已有的 deleteRequested 信号
            # self.log_output.append(f"通过键盘删除了 {len(self._selected_ids)} 个点") # 无法访问 log_output
            self.deleteRequested.emit(self._selected_ids.copy())
            event.accept()
        else:
            super().keyPressEvent(event)