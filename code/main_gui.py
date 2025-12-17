import sys
import os
from pathlib import Path
import json
import cv2
import traceback
import time
import math
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QTabWidget, QScrollArea, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSizePolicy, QAction, QMenu, QStyle, QToolBar,
    QMenuBar, QToolButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
    QGraphicsView
)
from PyQt5.QtGui import (
    QPixmap, QImage, QTextCursor, QIcon, QFontDatabase, QPainter
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSlot, QSettings, pyqtSignal, QSize, QThreadPool, QPointF
)

# --- 导入面板组件 ---
from gui_panels import (
    create_sahi_params_group, create_ransac_params_widget,
    create_debug_info_panel
)

# --- 步骤 1: 导入后端处理器 ---
try:
    from backend_processor import LatticeProcessor, generate_intensity_heatmap_from_file
except ImportError:
    print("CRITICAL: backend_processor.py 未找到或导入出错。")


    class LatticeProcessor:
        def load_model(self, *args, **kwargs):
            return False, "错误: backend_processor.py 未找到或导入失败。"

        def run_full_process(self, *args, **kwargs):
            return {"success": False, "error": "Backend processor 未加载。"}


    # 定义一个假的函数以防报错
    def generate_intensity_heatmap_from_file(image_path):
        return None

# --- 导入新的 GUI 模块 ---
from gui_image_viewer import ImageViewer
from gui_workers import (
    WorkerSignals, ModelLoaderWorker, DataEditSignals, DataEditWorker,
    ProcessingWorker, SaveAllWorker, SaveAllSignals
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("原子检测系统")
        if os.path.exists("tubiao/LogosAtomIcon.png"):
            self.setWindowIcon(QIcon("tubiao/LogosAtomIcon.png"))
        self.setGeometry(100, 100, 1800, 900)
        self.settings = QSettings("MyCompany", "SAHILatticeDetector")

        # --- 步骤 1: 后端与状态变量 ---
        self.processor = LatticeProcessor()
        self.worker_thread = None
        self.thread_pool = QThreadPool()
        print(f"主线程：初始化线程池，最大线程数: {self.thread_pool.maxThreadCount()}")

        self.model_path = ""
        self.source_dir = ""
        self.output_dir = ""

        self.image_list = []
        self.processing_results = []
        self.current_image_index = -1
        self.selected_point_ids = set()
        self.is_edit_mode = False

        self.manual_center_id = None  # [新增] 用于存储手动指定的中心点ID

        # --- 步骤 3: 视图状态变量 ---
        self.show_analysis_view = False

        self.is_processing = False

        # --- 菜单栏 ---
        self.file_menu = None
        self._init_menu_bar()
        self.menuBar().setVisible(False)

        # --- 初始化UI ---
        self._init_ui()

        # --- 信号处理 ---
        self.data_edit_signals = DataEditSignals()
        self.data_edit_signals.logMessage.connect(self.log_output.append)
        self.data_edit_signals.finished.connect(self.on_data_edit_finished)

        self._connect_signals_and_slots()
        self._load_settings()
        self._apply_stylesheet()

        self.log_output.append("欢迎使用晶格分析工具。")
        self.log_output.append("请先从 [文件] 菜单设置模型和图片目录。")

    def _apply_stylesheet(self):
        try:
            qss_path = Path(__file__).parent / "style.qss"
            if qss_path.exists():
                with open(qss_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            else:
                print(f"Warning: Stylesheet 'style.qss' not found at {qss_path}")
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

    def _init_menu_bar(self):
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu('文件 (&F)')

        clear_menu = menubar.addMenu('清空 (&C)')
        view_menu = menubar.addMenu('视图 (&V)')
        window_menu = menubar.addMenu('窗口 (&W)')
        help_menu = menubar.addMenu('帮助 (&H)')

        self.open_dir_action = QAction(QIcon.fromTheme("folder-open", self.style().standardIcon(QStyle.SP_DirOpenIcon)),
                                       '打开图片目录 (&O)...', self)
        self.file_menu.addAction(self.open_dir_action)

        self.select_model_action = QAction(
            QIcon.fromTheme("document-open", self.style().standardIcon(QStyle.SP_FileIcon)), '选择模型文件 (&M)...',
            self)
        self.file_menu.addAction(self.select_model_action)

        self.select_output_action = QAction(
            QIcon.fromTheme("folder-save", self.style().standardIcon(QStyle.SP_DirIcon)), '选择输出目录 (&S)...', self)
        self.file_menu.addAction(self.select_output_action)

        self.file_menu.addSeparator()

        self.load_config_action = QAction(
            QIcon.fromTheme("document-open", self.style().standardIcon(QStyle.SP_DialogOpenButton)), '加载配置 (&L)...',
            self)
        self.file_menu.addAction(self.load_config_action)

        self.save_config_action = QAction(
            QIcon.fromTheme("document-save", self.style().standardIcon(QStyle.SP_DialogSaveButton)), '保存配置 (&A)...',
            self)
        self.file_menu.addAction(self.save_config_action)

        # self.file_menu.addSeparator()

        # self.exit_action = QAction(
        #    QIcon.fromTheme("application-exit", self.style().standardIcon(QStyle.SP_DialogCloseButton)), '退出 (&X)',
        #    self)
        # self.file_menu.addAction(self.exit_action)

        self.about_action = QAction('关于 (&A)', self)
        help_menu.addAction(self.about_action)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ===== 主工具栏 =====
        main_toolbar = QToolBar("主工具栏")
        main_toolbar.setIconSize(QSize(18, 18))
        main_toolbar.setMovable(False)  # 禁止移动/拖拽
        main_toolbar.setFloatable(False)  # 禁止变为浮动窗口
        main_toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(Qt.TopToolBarArea, main_toolbar)

        self.action_tb_open_menu_btn = QToolButton()
        self.action_tb_open_menu_btn.setIcon(QIcon("tubiao/打开.png"))
        self.action_tb_open_menu_btn.setToolTip("文件 (打开目录, 加载/保存配置, 退出等)")
        self.action_tb_open_menu_btn.setMenu(self.file_menu)
        self.action_tb_open_menu_btn.setPopupMode(QToolButton.InstantPopup)
        main_toolbar.addWidget(self.action_tb_open_menu_btn)
        main_toolbar.addSeparator()

        self.action_tb_load_model = QAction(QIcon("tubiao/加载.png"), "加载模型", self)
        self.action_tb_run = QAction(QIcon("tubiao/play.png"), "开始处理", self)
        self.action_tb_switch = QAction(QIcon("tubiao/切换.png"), "切换分析视图", self)
        self.action_tb_exit = QAction(QIcon("tubiao/退出.png"), "退出", self)

        self.action_tb_pen = QAction(QIcon("tubiao/画笔.png"), "切换编辑模式 (添加/删除点)", self)
        self.action_tb_pen.setCheckable(True)
        self.action_tb_pen.setChecked(self.is_edit_mode)

        self.action_tb_heatmap = QAction(QIcon("tubiao/渐变.png"), "切换热力图显示 (原图强度)", self)
        self.action_tb_heatmap.setCheckable(True)
        self.action_tb_heatmap.setChecked(False)

        self.action_tb_prev = QAction(QIcon("tubiao/向左.png"), "上一张", self)
        self.action_tb_next = QAction(QIcon("tubiao/向右.png"), "下一张", self)

        self.save_menu_btn = QToolButton()
        self.save_menu_btn.setIcon(QIcon("tubiao/save.png"))
        self.save_menu_btn.setToolTip("保存图像或数据")
        self.save_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self.save_menu = QMenu(self.save_menu_btn)
        self.save_menu_btn.setMenu(self.save_menu)
        main_toolbar.addWidget(self.save_menu_btn)

        self.action_save_current_image = QAction("保存当前标注图像...", self)
        self.action_save_all_images = QAction("保存所有标注图像...", self)
        self.action_save_current_debug = QAction("保存当前调试信息 (JSON)...", self)
        self.action_save_all_debug = QAction("保存所有调试信息 (JSON)...", self)

        self.save_menu.addAction(self.action_save_current_image)
        self.save_menu.addAction(self.action_save_all_images)
        self.save_menu.addSeparator()
        self.save_menu.addAction(self.action_save_current_debug)
        self.save_menu.addAction(self.action_save_all_debug)

        main_toolbar.addAction(self.action_tb_load_model)
        main_toolbar.addAction(self.action_tb_run)
        main_toolbar.addAction(self.action_tb_switch)
        main_toolbar.addSeparator()

        # --- [修改] 添加编辑模式按钮，并设置 ID 以便在 QSS 中控制样式 ---
        main_toolbar.addAction(self.action_tb_pen)
        # 获取 QAction 对应的 QToolButton 控件
        widget_pen = main_toolbar.widgetForAction(self.action_tb_pen)
        if widget_pen:
            widget_pen.setObjectName("btn_edit_mode")  # 设置唯一的 ObjectName

        main_toolbar.addAction(self.action_tb_heatmap)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.action_tb_exit)
        main_toolbar.addSeparator()

        main_toolbar.addWidget(QLabel("标注大小:"))
        self.view_sphere_radius_sb = QSpinBox()
        self.view_sphere_radius_sb.setRange(1, 50)
        self.view_sphere_radius_sb.setValue(8)
        main_toolbar.addWidget(self.view_sphere_radius_sb)

        main_toolbar.addSeparator()
        main_toolbar.addAction(self.action_tb_prev)
        self.action_tb_prev.setEnabled(False)
        self.image_index_label = QLabel("0/0")
        self.image_index_label.setMinimumWidth(50)
        self.image_index_label.setAlignment(Qt.AlignCenter)
        main_toolbar.addWidget(self.image_index_label)
        main_toolbar.addAction(self.action_tb_next)
        self.action_tb_next.setEnabled(False)

        main_toolbar.addSeparator()
        white_spacer = QWidget()
        white_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        white_spacer.setObjectName("toolbarWhiteSpacer")
        main_toolbar.addWidget(white_spacer)

        # ===== 左侧面板 =====
        self.left_tabs = QTabWidget()
        self.left_tabs.setMinimumWidth(350)
        self.left_tabs.setMaximumWidth(450)
        main_layout.addWidget(self.left_tabs)

        # --- Tab 1: "参数" ---
        self.params_tab_scroll = QScrollArea()
        self.params_tab_scroll.setWidgetResizable(True)
        self.params_tab_widget = QWidget()
        self.params_tab_scroll.setWidget(self.params_tab_widget)
        params_layout = QVBoxLayout(self.params_tab_widget)
        params_layout.setAlignment(Qt.AlignTop)

        sahi_params_group, sahi_controls = create_sahi_params_group()
        self.slice_height_sb = sahi_controls["slice_height_sb"]
        self.slice_width_sb = sahi_controls["slice_width_sb"]
        self.overlap_height_ratio_dsb = sahi_controls["overlap_height_ratio_dsb"]
        self.overlap_width_ratio_dsb = sahi_controls["overlap_width_ratio_dsb"]
        self.confidence_threshold_dsb = sahi_controls["confidence_threshold_dsb"]
        self.iou_threshold_dsb = sahi_controls["iou_threshold_dsb"]
        self.size_threshold_sb = sahi_controls["size_threshold_sb"]
        params_layout.addWidget(sahi_params_group)

        # --- [修改] RANSAC 参数组：改为使用 self.post_process_group 并移除内部复选框 ---
        self.post_process_group = QGroupBox("RANSAC 后处理参数")
        self.post_process_group.setCheckable(True)
        self.post_process_group.setChecked(True)
        post_process_group_layout = QVBoxLayout(self.post_process_group)

        post_process_container = QWidget()
        post_process_layout = QVBoxLayout(post_process_container)

        # [删除] 原有的 self.enable_post_processing_cb 及其布局添加
        # self.enable_post_processing_cb = QCheckBox("启用RANSAC后处理")
        # self.enable_post_processing_cb.setChecked(True)
        # post_process_layout.addWidget(self.enable_post_processing_cb)

        self.post_process_tabs = QTabWidget()
        self.params_small = create_ransac_params_widget("小图参数")
        self.params_large = create_ransac_params_widget("大图参数")
        self.post_process_tabs.addTab(self.params_small, "小图")
        self.post_process_tabs.addTab(self.params_large, "大图")
        post_process_layout.addWidget(self.post_process_tabs)
        post_process_group_layout.addWidget(post_process_container)
        post_process_container.setVisible(True)
        params_layout.addWidget(self.post_process_group)
        # --- [修改结束] ---

        config_group = QGroupBox("参数配置管理")
        config_layout = QHBoxLayout(config_group)
        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        config_layout.addWidget(self.load_config_btn)
        config_layout.addWidget(self.save_config_btn)
        params_layout.addWidget(config_group)

        params_layout.addStretch(1)
        self.left_tabs.addTab(self.params_tab_scroll, "参数")

        # --- Tab 2: "日志" ---
        log_group = QGroupBox("日志")
        log_group.setFlat(True)
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        self.left_tabs.addTab(log_group, "日志")

        # --- 中间图像面板 ---
        image_display_container = QVBoxLayout()
        self.image_viewer = ImageViewer()
        image_display_container.addWidget(self.image_viewer)

        self.points_list_group = QGroupBox("选中点列表")
        points_list_layout = QVBoxLayout(self.points_list_group)
        self.point_list_table = QTableWidget()
        self.point_list_table.setColumnCount(3)
        self.point_list_table.setHorizontalHeaderLabels(["ID", "X", "Y"])
        self.point_list_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.point_list_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.point_list_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.point_list_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.point_list_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.point_list_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        points_list_layout.addWidget(self.point_list_table)

        self.debug_info_group_right, debug_labels = create_debug_info_panel()
        # 映射所有 debug labels
        self.debug_r_header = debug_labels["debug_r_header"]
        for i in range(8):
            setattr(self, f"debug_r_name_{i + 1}", debug_labels[f"debug_r_name_{i + 1}"])
            setattr(self, f"debug_r_val_{i + 1}", debug_labels[f"debug_r_val_{i + 1}"])

        self.debug_angle_header = debug_labels["debug_angle_header"]
        for i in range(4):
            setattr(self, f"debug_angle_name_{i + 1}", debug_labels[f"debug_angle_name_{i + 1}"])
            setattr(self, f"debug_angle_val1_{i + 1}", debug_labels[f"debug_angle_val1_{i + 1}"])  # [新]
            setattr(self, f"debug_angle_val2_{i + 1}", debug_labels[f"debug_angle_val2_{i + 1}"])  # [新]
            setattr(self, f"debug_angle_diff_{i + 1}", debug_labels[f"debug_angle_diff_{i + 1}"])

        self.debug_angle_avg_label = debug_labels["debug_angle_avg_label"]
        self.debug_angle_avg_value = debug_labels["debug_angle_avg_value"]

        bottom_panel_layout = QHBoxLayout()
        target_height = 300
        self.points_list_group.setMaximumHeight(target_height)
        self.debug_info_group_right.setMaximumHeight(target_height)
        bottom_panel_layout.addWidget(self.points_list_group, 0)
        bottom_panel_layout.addWidget(self.debug_info_group_right, 1)
        image_display_container.addLayout(bottom_panel_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("正在处理... %p%")
        image_display_container.addWidget(self.progress_bar)

        image_display_container.setStretch(0, 1)
        image_display_container.setStretch(1, 0)
        image_display_container.setStretch(2, 0)

        main_layout.addLayout(image_display_container)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 5)

    def _connect_signals_and_slots(self):
        # --- 菜单和工具栏 ---
        self.open_dir_action.triggered.connect(self._select_source_dir)
        self.select_model_action.triggered.connect(self._select_model_path)
        self.select_output_action.triggered.connect(self._select_output_dir)
        # self.exit_action.triggered.connect(self.close)
        self.load_config_action.triggered.connect(self.on_load_config)
        self.save_config_action.triggered.connect(self.on_save_config)

        self.load_config_btn.clicked.connect(self.on_load_config)
        self.save_config_btn.clicked.connect(self.on_save_config)

        self.action_tb_load_model.triggered.connect(self.on_load_model)
        self.action_tb_run.triggered.connect(self.on_start_batch_processing)
        self.action_tb_exit.triggered.connect(self.close)
        self.action_tb_pen.toggled.connect(self.on_toggle_edit_mode)
        self.action_tb_switch.triggered.connect(self._toggle_analysis_view)
        self.action_tb_heatmap.toggled.connect(self.on_toggle_heatmap)

        self.action_save_current_image.triggered.connect(self.on_save_current_image)
        self.action_save_all_images.triggered.connect(self.on_save_all_images)
        self.action_save_current_debug.triggered.connect(self.on_save_current_debug_info)
        self.action_save_all_debug.triggered.connect(self.on_save_all_debug_info)

        self.action_tb_prev.triggered.connect(self.on_prev_image)
        self.action_tb_next.triggered.connect(self.on_next_image)

        self.view_sphere_radius_sb.valueChanged.connect(self._trigger_rerender)

        # --- 交互连接 ---
        self.image_viewer.selectionChanged.connect(self.on_view_selection_changed)
        self.image_viewer.deleteRequested.connect(self.on_delete_requested_from_signal)
        self.image_viewer.addPointRequested.connect(self.on_add_point_requested)
        self.image_viewer.pointMoved.connect(self.on_point_moved)
        self.point_list_table.itemSelectionChanged.connect(self.on_list_selection_changed)

        # [新增] 连接 ImageViewer 的新信号 (设置中心 & 重置中心)
        self.image_viewer.centerPointSet.connect(self.on_set_manual_center)
        self.image_viewer.resetCenterRequested.connect(self.on_reset_manual_center)

    def _select_source_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录", self.settings.value("paths/source_dir", "."))
        if dir_path:
            self.source_dir = dir_path
            self.settings.setValue("paths/source_dir", dir_path)
            self.log_output.append(f"设置图片目录: {self.source_dir}")
            self.image_list = sorted([str(p) for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"] for p in
                                      Path(self.source_dir).glob(ext)])
            self.log_output.append(f"在目录中找到 {len(self.image_list)} 张图片。")

    def _select_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", self.settings.value("paths/model_path", "."),
                                                   "PyTorch Models (*.pt)")
        if file_path:
            self.model_path = file_path
            self.settings.setValue("paths/model_path", file_path)
            self.log_output.append(f"设置模型文件: {self.model_path}")

    def _select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.settings.value("paths/output_dir", "."))
        if dir_path:
            self.output_dir = dir_path
            self.settings.setValue("paths/output_dir", dir_path)
            self.log_output.append(f"设置输出目录: {self.output_dir}")

    @pyqtSlot()
    def on_load_model(self):
        if not self.model_path:
            self.log_output.append("错误: 请先选择模型文件路径。")
            return
        self.action_tb_load_model.setEnabled(False)
        self.log_output.append(f"主线程：开始加载模型: {self.model_path}")
        worker = ModelLoaderWorker(self.processor, self.model_path, self.confidence_threshold_dsb.value(),
                                   device="cuda:0")
        worker.signals.logMessage.connect(self.log_output.append)
        worker.signals.finished.connect(self.on_model_loaded)
        self.thread_pool.start(worker)

    @pyqtSlot(bool, str)
    def on_model_loaded(self, success, message):
        self.log_output.append(message)
        self.action_tb_load_model.setEnabled(True)
        if success:
            self.action_tb_run.setEnabled(True)
        else:
            self.action_tb_run.setEnabled(False)

    @pyqtSlot()
    def on_start_batch_processing(self):
        if self.is_processing:
            self.log_output.append("错误：一个处理任务已经在运行中。")
            return
        if not self.source_dir or len(self.image_list) == 0:
            self.log_output.append("错误: 请先选择包含图片的目录。")
            return
        if not self.processor.model:
            self.log_output.append("错误: 模型未加载。")
            return
        self.is_processing = True
        self.action_tb_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.log_output.append("--- 开始批量处理 (自适应大小图参数) ---")

        sahi_params = {
            "slice_height": self.slice_height_sb.value(),
            "slice_width": self.slice_width_sb.value(),
            "overlap_height_ratio": self.overlap_height_ratio_dsb.value(),
            "overlap_width_ratio": self.overlap_width_ratio_dsb.value(),
            "confidence_threshold": self.confidence_threshold_dsb.value(),
            "iou_threshold": self.iou_threshold_dsb.value()
        }

        # --- 辅助函数：从参数面板提取字典 ---
        def get_ransac_params_from_widget(widget):
            return {
                "num_iterations": widget.num_iterations_sb.value(),
                "inlier_threshold": widget.inlier_threshold_dsb.value(),
                "min_inliers_ratio": widget.min_inliers_ratio_dsb.value(),
                "k_neighbors_for_basis": widget.k_neighbors_for_basis_sb.value(),
                "basis_angle_tolerance_deg": widget.basis_angle_tolerance_deg_sb.value(),
                "min_basis_len_override": widget.min_basis_len_override_dsb.value(),
                "max_basis_len_ratio": widget.max_basis_len_ratio_dsb.value(),
                "adaptive_min_basis_len_neighbors": widget.adaptive_min_basis_len_neighbors_sb.value(),
                "adaptive_min_basis_len_percentile": widget.adaptive_min_basis_len_percentile_sb.value()
            }

        # --- 构建自适应配置字典 (包含两套参数) ---
        ransac_config = {
            "is_adaptive": True,  # 标记为自适应模式
            "size_threshold": 1030,  # 大小图判断阈值 (与 run_detection.py 保持一致)
            "params_small": get_ransac_params_from_widget(self.params_small),
            "params_large": get_ransac_params_from_widget(self.params_large)
        }

        # [修改] 直接从 self.post_process_group 读取启用状态
        enable_pp = self.post_process_group.isChecked()

        self.left_tabs.setCurrentIndex(1)
        self.worker_thread = QThread()
        # 将打包好的 ransac_config 传递给 Worker
        self.worker = ProcessingWorker(self.processor, self.image_list, sahi_params, ransac_config, enable_pp)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_batch_processing_complete)
        self.worker.progressUpdated.connect(self.on_processing_progress)
        self.worker.logMessage.connect(self.log_output.append)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    @pyqtSlot(int, int, str)
    def on_processing_progress(self, current, total, filename):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"正在处理: {filename} ({current}/{total}) - %p%")
        else:
            self.progress_bar.setValue(0)

    @pyqtSlot(list)
    def on_batch_processing_complete(self, results):
        self.is_processing = False
        self.action_tb_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.processing_results = results
        if not self.processing_results:
            self.log_output.append("错误: 处理完成，但没有返回任何结果。")
            return
        self.log_output.append(f"成功处理 {len(self.processing_results)} 张图片。")
        self.current_image_index = 0
        self.action_tb_prev.setEnabled(True)
        self.action_tb_next.setEnabled(True)
        self._display_current_image_result()

    @pyqtSlot()
    def on_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.selected_point_ids = set()
            self.manual_center_id = None  # [新增] 重置
            self._display_current_image_result()
        else:
            self.log_output.append("已经是第一张图片。")

    @pyqtSlot()
    def on_next_image(self):
        if self.current_image_index < len(self.processing_results) - 1:
            self.current_image_index += 1
            self.selected_point_ids = set()
            self.manual_center_id = None  # [新增] 重置
            self._display_current_image_result()
        else:
            self.log_output.append("已经是最后一张图片。")

    @pyqtSlot()
    def _trigger_rerender(self):
        self._display_current_image_result()

    @pyqtSlot()
    def _toggle_analysis_view(self):
        self.show_analysis_view = not self.show_analysis_view
        self.log_output.append(f"分析视图: {'打开' if self.show_analysis_view else '关闭'}")
        if self.show_analysis_view and self.is_edit_mode:
            self.log_output.append("切换到分析视图，自动关闭 [编辑模式]。")
            self.action_tb_pen.setChecked(False)
        self._display_current_image_result()

    def _clear_debug_labels(self):
        for i in range(8):
            getattr(self, f"debug_r_name_{i + 1}").setText("N/A:")
            getattr(self, f"debug_r_val_{i + 1}").setText("N/A")
        for i in range(4):
            getattr(self, f"debug_angle_name_{i + 1}").setText("N/A:")
            getattr(self, f"debug_angle_val1_{i + 1}").setText("N/A")  # [新]
            getattr(self, f"debug_angle_val2_{i + 1}").setText("N/A")  # [新]
            getattr(self, f"debug_angle_diff_{i + 1}").setText("N/A")
        self.debug_angle_avg_label.setText("平均差值:")
        self.debug_angle_avg_value.setText("N/A")

    def _display_current_image_result(self):
        """
        根据 self.current_image_index 从 self.processing_results 加载数据并更新UI。
        """
        if not (0 <= self.current_image_index < len(self.processing_results)):
            return

        result = self.processing_results[self.current_image_index]
        self.image_index_label.setText(f"{self.current_image_index + 1}/{len(self.processing_results)}")

        if not result["success"]:
            self.log_output.append(
                f"无法显示结果: {Path(result.get('source_image_path', 'N/A')).name}. 错误: {result.get('error', '未知')}")
            self.image_viewer.scene().clear()
            self._clear_debug_labels()
            return

        # --- 更新调试信息面板 ---
        if not self.show_analysis_view:
            self._clear_debug_labels()
        else:
            local_lattice_info = result.get("local_lattice_analysis", {})
            angles = local_lattice_info.get("a_center_angles_degrees", {})
            ordered_points = local_lattice_info.get("ordered_points", [])
            num_neighbors = len(ordered_points) - 1
            r_values = local_lattice_info.get("r_distances_center_to_neighbor", {})

            # 更新 r-距离 标签
            key_names = [f"r1{i + 2}" for i in range(8)]
            for i, key in enumerate(key_names):
                label_name = getattr(self, f"debug_r_name_{i + 1}")
                label_val = getattr(self, f"debug_r_val_{i + 1}")
                label_name.setText(f"{key}:")
                if key in r_values:
                    label_val.setText(f"{r_values[key]:.4f}")
                else:
                    label_val.setText("N/A")

            # 更新角度差值 (及新增的角度值)
            diff_values = []
            pairs = []
            if num_neighbors == 8:
                pairs = [("a213", "a617"), ("a314", "a718"), ("a415", "a819"), ("a516", "a912")]
            elif num_neighbors == 6:
                pairs = [("a213", "a516"), ("a314", "a617"), ("a415", "a712")]

            for i in range(4):
                name_label = getattr(self, f"debug_angle_name_{i + 1}")
                val1_label = getattr(self, f"debug_angle_val1_{i + 1}")  # [新]
                val2_label = getattr(self, f"debug_angle_val2_{i + 1}")  # [新]
                diff_label = getattr(self, f"debug_angle_diff_{i + 1}")

                if i < len(pairs):
                    ang1_name, ang2_name = pairs[i]
                    name_label.setText(f"{ang1_name}/{ang2_name}:")

                    # 获取具体角度值
                    val1 = angles.get(ang1_name)
                    val2 = angles.get(ang2_name)

                    if val1 is not None and val2 is not None:
                        val1_label.setText(f"{val1:.2f}°")
                        val2_label.setText(f"{val2:.2f}°")
                        diff = math.fabs(val1 - val2)
                        diff_label.setText(f"{diff:.2f}°")
                        diff_values.append(diff)
                    else:
                        val1_label.setText("N/A")
                        val2_label.setText("N/A")
                        diff_label.setText("N/A")
                else:
                    name_label.setText("N/A:")
                    val1_label.setText("N/A")
                    val2_label.setText("N/A")
                    diff_label.setText("N/A")

            if diff_values:
                avg_diff = sum(diff_values) / len(diff_values)
                self.debug_angle_avg_value.setText(f"{avg_diff:.4f}°")
            else:
                self.debug_angle_avg_value.setText("N/A")

        # --- 更新选中点列表 ---
        self.point_list_table.blockSignals(True)
        self.point_list_table.setRowCount(0)
        inlier_data = result.get("filtered_detections", [])
        self.point_list_table.setRowCount(len(inlier_data))

        for row, point in enumerate(inlier_data):
            point_id = point["id"]
            x = point["midpoint"]["x"]
            y = point["midpoint"]["y"]
            item_id = QTableWidgetItem(point_id)
            item_x = QTableWidgetItem(str(x))
            item_y = QTableWidgetItem(str(y))
            item_id.setData(Qt.UserRole, point_id)
            self.point_list_table.setItem(row, 0, item_id)
            self.point_list_table.setItem(row, 1, item_x)
            self.point_list_table.setItem(row, 2, item_y)
            if point_id in self.selected_point_ids:
                self.point_list_table.selectRow(row)
        self.point_list_table.blockSignals(False)

        # --- 更新图像视图 (支持原图强度热力图) ---
        image_path = result["source_image_path"]
        is_heatmap_on = self.action_tb_heatmap.isChecked()
        heatmap_type = "无"

        if is_heatmap_on:
            bgra_data = generate_intensity_heatmap_from_file(image_path)
            if bgra_data is not None:
                h, w, ch = bgra_data.shape
                q_image = QImage(bgra_data.data, w, h, w * 4, QImage.Format_ARGB32)
                pixmap = QPixmap.fromImage(q_image.copy())
            else:
                self.log_output.append("错误: 无法生成强度热力图，已回退到灰度原图。")
                pixmap = QPixmap(image_path)
        else:
            pixmap = QPixmap(image_path)

        sphere_radius = self.view_sphere_radius_sb.value()
        heatmap_opacity = 1.0

        self.image_viewer.set_image_data(
            pixmap,
            inlier_data,
            result.get("lattice_nodes_for_viz", []),
            result.get("local_lattice_analysis", {}),
            None,
            heatmap_type,
            heatmap_opacity,
            selected_ids=self.selected_point_ids,
            sphere_radius=sphere_radius,
            show_analysis=self.show_analysis_view
        )

    @pyqtSlot(set)
    def on_view_selection_changed(self, selected_ids):
        if selected_ids == self.selected_point_ids:
            return
        self.selected_point_ids = selected_ids
        self.point_list_table.blockSignals(True)
        self.point_list_table.clearSelection()
        rows_to_select = []
        for row in range(self.point_list_table.rowCount()):
            item = self.point_list_table.item(row, 0)
            if item and item.data(Qt.UserRole) in selected_ids:
                rows_to_select.append(row)
        for row in rows_to_select:
            self.point_list_table.selectRow(row)
        if rows_to_select:
            self.point_list_table.scrollToItem(self.point_list_table.item(rows_to_select[0], 0),
                                               QAbstractItemView.EnsureVisible)
        self.point_list_table.blockSignals(False)

    @pyqtSlot()
    def on_list_selection_changed(self):
        selected_rows = sorted(list(set(index.row() for index in self.point_list_table.selectedIndexes())))
        new_selection = set()
        for row in selected_rows:
            item = self.point_list_table.item(row, 0)
            if item:
                new_selection.add(item.data(Qt.UserRole))
        if new_selection == self.selected_point_ids:
            return
        self.selected_point_ids = new_selection
        self.image_viewer._update_point_selection(new_selection)

    @pyqtSlot(set)
    def on_delete_requested_from_signal(self, ids_to_delete):
        self._trigger_delete(ids_to_delete)

    def _trigger_delete(self, ids_to_delete):
        if not ids_to_delete or not (0 <= self.current_image_index < len(self.processing_results)):
            return
        self.selected_point_ids = set()
        current_result = self.processing_results[self.current_image_index]
        # [修改] 传入 forced_center_id
        worker = DataEditWorker(
            current_result,
            self.data_edit_signals,
            ids_to_remove=ids_to_delete,
            forced_center_id=self.manual_center_id
        )
        self.thread_pool.start(worker)

    @pyqtSlot(QPointF)
    def on_add_point_requested(self, scene_pos):
        if not (0 <= self.current_image_index < len(self.processing_results)):
            return
        current_result = self.processing_results[self.current_image_index]
        new_id = f"manual_{int(time.time() * 1000)}"
        new_point_data = {"id": new_id, "midpoint": {"x": int(scene_pos.x()), "y": int(scene_pos.y())}}
        self.log_output.append(
            f"添加新点: {new_id} at ({new_point_data['midpoint']['x']}, {new_point_data['midpoint']['y']})")
        self.selected_point_ids = {new_id}
        # [修改] 传入 forced_center_id
        worker = DataEditWorker(
            current_result,
            self.data_edit_signals,
            point_to_add=new_point_data,
            forced_center_id=self.manual_center_id
        )
        self.thread_pool.start(worker)

    # [新增] 槽函数：处理手动设置中心点
    @pyqtSlot(str)
    def on_set_manual_center(self, point_id):
        if not (0 <= self.current_image_index < len(self.processing_results)):
            return

        self.manual_center_id = point_id
        self.log_output.append(f"已手动指定中心点 (Point 1): {point_id}")

        # 触发重算，仅传递 forced_center_id，不增删改点
        current_result = self.processing_results[self.current_image_index]
        worker = DataEditWorker(
            current_result,
            self.data_edit_signals,
            forced_center_id=self.manual_center_id
        )
        self.thread_pool.start(worker)

    # [新增] 槽函数：处理重置中心点请求
    @pyqtSlot()
    def on_reset_manual_center(self):
        if not (0 <= self.current_image_index < len(self.processing_results)):
            return

        # 1. 核心逻辑：将手动ID设为 None，恢复自动逻辑
        self.manual_center_id = None
        self.log_output.append("已重置为自动计算中心点。")

        # 2. 触发重算 (传入 forced_center_id=None)
        current_result = self.processing_results[self.current_image_index]
        worker = DataEditWorker(
            current_result,
            self.data_edit_signals,
            forced_center_id=None  # 明确传入 None
        )
        self.thread_pool.start(worker)

    @pyqtSlot(str, QPointF)
    def on_point_moved(self, point_id, scene_pos):
        if not (0 <= self.current_image_index < len(self.processing_results)):
            return
        current_result = self.processing_results[self.current_image_index]
        modified_point_data = None
        for p in current_result['filtered_detections']:
            if p['id'] == point_id:
                modified_point_data = p.copy()
                break
        if modified_point_data is None:
            self.log_output.append(f"错误：找不到要移动的点 {point_id}")
            return
        modified_point_data['midpoint'] = {'x': int(scene_pos.x()), 'y': int(scene_pos.y())}
        self.log_output.append(
            f"移动点: {point_id} 至 ({modified_point_data['midpoint']['x']}, {modified_point_data['midpoint']['y']})")
        # [修改] 传入 forced_center_id
        worker = DataEditWorker(
            current_result,
            self.data_edit_signals,
            point_to_modify=modified_point_data,
            forced_center_id=self.manual_center_id
        )
        self.thread_pool.start(worker)

    @pyqtSlot(dict)
    def on_data_edit_finished(self, new_result):
        self.processing_results[self.current_image_index] = new_result
        self._display_current_image_result()

    @pyqtSlot()
    def on_save_current_image(self):
        if self.current_image_index < 0 or not self.image_viewer._pixmap_item:
            self.log_output.append("错误: 没有可保存的当前图像。")
            return
        current_result = self.processing_results[self.current_image_index]
        original_path = Path(current_result.get("source_image_path", "image.png"))
        save_dir = self.output_dir if self.output_dir else "."
        default_path = os.path.join(save_dir, f"{original_path.stem}_annotated.png")
        file_path, _ = QFileDialog.getSaveFileName(self, "保存当前图像", default_path,
                                                   "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)")
        if not file_path: return
        scene = self.image_viewer.scene()
        image = QImage(scene.sceneRect().size().toSize(), QImage.Format_ARGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        scene.render(painter)
        painter.end()
        image.save(file_path)
        self.log_output.append(f"当前图像已保存到: {file_path}")

    @pyqtSlot()
    def on_save_all_images(self):
        if not self.processing_results:
            self.log_output.append("错误: 没有可保存的处理结果。")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择保存目录",
                                                      self.settings.value("paths/output_dir", "."))
        default_open_dir = self.output_dir if self.output_dir else "."
        output_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", default_open_dir)
        if not output_dir: return
        self.log_output.append(f"开始批量保存在...: {output_dir}")
        view_settings = {"sphere_radius": self.view_sphere_radius_sb.value(), "show_analysis": self.show_analysis_view}
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        worker = SaveAllWorker(self.processing_results, view_settings, output_dir)
        worker.signals.progress.connect(self.on_save_all_progress)
        worker.signals.finished.connect(self.on_save_all_finished)
        worker.signals.logMessage.connect(self.log_output.append)
        self.thread_pool.start(worker)

    @pyqtSlot(int, int, str)
    def on_save_all_progress(self, current, total, filename):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)

    @pyqtSlot(int, str)
    def on_save_all_finished(self, count, output_dir):
        self.progress_bar.setVisible(False)
        self.log_output.append(f"批量保存完成: {count} 张图片。")

    @pyqtSlot()
    def on_save_current_debug_info(self):
        if not (0 <= self.current_image_index < len(self.processing_results)):
            self.log_output.append("错误: 没有可保存的当前调试信息。")
            return
        result = self.processing_results[self.current_image_index]
        original_path = Path(result.get("source_image_path", "image.json"))
        default_name = f"{original_path.stem}_debug.json"
        default_path = os.path.join(self.settings.value("paths/output_dir", "."), default_name)
        file_path, _ = QFileDialog.getSaveFileName(self, "保存调试信息", default_path, "JSON Files (*.json)")
        if not file_path: return
        debug_data = {
            "source_image": result.get("source_image_path"),
            "ransac_debug_info": result.get("ransac_debug_info", {}),
            "local_lattice_analysis": result.get("local_lattice_analysis", {})
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=4, ensure_ascii=False)
            self.log_output.append(f"调试信息已保存到: {file_path}")
        except Exception as e:
            self.log_output.append(f"错误: 保存调试信息失败: {e}")

    @pyqtSlot()
    def on_save_all_debug_info(self):
        if not self.processing_results:
            self.log_output.append("错误: 没有可保存的处理结果。")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "选择保存所有调试信息的目录",
                                                      self.settings.value("paths/output_dir", "."))
        if not output_dir: return
        self.log_output.append(f"开始批量保存调试信息到: {output_dir}")
        count = 0
        for result in self.processing_results:
            if not result.get("success"): continue
            original_path = Path(result.get("source_image_path", f"image_{count}.json"))
            file_name = f"{original_path.stem}_debug.json"
            file_path = os.path.join(output_dir, file_name)
            debug_data = {
                "source_image": result.get("source_image_path"),
                "ransac_debug_info": result.get("ransac_debug_info", {}),
                "local_lattice_analysis": result.get("local_lattice_analysis", {})
            }
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(debug_data, f, indent=4, ensure_ascii=False)
                count += 1
            except Exception as e:
                self.log_output.append(f"错误: 保存 {file_name} 失败: {e}")
        self.log_output.append(f"批量保存完成。共保存 {count} 个调试文件。")

    @pyqtSlot(bool)
    def on_toggle_heatmap(self, checked):

        self._trigger_rerender()

    @pyqtSlot(bool)
    def on_toggle_edit_mode(self, checked):
        self.is_edit_mode = checked
        self.image_viewer.set_edit_mode(checked)
        self.log_output.append(f"--- 编辑模式: {'开启' if checked else '关闭'} ---")
        if checked:
            self.image_viewer.setInteractive(True)
            rerender = False

            # (保留) 分析视图仍然需要互斥，因为视图逻辑不同
            if self.show_analysis_view:
                self.show_analysis_view = False
                self.log_output.append("进入编辑模式，自动关闭分析视图。")
                rerender = True

            # [修改] 移除了热力图互斥逻辑
            # if self.action_tb_heatmap.isChecked(): ... (已删除)

            if rerender:
                self._trigger_rerender()
        else:
            self.image_viewer.setDragMode(QGraphicsView.NoDrag)
            self.image_viewer.setInteractive(True)

    def on_save_config(self):
        default_path = self.settings.value("paths/config_path", "config.json")
        file_path, _ = QFileDialog.getSaveFileName(self, "保存配置", default_path, "JSON Files (*.json)")
        if not file_path: return
        self.settings.setValue("paths/config_path", file_path)

        def get_ransac_params(widget):
            return {
                "num_iterations": widget.num_iterations_sb.value(),
                "inlier_threshold": widget.inlier_threshold_dsb.value(),
                "min_inliers_ratio": widget.min_inliers_ratio_dsb.value(),
                "k_neighbors_for_basis": widget.k_neighbors_for_basis_sb.value(),
                "basis_angle_tolerance_deg": widget.basis_angle_tolerance_deg_sb.value(),
                "min_basis_len_override": widget.min_basis_len_override_dsb.value(),
                "max_basis_len_ratio": widget.max_basis_len_ratio_dsb.value(),
                "adaptive_min_basis_len_neighbors": widget.adaptive_min_basis_len_neighbors_sb.value(),
                "adaptive_min_basis_len_percentile": widget.adaptive_min_basis_len_percentile_sb.value()
            }

        config_data = {
            "paths": {"model_path": self.model_path, "source_dir": self.source_dir, "output_dir": self.output_dir},
            "sahi_params": {
                "slice_height": self.slice_height_sb.value(), "slice_width": self.slice_width_sb.value(),
                "overlap_height_ratio": self.overlap_height_ratio_dsb.value(),
                "overlap_width_ratio": self.overlap_width_ratio_dsb.value(),
                "confidence_threshold": self.confidence_threshold_dsb.value(),
                "iou_threshold": self.iou_threshold_dsb.value(),
                "size_threshold": self.size_threshold_sb.value()
            },
            "post_processing": {
                # [修改] 直接从 group box 保存状态
                "enable": self.post_process_group.isChecked(),
                "params_small": get_ransac_params(self.params_small),
                "params_large": get_ransac_params(self.params_large)
            },
            "view_settings": {"sphere_radius": self.view_sphere_radius_sb.value()}
        }
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
            self.log_output.append(f"配置已保存到: {file_path}")
        except Exception as e:
            self.log_output.append(f"错误: 保存配置失败: {e}")

    def on_load_config(self):
        default_path = self.settings.value("paths/config_path", ".")
        file_path, _ = QFileDialog.getOpenFileName(self, "加载配置", default_path, "JSON Files (*.json)")
        if not file_path: return
        self.settings.setValue("paths/config_path", file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception as e:
            self.log_output.append(f"错误: 加载配置失败: {e}")
            return

        def set_ransac_params(widget, params_dict):
            widget.num_iterations_sb.setValue(params_dict.get("num_iterations", 100))
            widget.inlier_threshold_dsb.setValue(params_dict.get("inlier_threshold", 5.0))
            widget.min_inliers_ratio_dsb.setValue(params_dict.get("min_inliers_ratio", 0.7))
            widget.k_neighbors_for_basis_sb.setValue(params_dict.get("k_neighbors_for_basis", 10))
            widget.basis_angle_tolerance_deg_sb.setValue(params_dict.get("basis_angle_tolerance_deg", 30))
            widget.min_basis_len_override_dsb.setValue(params_dict.get("min_basis_len_override", 20.0))
            widget.max_basis_len_ratio_dsb.setValue(params_dict.get("max_basis_len_ratio", 1.5))
            widget.adaptive_min_basis_len_neighbors_sb.setValue(params_dict.get("adaptive_min_basis_len_neighbors", 10))
            widget.adaptive_min_basis_len_percentile_sb.setValue(
                params_dict.get("adaptive_min_basis_len_percentile", 16))

        paths = config_data.get("paths", {})
        self.model_path = paths.get("model_path", "")
        self.source_dir = paths.get("source_dir", "")
        self.output_dir = paths.get("output_dir", "")
        self.settings.setValue("paths/model_path", self.model_path)
        self.settings.setValue("paths/source_dir", self.source_dir)
        self.settings.setValue("paths/output_dir", self.output_dir)
        self.log_output.append("--- 已加载路径 ---")
        self.log_output.append(f"模型: {self.model_path}")
        self.log_output.append(f"图片目录: {self.source_dir}")
        self.log_output.append(f"输出目录: {self.output_dir}")

        sahi = config_data.get("sahi_params", {})
        self.slice_height_sb.setValue(sahi.get("slice_height", 1024))
        self.slice_width_sb.setValue(sahi.get("slice_width", 1024))
        self.overlap_height_ratio_dsb.setValue(sahi.get("overlap_height_ratio", 0.25))
        self.overlap_width_ratio_dsb.setValue(sahi.get("overlap_width_ratio", 0.25))
        self.confidence_threshold_dsb.setValue(sahi.get("confidence_threshold", 0.25))
        self.iou_threshold_dsb.setValue(sahi.get("iou_threshold", 0.01))
        self.size_threshold_sb.setValue(sahi.get("size_threshold", 1030))

        pp = config_data.get("post_processing", {})
        # [修改] 加载配置到 group box
        self.post_process_group.setChecked(pp.get("enable", True))
        set_ransac_params(self.params_small, pp.get("params_small", {}))
        set_ransac_params(self.params_large, pp.get("params_large", {}))

        view = config_data.get("view_settings", {})
        self.view_sphere_radius_sb.setValue(view.get("sphere_radius", 8))
        self.log_output.append(f"配置已从 {file_path} 加载并应用。")

    def _save_settings(self):
        self.settings.setValue("view/sphere_radius", self.view_sphere_radius_sb.value())
        self.settings.setValue("paths/source_dir", self.source_dir)
        self.settings.setValue("paths/model_path", self.model_path)
        self.settings.setValue("paths/output_dir", self.output_dir)

    def _load_settings(self):
        self.view_sphere_radius_sb.setValue(int(self.settings.value("view/sphere_radius", 8)))
        self.source_dir = self.settings.value("paths/source_dir", "")
        self.model_path = self.settings.value("paths/model_path", "")
        self.output_dir = self.settings.value("paths/output_dir", "")
        if self.source_dir and os.path.exists(self.source_dir):
            try:
                # 重新执行扫描逻辑 (复制自 _select_source_dir)
                self.image_list = sorted([str(p) for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"] for p in
                                          Path(self.source_dir).glob(ext)])
                if self.image_list:
                    self.log_output.append(f"已自动加载上次的图片目录: {self.source_dir}")
                    self.log_output.append(f"在目录中找到 {len(self.image_list)} 张图片。")
            except Exception as e:
                self.log_output.append(f"自动加载图片目录失败: {e}")
        if self.model_path:
            if os.path.exists(self.model_path):
                self.log_output.append(f"已记录上次的模型路径: {Path(self.model_path).name}")
                # 可选：如果希望启动时直接自动加载模型，可以取消下面这行的注释
                # self.on_load_model()
            else:
                self.log_output.append(f"警告: 上次使用的模型文件不存在: {self.model_path}")
            if self.output_dir:
                if os.path.exists(self.output_dir):
                    self.log_output.append(f"已自动加载输出保存目录: {self.output_dir}")

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support

        freeze_support()
    except ImportError:
        pass
    import ctypes
    import platform

    if platform.system() == 'Windows':
        try:
            myappid = 'mycompany.atom.detector.gui.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
    app = QApplication(sys.argv)
    if os.path.exists("tubiao/LogosAtomIcon.png"):
        app.setWindowIcon(QIcon("tubiao/LogosAtomIcon.png"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())