from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTabWidget, QLabel, QGridLayout
)
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtCore import Qt


def _add_form_row_with_tooltip(form_layout, label_text, widget, tooltip_text):
    """一个私有辅助函数，用于添加带工具提示的表单行"""
    label = QLabel(label_text)
    label.setToolTip(tooltip_text)
    widget.setToolTip(tooltip_text)
    form_layout.addRow(label, widget)


def create_sahi_params_group():
    """创建 SAHI 检测参数的 QGroupBox"""
    sahi_params_group = QGroupBox("SAHI 检测参数")
    sahi_params_group.setCheckable(True)
    sahi_params_group.setChecked(True)
    sahi_group_layout = QVBoxLayout(sahi_params_group)
    sahi_params_container = QWidget()
    sahi_params_layout = QFormLayout(sahi_params_container)

    slice_height_label = QLabel("切片高度:")
    slice_height_sb = QSpinBox()
    slice_height_sb.setRange(128, 4096)
    slice_height_sb.setValue(1024)
    sahi_params_layout.addRow(slice_height_label, slice_height_sb)

    slice_width_label = QLabel("切片宽度:")
    slice_width_sb = QSpinBox()
    slice_width_sb.setRange(128, 4096)
    slice_width_sb.setValue(1024)
    sahi_params_layout.addRow(slice_width_label, slice_width_sb)

    overlap_height_label = QLabel("重叠高度比例:")
    overlap_height_ratio_dsb = QDoubleSpinBox()
    overlap_height_ratio_dsb.setRange(0.0, 0.9)
    overlap_height_ratio_dsb.setSingleStep(0.05)
    overlap_height_ratio_dsb.setValue(0.25)
    sahi_params_layout.addRow(overlap_height_label, overlap_height_ratio_dsb)

    overlap_width_label = QLabel("重叠宽度比例:")
    overlap_width_ratio_dsb = QDoubleSpinBox()
    overlap_width_ratio_dsb.setRange(0.0, 0.9)
    overlap_width_ratio_dsb.setSingleStep(0.05)
    overlap_width_ratio_dsb.setValue(0.25)
    sahi_params_layout.addRow(overlap_width_label, overlap_width_ratio_dsb)

    conf_label = QLabel("置信度阈值:")
    confidence_threshold_dsb = QDoubleSpinBox()
    confidence_threshold_dsb.setRange(0.0, 1.0)
    confidence_threshold_dsb.setSingleStep(0.01)
    confidence_threshold_dsb.setValue(0.25)
    sahi_params_layout.addRow(conf_label, confidence_threshold_dsb)

    iou_label = QLabel("IOU阈值:")
    iou_threshold_dsb = QDoubleSpinBox()
    iou_threshold_dsb.setRange(0.0, 1.0)
    iou_threshold_dsb.setSingleStep(0.01)
    iou_threshold_dsb.setValue(0.01)
    sahi_params_layout.addRow(iou_label, iou_threshold_dsb)

    size_thresh_label = QLabel("尺寸阈值 (小图判断):")
    size_threshold_sb = QSpinBox()
    size_threshold_sb.setRange(0, 4096)
    size_threshold_sb.setValue(1030)
    sahi_params_layout.addRow(size_thresh_label, size_threshold_sb)

    sahi_group_layout.addWidget(sahi_params_container)
    sahi_params_container.setVisible(True)

    # 将控件打包返回
    controls = {
        "slice_height_sb": slice_height_sb,
        "slice_width_sb": slice_width_sb,
        "overlap_height_ratio_dsb": overlap_height_ratio_dsb,
        "overlap_width_ratio_dsb": overlap_width_ratio_dsb,
        "confidence_threshold_dsb": confidence_threshold_dsb,
        "iou_threshold_dsb": iou_threshold_dsb,
        "size_threshold_sb": size_threshold_sb
    }
    return sahi_params_group, controls


def create_ransac_params_widget(title):
    """创建一个 RANSAC 参数配置 QWidget"""
    widget = QWidget()
    layout = QFormLayout(widget)
    tooltips = {
        "迭代次数:": "RANSAC算法随机采样的迭代次数，次数越多找到最优解的概率越大。",
        "内点阈值:": "一个点被视为‘内点’（inlier）时，其到理论晶格点的最大允许距离（像素）。",
        "最小内点比例:": "一个候选晶格模型被接受所需的最低内点比例。",
        "基矢量邻居数:": "从中心采样点周围寻找多少个最近邻居来推断候选的基矢量。",
        "基矢量角度容忍度 (度):": "两个候选基矢量之间允许的最小/最大夹角，以避免共线。",
        "最小基矢量长度:": "基矢量的最小长度硬性下限（像素）。",
        "最大基矢量长度比:": "两个基矢量长度之间允许的最大比例。",
        "自适应基长邻居数:": "用于自适应计算最小基矢量长度时，所考虑的最近邻G数量。",
        "自适应基长百分位数:": "在邻居距离分布中，用于估算最小基矢量的百分位数数值。"
    }

    # 将控件直接附加到 widget 对象上，以便 MainWindow 访问
    widget.num_iterations_sb = QSpinBox()
    widget.num_iterations_sb.setRange(10, 5000)
    _add_form_row_with_tooltip(layout, "迭代次数:", widget.num_iterations_sb, tooltips["迭代次数:"])

    widget.inlier_threshold_dsb = QDoubleSpinBox()
    widget.inlier_threshold_dsb.setRange(1.0, 100.0)
    widget.inlier_threshold_dsb.setSingleStep(0.5)
    _add_form_row_with_tooltip(layout, "内点阈值:", widget.inlier_threshold_dsb, tooltips["内点阈值:"])

    widget.min_inliers_ratio_dsb = QDoubleSpinBox()
    widget.min_inliers_ratio_dsb.setRange(0.0, 1.0)
    widget.min_inliers_ratio_dsb.setSingleStep(0.01)
    _add_form_row_with_tooltip(layout, "最小内点比例:", widget.min_inliers_ratio_dsb,
                               tooltips["最小内点比例:"])

    widget.k_neighbors_for_basis_sb = QSpinBox()
    widget.k_neighbors_for_basis_sb.setRange(3, 50)
    _add_form_row_with_tooltip(layout, "基矢量邻居数:", widget.k_neighbors_for_basis_sb,
                               tooltips["基矢量邻居数:"])

    widget.basis_angle_tolerance_deg_sb = QSpinBox()
    widget.basis_angle_tolerance_deg_sb.setRange(1, 89)
    _add_form_row_with_tooltip(layout, "基矢量角度容忍度 (度):", widget.basis_angle_tolerance_deg_sb,
                               tooltips["基矢量角度容忍度 (度):"])

    widget.min_basis_len_override_dsb = QDoubleSpinBox()
    widget.min_basis_len_override_dsb.setRange(1.0, 500.0)
    widget.min_basis_len_override_dsb.setSingleStep(1.0)
    _add_form_row_with_tooltip(layout, "最小基矢量长度:", widget.min_basis_len_override_dsb,
                               tooltips["最小基矢量长度:"])

    widget.max_basis_len_ratio_dsb = QDoubleSpinBox()
    widget.max_basis_len_ratio_dsb.setRange(1.0, 5.0)
    widget.max_basis_len_ratio_dsb.setSingleStep(0.1)
    _add_form_row_with_tooltip(layout, "最大基矢量长度比:", widget.max_basis_len_ratio_dsb,
                               tooltips["最大基矢量长度比:"])

    widget.adaptive_min_basis_len_neighbors_sb = QSpinBox()
    widget.adaptive_min_basis_len_neighbors_sb.setRange(1, 20)
    _add_form_row_with_tooltip(layout, "自适应基长邻居数:", widget.adaptive_min_basis_len_neighbors_sb,
                               tooltips["自适应基长邻居数:"])

    widget.adaptive_min_basis_len_percentile_sb = QSpinBox()
    widget.adaptive_min_basis_len_percentile_sb.setRange(1, 99)
    _add_form_row_with_tooltip(layout, "自适应基长百分位数:", widget.adaptive_min_basis_len_percentile_sb,
                               tooltips["自适应基长百分位数:"])

    # 根据 title 设置默认值
    if "小图" in title:
        widget.num_iterations_sb.setValue(100)
        widget.inlier_threshold_dsb.setValue(5.0)
        widget.min_inliers_ratio_dsb.setValue(0.7)
        widget.k_neighbors_for_basis_sb.setValue(10)
        widget.basis_angle_tolerance_deg_sb.setValue(30)
        widget.min_basis_len_override_dsb.setValue(20.0)
        widget.max_basis_len_ratio_dsb.setValue(1.5)
        widget.adaptive_min_basis_len_neighbors_sb.setValue(10)
        widget.adaptive_min_basis_len_percentile_sb.setValue(16)
    else:  # 大图
        widget.num_iterations_sb.setValue(100)
        widget.inlier_threshold_dsb.setValue(40.0)
        widget.min_inliers_ratio_dsb.setValue(0.7)
        widget.k_neighbors_for_basis_sb.setValue(10)
        widget.basis_angle_tolerance_deg_sb.setValue(30)
        widget.min_basis_len_override_dsb.setValue(100.0)
        widget.max_basis_len_ratio_dsb.setValue(1.5)
        widget.adaptive_min_basis_len_neighbors_sb.setValue(10)
        widget.adaptive_min_basis_len_percentile_sb.setValue(16)

    return widget


def create_debug_info_panel():
    """创建调试信息面板 (包含角度和r-距离)"""

    debug_info_group_right = QGroupBox("调试信息")
    main_debug_layout = QHBoxLayout(debug_info_group_right)

    # 1. 左侧 (角度) 布局
    angle_layout_container = QWidget()
    angle_layout = QFormLayout(angle_layout_container)
    angle_layout.setContentsMargins(0, 0, 0, 0)

    # 2. 右侧 (r-距离) 布局
    r_distance_container = QWidget()
    r_distance_v_layout = QVBoxLayout(r_distance_container)
    r_distance_v_layout.setContentsMargins(0, 0, 0, 0)
    r_distance_grid_layout = QGridLayout()
    r_distance_grid_layout.setContentsMargins(0, 0, 0, 0)

    mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)

    labels = {}  # 用于存储所有标签并返回

    # 3. 添加 r-距离 标签
    labels["debug_r_header"] = QLabel("中心点-邻居距离 (r):")
    labels["debug_r_header"].setStyleSheet("font-weight: bold; margin-top: 5px;")
    r_distance_v_layout.addWidget(labels["debug_r_header"])

    r_name_labels = []
    r_val_labels = []

    for i in range(8):
        labels[f"debug_r_name_{i + 1}"] = QLabel("N/A:")
        labels[f"debug_r_val_{i + 1}"] = QLabel("N/A")
        labels[f"debug_r_val_{i + 1}"].setFont(mono_font)
        r_name_labels.append(labels[f"debug_r_name_{i + 1}"])
        r_val_labels.append(labels[f"debug_r_val_{i + 1}"])

    for i in range(len(r_name_labels)):
        row = i // 3
        col = i % 3
        r_distance_grid_layout.addWidget(r_name_labels[i], row, col * 2)
        r_distance_grid_layout.addWidget(r_val_labels[i], row, col * 2 + 1)

    r_distance_grid_layout.setColumnStretch(1, 1)
    r_distance_grid_layout.setColumnStretch(3, 1)
    r_distance_grid_layout.setColumnStretch(5, 1)

    r_distance_v_layout.addLayout(r_distance_grid_layout)
    r_distance_v_layout.addStretch(1)

    # 4. 添加 角度 标签
    labels["debug_angle_header"] = QLabel("对角差值:")
    labels["debug_angle_header"].setStyleSheet("font-weight: bold; margin-top: 5px;")
    angle_layout.addRow(labels["debug_angle_header"])

    for i in range(4):
        labels[f"debug_angle_name_{i + 1}"] = QLabel("N/A:")
        labels[f"debug_angle_diff_{i + 1}"] = QLabel("N/A")
        labels[f"debug_angle_diff_{i + 1}"].setFont(mono_font)
        angle_layout.addRow(labels[f"debug_angle_name_{i + 1}"], labels[f"debug_angle_diff_{i + 1}"])

    labels["debug_angle_avg_label"] = QLabel("平均差值:")
    labels["debug_angle_avg_label"].setStyleSheet("font-weight: bold;")
    labels["debug_angle_avg_value"] = QLabel("N/A")
    labels["debug_angle_avg_value"].setFont(mono_font)
    labels["debug_angle_avg_value"].setStyleSheet("font-weight: bold; color: #006666;")
    angle_layout.addRow(labels["debug_angle_avg_label"], labels[f"debug_angle_avg_value"])

    # 5. 组合主布局
    main_debug_layout.addWidget(angle_layout_container)
    main_debug_layout.addWidget(r_distance_container)
    main_debug_layout.setStretch(0, 1)  # 角度列
    main_debug_layout.setStretch(1, 2)  # r-距离列 (更宽)

    return debug_info_group_right, labels