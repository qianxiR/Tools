#!/usr/bin/env python3
"""
SegEarth-OV3 GUI - 基于 SAM 3 的交互式语义分割工具
输入图片 + 类别文本提示，调用 SAM 3 模型进行多类语义分割，界面展示彩色掩码结果。

入参:
    --ckpt       : SAM 3 权重路径 (默认 sam3/sam3.pt)
    --bpe        : BPE 词表路径
    --conf       : 置信度阈值 (默认 0.5)
    --classes    : 类别文件路径 (默认使用内置 7 类)

方法:
    基于 PySide6 构建图形界面，后台线程加载 SAM 3 模型并执行多轮迭代推理，
    复用 infer.py 中的 load_model() / multipass_inference() / postprocess() 等核心逻辑。

出参:
    无返回值，结果直接在 GUI 界面中展示 (原图 / 彩色掩码 / 叠加效果)。
"""

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# 日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gui.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("segearth.gui")


# ---------------------------------------------------------------------------
# 后台工作线程
# ---------------------------------------------------------------------------

class ModelLoadWorker(QThread):
    """
    后台线程加载 SAM 3 模型，避免阻塞 UI。

    入参:
        ckpt: SAM 3 权重路径
        bpe: BPE 词表路径
        conf: 置信度阈值
        device: 推理设备

    方法:
        run() 中调用 infer.py 的 load_model() 加载模型

    出参:
        finished signal: (processor, device) 元组
        error signal: 错误信息字符串
    """
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, ckpt, bpe, conf, device):
        super().__init__()
        self.ckpt = ckpt
        self.bpe = bpe
        self.conf = conf
        self.device = device

    def run(self):
        log.info("ModelLoadWorker 开始执行, ckpt=%s, device=%s", self.ckpt, self.device)
        t0 = time.time()
        try:
            from infer import load_model
            processor, device = load_model(self.ckpt, self.bpe, self.conf, self.device)
            elapsed = time.time() - t0
            log.info("ModelLoadWorker 完成, 耗时 %.2f 秒", elapsed)
            self.finished.emit(processor, device)
        except Exception as e:
            elapsed = time.time() - t0
            log.exception("ModelLoadWorker 失败 (耗时 %.2f 秒): %s", elapsed, e)
            self.error.emit(str(e))


class InferenceWorker(QThread):
    """
    后台线程执行多类语义分割推理，避免阻塞 UI。

    入参:
        processor: Sam3Processor 实例
        device: 推理设备
        image_path: 输入图片路径
        query_words: 文本提示列表
        query_idx_tensor: 查询→类别映射 tensor
        num_cls: 类别总数 (含 background)
        num_queries: 查询总数
        prob: 初始置信度阈值
        max_passes: 多轮迭代最大轮数
        coverage_target: 目标覆盖率
        postprocess_kwargs: 后处理参数 dict

    方法:
        run() 中调用 multipass_inference() + postprocess() + build_palette()

    出参:
        result signal: (rgb_orig, color_mask, overlay, seg_pred_np, cls_name_map)
        error signal: 错误信息字符串
    """
    result = Signal(object, object, object, object, object)
    error = Signal(str)

    def __init__(self, processor, device, image_path, query_words, query_idx_tensor,
                 num_cls, num_queries, prob=0.1, max_passes=10, coverage_target=0.95,
                 postprocess_kwargs=None):
        super().__init__()
        self.processor = processor
        self.device = device
        self.image_path = image_path
        self.query_words = query_words
        self.query_idx_tensor = query_idx_tensor
        self.num_cls = num_cls
        self.num_queries = num_queries
        self.prob = prob
        self.max_passes = max_passes
        self.coverage_target = coverage_target
        self.postprocess_kwargs = postprocess_kwargs or {}

    def run(self):
        log.info("InferenceWorker 开始执行, image=%s, num_cls=%d, num_queries=%d",
                 self.image_path, self.num_cls, self.num_queries)
        t0 = time.time()
        try:
            from PIL import Image
            from infer import multipass_inference, postprocess, build_palette

            # 读取图片
            Image.MAX_IMAGE_PIXELS = None
            image = Image.open(self.image_path).convert("RGB")
            w, h = image.size
            rgb_orig = np.array(image)
            log.info("    图片尺寸: %dx%d", w, h)

            # 多轮迭代推理
            log.info("    开始多轮迭代推理 (max=%d, target=%.0f%%)...",
                     self.max_passes, self.coverage_target * 100)
            seg_pred_np = multipass_inference(
                self.processor, self.query_words, self.query_idx_tensor,
                self.num_cls, self.num_queries, self.device, image,
                prob=self.prob, max_passes=self.max_passes,
                coverage_target=self.coverage_target,
            )

            # 后处理
            seg_pred_np = postprocess(seg_pred_np, self.num_cls, **self.postprocess_kwargs)
            log.info("    后处理完成")

            # 构建类别名映射
            cls_name_map = {}
            for idx, word in enumerate(self.query_words):
                cls_id = self.query_idx_tensor[idx].item()
                cls_name_map.setdefault(cls_id, [])
                cls_name_map[cls_id].append(word)
            # class 0 = background
            cls_name_map.setdefault(0, ["background"])

            # 着色：单类 → 用 NAME_COLOR 单色掩码；多类 → 多色 palette
            single_class = (self.num_cls == 2)  # 只有 background + 1 个前景类
            if single_class:
                from infer import NAME_COLOR
                fg_id = 1
                names = cls_name_map.get(fg_id, [])
                fg_color = NAME_COLOR.get(names[0], (255, 0, 0)) if names else (255, 0, 0)
                color_mask = np.full((*seg_pred_np.shape, 3), 255, dtype=np.uint8)
                fg_pixels = seg_pred_np == fg_id
                color_mask[fg_pixels] = fg_color
            else:
                palette = build_palette(self.num_cls, cls_name_map)
                palette[0] = [255, 255, 255]  # background → 白色
                color_mask = palette[seg_pred_np]

            # 叠加图
            overlay = _make_overlay(rgb_orig, color_mask)

            elapsed = time.time() - t0
            present_cls = sorted(set(seg_pred_np[seg_pred_np > 0].tolist()))
            log.info("    推理完成, 耗时 %.2f 秒, 检测到类别: %s", elapsed, present_cls)
            self.result.emit(rgb_orig, color_mask, overlay, seg_pred_np, cls_name_map)

        except Exception as e:
            elapsed = time.time() - t0
            log.exception("InferenceWorker 失败 (耗时 %.2f 秒): %s", elapsed, e)
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _make_overlay(rgb, color_mask, alpha=0.5):
    """
    将彩色掩码以半透明方式叠加到原图上。

    入参:
        rgb: H×W×3 uint8 RGB 图片
        color_mask: H×W×3 uint8 彩色掩码
        alpha: 叠加透明度 (默认 0.5)

    出参:
        overlay: H×W×3 uint8 叠加效果图
    """
    # 只对非背景区域叠加
    bg = (color_mask.sum(axis=-1) == 255 * 3)  # 白色=背景
    overlay = rgb.copy()
    overlay_f = overlay.astype(np.float32)
    mask_f = color_mask.astype(np.float32)
    # 非背景区域做混合
    fg = ~bg
    overlay_f[fg] = alpha * mask_f[fg] + (1 - alpha) * overlay_f[fg]
    return np.clip(overlay_f, 0, 255).astype(np.uint8)


def _np_to_pixmap(arr, target_size=None):
    """
    numpy RGB 数组 → QPixmap，用于在 QLabel 中展示。

    入参:
        arr: H×W×3 uint8 RGB 数组
        target_size: 可选 (w, h) 目标尺寸，等比缩放

    出参:
        QPixmap 实例
    """
    h, w = arr.shape[:2]
    data = np.ascontiguousarray(arr)
    qimg = QImage(data.data, w, h, 3 * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if target_size:
        tw, th = target_size
        pix = pix.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class SegEarthWindow(QWidget):
    """
    SegEarth-OV3 分割工具主窗口。

    入参:
        args: argparse.Namespace，包含模型路径等默认参数

    方法:
        组合模型配置区、输入区、结果展示区三个主要布局，
        通过后台线程加载模型和执行推理。

    出参:
        无返回值，所有结果在 GUI 界面中展示。
    """

    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("SegEarth-OV3 语义分割工具 (SAM 3)")
        self.setMinimumSize(1200, 800)
        self.processor = None
        self.device = None
        self._args = args
        self._init_ui(args)

    # ---- UI 初始化 ----

    def _init_ui(self, args):
        root = QVBoxLayout(self)

        # 模型配置区
        root.addWidget(self._build_config_group(args))
        # 输入区
        root.addWidget(self._build_input_group())
        # 结果展示区
        root.addWidget(self._build_result_group())
        # 状态栏
        self.status_bar = QStatusBar()
        root.addWidget(self.status_bar)
        self.status_bar.showMessage("就绪 — 请先加载模型")

    def _build_config_group(self, args):
        """构建模型配置区域"""
        group = QGroupBox("模型配置")
        layout = QVBoxLayout()

        def _path_row(label, default, row_tag, is_dir=False):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            le = QLineEdit(default)
            le.setObjectName(row_tag)
            row.addWidget(le, stretch=1)
            btn = QPushButton("浏览...")
            btn.clicked.connect(lambda: self._browse(le, f"选择{label.replace(':', '').strip()}", is_dir=is_dir))
            row.addWidget(btn)
            return row

        layout.addLayout(_path_row("SAM 3 权重:", args.ckpt, "ckpt"))
        layout.addLayout(_path_row("BPE 词表:", args.bpe, "bpe"))
        layout.addLayout(_path_row("类别文件(可选):", args.classes or "", "classes_file"))

        # 设备 + 置信度 + 加载按钮
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText(args.device)
        ctrl.addWidget(self.device_combo)

        ctrl.addWidget(QLabel("置信度:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(args.conf)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setToolTip("SAM 3 检测置信度阈值")
        ctrl.addWidget(self.conf_spin)

        self.load_btn = QPushButton("加载模型")
        self.load_btn.clicked.connect(self._on_load_model)
        ctrl.addWidget(self.load_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)
        group.setLayout(layout)
        return group

    def _build_input_group(self):
        """构建输入区域"""
        group = QGroupBox("输入")
        layout = QVBoxLayout()

        # 图片选择
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("选择图片:"))
        self.image_edit = QLineEdit()
        img_row.addWidget(self.image_edit, stretch=1)
        img_btn = QPushButton("浏览...")
        img_btn.clicked.connect(lambda: self._browse(self.image_edit, "选择图片", is_image=True))
        img_row.addWidget(img_btn)
        # 预览缩略图
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(64, 64)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet("border: 1px solid #ccc;")
        self.thumb_label.setText("预览")
        img_row.addWidget(self.thumb_label)
        layout.addLayout(img_row)

        # 文本提示 — 输入多个类别，逗号或空格分隔
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("分割类别:"))
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText(
            "输入类别名，逗号或空格分隔，如: building,road,water  (留空使用默认7类)"
        )
        prompt_row.addWidget(self.prompt_edit, stretch=1)
        layout.addLayout(prompt_row)

        # 推理参数 + 分割按钮
        btn_row = QHBoxLayout()

        btn_row.addWidget(QLabel("初始阈值:"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.01, 0.99)
        self.prob_spin.setSingleStep(0.05)
        self.prob_spin.setValue(0.7)
        self.prob_spin.setDecimals(2)
        self.prob_spin.setToolTip("多轮推理初始置信度阈值 (越低越宽松)")
        btn_row.addWidget(self.prob_spin)

        btn_row.addWidget(QLabel("最大轮数:"))
        self.max_passes_spin = QSpinBox()
        self.max_passes_spin.setRange(1, 30)
        self.max_passes_spin.setValue(10)
        self.max_passes_spin.setToolTip("多轮迭代推理最大轮数")
        btn_row.addWidget(self.max_passes_spin)

        btn_row.addWidget(QLabel("目标覆盖率:"))
        self.coverage_spin = QDoubleSpinBox()
        self.coverage_spin.setRange(0.5, 1.0)
        self.coverage_spin.setSingleStep(0.05)
        self.coverage_spin.setValue(0.95)
        self.coverage_spin.setDecimals(2)
        self.coverage_spin.setToolTip("目标覆盖率，达到后停止迭代")
        btn_row.addWidget(self.coverage_spin)

        btn_row.addStretch()
        self.seg_btn = QPushButton("开始分割")
        self.seg_btn.setEnabled(False)
        self.seg_btn.clicked.connect(self._on_segment)
        btn_row.addWidget(self.seg_btn)
        layout.addLayout(btn_row)

        # 后处理参数
        pp_row = QHBoxLayout()
        pp_row.addWidget(QLabel("后处理 —"))

        pp_row.addWidget(QLabel("平滑核:"))
        self.smooth_kernel_spin = QSpinBox()
        self.smooth_kernel_spin.setRange(0, 20)
        self.smooth_kernel_spin.setValue(3)
        self.smooth_kernel_spin.setToolTip("形态学平滑核半径 (0=关闭)")
        pp_row.addWidget(self.smooth_kernel_spin)

        pp_row.addWidget(QLabel("最小面积:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 10000)
        self.min_area_spin.setValue(64)
        self.min_area_spin.setToolTip("连通域最小面积，低于此值的碎片被去除")
        pp_row.addWidget(self.min_area_spin)

        pp_row.addWidget(QLabel("最大空洞:"))
        self.max_hole_spin = QSpinBox()
        self.max_hole_spin.setRange(0, 10000)
        self.max_hole_spin.setValue(256)
        self.max_hole_spin.setToolTip("内部空洞最大面积，超过的不填充")
        pp_row.addWidget(self.max_hole_spin)

        pp_row.addStretch()
        layout.addLayout(pp_row)

        group.setLayout(layout)
        return group

    def _build_result_group(self):
        """构建结果展示区域：原图 / 彩色掩码 / 叠加效果"""
        group = QGroupBox("结果")
        layout = QHBoxLayout()

        self.result_labels = {}
        for tag, title in [("orig", "原始图片"), ("mask", "彩色掩码"), ("overlay", "叠加效果")]:
            col = QVBoxLayout()
            header = QLabel(f"<b>{title}</b>")
            header.setAlignment(Qt.AlignCenter)
            col.addWidget(header)
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumSize(320, 240)
            img_label.setStyleSheet("border: 1px solid #ccc; background: #1a1a1a;")
            col.addWidget(img_label)
            self.result_labels[tag] = img_label
            layout.addLayout(col)

        # 第四栏：类别统计
        stat_col = QVBoxLayout()
        stat_header = QLabel("<b>类别统计</b>")
        stat_header.setAlignment(Qt.AlignCenter)
        stat_col.addWidget(stat_header)
        self.stat_label = QLabel()
        self.stat_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.stat_label.setMinimumSize(240, 240)
        self.stat_label.setWordWrap(True)
        self.stat_label.setStyleSheet(
            "border: 1px solid #ccc; background: #fafafa; padding: 8px; "
            "font-size: 13px; color: #333;"
        )
        self.stat_label.setText("分割后自动统计...")
        stat_col.addWidget(self.stat_label)
        layout.addLayout(stat_col)

        group.setLayout(layout)
        return group

    # ---- 文件浏览 ----

    def _browse(self, line_edit, title, is_image=False, is_dir=False):
        """弹出文件选择对话框并将路径写入 line_edit"""
        log.debug("文件浏览对话框: title='%s', target=%s", title, line_edit.objectName())
        if is_image:
            path, _ = QFileDialog.getOpenFileName(
                self, title, "",
                "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if path:
                log.info("选择图片: %s", path)
                line_edit.setText(path)
                self._update_thumb(path)
        elif is_dir:
            path = QFileDialog.getExistingDirectory(self, title)
            if path:
                line_edit.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title, "", "所有文件 (*)")
            if path:
                line_edit.setText(path)

    def _update_thumb(self, image_path):
        """更新图片预览缩略图，同时将原图显示到结果展示区"""
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                arr = np.array(im)
            pix = _np_to_pixmap(arr, (64, 64))
            self.thumb_label.setPixmap(pix)
            orig_label = self.result_labels["orig"]
            target = (orig_label.width(), orig_label.height())
            orig_label.setPixmap(_np_to_pixmap(arr, target))
        except Exception:
            self.thumb_label.setText("预览")

    # ---- 模型加载 ----

    def _on_load_model(self):
        """点击"加载模型"按钮：启动后台加载线程"""
        ckpt = self.findChild(QLineEdit, "ckpt").text().strip()
        bpe = self.findChild(QLineEdit, "bpe").text().strip()
        device = self.device_combo.currentText()
        conf = self.conf_spin.value()

        if not ckpt or not os.path.isfile(ckpt):
            QMessageBox.warning(self, "路径缺失", "请填写有效的 SAM 3 权重路径")
            return
        if not bpe or not os.path.isfile(bpe):
            QMessageBox.warning(self, "路径缺失", "请填写有效的 BPE 词表路径")
            return

        log.info("=== 开始加载 SAM 3 模型 ===")
        log.info("ckpt=%s, bpe=%s, device=%s, conf=%.2f", ckpt, bpe, device, conf)

        self.load_btn.setEnabled(False)
        self.seg_btn.setEnabled(False)
        self.status_bar.showMessage("正在加载 SAM 3 模型，请稍候...")

        self._load_worker = ModelLoadWorker(ckpt, bpe, conf, device)
        self._load_worker.finished.connect(self._on_model_loaded)
        self._load_worker.error.connect(self._on_model_error)
        self._load_worker.start()

    def _on_model_loaded(self, processor, device):
        """模型加载成功回调"""
        self.processor = processor
        self.device = device
        self.load_btn.setEnabled(True)
        self.seg_btn.setEnabled(True)
        self.status_bar.showMessage("SAM 3 模型加载完成 — 可以开始分割")
        log.info("=== SAM 3 模型加载完成 ===")

    def _on_model_error(self, msg):
        """模型加载失败回调"""
        self.load_btn.setEnabled(True)
        log.error("模型加载失败: %s", msg)
        QMessageBox.critical(self, "加载失败", msg)
        self.status_bar.showMessage("模型加载失败")

    # ---- 推理 ----

    def _on_segment(self):
        """点击"开始分割"按钮：解析类别 → 启动后台推理线程"""
        image_path = self.image_edit.text().strip()
        text = self.prompt_edit.text().strip()
        classes_file = self.findChild(QLineEdit, "classes_file").text().strip()

        log.info("=== 开始分割 ===")
        log.info("图片: %s", image_path)
        log.info("类别提示: '%s'", text)

        if not image_path or not os.path.isfile(image_path):
            QMessageBox.warning(self, "输入缺失", "请选择有效的图片文件")
            return
        if self.processor is None:
            QMessageBox.warning(self, "模型未就绪", "请先加载模型")
            return

        # 解析类别
        try:
            query_words, query_idx, num_cls, num_queries, cls_display = \
                self._parse_classes(text, classes_file)
        except Exception as e:
            QMessageBox.warning(self, "类别解析失败", str(e))
            return

        if num_queries == 0:
            QMessageBox.warning(self, "输入缺失", "请输入至少一个分割类别")
            return

        log.info("类别数: %d, 查询数: %d", num_cls, num_queries)
        query_idx_tensor = torch.Tensor(query_idx).to(torch.int64).to(self.device)

        self.seg_btn.setEnabled(False)
        self.status_bar.showMessage(
            f"正在推理中... ({num_cls} 类, {num_queries} 查询)"
        )

        postprocess_kwargs = {
            "min_area": self.min_area_spin.value(),
            "max_hole": self.max_hole_spin.value(),
            "smooth_kernel": self.smooth_kernel_spin.value(),
        }
        self._infer_worker = InferenceWorker(
            self.processor, self.device, image_path,
            query_words, query_idx_tensor,
            num_cls, num_queries,
            prob=self.prob_spin.value(),
            max_passes=self.max_passes_spin.value(),
            coverage_target=self.coverage_spin.value(),
            postprocess_kwargs=postprocess_kwargs,
        )
        self._infer_worker.result.connect(self._on_result)
        self._infer_worker.error.connect(self._on_infer_error)
        self._infer_worker.start()

    def _parse_classes(self, text, classes_file):
        """
        解析用户输入的类别信息。

        优先级: 类别文件 > 文本输入 > 默认 7 类
        入参:
            text: 用户输入的类别文本 (逗号/空格分隔)
            classes_file: 类别文件路径
        出参:
            (query_words, query_idx, num_cls, num_queries, cls_display)
        """
        from infer import DEFAULT_CLASSES, load_classes, NAME_COLOR

        if classes_file and os.path.isfile(classes_file):
            query_words, query_idx, num_cls, num_queries = load_classes(classes_file)
            name_list = open(classes_file).readlines()
            cls_display = {}
            for idx, line in enumerate(name_list):
                synonyms = [s.strip().lower() for s in str(line).strip().split(",")]
                cls_display[idx] = synonyms[0]
            return query_words, query_idx, num_cls, num_queries, cls_display

        if text.strip():
            # 用户输入的类别，逗号或空格分隔
            # 支持中文逗号
            text = text.replace("，", ",")
            user_classes = [c.strip() for c in text.replace(" ", ",").split(",") if c.strip()]
            # 去重
            seen = set()
            unique_classes = []
            for c in user_classes:
                if c.lower() not in seen:
                    seen.add(c.lower())
                    unique_classes.append(c)

            query_words = []
            query_idx = []
            # class 0 = background
            for idx, cls_name in enumerate(unique_classes, start=1):
                query_words.append(cls_name)
                query_idx.append(idx)
            num_cls = len(unique_classes) + 1  # +1 for background
            num_queries = len(query_words)
            cls_display = {0: "background"}
            for idx, cls_name in enumerate(unique_classes, start=1):
                cls_display[idx] = cls_name
            return query_words, query_idx, num_cls, num_queries, cls_display

        # 默认 7 类
        query_words, query_idx = [], []
        for idx, line in enumerate(DEFAULT_CLASSES):
            if idx == 0:
                continue  # background
            synonyms = [w.strip() for w in line.split(",")]
            query_words.extend(synonyms)
            query_idx.extend([idx] * len(synonyms))
        num_cls = len(DEFAULT_CLASSES)
        num_queries = len(query_words)
        cls_display = {}
        for idx, line in enumerate(DEFAULT_CLASSES):
            synonyms = [s.strip() for s in line.split(",")]
            cls_display[idx] = synonyms[0]
        return query_words, query_idx, num_cls, num_queries, cls_display

    def _on_result(self, rgb_orig, color_mask, overlay, seg_pred_np, cls_name_map):
        """推理完成回调：更新结果展示 + 类别统计"""
        log.info("=== 推理结果返回 ===")

        # 统一展示尺寸
        target = None
        for label in self.result_labels.values():
            target = (label.width(), label.height())
            break

        self.result_labels["orig"].setPixmap(_np_to_pixmap(rgb_orig, target))
        self.result_labels["mask"].setPixmap(_np_to_pixmap(color_mask, target))
        self.result_labels["overlay"].setPixmap(_np_to_pixmap(overlay, target))

        self.seg_btn.setEnabled(True)

        # 类别统计
        total_pixels = seg_pred_np.size
        present_cls = sorted(set(seg_pred_np[seg_pred_np > 0].tolist()))
        from infer import NAME_COLOR
        stat_lines = []
        for cls_id in present_cls:
            count = int((seg_pred_np == cls_id).sum())
            pct = count / total_pixels * 100
            names = cls_name_map.get(cls_id, [str(cls_id)])
            name = names[0] if names else str(cls_id)
            color = NAME_COLOR.get(name, (128, 128, 128))
            r, g, b = color
            stat_lines.append(
                f'<span style="color:rgb({r},{g},{b});">■</span> '
                f'{name}: {pct:.1f}% ({count:,} 像素)'
            )

        bg_pct = (seg_pred_np == 0).sum() / total_pixels * 100
        stat_text = f"<b>背景</b>: {bg_pct:.1f}%<br><hr>" + "<br>".join(stat_lines)
        self.stat_label.setText(stat_text)

        status_msg = f"分割完成 — 检测到 {len(present_cls)} 个类别"
        self.status_bar.showMessage(status_msg)
        log.info("分割完成, 检测到类别: %s", present_cls)

    def _on_infer_error(self, msg):
        """推理失败回调"""
        self.seg_btn.setEnabled(True)
        log.error("推理失败: %s", msg)
        QMessageBox.critical(self, "推理失败", msg)
        self.status_bar.showMessage("推理失败")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("SegEarth-OV3 GUI - SAM 3 语义分割工具")
    p.add_argument("--ckpt", type=str, default="sam3/sam3.pt",
                   help="SAM 3 权重路径")
    p.add_argument("--bpe", type=str, default="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                   help="BPE 词表路径")
    p.add_argument("--conf", type=float, default=0.5,
                   help="SAM 3 置信度阈值")
    p.add_argument("--classes", type=str, default=None,
                   help="类别文件路径 (默认使用内置 7 类)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    log.info("========================================")
    log.info("SegEarth-OV3 GUI 启动 (SAM 3)")
    log.info("参数: ckpt=%s, device=%s, conf=%.2f", args.ckpt, args.device, args.conf)
    log.info("BPE: %s", args.bpe)
    log.info("类别文件: %s", args.classes)
    log.info("CUDA 可用: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))
        log.info("GPU 显存: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1024**3)
    log.info("========================================")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SegEarthWindow(args)
    win.show()
    log.info("窗口已显示，进入事件循环")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
