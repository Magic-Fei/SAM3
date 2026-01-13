"""
SAM3 Annotation Tool - A labelme-style annotation software powered by SAM3
支持目标检测和分割标注，生成 labelme 格式的 JSON 文件
"""

import sys
import os
import json
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QListWidget, QFileDialog, 
    QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QSplitter,
    QGroupBox, QScrollArea, QListWidgetItem, QInputDialog, QCheckBox
)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class Canvas(QLabel):
    """画布类 - 用于显示图像和标注结果"""
    
    box_drawn = pyqtSignal(tuple)    # (x1, y1, x2, y2)
    scale_changed = pyqtSignal(float)  # 缩放比例改变
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        
        self.image = None
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.scale_step = 0.1
        self.annotations = []  # 存储所有标注
        self.mode = "text"  # text, box, edit
        self.task_type = "segmentation"  # "segmentation" 或 "detection"
        self.temp_box_start = None
        self.temp_box_end = None
        self.drawing_box = False
        self.offset_x = 0  # 图像偏移（用于平移）
        self.offset_y = 0
        self.dragging = False
        self.dragging_image = False  # 是否正在拖动图像（中键）
        self.last_drag_pos = None  # 上次拖动位置
        self.performance_mode = True  # 性能模式（默认启用）
        self.update_counter = 0  # 更新计数器（用于限制更新频率）
        
        # 编辑模式相关
        self.selected_annotation_index = -1  # 当前选中的标注索引
        self.selected_point_index = -1  # 当前选中的点索引
        self.dragging_point = False  # 是否正在拖动点
        self.dragging_annotation = False  # 是否正在拖动整个标注
        self.drag_start_pos = None  # 拖动开始位置
        self.hover_point_index = -1  # 鼠标悬停的点索引
        self.hover_annotation_index = -1  # 鼠标悬停的标注索引
        
    def set_image(self, image: Image.Image):
        """设置要显示的图像"""
        self.image = image
        self.annotations = []
        self.scale = 1.0  # 重置缩放
        self.offset_x = 0  # 重置偏移
        self.offset_y = 0
        self.dragging_image = False
        self.last_drag_pos = None
        self.update_display()
        
    def set_mode(self, mode: str):
        """设置标注模式"""
        self.mode = mode
        if mode == "box":
            self.setCursor(QCursor(Qt.CrossCursor))
        elif mode == "edit":
            self.setCursor(QCursor(Qt.ArrowCursor))
            # 取消选择
            self.selected_annotation_index = -1
            self.selected_point_index = -1
            self.update_display()
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))
            
    def add_annotation(self, annotation: Dict):
        """添加标注"""
        self.annotations.append(annotation)
        self.update_display()
        
    def clear_annotations(self):
        """清除所有标注"""
        self.annotations = []
        self.update_display()
        
    def remove_annotation(self, index: int):
        """删除指定标注"""
        if 0 <= index < len(self.annotations):
            self.annotations.pop(index)
            self.update_display()
            
    def update_display(self):
        """更新显示"""
        if self.image is None:
            return
            
        # 创建副本用于绘制
        display_img = self.image.copy()
        draw = ImageDraw.Draw(display_img, 'RGBA')
        
        # 绘制所有标注
        for idx, ann in enumerate(self.annotations):
            label = ann.get('label', 'object')
            color = self._get_color_for_label(label)
            is_selected = (idx == self.selected_annotation_index)
            shape_type = ann.get('shape_type', 'polygon')
            
            # 选中的标注使用更亮的颜色
            if is_selected:
                color = tuple(min(c + 50, 255) for c in color)
            
            # 根据 shape_type 决定显示内容
            # 检测模式 (rectangle)：只显示 box
            # 分割模式 (polygon)：只显示 mask
            
            if shape_type == 'rectangle':
                # 目标检测：只绘制边界框
                if 'box' in ann and ann['box'] is not None:
                    box = ann['box']
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().tolist()
                    x1, y1, x2, y2 = box
                    width = 4  # 保持固定宽度，不因选中而变化
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                    
                    # 不显示标签文本
                    # draw.text((x1, y1 - 20), label, fill=color)
                    
                    # 在编辑模式、box模式或text模式下绘制控制点（只有左上和右下两个点）
                    if self.mode == "edit" or self.mode == "box" or self.mode == "text":
                        point_radius = 4  # 增大控制点半径，更容易点击
                        # 只显示左上和右下两个角点
                        corners = [(x1, y1), (x2, y2)]
                        for i, (px, py) in enumerate(corners):
                            # 判断是否是悬停的点
                            is_hover = (idx == self.hover_annotation_index and i == self.hover_point_index)
                            
                            # 使用浅蓝色
                            if is_hover:
                                # 悬停：最浅的蓝色
                                point_color = (50, 150, 255)
                            elif is_selected:
                                # 选中：浅蓝色
                                point_color = (30, 120, 220)
                            else:
                                # 未选中：中等浅蓝色
                                point_color = (20, 100, 180)
                            
                            draw.ellipse(
                                [px - point_radius, py - point_radius, 
                                 px + point_radius, py + point_radius],
                                fill=point_color
                            )
            
            elif shape_type == 'polygon':
                # 语义分割：只绘制轮廓线，不绘制 mask 填充（提升性能）
                if 'polygon_points' in ann and len(ann['polygon_points']) > 0:
                    points = ann['polygon_points']
                    
                    # 绘制轮廓多边形（只绘制轮廓线，不填充）
                    if len(points) > 2:
                        polygon_coords = [(x, y) for x, y in points]
                        width = 2  # 保持固定宽度，不因选中而变化
                        
                        # 只绘制轮廓线，不绘制填充
                        draw.line(polygon_coords + [polygon_coords[0]], fill=color, width=width)
                    
                    # 不显示标签文本
                    # x1, y1 = points[0]
                    # draw.text((x1, y1 - 20), label, fill=color)
                    
                    # 在编辑模式或text模式下绘制多边形的控制点
                    if self.mode == "edit" or self.mode == "text":
                        point_radius = 2  # 固定大小（原来3的70%）
                        for i, (px, py) in enumerate(points):
                            # 判断是否是悬停的点
                            is_hover = (idx == self.hover_annotation_index and i == self.hover_point_index)
                            
                            # 使用浅蓝色
                            if is_hover:
                                # 悬停：最浅的蓝色
                                point_color = (50, 150, 255)
                            elif is_selected:
                                # 选中：浅蓝色
                                point_color = (30, 120, 220)
                            else:
                                # 未选中：中等浅蓝色
                                point_color = (20, 100, 180)
                            
                            draw.ellipse(
                                [px - point_radius, py - point_radius, 
                                 px + point_radius, py + point_radius],
                                fill=point_color
                            )
                elif 'box' in ann and ann['box'] is not None:
                    # 如果没有多边形点，但有 box，不显示
                    pass
                    # box = ann['box']
                    # if isinstance(box, torch.Tensor):
                    #     box = box.cpu().tolist()
                    # x1, y1, x2, y2 = box
                    # draw.text((x1, y1 - 20), f"{label} (轮廓未提取)", fill=color)
        
        # 绘制临时框
        if self.drawing_box and self.temp_box_start and self.temp_box_end:
            draw.rectangle(
                [self.temp_box_start[0], self.temp_box_start[1],
                 self.temp_box_end[0], self.temp_box_end[1]],
                outline=(255, 0, 0), width=2
            )
        
        # 转换为 QPixmap
        display_img = display_img.convert('RGB')
        img_array = np.array(display_img)
        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 计算基于用户缩放和自适应窗口的最终尺寸
        canvas_size = self.size()
        
        # 根据性能模式选择缩放质量
        if self.performance_mode:
            transform_mode = Qt.FastTransformation  # 快速变换，降低质量
        else:
            transform_mode = Qt.SmoothTransformation  # 平滑变换，高质量
        
        # 先应用用户的缩放倍数
        if self.scale != 1.0:
            new_width = int(pixmap.width() * self.scale)
            new_height = int(pixmap.height() * self.scale)
            pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, transform_mode)
        else:
            # 当用户没有手动缩放时（scale == 1.0），自动适应窗口大小
            # 这样图片会填充满窗口，保持宽高比
            if canvas_size.width() > 0 and canvas_size.height() > 0:
                pixmap = pixmap.scaled(canvas_size, Qt.KeepAspectRatio, transform_mode)
        
        self.setPixmap(pixmap)
        
    def _get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        """为不同标签生成不同颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 255, 0), (0, 128, 255)
        ]
        hash_val = sum(ord(c) for c in label)
        return colors[hash_val % len(colors)]
        
    def _find_clicked_point(self, pos: Tuple[int, int], threshold: int = 15) -> Tuple[Optional[int], Optional[int]]:
        """查找点击位置附近的控制点，返回(点索引, 标注索引)"""
        x, y = pos
        
        # 检查所有标注的控制点（从后往前，后面的在上层）
        for ann_idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[ann_idx]
            shape_type = ann.get('shape_type', 'polygon')
            
            # 检查 box 的控制点
            if 'box' in ann and ann['box'] is not None and shape_type == 'rectangle':
                box = ann['box']
                if isinstance(box, torch.Tensor):
                    box = box.cpu().tolist()
                x1, y1, x2, y2 = box
                
                # 目标检测模式：只检查左上和右下两个点
                corners = [(x1, y1), (x2, y2)]
                
                for i, (px, py) in enumerate(corners):
                    dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                    if dist <= threshold:
                        return (i, ann_idx)
            
            # 检查多边形的顶点（分割模式）
            if 'polygon_points' in ann and shape_type == 'polygon':
                points = ann['polygon_points']
                for i, (px, py) in enumerate(points):
                    dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                    if dist <= threshold:
                        return (i, ann_idx)
        
        return (None, None)
        
    def _find_clicked_annotation(self, pos: Tuple[int, int]) -> Optional[int]:
        """查找点击位置的标注，返回标注索引"""
        x, y = pos
        
        # 从后往前遍历（后绘制的在上层）
        for idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[idx]
            
            # 检查是否在 box 内
            if 'box' in ann and ann['box'] is not None:
                box = ann['box']
                if isinstance(box, torch.Tensor):
                    box = box.cpu().tolist()
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return idx
            
            # 检查是否在 mask 内
            if 'mask' in ann and ann['mask'] is not None:
                mask = ann['mask']
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0.5:
                        return idx
        
        return None
        
    def _update_point_position(self, new_pos: Tuple[int, int]):
        """更新控制点位置"""
        # 使用正在拖动的标注索引（而不是选中的标注索引）
        ann_idx = self.selected_annotation_index
        point_idx = self.selected_point_index
        
        if ann_idx < 0 or point_idx < 0 or ann_idx >= len(self.annotations):
            return
            
        ann = self.annotations[ann_idx]
        shape_type = ann.get('shape_type', 'polygon')
        x, y = new_pos
        
        # 更新 box 的角点（检测模式）
        if 'box' in ann and ann['box'] is not None and shape_type == 'rectangle':
            box = ann['box']
            if isinstance(box, torch.Tensor):
                box = box.cpu().tolist()
            x1, y1, x2, y2 = box
            
            # 目标检测模式：只有两个控制点
            # 索引 0 = 左上角，索引 1 = 右下角
            if self.selected_point_index == 0:  # 左上
                x1, y1 = x, y
            elif self.selected_point_index == 1:  # 右下
                x2, y2 = x, y
            
            # 确保 x1 < x2, y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            ann['box'] = torch.tensor([x1, y1, x2, y2])
        
        # 更新多边形的顶点（分割模式）
        elif 'polygon_points' in ann and shape_type == 'polygon':
            points = ann['polygon_points']
            if self.selected_point_index < len(points):
                points[self.selected_point_index] = [x, y]
                ann['polygon_points'] = points
                
    def _move_annotation(self, ann_idx: int, delta_x: int, delta_y: int):
        """移动整个标注"""
        if ann_idx < 0 or ann_idx >= len(self.annotations):
            return
            
        ann = self.annotations[ann_idx]
        shape_type = ann.get('shape_type', 'polygon')
        
        # 移动 box
        if 'box' in ann and ann['box'] is not None:
            box = ann['box']
            if isinstance(box, torch.Tensor):
                box = box.cpu().tolist()
            x1, y1, x2, y2 = box
            
            # 应用偏移
            x1 += delta_x
            y1 += delta_y
            x2 += delta_x
            y2 += delta_y
            
            # 确保不超出图像边界
            if self.image:
                x1 = max(0, min(x1, self.image.width - 1))
                y1 = max(0, min(y1, self.image.height - 1))
                x2 = max(0, min(x2, self.image.width - 1))
                y2 = max(0, min(y2, self.image.height - 1))
            
            ann['box'] = torch.tensor([x1, y1, x2, y2])
        
        # 移动多边形控制点
        if 'polygon_points' in ann:
            points = ann['polygon_points']
            new_points = []
            for px, py in points:
                new_x = px + delta_x
                new_y = py + delta_y
                
                # 确保不超出图像边界
                if self.image:
                    new_x = max(0, min(new_x, self.image.width - 1))
                    new_y = max(0, min(new_y, self.image.height - 1))
                
                new_points.append([new_x, new_y])
            
            ann['polygon_points'] = new_points
        
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.image is None:
            return
        
        # 中键按下 - 开始拖动图像
        if event.button() == Qt.MiddleButton:
            self.dragging_image = True
            self.last_drag_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return
            
        if event.button() == Qt.LeftButton:
            # 获取图像坐标
            pos = self._get_image_coords(event.pos())
            if pos is None:
                return
            
            if self.mode == "edit" or self.mode == "text":
                # 编辑模式或文本提示模式：检查是否点击了控制点或标注
                clicked_point, clicked_ann = self._find_clicked_point(pos)
                if clicked_point is not None:
                    # 点击了控制点，开始拖动控制点
                    self.selected_annotation_index = clicked_ann
                    self.selected_point_index = clicked_point
                    self.dragging_point = True
                    self.dragging_annotation = False
                else:
                    # 检查是否点击了标注主体
                    clicked_ann_idx = self._find_clicked_annotation(pos)
                    if clicked_ann_idx is not None:
                        # 点击了标注主体，准备拖动整个标注
                        self.selected_annotation_index = clicked_ann_idx
                        self.selected_point_index = -1
                        self.dragging_annotation = True
                        self.drag_start_pos = pos
                        self.setCursor(QCursor(Qt.ClosedHandCursor))
                        self.update_display()
                    else:
                        # 取消选择
                        self.selected_annotation_index = -1
                        self.selected_point_index = -1
                        self.dragging_annotation = False
                        self.update_display()
                        
            elif self.mode == "box":
                # box模式：检查是否点击了控制点或已有标注
                clicked_point, clicked_ann = self._find_clicked_point(pos)
                if clicked_point is not None:
                    # 点击了控制点，开始拖动控制点
                    self.selected_annotation_index = clicked_ann
                    self.selected_point_index = clicked_point
                    self.dragging_point = True
                    self.dragging_annotation = False
                    self.drawing_box = False
                else:
                    # 检查是否点击了已有标注的中心区域
                    clicked_ann_idx = self._find_clicked_annotation(pos)
                    if clicked_ann_idx is not None:
                        # 点击了已有标注，选择它并准备拖动（不开始绘制新框）
                        self.selected_annotation_index = clicked_ann_idx
                        self.selected_point_index = -1
                        self.dragging_annotation = True
                        self.drag_start_pos = pos
                        self.drawing_box = False
                        self.setCursor(QCursor(Qt.ClosedHandCursor))
                        self.update_display()
                    else:
                        # 没有点击已有标注，开始绘制新框
                        self.drawing_box = True
                        self.temp_box_start = pos
                        self.temp_box_end = pos
                
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        # 中键拖动图像
        if self.dragging_image and self.last_drag_pos is not None:
            delta = event.pos() - self.last_drag_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_drag_pos = event.pos()
            
            # 性能模式：降低更新频率
            if self.performance_mode:
                self.update_counter += 1
                if self.update_counter % 3 == 0:  # 每 3 次移动更新一次
                    self.update_display()
            else:
                self.update_display()
            return
        
        pos = self._get_image_coords(event.pos())
        if pos is None:
            return
            
        if self.drawing_box and self.temp_box_start:
            # 优先处理绘制框的逻辑
            self.temp_box_end = pos
            
            # 性能模式：降低更新频率
            if self.performance_mode:
                self.update_counter += 1
                if self.update_counter % 3 == 0:
                    self.update_display()
            else:
                self.update_display()
        elif (self.mode == "edit" or self.mode == "box" or self.mode == "text") and self.dragging_point and self.selected_annotation_index >= 0:
            # 拖动控制点（编辑模式、box模式或文本提示模式）
            self._update_point_position(pos)
            
            # 性能模式：降低更新频率
            if self.performance_mode:
                self.update_counter += 1
                if self.update_counter % 2 == 0:  # 每 2 次移动更新一次
                    self.update_display()
            else:
                self.update_display()
        elif (self.mode == "edit" or self.mode == "box" or self.mode == "text") and self.dragging_annotation and self.selected_annotation_index >= 0 and self.drag_start_pos:
            # 拖动整个标注
            delta_x = pos[0] - self.drag_start_pos[0]
            delta_y = pos[1] - self.drag_start_pos[1]
            self._move_annotation(self.selected_annotation_index, delta_x, delta_y)
            self.drag_start_pos = pos
            
            # 性能模式：降低更新频率
            if self.performance_mode:
                self.update_counter += 1
                if self.update_counter % 2 == 0:
                    self.update_display()
            else:
                self.update_display()
        elif self.mode == "edit" or self.mode == "box" or self.mode == "text":
            # 更新悬停状态（编辑模式、box模式或文本提示模式）
            if not self.dragging_point and not self.dragging_annotation and not self.drawing_box:
                hover_point, hover_ann = self._find_clicked_point(pos)
                hover_annotation = self._find_clicked_annotation(pos)
                
                # 更新光标
                if hover_point is not None:
                    # 悬停在控制点上，显示可拖拽光标
                    self.setCursor(QCursor(Qt.SizeAllCursor))
                elif hover_annotation is not None:
                    # 悬停在标注上，显示可移动光标
                    self.setCursor(QCursor(Qt.OpenHandCursor))
                elif self.mode == "box":
                    # box模式下，其他情况显示十字光标
                    self.setCursor(QCursor(Qt.CrossCursor))
                else:
                    # 编辑模式下，其他情况显示箭头光标
                    self.setCursor(QCursor(Qt.ArrowCursor))
                
                if hover_point != self.hover_point_index or hover_ann != self.hover_annotation_index:
                    self.hover_point_index = hover_point
                    self.hover_annotation_index = hover_ann
                    
                    # 悬停状态变化不需要高频更新
                    if not self.performance_mode:
                        self.update_display()
                
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        # 中键释放 - 停止拖动图像
        if event.button() == Qt.MiddleButton:
            self.dragging_image = False
            self.last_drag_pos = None
            # 确保最终更新显示（性能模式下可能有跳帧）
            if self.performance_mode:
                self.update_display()
            # 恢复鼠标光标
            self.set_mode(self.mode)
            return
            
        if event.button() == Qt.LeftButton:
            if (self.mode == "edit" or self.mode == "box" or self.mode == "text") and self.dragging_point:
                # 停止拖动控制点（编辑模式、box模式或文本提示模式）
                self.dragging_point = False
                self.selected_point_index = -1
                # 确保最终更新显示
                if self.performance_mode:
                    self.update_display()
            elif (self.mode == "edit" or self.mode == "box" or self.mode == "text") and self.dragging_annotation:
                # 停止拖动标注（编辑模式、box模式或文本提示模式）
                self.dragging_annotation = False
                self.drag_start_pos = None
                self.set_mode(self.mode)  # 恢复光标
                # 确保最终更新显示
                if self.performance_mode:
                    self.update_display()
            elif self.drawing_box:
                self.drawing_box = False
                if self.temp_box_start and self.temp_box_end:
                    x1, y1 = self.temp_box_start
                    x2, y2 = self.temp_box_end
                    # 确保 x1 < x2, y1 < y2
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    self.box_drawn.emit((x1, y1, x2, y2))
                    self.temp_box_start = None
                    self.temp_box_end = None
                
        self.dragging = False
        
    def wheelEvent(self, event):
        """鼠标滚轮事件 - 缩放图像"""
        if self.image is None:
            return
        
        # 获取滚轮滚动方向
        delta = event.angleDelta().y()
        
        # 计算新的缩放比例
        old_scale = self.scale
        if delta > 0:
            # 向上滚动，放大
            self.scale = min(self.scale + self.scale_step, self.max_scale)
        else:
            # 向下滚动，缩小
            self.scale = max(self.scale - self.scale_step, self.min_scale)
        
        # 更新显示
        if self.scale != old_scale:
            self.update_display()
            self.scale_changed.emit(self.scale)  # 发送缩放改变信号
            
    def resizeEvent(self, event):
        """窗口大小改变事件 - 重新调整图像大小"""
        super().resizeEvent(event)
        if self.image is not None:
            self.update_display()
            
    def paintEvent(self, event):
        """自定义绘制事件 - 支持图像偏移"""
        if self.pixmap() is None:
            super().paintEvent(event)
            return
        
        painter = QPainter(self)
        
        # 计算图像位置（考虑偏移）
        pixmap = self.pixmap()
        label_rect = self.rect()
        pixmap_rect = pixmap.rect()
        
        # 默认居中位置
        x = (label_rect.width() - pixmap_rect.width()) // 2
        y = (label_rect.height() - pixmap_rect.height()) // 2
        
        # 应用偏移
        x += int(self.offset_x)
        y += int(self.offset_y)
        
        # 绘制图像
        painter.drawPixmap(x, y, pixmap)
                
    def _get_image_coords(self, widget_pos: QPoint) -> Optional[Tuple[int, int]]:
        """将窗口坐标转换为图像坐标"""
        if self.pixmap() is None or self.image is None:
            return None
            
        pixmap = self.pixmap()
        # 计算实际图像在 widget 中的位置（考虑偏移）
        label_rect = self.rect()
        pixmap_rect = pixmap.rect()
        
        # 计算偏移（包括居中偏移和用户拖动偏移）
        x_offset = (label_rect.width() - pixmap_rect.width()) / 2 + self.offset_x
        y_offset = (label_rect.height() - pixmap_rect.height()) / 2 + self.offset_y
        
        # 转换坐标
        x = widget_pos.x() - x_offset
        y = widget_pos.y() - y_offset
        
        # 检查是否在图像范围内
        if x < 0 or y < 0 or x >= pixmap_rect.width() or y >= pixmap_rect.height():
            return None
            
        # 计算从显示的 pixmap 到原始图像的缩放比例
        scale_x = self.image.width / pixmap_rect.width()
        scale_y = self.image.height / pixmap_rect.height()
        
        # 转换到原始图像坐标
        img_x = int(x * scale_x)
        img_y = int(y * scale_y)
        
        # 确保坐标在图像范围内
        img_x = max(0, min(img_x, self.image.width - 1))
        img_y = max(0, min(img_y, self.image.height - 1))
        
        return (img_x, img_y)


class SAM3AnnotatorApp(QMainWindow):
    """SAM3 标注工具主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Annotation Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置状态栏
        self.statusBar().showMessage("快捷键: A/D=切换 | E=编辑 | Delete=删除 | R=重置视图 | 中键=拖动 | 滚轮=缩放")
        
        # 模型和处理器
        self.model = None
        self.processor = None
        self.inference_state = None
        
        # 当前图像信息
        self.current_image = None
        self.current_image_path = None
        self.image_width = 0
        self.image_height = 0
        
        # 图像列表管理
        self.image_list = []  # 所有图像路径列表
        self.current_image_index = -1  # 当前图像索引
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # 任务类型
        self.task_type = "segmentation"  # "segmentation" 或 "detection"
        
        self.init_ui()
        
    def init_ui(self):
        """初始化 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # 中间画布区域
        self.canvas = Canvas()
        main_layout.addWidget(self.canvas, stretch=3)
        
        # 右侧标注列表
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        # 连接信号
        self.canvas.box_drawn.connect(self.on_box_drawn)
        self.canvas.scale_changed.connect(self.on_scale_changed)
        
    def _create_left_panel(self) -> QWidget:
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型加载区域
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("模型文件路径")
        # 设置默认模型路径
        default_model_path = r"D:\qianpf\code\sam3-main\model\sam3\sam3.pt"
        self.model_path_input.setText(default_model_path)
        model_layout.addWidget(QLabel("模型路径:"))
        model_layout.addWidget(self.model_path_input)
        
        model_path_btn = QPushButton("浏览模型")
        model_path_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_path_btn)
        
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        
        self.model_status_label = QLabel("模型: 未加载")
        self.model_status_label.setStyleSheet("color: red;")
        model_layout.addWidget(self.model_status_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 图像加载区域
        image_group = QGroupBox("图像操作")
        image_layout = QVBoxLayout()
        
        load_image_btn = QPushButton("加载单张图像")
        load_image_btn.clicked.connect(self.load_image)
        image_layout.addWidget(load_image_btn)
        
        load_folder_btn = QPushButton("加载文件夹")
        load_folder_btn.clicked.connect(self.load_folder)
        image_layout.addWidget(load_folder_btn)
        
        # 图像导航
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("← 上一张 (A)")
        self.prev_btn.clicked.connect(self.previous_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("下一张 (D) →")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        image_layout.addLayout(nav_layout)
        
        self.image_info_label = QLabel("未加载图像")
        image_layout.addWidget(self.image_info_label)
        
        self.image_count_label = QLabel("0/0")
        self.image_count_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        image_layout.addWidget(self.image_count_label)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # 标注模式区域
        mode_group = QGroupBox("标注模式")
        mode_layout = QVBoxLayout()
        
        # 任务类型选择
        mode_layout.addWidget(QLabel("任务类型:"))
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["分割 (Mask)", "检测 (Box)"])
        self.task_type_combo.currentTextChanged.connect(self.on_task_type_changed)
        mode_layout.addWidget(self.task_type_combo)
        
        mode_layout.addWidget(QLabel("提示方式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["文本提示", "框提示", "编辑模式"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        
        # 文本提示输入
        self.text_prompt_label = QLabel("文本提示:")
        mode_layout.addWidget(self.text_prompt_label)
        self.text_prompt_input = QLineEdit()
        self.text_prompt_input.setPlaceholderText("输入类别名称，如: car, person")
        self.text_prompt_input.setText("rectangle")  # 设置默认提示词
        mode_layout.addWidget(self.text_prompt_input)
        
        self.text_run_btn = QPushButton("运行文本提示")
        self.text_run_btn.clicked.connect(self.run_text_prompt)
        mode_layout.addWidget(self.text_run_btn)
        
        # 清除标注选项
        self.clear_before_text = QCheckBox("运行前清除已有标注")
        self.clear_before_text.setChecked(True)  # 默认勾选
        mode_layout.addWidget(self.clear_before_text)
        
        # 自动进入编辑模式选项
        self.auto_edit_mode = QCheckBox("标注后自动进入编辑模式")
        self.auto_edit_mode.setChecked(True)  # 默认勾选
        mode_layout.addWidget(self.auto_edit_mode)
        
        # 提示区域
        self.mode_hint_label = QLabel("文本提示: 输入类别名称自动检测")
        mode_layout.addWidget(self.mode_hint_label)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # 标注参数
        param_group = QGroupBox("标注参数")
        param_layout = QVBoxLayout()
        
        param_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.0, 1.0)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setValue(0.3)
        param_layout.addWidget(self.conf_threshold)
        
        self.current_label_input = QLineEdit()
        self.current_label_input.setPlaceholderText("当前标签名称")
        self.current_label_input.setText("object")
        param_layout.addWidget(QLabel("当前标签:"))
        param_layout.addWidget(self.current_label_input)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        layout.addStretch()
        
        return panel
        
    def _create_right_panel(self) -> QWidget:
        """创建右侧标注列表面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 缩放控制
        zoom_group = QGroupBox("视图控制")
        zoom_layout = QVBoxLayout()
        
        self.zoom_label = QLabel("缩放: 100%")
        zoom_layout.addWidget(self.zoom_label)
        
        zoom_btn_layout = QHBoxLayout()
        
        zoom_in_btn = QPushButton("放大 (+)")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_btn_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("缩小 (-)")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_btn_layout.addWidget(zoom_out_btn)
        
        zoom_reset_btn = QPushButton("重置 (1:1)")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        zoom_btn_layout.addWidget(zoom_reset_btn)
        
        zoom_layout.addLayout(zoom_btn_layout)
        
        # 重置视图按钮
        reset_view_btn = QPushButton("重置视图 (R)")
        reset_view_btn.clicked.connect(self.reset_view)
        zoom_layout.addWidget(reset_view_btn)
        
        zoom_layout.addWidget(QLabel("提示: 滚轮缩放 | 中键拖动"))
        
        # 性能选项
        self.performance_mode = QCheckBox("高性能模式（大图推荐）")
        self.performance_mode.setChecked(True)  # 默认启用高性能模式
        self.performance_mode.stateChanged.connect(self.on_performance_mode_changed)
        zoom_layout.addWidget(self.performance_mode)
        
        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)
        
        # 标注列表
        layout.addWidget(QLabel("标注列表:"))
        
        self.annotation_list = QListWidget()
        self.annotation_list.itemDoubleClicked.connect(self.edit_annotation_label)
        layout.addWidget(self.annotation_list)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self.delete_annotation)
        btn_layout.addWidget(delete_btn)
        
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_annotations)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        
        # 保存按钮
        save_btn = QPushButton("保存为 JSON")
        save_btn.clicked.connect(self.save_annotations)
        layout.addWidget(save_btn)
        
        load_json_btn = QPushButton("加载 JSON")
        load_json_btn.clicked.connect(self.load_annotations)
        layout.addWidget(load_json_btn)
        
        return panel
        
    def browse_model(self):
        """浏览选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch Models (*.pt *.pth)"
        )
        if file_path:
            self.model_path_input.setText(file_path)
            
    def load_model(self):
        """加载 SAM3 模型"""
        model_path = self.model_path_input.text()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的模型文件")
            return
            
        try:
            self.model = build_sam3_image_model(
                checkpoint_path=model_path,
                load_from_HF=False
            )
            self.processor = Sam3Processor(self.model)
            self.model_status_label.setText("模型: 已加载")
            self.model_status_label.setStyleSheet("color: green;")
            self.statusBar().showMessage("✓ 模型加载成功", 3000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            
    def load_image(self):
        """加载单张图像"""
        if self.processor is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not file_path:
            return
            
        # 单张图像模式
        self.image_list = [file_path]
        self.current_image_index = 0
        self._load_current_image()
        self._update_navigation_buttons()
        
    def load_folder(self):
        """加载文件夹中的所有图像"""
        if self.processor is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择图像文件夹"
        )
        if not folder_path:
            return
            
        # 查找所有图像文件
        image_set = set()  # 使用 set 去重
        for ext in self.image_extensions:
            image_set.update(Path(folder_path).glob(f"*{ext}"))
            image_set.update(Path(folder_path).glob(f"*{ext.upper()}"))
        
        # 转换为字符串并排序（使用 set 去重后的结果）
        self.image_list = sorted([str(p) for p in image_set])
        
        if len(self.image_list) == 0:
            QMessageBox.warning(self, "警告", f"文件夹中没有找到图像文件")
            return
            
        # 加载第一张图像
        self.current_image_index = 0
        self._load_current_image()
        self._update_navigation_buttons()
        
        QMessageBox.information(
            self, "成功", 
            f"已加载 {len(self.image_list)} 张图像\n使用 A/D 键或按钮切换"
        )
        
    def _load_current_image(self):
        """加载当前索引的图像"""
        if not self.image_list or self.current_image_index < 0:
            return
            
        try:
            file_path = self.image_list[self.current_image_index]
            
            # 在切换图片前，自动保存当前标注（如果有的话）
            if self.current_image_path and len(self.canvas.annotations) > 0:
                self._auto_save_annotations()
                self.statusBar().showMessage(f"✓ 已自动保存上一张的标注", 2000)
            
            # 加载图像并处理 EXIF 方向信息
            self.current_image = Image.open(file_path)
            self.current_image = ImageOps.exif_transpose(self.current_image)
            self.current_image = self.current_image.convert('RGB')
            self.current_image_path = file_path
            self.image_width = self.current_image.width
            self.image_height = self.current_image.height
            
            # 设置图像到处理器
            self.inference_state = self.processor.set_image(self.current_image)
            
            # 显示图像
            self.canvas.set_image(self.current_image)
            
            # 更新信息显示
            self.image_info_label.setText(
                f"图像: {os.path.basename(file_path)}\n"
                f"尺寸: {self.image_width}x{self.image_height}"
            )
            
            self.image_count_label.setText(
                f"{self.current_image_index + 1}/{len(self.image_list)}"
            )
            
            # 清空当前标注
            self.canvas.clear_annotations()
            self.annotation_list.clear()
            
            # 尝试加载已有的标注
            has_annotation = self._try_load_existing_annotations()
            
            # 更新状态栏
            if has_annotation:
                self.statusBar().showMessage(
                    f"✓ 已加载: {os.path.basename(file_path)} ({self.current_image_index + 1}/{len(self.image_list)}) - 发现已有标注", 
                    3000
                )
            else:
                self.statusBar().showMessage(
                    f"已加载: {os.path.basename(file_path)} ({self.current_image_index + 1}/{len(self.image_list)})", 
                    3000
                )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图像加载失败: {str(e)}")
            
    def _auto_save_annotations(self):
        """自动保存当前标注"""
        if not self.current_image_path or len(self.canvas.annotations) == 0:
            return
            
        try:
            json_path = self.current_image_path.rsplit('.', 1)[0] + '.json'
            labelme_data = self._convert_to_labelme_format()
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"自动保存失败: {e}")
            
    def _try_load_existing_annotations(self):
        """尝试加载已有的标注文件，返回是否成功加载"""
        if not self.current_image_path:
            return False
            
        json_path = self.current_image_path.rsplit('.', 1)[0] + '.json'
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    labelme_data = json.load(f)
                    
                # 加载标注
                loaded_count = 0
                for shape in labelme_data.get('shapes', []):
                    label = shape['label']
                    points = shape['points']
                    shape_type = shape.get('shape_type', 'polygon')
                    
                    annotation = {
                        'label': label,
                        'shape_type': shape_type,
                        'score': 1.0
                    }
                    
                    if shape_type == 'rectangle' and len(points) == 2:
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        annotation['box'] = torch.tensor([x1, y1, x2, y2])
                    elif shape_type == 'polygon':
                        # 从多边形创建 mask
                        mask = self._polygon_to_mask(points, self.image_width, self.image_height)
                        annotation['mask'] = mask
                        # 保存多边形控制点
                        annotation['polygon_points'] = points
                        
                    self.canvas.add_annotation(annotation)
                    self._add_annotation_to_list(annotation, len(self.canvas.annotations) - 1)
                    loaded_count += 1
                
                return loaded_count > 0
                    
            except Exception as e:
                print(f"加载已有标注失败: {e}")
                return False
        
        return False
                
    def previous_image(self):
        """切换到上一张图像"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._load_current_image()
            self._update_navigation_buttons()
            
    def next_image(self):
        """切换到下一张图像"""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self._load_current_image()
            self._update_navigation_buttons()
            
    def _update_navigation_buttons(self):
        """更新导航按钮状态"""
        if len(self.image_list) == 0:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.prev_btn.setEnabled(self.current_image_index > 0)
            self.next_btn.setEnabled(self.current_image_index < len(self.image_list) - 1)
            
    def on_task_type_changed(self, task_text: str):
        """任务类型改变"""
        if task_text == "检测 (Box)":
            self.task_type = "detection"
            self.canvas.task_type = "detection"
        else:
            self.task_type = "segmentation"
            self.canvas.task_type = "segmentation"
        
        # 更新显示以反映任务类型变化
        self.canvas.update_display()
    
    def on_mode_changed(self, mode_text: str):
        """标注模式改变"""
        if mode_text == "框提示":
            self.canvas.mode = "box"
            self.mode_hint_label.setText("框提示: 拖拽鼠标绘制边界框")
            # 隐藏文本提示相关控件
            self.text_prompt_label.setVisible(False)
            self.text_prompt_input.setVisible(False)
            self.text_run_btn.setVisible(False)
            self.clear_before_text.setVisible(False)
        elif mode_text == "编辑模式":
            self.canvas.mode = "edit"
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))
            self.mode_hint_label.setText("编辑模式: 拖动控制点调整形状 | 拖动标注移动位置")
            # 隐藏文本提示相关控件
            self.text_prompt_label.setVisible(False)
            self.text_prompt_input.setVisible(False)
            self.text_run_btn.setVisible(False)
            self.clear_before_text.setVisible(False)
        else:
            self.canvas.mode = "text"
            self.mode_hint_label.setText("文本提示: 输入类别名称自动检测")
            # 显示文本提示相关控件
            self.text_prompt_label.setVisible(True)
            self.text_prompt_input.setVisible(True)
            self.text_run_btn.setVisible(True)
            self.text_run_btn.setText("运行文本提示")
            self.clear_before_text.setVisible(True)
        self.canvas.set_mode(self.canvas.mode)
        
    def run_text_prompt(self):
        """运行文本提示"""
        if self.inference_state is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
            
        text_prompt = self.text_prompt_input.text().strip()
        if not text_prompt:
            QMessageBox.warning(self, "警告", "请输入文本提示")
            return
        
        # 如果勾选了清除选项，清空当前标注
        if self.clear_before_text.isChecked():
            self.canvas.clear_annotations()
            self.annotation_list.clear()
            
        try:
            output = self.processor.set_text_prompt(
                state=self.inference_state,
                prompt=text_prompt
            )
            
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            threshold = self.conf_threshold.value()
            label = self.current_label_input.text() or text_prompt
            
            # 添加检测结果
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if score.item() >= threshold:
                    annotation = {
                        'label': label,
                        'score': score.item(),
                    }
                    
                    # 根据任务类型决定保存什么
                    if self.task_type == "detection":
                        # 检测模式：只保存 box
                        annotation['box'] = box
                        annotation['shape_type'] = 'rectangle'
                    else:
                        # 分割模式：保存 mask 和提取多边形控制点
                        annotation['mask'] = mask
                        annotation['box'] = box
                        annotation['shape_type'] = 'polygon'
                        # 从 mask 提取多边形控制点
                        polygon_points = self._extract_polygon_points_from_mask(mask)
                        annotation['polygon_points'] = polygon_points
                    
                    self.canvas.add_annotation(annotation)
                    self._add_annotation_to_list(annotation, len(self.canvas.annotations) - 1)
                    
            task_name = "检测" if self.task_type == "detection" else "分割"
            detected_count = len([s for s in scores if s >= threshold])
            
            # 在文本提示模式下，保持当前模式不变，不自动切换
            # 这样用户可以继续使用文本提示功能，而不会意外切换到其他模式
            self.statusBar().showMessage(
                f"✓ {task_name}到 {detected_count} 个对象 | 提示: 按 E 键或选择'编辑模式'可以查看和调整控制点", 
                5000
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理失败: {str(e)}")
            
    def on_box_drawn(self, box_coords: Tuple[int, int, int, int]):
        """框被绘制"""
        if self.inference_state is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
            
        try:
            x1, y1, x2, y2 = box_coords
            
            label = self.current_label_input.text() or "object"
            
            # 框提示模式：总是使用框提示生成分割 mask（不管任务类型是什么）
            # 因为"框提示"本身就是用来提示模型分割框内物体的
            self.statusBar().showMessage("正在使用框提示生成分割...", 0)
            
            # 转换框格式：从 [x1, y1, x2, y2] 到归一化的 [center_x, center_y, width, height]
            img_width = self.image_width
            img_height = self.image_height
            
            # 计算中心点和宽高（归一化到 [0, 1]）
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # SAM3 的 add_geometric_prompt 需要 [center_x, center_y, width, height] 格式
            box_normalized = [center_x, center_y, width, height]
            
            # 根据 SAM3 API 调用（label=True 表示正框）
            output = self.processor.add_geometric_prompt(
                box=box_normalized,
                label=True,
                state=self.inference_state
            )
            
            masks = output.get("masks", [])
            boxes = output.get("boxes", [])
            scores = output.get("scores", [])
            
            if len(masks) > 0:
                # 框提示模式总是生成分割结果，保存为多边形格式
                annotation = {
                    'label': label,
                    'mask': masks[0],
                    'box': boxes[0] if len(boxes) > 0 else torch.tensor([x1, y1, x2, y2]),
                    'score': scores[0].item() if len(scores) > 0 else 1.0,
                    'shape_type': 'polygon'
                }
                # 从 mask 提取多边形控制点
                polygon_points = self._extract_polygon_points_from_mask(masks[0])
                annotation['polygon_points'] = polygon_points
                
                self.canvas.add_annotation(annotation)
                self._add_annotation_to_list(annotation, len(self.canvas.annotations) - 1)
                
                # 自动切换到编辑模式
                if self.auto_edit_mode.isChecked():
                    self.mode_combo.setCurrentText("编辑模式")
                    
                self.statusBar().showMessage(f"✓ 框提示生成分割成功", 3000)
            else:
                self.statusBar().showMessage("⚠ 框提示未生成有效分割，请检查框是否在图像范围内", 5000)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"框提示推理失败: {str(e)}\n\n请确保：\n1. 已加载模型\n2. 已加载图像\n3. 绘制的框在图像范围内")
            self.statusBar().showMessage(f"✗ 框提示失败: {str(e)}", 5000)
            
    def _add_annotation_to_list(self, annotation: Dict, index: int):
        """添加标注到列表"""
        label = annotation['label']
        score = annotation.get('score', 1.0)
        item = QListWidgetItem(f"{index}: {label} ({score:.3f})")
        self.annotation_list.addItem(item)
        
    def edit_annotation_label(self, item: QListWidgetItem):
        """编辑标注标签"""
        index = self.annotation_list.row(item)
        if 0 <= index < len(self.canvas.annotations):
            current_label = self.canvas.annotations[index]['label']
            new_label, ok = QInputDialog.getText(
                self, "编辑标签", "输入新标签:", text=current_label
            )
            if ok and new_label:
                self.canvas.annotations[index]['label'] = new_label
                self._refresh_annotation_list()
                self.canvas.update_display()
                
    def delete_annotation(self):
        """删除选中的标注"""
        current_row = self.annotation_list.currentRow()
        if current_row >= 0:
            self.canvas.remove_annotation(current_row)
            self.annotation_list.takeItem(current_row)
            self._refresh_annotation_list()
            
    def clear_annotations(self):
        """清空所有标注"""
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有标注吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.canvas.clear_annotations()
            self.annotation_list.clear()
            
    def _refresh_annotation_list(self):
        """刷新标注列表"""
        self.annotation_list.clear()
        for i, ann in enumerate(self.canvas.annotations):
            self._add_annotation_to_list(ann, i)
            
    def zoom_in(self):
        """放大图像"""
        if self.canvas.image is not None:
            old_scale = self.canvas.scale
            self.canvas.scale = min(self.canvas.scale + 0.1, self.canvas.max_scale)
            if self.canvas.scale != old_scale:
                self.canvas.update_display()
                self._update_zoom_label()
            
    def zoom_out(self):
        """缩小图像"""
        if self.canvas.image is not None:
            old_scale = self.canvas.scale
            self.canvas.scale = max(self.canvas.scale - 0.1, self.canvas.min_scale)
            if self.canvas.scale != old_scale:
                self.canvas.update_display()
                self._update_zoom_label()
            
    def zoom_reset(self):
        """重置缩放"""
        if self.canvas.image is not None:
            self.canvas.scale = 1.0
            self.canvas.update_display()
            self._update_zoom_label()
            
    def reset_view(self):
        """重置视图（缩放和偏移）"""
        if self.canvas.image is not None:
            self.canvas.scale = 1.0
            self.canvas.offset_x = 0
            self.canvas.offset_y = 0
            self.canvas.update_display()
            self._update_zoom_label()
            self.statusBar().showMessage("已重置视图", 2000)
            
    def _update_zoom_label(self):
        """更新缩放标签"""
        zoom_percent = int(self.canvas.scale * 100)
        self.zoom_label.setText(f"缩放: {zoom_percent}%")
        
    def on_scale_changed(self, scale: float):
        """缩放比例改变时的回调"""
        zoom_percent = int(scale * 100)
        self.zoom_label.setText(f"缩放: {zoom_percent}%")
        
    def on_performance_mode_changed(self, state):
        """性能模式切换"""
        is_enabled = (state == 2)  # Qt.Checked = 2
        self.canvas.performance_mode = is_enabled
        
        if is_enabled:
            self.statusBar().showMessage(
                "⚡ 高性能模式已启用 - 降低绘制质量以提升速度", 
                3000
            )
        else:
            self.statusBar().showMessage(
                "🎨 已切换到高质量模式", 
                3000
            )
        
        # 刷新显示
        if self.canvas.image is not None:
            self.canvas.update_display()
        
    def keyPressEvent(self, event):
        """处理键盘事件"""
        key = event.key()
        
        # A 键 - 上一张图片
        if key == Qt.Key_A:
            self.previous_image()
        # D 键 - 下一张图片
        elif key == Qt.Key_D:
            self.next_image()
        # S 键 - 保存当前标注
        elif key == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_annotations()
        # Delete 键 - 删除选中的标注
        elif key == Qt.Key_Delete or key == Qt.Key_Backspace:
            if self.canvas.selected_annotation_index >= 0:
                self.canvas.remove_annotation(self.canvas.selected_annotation_index)
                self._refresh_annotation_list()
                self.canvas.selected_annotation_index = -1
                self.statusBar().showMessage("已删除选中的标注", 2000)
        # E 键 - 切换到编辑模式
        elif key == Qt.Key_E:
            self.mode_combo.setCurrentText("编辑模式")
        # + 键 - 放大
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.zoom_in()
        # - 键 - 缩小
        elif key == Qt.Key_Minus:
            self.zoom_out()
        # 0 键 - 重置缩放
        elif key == Qt.Key_0:
            self.zoom_reset()
        # R 键 - 重置视图
        elif key == Qt.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)
            
    def save_annotations(self):
        """保存标注为 labelme 格式 JSON"""
        if self.current_image_path is None:
            QMessageBox.warning(self, "警告", "没有可保存的标注")
            return
            
        # 默认保存路径
        default_path = self.current_image_path.rsplit('.', 1)[0] + '.json'
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存标注", default_path, "JSON Files (*.json)"
        )
        if not file_path:
            return
            
        try:
            labelme_data = self._convert_to_labelme_format()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            self.statusBar().showMessage(f"✓ 已保存: {os.path.basename(file_path)}", 3000)
            QMessageBox.information(self, "成功", f"标注已保存到: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
            
    def _convert_to_labelme_format(self) -> Dict:
        """转换为 labelme 格式"""
        # 不保存图像数据，减小文件大小
        image_base64 = None
        
        shapes = []
        for ann in self.canvas.annotations:
            label = ann['label']
            
            # 从 mask 提取轮廓点
            if 'mask' in ann and ann['mask'] is not None:
                mask = ann['mask']
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                    
                if mask.ndim == 3:
                    mask = mask[0]
                    
                # 提取轮廓
                points = self._mask_to_polygon(mask)
                
                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)
            elif 'box' in ann and ann['box'] is not None:
                # 如果只有 box，保存为矩形
                box = ann['box']
                if isinstance(box, torch.Tensor):
                    box = box.cpu().tolist()
                x1, y1, x2, y2 = box
                
                shape = {
                    "label": label,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                shapes.append(shape)
        
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(self.current_image_path),
            "imageData": None,  # 不保存图像数据，减小文件大小
            "imageHeight": self.image_height,
            "imageWidth": self.image_width
        }
        
        return labelme_data
        
    def _mask_to_polygon(self, mask: np.ndarray, tolerance: float = 2.0) -> List[List[float]]:
        """将 mask 转换为多边形点"""
        from skimage import measure
        
        # 确保 mask 是二值的
        mask_bool = mask > 0.5
        
        # 查找轮廓
        contours = measure.find_contours(mask_bool.astype(np.uint8), 0.5)
        
        if len(contours) == 0:
            return []
            
        # 选择最长的轮廓
        contour = max(contours, key=len)
        
        # 简化轮廓
        from skimage.measure import approximate_polygon
        contour = approximate_polygon(contour, tolerance=tolerance)
        
        # 转换为 [x, y] 格式（注意 contours 返回的是 [y, x]）
        points = [[float(x), float(y)] for y, x in contour]
        
        return points
        
    def load_annotations(self):
        """加载 labelme 格式的 JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载标注", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
                
            # 加载对应的图像
            image_path = labelme_data.get('imagePath')
            if image_path:
                # 尝试在相同目录下查找图像
                json_dir = os.path.dirname(file_path)
                full_image_path = os.path.join(json_dir, image_path)
                
                if os.path.exists(full_image_path):
                    # 加载图像并处理 EXIF 方向信息
                    self.current_image = Image.open(full_image_path)
                    self.current_image = ImageOps.exif_transpose(self.current_image)
                    self.current_image = self.current_image.convert('RGB')
                    self.current_image_path = full_image_path
                    self.image_width = self.current_image.width
                    self.image_height = self.current_image.height
                    
                    if self.processor:
                        self.inference_state = self.processor.set_image(self.current_image)
                    
                    self.canvas.set_image(self.current_image)
            
            # 清空现有标注
            self.canvas.clear_annotations()
            self.annotation_list.clear()
            
            # 加载标注
            for shape in labelme_data.get('shapes', []):
                label = shape['label']
                points = shape['points']
                shape_type = shape.get('shape_type', 'polygon')
                
                annotation = {
                    'label': label,
                    'shape_type': shape_type,
                    'score': 1.0
                }
                
                if shape_type == 'rectangle' and len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    annotation['box'] = torch.tensor([x1, y1, x2, y2])
                elif shape_type == 'polygon':
                    # 从多边形创建 mask
                    mask = self._polygon_to_mask(points, self.image_width, self.image_height)
                    annotation['mask'] = mask
                    # 保存多边形控制点
                    annotation['polygon_points'] = points
                    
                self.canvas.add_annotation(annotation)
                self._add_annotation_to_list(annotation, len(self.canvas.annotations) - 1)
                
            QMessageBox.information(self, "成功", f"已加载 {len(labelme_data.get('shapes', []))} 个标注")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
            
    def _polygon_to_mask(self, points: List[List[float]], width: int, height: int) -> np.ndarray:
        """将多边形转换为 mask"""
        from PIL import ImageDraw
        
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # 转换为元组列表
        polygon = [(x, y) for x, y in points]
        draw.polygon(polygon, fill=255)
        
        mask = np.array(mask_img) / 255.0
        return mask
        
    def _extract_polygon_points_from_mask(self, mask: torch.Tensor, tolerance: float = 2.0, max_points: int = 100) -> List[List[float]]:
        """从 mask 提取多边形控制点"""
        try:
            from skimage import measure
            from skimage.measure import approximate_polygon
            
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            if mask.ndim == 3:
                mask = mask[0]
            
            # 确保 mask 是二值的
            mask_bool = mask > 0.5
            
            # 查找轮廓
            contours = measure.find_contours(mask_bool.astype(np.uint8), 0.5)
            
            if len(contours) == 0:
                return []
            
            # 选择最长的轮廓
            contour = max(contours, key=len)
            
            # 简化轮廓
            contour = approximate_polygon(contour, tolerance=tolerance)
            
            # 如果点太多，进一步简化
            while len(contour) > max_points and tolerance < 10.0:
                tolerance += 1.0
                contour = approximate_polygon(contour, tolerance=tolerance)
            
            # 转换为 [x, y] 格式（注意 contours 返回的是 [y, x]）
            points = [[float(x), float(y)] for y, x in contour]
            
            return points
            
        except Exception as e:
            print(f"提取多边形控制点失败: {e}")
            return []


def main():
    app = QApplication(sys.argv)
    
    # 设置样式
    app.setStyle('Fusion')
    
    window = SAM3AnnotatorApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

