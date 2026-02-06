#!/usr/bin/env python
"""
Batch inference script for SAM3 image model with hardcoded paths/settings.
Edit the constants below to match your environment, then run:

    python tools/batch_inference.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============================================================================
# Hardcoded configuration
# ============================================================================
# 标准模型的checkpoint（已经训练好，可以直接使用）
CHECKPOINT_PATH = Path(r"D:\qianpf\code\sam3-main\experiments_jiaodai\checkpoints\checkpoint_fp.pt")

INPUT_DIR = Path(r"D:\qianpf\data\auxx\images")  # folder containing source images
OUTPUT_DIR = Path(r"D:\qianpf\data\auxx\res")
PROMPT = "visual"  # e.g. "visual", "car", "guajia", etc.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.5  # 降低阈值以看到更多检测结果
USE_FP16 = False # 如果加载的是 FP16 模型，设置为 True

# 保存配置
SAVE_MASKS = False  # 是否保存 masks 图像（默认关闭）
SAVE_OVERLAYS = True  # 是否保存 overlays 可视化图像（默认关闭）

# Labelme 转换配置
CONVERT_TO_LABELME = False # 是否转换为 labelme 格式
LABELME_ANNOTATION_TYPE = "segmentation"  # 标注类型: "segmentation"（分割）或 "detection"（目标检测）

# 多类别配置（如果训练了多类别模型）
# 方式1: 使用类别列表（推荐）- 会循环分配类别标签
LABELME_CLASS_LABELS = ["guajia","insulatingTube_fore", "jiaodai"]  # 类别名称列表，必须与训练时的类别数量一

# 方式2: 使用单个类别标签（单类别或所有检测结果使用同一标签）
# LABELME_CLASS_LABEL = "1"  # 如果使用此方式，注释掉上面的 LABELME_CLASS_LABELS

# 类别分配策略（当使用 LABELME_CLASS_LABELS 时）
LABELME_CLASS_ASSIGNMENT = "by_area"  # "round_robin"（循环分配）、"by_score"（按置信度分配）或 "by_area"（按面积分组）
# "round_robin": 按顺序循环分配类别（第1个检测=class1, 第2个=class2, 第3个=class3, 第4个=class1...）
# "by_score": 根据置信度分数分配（需要设置 LABELME_CLASS_SCORE_THRESHOLDS）
# "by_area": 按面积从大到小排序，面积相近的（相差在 AREA_TOLERANCE 以内）归为同一类别，每组分配一个类别（class1, class2, class3...）
LABELME_CLASS_SCORE_THRESHOLDS = [0.5, 0.5, 0.5]  # 每个类别的置信度阈值（仅用于 "by_score" 模式）
AREA_TOLERANCE = 0.8  # 面积容差（50%），用于 "by_area" 模式，面积相差在此范围内的归为同一类别


def load_model(ckpt_path: Path, device: torch.device) -> Sam3Processor:
    """Load trained SAM3 model from checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Build empty model structure
    print("🚀 Building standard SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
        device=str(device),
        eval_mode=True,
    )
    
    # Load checkpoint file
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract state_dict from different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"✓ Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("✓ Loaded checkpoint (state_dict format)")
    else:
        state_dict = checkpoint
        print("✓ Loaded checkpoint (raw weights)")
    
    # Load weights into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"⚠ Missing keys ({len(missing)} keys):")
        for key in missing[:5]:  # Show first 5
            print(f"  - {key}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    if unexpected:
        print(f"⚠ Unexpected keys ({len(unexpected)} keys):")
        for key in unexpected[:5]:  # Show first 5
            print(f"  - {key}")
        if len(unexpected) > 5:
            print(f"  ... and {len(unexpected) - 5} more")
    
    if not missing and not unexpected:
        print("✓ All weights loaded successfully!")
    
    # 如果使用 FP16 模型
    if USE_FP16:
        if device.type == "cuda":
            model = model.half()
            print("✓ Using FP16 (half precision)")
        else:
            print("⚠ FP16 only supported on CUDA, using FP32 on CPU")
    
    model.eval()
    processor = Sam3Processor(model, device=str(device))
    print("✓ Model ready for inference\n")
    return processor


def iter_images(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in exts:
            yield path


def blend_masks(image: Image.Image, masks: np.ndarray, colors: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Blend colored masks on top of the original image.
    """
    base = np.array(image).astype(np.float32)
    overlay = base.copy()

    for color, mask in zip(colors, masks):
        if mask.ndim == 3:
            mask = np.squeeze(mask, axis=0)
        mask_bin = mask > 0.5
        if not mask_bin.any():
            continue
        color_px = (color * 255).astype(np.float32)
        overlay[mask_bin] = overlay[mask_bin] * (1 - alpha) + color_px * alpha

    return Image.fromarray(overlay.astype(np.uint8))


def save_masks(masks: np.ndarray, out_dir: Path, stem: str):
    """Save each mask as a separate binary image."""
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # Ensure mask is 2D (H, W)
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            print(f"[WARN] Unexpected mask shape {mask.shape}, skipping...")
            continue
        # Convert to binary image
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_binary, mode='L')
        mask_img.save(mask_dir / f"{stem}_mask_{idx:02d}.png")


def run_inference(
    processor: Sam3Processor,
    image_path: Path,
    prompt: str,
    score_threshold: float,
) -> Dict[str, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    
    # 如果使用 FP16，需要使用 autocast 包裹推理过程
    if USE_FP16 and DEVICE.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            state = processor.set_image(image)
            outputs = processor.set_text_prompt(state=state, prompt=prompt)
    else:
        state = processor.set_image(image)
        outputs = processor.set_text_prompt(state=state, prompt=prompt)
    
    # 调试信息：显示原始检测结果
    raw_scores = outputs["scores"].cpu().numpy()
    print(f"  {image_path.name}: 检测到 {len(raw_scores)} 个物体")
    if len(raw_scores) > 0:
        print(f"    分数范围: [{raw_scores.min():.3f}, {raw_scores.max():.3f}]")
        print(f"    平均分数: {raw_scores.mean():.3f}")
    
    # Filter by confidence
    scores = outputs["scores"]
    keep = scores >= score_threshold
    num_kept = keep.sum().item()
    print(f"    阈值 {score_threshold} 后保留: {num_kept} 个")
    
    for key in ("masks", "boxes", "scores"):
        outputs[key] = outputs[key][keep]
    outputs["image"] = image
    return outputs


def visualize_and_save(image: Image.Image, outputs: Dict[str, torch.Tensor], out_path: Path):
    masks = outputs["masks"].cpu().numpy()
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    if len(masks) == 0:
        image.save(out_path)
        return

    colors = plt_colormap(len(masks))
    overlay = blend_masks(image, masks, colors)
    draw = ImageDraw.Draw(overlay)
    for color, box, score in zip(colors, boxes, scores):
        x1, y1, x2, y2 = box.tolist()
        rgb = tuple(int(c * 255) for c in color)
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill=rgb)
    overlay.save(out_path)


def plt_colormap(n: int) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 3))
    cmap = np.linspace(0, 1, n)
    colors = np.stack([np.sin(2 * np.pi * (cmap + shift)) for shift in (0, 0.33, 0.66)], axis=1)
    colors = (colors * 0.5 + 0.5).clip(0, 1)
    return colors


def calculate_mask_area(mask: np.ndarray) -> float:
    """
    计算 mask 的面积（像素数）。
    
    Args:
        mask: mask 数组 (H, W) 或 (1, H, W)
    
    Returns:
        面积（像素数）
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0)
    if mask.ndim != 2:
        return 0.0
    
    mask_binary = (mask > 0.5).astype(np.uint8)
    area = np.sum(mask_binary)
    return float(area)


def group_by_area(masks: np.ndarray, tolerance: float = 0.3) -> List[List[int]]:
    """
    根据面积对 mask 进行分组，面积相差在 tolerance（百分比）以内的归为一组。
    
    Args:
        masks: mask 数组 (N, H, W) 或 (N, 1, H, W)
        tolerance: 面积容差（百分比），例如 0.3 表示 30%
    
    Returns:
        分组列表，每个元素是一个索引列表，表示属于同一组的 mask 索引
    """
    if len(masks) == 0:
        return []
    
    # 计算每个 mask 的面积
    areas = []
    for i, mask in enumerate(masks):
        area = calculate_mask_area(mask)
        areas.append((i, area))
    
    # 按面积排序
    areas.sort(key=lambda x: x[1])
    
    # 分组：面积相差在 tolerance 以内的归为一组
    groups = []
    current_group = [areas[0][0]]  # 第一个 mask 的索引
    current_base_area = areas[0][1]
    
    for i in range(1, len(areas)):
        idx, area = areas[i]
        
        # 检查是否与当前组的基准面积相差在容差范围内
        if current_base_area > 0:
            area_diff_ratio = abs(area - current_base_area) / current_base_area
        else:
            area_diff_ratio = float('inf') if area > 0 else 0.0
        
        if area_diff_ratio <= tolerance:
            # 属于当前组
            current_group.append(idx)
        else:
            # 开始新组
            groups.append(current_group)
            current_group = [idx]
            current_base_area = area
    
    # 添加最后一组
    if current_group:
        groups.append(current_group)
    
    return groups


def calculate_box_area(box: np.ndarray) -> float:
    """
    计算边界框的面积。
    
    Args:
        box: 边界框数组 [x1, y1, x2, y2]
    
    Returns:
        面积（像素数）
    """
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return float(width * height)


def group_boxes_by_area(boxes: np.ndarray, tolerance: float = 0.3) -> List[List[int]]:
    """
    根据面积对边界框进行分组，面积相差在 tolerance（百分比）以内的归为一组。
    
    Args:
        boxes: 边界框数组 (N, 4)，格式为 [x1, y1, x2, y2]
        tolerance: 面积容差（百分比），例如 0.3 表示 30%
    
    Returns:
        分组列表，每个元素是一个索引列表，表示属于同一组的边界框索引
    """
    if len(boxes) == 0:
        return []
    
    # 计算每个边界框的面积
    areas = []
    for i, box in enumerate(boxes):
        area = calculate_box_area(box)
        areas.append((i, area))
    
    # 按面积排序
    areas.sort(key=lambda x: x[1])
    
    # 分组：面积相差在 tolerance 以内的归为一组
    groups = []
    current_group = [areas[0][0]]  # 第一个边界框的索引
    current_base_area = areas[0][1]
    
    for i in range(1, len(areas)):
        idx, area = areas[i]
        
        # 检查是否与当前组的基准面积相差在容差范围内
        if current_base_area > 0:
            area_diff_ratio = abs(area - current_base_area) / current_base_area
        else:
            area_diff_ratio = float('inf') if area > 0 else 0.0
        
        if area_diff_ratio <= tolerance:
            # 属于当前组
            current_group.append(idx)
        else:
            # 开始新组
            groups.append(current_group)
            current_group = [idx]
            current_base_area = area
    
    # 添加最后一组
    if current_group:
        groups.append(current_group)
    
    return groups


def assign_labels_by_area(
    masks: np.ndarray,
    class_labels: List[str],
    tolerance: float = 0.3,
) -> List[str]:
    """
    根据面积从大到小排序，面积相近的归为同一类别。
    
    Args:
        masks: mask 数组 (N, H, W) 或 (N, 1, H, W)
        class_labels: 类别标签列表
        tolerance: 面积容差（百分比），例如 0.3 表示 30%，面积相差在此范围内的归为同一类别
    
    Returns:
        每个 mask 对应的类别标签列表（面积相近的归为同一类别）
    """
    if len(masks) == 0:
        return []
    
    # 计算每个 mask 的面积
    areas = []
    for i, mask in enumerate(masks):
        area = calculate_mask_area(mask)
        areas.append((i, area))
    
    # 按面积从大到小排序
    areas.sort(key=lambda x: x[1], reverse=True)
    
    # 分组：面积相近的（相差在 tolerance 以内）归为一组
    groups = []
    if len(areas) > 0:
        current_group = [areas[0][0]]  # 第一个 mask 的索引
        current_base_area = areas[0][1]
        
        for i in range(1, len(areas)):
            idx, area = areas[i]
            
            # 检查是否与当前组的基准面积相差在容差范围内
            if current_base_area > 0:
                area_diff_ratio = abs(area - current_base_area) / current_base_area
            else:
                area_diff_ratio = float('inf') if area > 0 else 0.0
            
            if area_diff_ratio <= tolerance:
                # 属于当前组（面积相近）
                current_group.append(idx)
            else:
                # 开始新组（面积差距较大）
                groups.append(current_group)
                current_group = [idx]
                current_base_area = area
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
    
    # 为每个 mask 分配标签
    num_masks = len(masks)
    labels = [None] * num_masks
    
    # 为每个组分配一个类别标签（循环使用 class1, class2, class3...）
    for group_idx, group in enumerate(groups):
        class_label = class_labels[group_idx % len(class_labels)]
        for mask_idx in group:
            labels[mask_idx] = class_label
    
    return labels


def assign_labels_by_box_area(
    boxes: np.ndarray,
    class_labels: List[str],
    tolerance: float = 0.3,
) -> List[str]:
    """
    根据边界框面积从大到小排序，面积相近的归为同一类别。
    
    Args:
        boxes: 边界框数组 (N, 4)，格式为 [x1, y1, x2, y2]
        class_labels: 类别标签列表
        tolerance: 面积容差（百分比），例如 0.3 表示 30%，面积相差在此范围内的归为同一类别
    
    Returns:
        每个边界框对应的类别标签列表（面积相近的归为同一类别）
    """
    if len(boxes) == 0:
        return []
    
    # 计算每个边界框的面积
    areas = []
    for i, box in enumerate(boxes):
        area = calculate_box_area(box)
        areas.append((i, area))
    
    # 按面积从大到小排序
    areas.sort(key=lambda x: x[1], reverse=True)
    
    # 分组：面积相近的（相差在 tolerance 以内）归为一组
    groups = []
    if len(areas) > 0:
        current_group = [areas[0][0]]  # 第一个边界框的索引
        current_base_area = areas[0][1]
        
        for i in range(1, len(areas)):
            idx, area = areas[i]
            
            # 检查是否与当前组的基准面积相差在容差范围内
            if current_base_area > 0:
                area_diff_ratio = abs(area - current_base_area) / current_base_area
            else:
                area_diff_ratio = float('inf') if area > 0 else 0.0
            
            if area_diff_ratio <= tolerance:
                # 属于当前组（面积相近）
                current_group.append(idx)
            else:
                # 开始新组（面积差距较大）
                groups.append(current_group)
                current_group = [idx]
                current_base_area = area
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
    
    # 为每个边界框分配标签
    num_boxes = len(boxes)
    labels = [None] * num_boxes
    
    # 为每个组分配一个类别标签（循环使用 class1, class2, class3...）
    for group_idx, group in enumerate(groups):
        class_label = class_labels[group_idx % len(class_labels)]
        for box_idx in group:
            labels[box_idx] = class_label
    
    return labels


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.002) -> Optional[List[List[float]]]:
    """
    将二值 mask 转换为多边形点列表。
    
    Args:
        mask: 二值 mask 数组 (H, W)
        epsilon_factor: 多边形简化系数，越大点数越少（默认 0.002，约为原来的一半点数）
    
    Returns:
        多边形点列表 [[x1, y1], [x2, y2], ...]，如果 mask 为空则返回 None
    """
    # 确保 mask 是二值的
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        return None
    
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 简化轮廓（减少点数，epsilon_factor 越大，点数越少）
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 转换为点列表格式 [[x, y], ...]
    polygon = approx.reshape(-1, 2).tolist()
    
    return polygon


def boxes_to_labelme_rectangles(boxes: np.ndarray, label: str) -> List[Dict]:
    """
    将边界框转换为 labelme 矩形格式。
    
    Args:
        boxes: 边界框数组 (N, 4)，格式为 [x1, y1, x2, y2]
        label: 类别标签
    
    Returns:
        labelme shape 列表
    """
    shapes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        
        # labelme 矩形格式：points 是 [[x1, y1], [x2, y2]]
        shape = {
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        }
        shapes.append(shape)
    
    return shapes


def masks_to_labelme_json(
    masks: np.ndarray,
    image_path: Path,
    image: Image.Image,
    label: str,
    annotation_type: str = "segmentation",
    boxes: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    scores: Optional[np.ndarray] = None,
) -> Dict:
    """
    将多个 mask 或 boxes 转换为 labelme 格式的 JSON。
    
    Args:
        masks: mask 数组 (N, H, W) 或 (N, 1, H, W)，用于分割格式
        image_path: 图像文件路径
        image: PIL Image 对象
        label: 类别标签（单类别时使用，如果提供了 labels 则忽略）
        annotation_type: 标注类型，"segmentation"（分割）或 "detection"（目标检测）
        boxes: 边界框数组 (N, 4)，格式为 [x1, y1, x2, y2]，用于目标检测格式
        labels: 类别标签列表（多类别时使用），长度应与 masks/boxes 数量一致
        scores: 置信度分数数组（可选），用于按分数分配类别
    
    Returns:
        labelme 格式的字典
    """
    shapes = []
    
    # 确定每个检测结果的类别标签
    num_detections = len(masks) if annotation_type == "segmentation" else (len(boxes) if boxes is not None else 0)
    
    if labels is not None and len(labels) > 0:
        # 使用提供的类别标签列表
        detection_labels = labels
    else:
        # 使用单个类别标签
        detection_labels = [label] * num_detections
    
    if annotation_type == "detection":
        # 目标检测格式：使用边界框
        if boxes is not None and len(boxes) > 0:
            shapes = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.tolist()
                shape = {
                    "label": detection_labels[i] if i < len(detection_labels) else label,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
                shapes.append(shape)
    else:
        # 分割格式：使用 mask 转换为多边形
        for i, mask in enumerate(masks):
            # 确保 mask 是 2D
            mask_2d = np.squeeze(mask)
            if mask_2d.ndim != 2:
                continue
            
            polygon = mask_to_polygon(mask_2d)
            if polygon is None or len(polygon) < 3:  # 至少需要3个点才能形成多边形
                continue
            
            shape = {
                "label": detection_labels[i] if i < len(detection_labels) else label,
                "points": polygon,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
            shapes.append(shape)
    
    labelme_json = {
        "version": "5.8.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": image.height,
        "imageWidth": image.width,
    }
    
    return labelme_json


def save_labelme_json(
    labelme_data: Dict,
    output_dir: Path,
    stem: str,
):
    """保存 labelme 格式的 JSON 文件。"""
    labelme_dir = output_dir / "labelme_annotations"
    labelme_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = labelme_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)
    
    return json_path


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    processor = load_model(CHECKPOINT_PATH, DEVICE)

    for img_path in tqdm(list(iter_images(input_dir)), desc="Infer"):
        outputs = run_inference(processor, img_path, PROMPT, SCORE_THRESHOLD)
        image = outputs.pop("image")
        stem = img_path.stem
        
        # 保存可视化结果（如果启用）
        if SAVE_OVERLAYS:
            visualize_and_save(image, outputs, overlay_dir / f"{stem}_overlay.png")
        
        # 保存 mask 图像（如果启用）
        if SAVE_MASKS:
            save_masks(outputs["masks"].cpu().numpy(), output_dir, stem)
        
        # 转换为 labelme 格式（如果启用）
        if CONVERT_TO_LABELME:
            masks_np = outputs["masks"].cpu().numpy()
            boxes_np = outputs["boxes"].cpu().numpy() if "boxes" in outputs else None
            scores_np = outputs["scores"].cpu().numpy() if "scores" in outputs else None
            
            # 确定类别标签
            if "LABELME_CLASS_LABELS" in globals() and len(LABELME_CLASS_LABELS) > 0:
                # 多类别模式：根据策略分配类别标签
                num_detections = len(masks_np) if LABELME_ANNOTATION_TYPE == "segmentation" else (len(boxes_np) if boxes_np is not None else 0)
                
                if LABELME_CLASS_ASSIGNMENT == "by_area":
                    if LABELME_ANNOTATION_TYPE == "segmentation":
                        # 根据 mask 面积分组分类（面积相近的归为同一类别）
                        detection_labels = assign_labels_by_area(
                            masks_np,
                            LABELME_CLASS_LABELS,
                            tolerance=AREA_TOLERANCE
                        )
                        # 打印分组信息（用于调试）
                        groups = group_by_area(masks_np, tolerance=AREA_TOLERANCE)
                        print(f"    面积分组分类（容差 {AREA_TOLERANCE*100:.0f}%）: {len(groups)} 个组，共 {num_detections} 个检测")
                        for group_idx, group in enumerate(groups):
                            areas = [calculate_mask_area(masks_np[i]) for i in group]
                            min_area = min(areas)
                            max_area = max(areas)
                            avg_area = np.mean(areas)
                            label = LABELME_CLASS_LABELS[group_idx % len(LABELME_CLASS_LABELS)]
                            print(f"      组 {group_idx + 1} ({label}): {len(group)} 个mask，面积范围 [{min_area:.0f}, {max_area:.0f}]，平均 {avg_area:.0f} 像素")
                    elif LABELME_ANNOTATION_TYPE == "detection" and boxes_np is not None:
                        # 根据边界框面积分组分类（面积相近的归为同一类别）
                        detection_labels = assign_labels_by_box_area(
                            boxes_np,
                            LABELME_CLASS_LABELS,
                            tolerance=AREA_TOLERANCE
                        )
                        # 打印分组信息（用于调试）
                        groups = group_boxes_by_area(boxes_np, tolerance=AREA_TOLERANCE)
                        print(f"    面积分组分类（容差 {AREA_TOLERANCE*100:.0f}%）: {len(groups)} 个组，共 {num_detections} 个检测")
                        for group_idx, group in enumerate(groups):
                            areas = [calculate_box_area(boxes_np[i]) for i in group]
                            min_area = min(areas)
                            max_area = max(areas)
                            avg_area = np.mean(areas)
                            label = LABELME_CLASS_LABELS[group_idx % len(LABELME_CLASS_LABELS)]
                            print(f"      组 {group_idx + 1} ({label}): {len(group)} 个边界框，面积范围 [{min_area:.0f}, {max_area:.0f}]，平均 {avg_area:.0f} 像素")
                    else:
                        # 如果无法使用面积分组，回退到循环分配
                        detection_labels = [LABELME_CLASS_LABELS[i % len(LABELME_CLASS_LABELS)] for i in range(num_detections)]
                elif LABELME_CLASS_ASSIGNMENT == "by_score" and scores_np is not None:
                    # 根据置信度分数分配类别
                    detection_labels = []
                    for score in scores_np:
                        assigned = False
                        for i, threshold in enumerate(LABELME_CLASS_SCORE_THRESHOLDS):
                            if score >= threshold:
                                detection_labels.append(LABELME_CLASS_LABELS[i])
                                assigned = True
                                break
                        if not assigned:
                            # 如果分数低于所有阈值，使用最后一个类别
                            detection_labels.append(LABELME_CLASS_LABELS[-1])
                else:
                    # 循环分配（round_robin）
                    detection_labels = [LABELME_CLASS_LABELS[i % len(LABELME_CLASS_LABELS)] for i in range(num_detections)]
                
                class_label = None  # 不使用单个标签
            else:
                # 单类别模式：使用单个类别标签
                detection_labels = None
                class_label = globals().get("LABELME_CLASS_LABEL", "object")
            
            # 根据标注类型决定使用哪个数据
            if LABELME_ANNOTATION_TYPE == "detection":
                if boxes_np is not None and len(boxes_np) > 0:
                    labelme_data = masks_to_labelme_json(
                        masks_np,  # 虽然不使用，但保持接口一致
                        img_path,
                        image,
                        class_label or "object",
                        annotation_type="detection",
                        boxes=boxes_np,
                        labels=detection_labels,
                        scores=scores_np,
                    )
                    json_path = save_labelme_json(labelme_data, output_dir, stem)
                    print(f"    ✓ 已保存 labelme 检测标注: {json_path.name} ({len(labelme_data['shapes'])} 个目标)")
            else:  # segmentation
                if len(masks_np) > 0:
                    labelme_data = masks_to_labelme_json(
                        masks_np,
                        img_path,
                        image,
                        class_label or "object",
                        annotation_type="segmentation",
                        labels=detection_labels,
                        scores=scores_np,
                    )
                    json_path = save_labelme_json(labelme_data, output_dir, stem)
                    print(f"    ✓ 已保存 labelme 分割标注: {json_path.name} ({len(labelme_data['shapes'])} 个目标)")

    print(f"\n推理完成。结果已保存到: {output_dir}")
    if CONVERT_TO_LABELME:
        print(f"Labelme 标注文件保存在: {output_dir / 'labelme_annotations'}")


if __name__ == "__main__":
    main()

