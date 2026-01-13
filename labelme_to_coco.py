#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Labelme 转 COCO 格式转换脚本
支持单类别和多类别分割数据集
自动检测所有类别
"""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
from tqdm import tqdm


def polygon_to_rle(polygon, height, width):
    """将多边形转换为 RLE 格式"""
    # 将多边形转换为二进制掩码
    from PIL import Image, ImageDraw
    
    mask = Image.new('L', (width, height), 0)
    # polygon 格式: [[x1, y1], [x2, y2], ...]
    if isinstance(polygon[0], (int, float)):
        # 如果是扁平列表 [x1, y1, x2, y2, ...]
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    else:
        # 如果是嵌套列表 [[x1, y1], [x2, y2], ...]
        points = polygon
    
    ImageDraw.Draw(mask).polygon(points, fill=1)
    mask = np.array(mask, dtype=np.uint8)
    
    # 转换为 RLE
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def labelme_to_coco(labelme_dir, output_dir, class_names=None, train_split=0.8):
    """
    将 Labelme 标注转换为 COCO 格式（支持多类别）
    
    Args:
        labelme_dir: Labelme 标注文件目录
        output_dir: 输出目录
        class_names: 类别名称列表（可选，如果为 None 则自动从标注中检测）
        train_split: 训练集比例
    """
    labelme_dir = Path(labelme_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有 JSON 文件
    json_files = list(labelme_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个标注文件")
    
    if len(json_files) == 0:
        raise ValueError(f"在 {labelme_dir} 中未找到 JSON 文件")
    
    # 第一步：扫描所有文件，收集所有类别名称
    print("正在扫描所有类别...")
    all_labels = set()
    for json_file in tqdm(json_files, desc="扫描类别"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            for shape in labelme_data.get('shapes', []):
                if shape.get('shape_type') == 'polygon' and 'label' in shape:
                    all_labels.add(shape['label'])
        except Exception as e:
            print(f"警告: 读取 {json_file} 失败: {e}")
            continue
    
    # 处理类别名称
    if class_names is None:
        # 自动检测：使用所有找到的类别
        class_names = sorted(list(all_labels))
        print(f"自动检测到 {len(class_names)} 个类别: {class_names}")
    else:
        # 使用用户指定的类别
        if isinstance(class_names, str):
            class_names = [class_names]  # 兼容单类别输入
        print(f"使用指定的 {len(class_names)} 个类别: {class_names}")
        # 检查是否有未指定的类别
        missing_labels = all_labels - set(class_names)
        if missing_labels:
            print(f"警告: 发现未指定的类别: {missing_labels}")
            print(f"这些类别将被忽略，或添加到 class_names 列表中")
    
    # 创建类别映射（ID 从 1 开始）
    category_map = {}
    categories = []
    for idx, class_name in enumerate(class_names, start=1):
        category_map[class_name] = idx
        categories.append({
            "id": idx,
            "name": class_name,
            "supercategory": "none"
        })
    
    print(f"\n类别映射:")
    for class_name, cat_id in category_map.items():
        print(f"  {class_name} -> ID {cat_id}")
    
    # 分割训练集和验证集
    np.random.seed(42)
    np.random.shuffle(json_files)
    split_idx = int(len(json_files) * train_split)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_files)} 张图片")
    print(f"  验证集: {len(val_files)} 张图片")
    
    # 处理训练集和验证集
    for split_name, files in [("train", train_files), ("val", val_files)]:
        if len(files) == 0:
            continue
            
        # 创建输出目录
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        images_dir = split_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # COCO 格式数据结构
        class_names_str = ", ".join(class_names) if len(class_names) <= 5 else f"{len(class_names)} classes"
        coco_data = {
            "info": {
                "description": f"Multi-class segmentation dataset ({class_names_str})",
                "version": "1.0",
                "year": 2024,
            },
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        image_id = 1
        annotation_id = 1
        
        for json_file in tqdm(files, desc=f"处理 {split_name} 集"):
            # 读取 Labelme JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            # 获取图像信息
            image_path = labelme_dir / labelme_data['imagePath']
            if not image_path.exists():
                # 尝试其他可能的路径
                image_path = json_file.parent / labelme_data['imagePath']
                if not image_path.exists():
                    print(f"警告: 找不到图像文件 {labelme_data['imagePath']}")
                    continue
            
            # 复制图像到输出目录
            img = Image.open(image_path)
            width, height = img.size
            img_filename = image_path.name
            output_img_path = images_dir / img_filename
            img.save(output_img_path)
            
            # 添加图像信息到 COCO
            coco_image = {
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height,
            }
            coco_data["images"].append(coco_image)
            
            # 处理标注
            for shape in labelme_data.get('shapes', []):
                if shape['shape_type'] != 'polygon':
                    print(f"警告: 跳过非多边形标注 {shape['shape_type']}")
                    continue
                
                # 获取类别名称
                label = shape.get('label', '')
                if not label:
                    print(f"警告: 标注缺少 label 字段，跳过")
                    continue
                
                # 检查类别是否在映射中
                if label not in category_map:
                    print(f"警告: 类别 '{label}' 不在类别列表中，跳过此标注")
                    continue
                
                category_id = category_map[label]
                
                # 获取多边形点
                points = shape['points']
                if len(points) < 3:
                    print(f"警告: 多边形点数不足，跳过")
                    continue
                
                # 转换为 RLE
                try:
                    rle = polygon_to_rle(points, height, width)
                except Exception as e:
                    print(f"警告: 转换 RLE 失败: {e}")
                    continue
                
                # 计算边界框
                mask = mask_utils.decode(rle)
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue
                
                x_min = float(np.min(x_indices))
                y_min = float(np.min(y_indices))
                x_max = float(np.max(x_indices))
                y_max = float(np.max(y_indices))
                
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = float(np.sum(mask))
                
                # 添加标注到 COCO
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,  # 多类别支持
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1
            
            image_id += 1
        
        # 保存 COCO JSON
        output_json = split_dir / "annotations.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        # 统计每个类别的标注数量
        category_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in categories if c['id'] == cat_id), f"ID_{cat_id}")
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        print(f"\n{split_name} 集转换完成:")
        print(f"  - 图像数量: {len(coco_data['images'])}")
        print(f"  - 标注数量: {len(coco_data['annotations'])}")
        print(f"  - 类别数量: {len(categories)}")
        print(f"  - 各类别标注统计:")
        for cat_name, count in sorted(category_counts.items()):
            print(f"    {cat_name}: {count} 个标注")
        print(f"  - 输出路径: {output_json}")
        print(f"  - 图像目录: {images_dir}")


def main():
    # ============================================================================
    # 硬编码配置 - 请修改以下参数
    # ============================================================================
    LABELME_DIR = r"C:\Users\29923\Desktop\3d_guajia_img\guajia_aug"  # Labelme 标注文件目录
    OUTPUT_DIR = r"C:\Users\29923\Desktop\3d_guajia_img\guajia_aug\coco"   # 输出 COCO 数据集目录
    CLASS_NAMES = None  # 类别名称列表（None 表示自动检测，或指定为 ["class1", "class2", "class3"]）
    TRAIN_SPLIT = 0.85   # 训练集比例 (0.8 = 80% 训练, 20% 验证)
    # ============================================================================
    
    # 检查路径是否存在
    if not os.path.exists(LABELME_DIR):
        print(f"错误: Labelme 目录不存在: {LABELME_DIR}")
        print("请修改脚本中的 LABELME_DIR 变量")
        return
    
    labelme_to_coco(
        labelme_dir=LABELME_DIR,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES,
        train_split=TRAIN_SPLIT
    )
    
    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)
    print(f"\n数据集结构:")
    print(f"{OUTPUT_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/")
    print(f"  │   └── annotations.json")
    print(f"  └── val/")
    print(f"      ├── images/")
    print(f"      └── annotations.json")
    print(f"\n使用说明:")
    print(f"  1. CLASS_NAMES = None: 自动检测所有类别（推荐）")
    print(f"  2. CLASS_NAMES = ['class1', 'class2']: 指定类别列表")
    print(f"  3. 确保训练配置中的 num_classes 与类别数量一致")


if __name__ == '__main__':
    main()

