"""
批量自动标注脚本
使用 SAM3 模型对整个文件夹的图像进行自动标注
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def mask_to_polygon(mask: np.ndarray, tolerance: float = 2.0) -> List[List[float]]:
    """将 mask 转换为多边形点"""
    try:
        from skimage import measure
        from skimage.measure import approximate_polygon
        
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
        
        # 转换为 [x, y] 格式（注意 contours 返回的是 [y, x]）
        points = [[float(x), float(y)] for y, x in contour]
        
        return points
    except Exception as e:
        print(f"Warning: Failed to convert mask to polygon: {e}")
        return []


def convert_to_labelme_format(
    image_path: str,
    annotations: List[Dict],
    save_image_data: bool = False
) -> Dict:
    """转换为 labelme 格式"""
    
    # 读取图像信息并处理 EXIF 方向
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image_width, image_height = image.size
    
    # 读取图像数据
    image_base64 = ""
    if save_image_data:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    shapes = []
    for ann in annotations:
        label = ann['label']
        
        # 从 mask 提取轮廓点
        if 'mask' in ann and ann['mask'] is not None:
            mask = ann['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
                
            if mask.ndim == 3:
                mask = mask[0]
                
            # 提取轮廓
            points = mask_to_polygon(mask, tolerance=2.0)
            
            if len(points) > 0:
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
        "imagePath": os.path.basename(image_path),
        "imageData": image_base64,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    return labelme_data


def batch_annotate(
    image_dir: str,
    output_dir: str,
    model_path: str,
    text_prompts: List[str],
    confidence_threshold: float = 0.3,
    save_image_data: bool = False,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
):
    """批量标注图像"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    model = build_sam3_image_model(
        checkpoint_path=model_path,
        load_from_HF=False
    )
    processor = Sam3Processor(model)
    print("Model loaded successfully!")
    
    # 获取所有图像文件（使用 set 去重，避免在不区分大小写的文件系统中重复）
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(Path(image_dir).glob(f"*{ext}"))
        image_files_set.update(Path(image_dir).glob(f"*{ext.upper()}"))
    
    image_files = sorted(list(image_files_set))
    
    print(f"\nFound {len(image_files)} images in {image_dir}")
    print(f"Text prompts: {text_prompts}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Output directory: {output_dir}\n")
    
    # 处理每张图像
    success_count = 0
    fail_count = 0
    
    for image_path in tqdm(image_files, desc="Annotating images"):
        try:
            # 加载图像并处理 EXIF 方向信息
            image = Image.open(image_path)
            image = ImageOps.exif_transpose(image)
            image = image.convert('RGB')
            inference_state = processor.set_image(image)
            
            # 收集所有检测结果
            all_annotations = []
            
            # 对每个文本提示运行推理
            for prompt in text_prompts:
                try:
                    output = processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt
                    )
                    
                    masks = output["masks"]
                    boxes = output["boxes"]
                    scores = output["scores"]
                    
                    # 过滤低置信度结果
                    for mask, box, score in zip(masks, boxes, scores):
                        if score.item() >= confidence_threshold:
                            annotation = {
                                'label': prompt,
                                'mask': mask,
                                'box': box,
                                'score': score.item()
                            }
                            all_annotations.append(annotation)
                            
                except Exception as e:
                    print(f"\nWarning: Failed to process prompt '{prompt}' for {image_path.name}: {e}")
                    continue
            
            # 转换为 labelme 格式
            labelme_data = convert_to_labelme_format(
                str(image_path),
                all_annotations,
                save_image_data=save_image_data
            )
            
            # 保存 JSON
            json_filename = image_path.stem + '.json'
            json_path = os.path.join(output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            
            success_count += 1
            
            # 打印统计信息
            tqdm.write(f"✓ {image_path.name}: {len(all_annotations)} objects detected")
            
        except Exception as e:
            fail_count += 1
            tqdm.write(f"✗ {image_path.name}: Error - {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch annotation completed!")
    print(f"  Success: {success_count}/{len(image_files)}")
    print(f"  Failed: {fail_count}/{len(image_files)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 批量自动标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 标注单个类别:
   python batch_annotate.py \\
       --image_dir ./images \\
       --output_dir ./annotations \\
       --model_path ./experiments/checkpoints/model_fp16.pt \\
       --prompts car

2. 标注多个类别:
   python batch_annotate.py \\
       --image_dir ./images \\
       --output_dir ./annotations \\
       --model_path ./experiments/checkpoints/model_fp16.pt \\
       --prompts person car truck bus bicycle \\
       --threshold 0.4

3. 保存图像数据到 JSON（增大文件但完全独立）:
   python batch_annotate.py \\
       --image_dir ./images \\
       --output_dir ./annotations \\
       --model_path ./experiments/checkpoints/model_fp16.pt \\
       --prompts car \\
       --save_image_data
       
注意: 默认不保存图像数据以减小 JSON 文件大小
        """
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='输入图像目录'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出 JSON 文件目录'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='SAM3 模型文件路径'
    )
    
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        required=True,
        help='文本提示列表（类别名称）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='置信度阈值 (default: 0.3)'
    )
    
    parser.add_argument(
        '--save_image_data',
        action='store_true',
        help='在 JSON 中保存 base64 图像数据（默认不保存以减小文件大小）'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='图像文件扩展名 (default: .jpg .jpeg .png .bmp)'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist!")
        return
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist!")
        return
    
    # 运行批量标注
    batch_annotate(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        text_prompts=args.prompts,
        confidence_threshold=args.threshold,
        save_image_data=args.save_image_data,
        image_extensions=args.extensions
    )


if __name__ == '__main__':
    main()

