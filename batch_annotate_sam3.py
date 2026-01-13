"""
SAM3 批量标注工具
从 sam3_labelme.py 提取的批量标注功能，支持非交互式批量处理图片文件夹
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有安装 tqdm，使用简单的进度显示
    def tqdm(iterable, desc=""):
        print(f"{desc}: 开始处理...")
        return iterable

import torch
import numpy as np
from PIL import Image, ImageOps

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class BatchAnnotator:
    """批量标注器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化批量标注器
        
        Args:
            model_path: SAM3 模型路径
            device: 设备 ("cuda" 或 "cpu")
        """
        print(f"正在加载模型: {model_path}")
        self.model = build_sam3_image_model(
            checkpoint_path=model_path,
            load_from_HF=False
        )
        self.model.to(device)
        self.model.eval()
        self.processor = Sam3Processor(self.model)
        self.device = device
        print("✓ 模型加载成功")
        
    def annotate_image(
        self,
        image_path: str,
        text_prompt: str,
        shape_type: str = "polygon",  # "polygon" 或 "rectangle"
        conf_threshold: float = 0.3,
        max_objects: Optional[int] = None,
        label: Optional[str] = None
    ) -> Dict:
        """
        对单张图片进行标注
        
        Args:
            image_path: 图片路径
            text_prompt: 文本提示词
            shape_type: 标注类型 ("polygon" 或 "rectangle")
            conf_threshold: 置信度阈值
            max_objects: 最大标注数量（None 表示不限制）
            label: 标签名称（None 则使用 text_prompt）
            
        Returns:
            labelme 格式的标注字典
        """
        # 加载图像并处理 EXIF 方向信息
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        image_width = image.width
        image_height = image.height
        
        # 设置图像到处理器
        inference_state = self.processor.set_image(image)
        
        # 运行文本提示
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        # 过滤和排序结果
        filtered_results = []
        for mask, box, score in zip(masks, boxes, scores):
            if score.item() >= conf_threshold:
                filtered_results.append({
                    'mask': mask,
                    'box': box,
                    'score': score.item()
                })
        
        # 按置信度排序（从高到低）
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 限制数量
        if max_objects is not None:
            filtered_results = filtered_results[:max_objects]
        
        # 转换为 labelme 格式
        label = label or text_prompt
        shapes = []
        
        for result in filtered_results:
            mask = result['mask']
            box = result['box']
            score = result['score']
            
            if shape_type == "rectangle":
                # 检测模式：只保存 box
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
            else:
                # 分割模式：保存 polygon
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                
                # 提取轮廓点
                points = self._mask_to_polygon(mask)
                
                if len(points) > 0:
                    shape = {
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    shapes.append(shape)
        
        # 构建 labelme 格式数据
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_path),
            "imageData": None,  # 不保存图像数据，减小文件大小
            "imageHeight": image_height,
            "imageWidth": image_width
        }
        
        return labelme_data
    
    def _mask_to_polygon(self, mask: np.ndarray, tolerance: float = 6.0) -> List[List[float]]:
        """将 mask 转换为多边形点"""
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
        
        # 简化轮廓（tolerance 越大，点越少）
        contour = approximate_polygon(contour, tolerance=tolerance)
        
        # 转换为 [x, y] 格式（注意 contours 返回的是 [y, x]）
        points = [[float(x), float(y)] for y, x in contour]
        
        return points
    
    def process_folder(
        self,
        folder_path: str,
        text_prompt: str,
        shape_type: str = "polygon",
        conf_threshold: float = 0.3,
        max_objects: Optional[int] = None,
        label: Optional[str] = None,
        output_folder: Optional[str] = None,
        image_extensions: List[str] = None
    ):
        """
        批量处理文件夹中的图片
        
        Args:
            folder_path: 图片文件夹路径
            text_prompt: 文本提示词
            shape_type: 标注类型 ("polygon" 或 "rectangle")
            conf_threshold: 置信度阈值
            max_objects: 每张图最大标注数量
            label: 标签名称
            output_folder: 输出文件夹（None 则保存到图片同目录）
            image_extensions: 图片扩展名列表
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # 查找所有图片文件
        folder = Path(folder_path)
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(list(set(image_files)))  # 去重并排序
        
        if len(image_files) == 0:
            print(f"警告: 在文件夹 {folder_path} 中没有找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始批量标注...")
        
        # 处理每张图片
        success_count = 0
        fail_count = 0
        
        for image_file in tqdm(image_files, desc="处理进度"):
            try:
                # 生成标注
                labelme_data = self.annotate_image(
                    image_path=str(image_file),
                    text_prompt=text_prompt,
                    shape_type=shape_type,
                    conf_threshold=conf_threshold,
                    max_objects=max_objects,
                    label=label
                )
                
                # 确定输出路径
                if output_folder:
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = Path(output_folder) / f"{image_file.stem}.json"
                else:
                    output_path = image_file.with_suffix('.json')
                
                # 保存 JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                
            except Exception as e:
                print(f"\n错误: 处理 {image_file.name} 时失败: {str(e)}")
                fail_count += 1
                continue
        
        print(f"\n完成! 成功: {success_count}, 失败: {fail_count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SAM3 批量标注工具")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="SAM3 模型路径"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="图片文件夹路径"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文本提示词（例如: 'car', 'person', 'dog'）"
    )
    parser.add_argument(
        "--shape_type",
        type=str,
        choices=["polygon", "rectangle"],
        default="polygon",
        help="标注类型: polygon (分割) 或 rectangle (检测框)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.3,
        help="置信度阈值 (默认: 0.3)"
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="每张图最大标注数量 (默认: 不限制)"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="标签名称 (默认: 使用提示词)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="输出文件夹 (默认: 保存到图片同目录)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备 (默认: cuda)"
    )
    
    args = parser.parse_args()
    
    # 创建标注器
    annotator = BatchAnnotator(model_path=args.model_path, device=args.device)
    
    # 批量处理
    annotator.process_folder(
        folder_path=args.folder,
        text_prompt=args.prompt,
        shape_type=args.shape_type,
        conf_threshold=args.conf_threshold,
        max_objects=args.max_objects,
        label=args.label,
        output_folder=args.output_folder
    )


if __name__ == "__main__":
    main()

