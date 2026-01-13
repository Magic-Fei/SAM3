#!/usr/bin/env python
"""
模型诊断脚本：对比标准模型和轻量级模型的输出

用法:
    python tools/diagnose_model.py
"""

from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps

from sam3.model_builder import build_sam3_image_model
from sam3.model_builder_lite import build_sam3_lite_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============================================================================
# 配置
# ============================================================================
STANDARD_CKPT = Path(r"D:\qianpf\code\sam3-main\experiments\checkpoints\checkpoint_7.pt")
LITE_CKPT = Path(r"D:\qianpf\code\sam3-main\experiments_lite\checkpoints\checkpoint_10.pt")
TEST_IMAGE = Path(r"C:\Users\29923\Desktop\1")  # 找第一张图片测试
PROMPT = "visual"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint_info(ckpt_path):
    """加载checkpoint并显示关键信息"""
    print(f"\n{'='*80}")
    print(f"检查 checkpoint: {ckpt_path.name}")
    print(f"{'='*80}")
    
    if not ckpt_path.exists():
        print(f"❌ 文件不存在: {ckpt_path}")
        return None
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 显示基本信息
    print(f"✓ 文件大小: {ckpt_path.stat().st_size / (1024**3):.2f} GB")
    
    if "epoch" in ckpt:
        print(f"✓ Epoch: {ckpt['epoch']}")
    
    # 检查是否有loss信息
    if "train_loss" in ckpt:
        print(f"✓ 训练Loss: {ckpt['train_loss']:.4f}")
    
    # 检查模型权重
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    # 统计权重信息
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"✓ 模型参数量: {total_params / 1e6:.1f}M")
    
    # 检查权重是否全零或异常
    weight_stats = []
    for name, param in list(state_dict.items())[:10]:
        if isinstance(param, torch.Tensor) and param.numel() > 0:
            weight_stats.append({
                'name': name,
                'mean': param.float().mean().item(),
                'std': param.float().std().item(),
                'min': param.float().min().item(),
                'max': param.float().max().item(),
            })
    
    print(f"\n前10个权重统计:")
    for stat in weight_stats[:3]:
        print(f"  {stat['name'][:60]}")
        print(f"    mean={stat['mean']:.4f}, std={stat['std']:.4f}, range=[{stat['min']:.4f}, {stat['max']:.4f}]")
    
    return state_dict


def test_model(model_type, ckpt_path, test_image):
    """测试模型推理"""
    print(f"\n{'='*80}")
    print(f"测试 {model_type} 模型推理")
    print(f"{'='*80}")
    
    # 构建模型
    if model_type == "standard":
        model = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            device=DEVICE,
            eval_mode=True,
        )
    else:
        model = build_sam3_lite_model(
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            device=DEVICE,
            eval_mode=True,
        )
    
    # 加载权重
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✓ 权重加载: missing={len(missing)}, unexpected={len(unexpected)}")
    
    if missing:
        print(f"  Missing keys (前5个): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys (前5个): {unexpected[:5]}")
    
    # 创建processor
    model.eval()
    processor = Sam3Processor(model, device=DEVICE)
    
    # 测试推理并处理 EXIF 方向信息
    image = Image.open(test_image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    print(f"✓ 测试图片: {test_image.name} ({image.size})")
    
    with torch.no_grad():
        state = processor.set_image(image)
        outputs = processor.set_text_prompt(state=state, prompt=PROMPT)
    
    # 分析输出
    scores = outputs["scores"].cpu().numpy()
    boxes = outputs["boxes"].cpu().numpy()
    masks = outputs["masks"].cpu().numpy()
    
    print(f"\n推理结果:")
    print(f"  检测数量: {len(scores)}")
    if len(scores) > 0:
        print(f"  分数统计: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        print(f"  分数分布: {np.histogram(scores, bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0])[0]}")
        print(f"  Boxes shape: {boxes.shape}")
        print(f"  Masks shape: {masks.shape}")
        
        # 显示前5个检测
        print(f"\n  前5个检测:")
        for i, score in enumerate(scores[:5]):
            print(f"    [{i}] score={score:.4f}, box={boxes[i]}")
    else:
        print(f"  ⚠️  没有检测到任何物体！")
    
    return scores, boxes, masks


def main():
    # 找第一张测试图片
    test_img = None
    if TEST_IMAGE.is_dir():
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            imgs = list(TEST_IMAGE.glob(f'*{ext}'))
            if imgs:
                test_img = imgs[0]
                break
    else:
        test_img = TEST_IMAGE
    
    if not test_img or not test_img.exists():
        print(f"❌ 找不到测试图片: {TEST_IMAGE}")
        return
    
    print(f"\n🔍 模型诊断开始")
    print(f"测试图片: {test_img}")
    print(f"提示词: {PROMPT}")
    print(f"设备: {DEVICE}")
    
    # 检查checkpoint信息
    print(f"\n" + "="*80)
    print(f"第一步: 检查 Checkpoint 信息")
    print(f"="*80)
    
    std_state = load_checkpoint_info(STANDARD_CKPT)
    lite_state = load_checkpoint_info(LITE_CKPT)
    
    if std_state is None or lite_state is None:
        print("\n❌ 缺少必要的checkpoint文件")
        return
    
    # 测试推理
    print(f"\n" + "="*80)
    print(f"第二步: 测试模型推理")
    print(f"="*80)
    
    print("\n" + "-"*80)
    print("标准模型:")
    print("-"*80)
    std_scores, std_boxes, std_masks = test_model("standard", STANDARD_CKPT, test_img)
    
    print("\n" + "-"*80)
    print("轻量级模型:")
    print("-"*80)
    lite_scores, lite_boxes, lite_masks = test_model("lite", LITE_CKPT, test_img)
    
    # 对比结果
    print(f"\n" + "="*80)
    print(f"第三步: 对比分析")
    print(f"="*80)
    
    print(f"\n标准模型 vs 轻量级模型:")
    print(f"  检测数量: {len(std_scores)} vs {len(lite_scores)}")
    
    if len(std_scores) > 0:
        print(f"  标准模型 - 平均分数: {std_scores.mean():.4f}")
    else:
        print(f"  标准模型 - 没有检测")
    
    if len(lite_scores) > 0:
        print(f"  轻量级模型 - 平均分数: {lite_scores.mean():.4f}")
    else:
        print(f"  轻量级模型 - 没有检测")
    
    # 诊断结论
    print(f"\n" + "="*80)
    print(f"诊断结论:")
    print(f"="*80)
    
    if len(std_scores) > 0 and len(lite_scores) == 0:
        print(f"\n❌ 问题: 轻量级模型训练失败")
        print(f"   标准模型能检测到物体，但轻量级模型检测不到")
        print(f"   可能原因:")
        print(f"   1. 训练配置有误（学习率、损失函数等）")
        print(f"   2. 模型架构不兼容")
        print(f"   3. 训练时间不够（只有10 epochs）")
        print(f"   4. 数据加载有问题")
        print(f"\n   建议:")
        print(f"   1. 检查训练日志中的loss值是否正常下降")
        print(f"   2. 增加训练轮数到20-30 epochs")
        print(f"   3. 对比两个配置文件的损失函数配置")
    elif len(std_scores) == 0 and len(lite_scores) == 0:
        print(f"\n❌ 问题: 两个模型都无法检测")
        print(f"   可能是推理配置问题或测试图片问题")
    elif len(lite_scores) > 0 and lite_scores.mean() < 0.1:
        print(f"\n⚠️  警告: 轻量级模型置信度很低")
        print(f"   轻量级模型能检测到物体，但分数很低（<0.1）")
        print(f"   说明模型还在训练早期，需要更多训练")
    else:
        print(f"\n✅ 两个模型都能正常推理")


if __name__ == "__main__":
    main()

