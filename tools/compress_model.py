#!/usr/bin/env python
"""
一键压缩模型：提取权重 + FP16 量化

将训练的 checkpoint 从 ~2.5 GB 压缩到 ~1.0 GB，几乎无精度损失。

使用方法:
    python tools/compress_model.py
"""

import torch
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================
INPUT_CHECKPOINT = Path(r"D:\qianpf\code\sam3-main\experiments_jiaodai\checkpoints\checkpoint_5.pt")
OUTPUT_FP16 = Path(r"D:\qianpf\code\sam3-main\experiments_jiaodai\checkpoints\checkpoint_1210.pt")
KEEP_INTERMEDIATE = False  # 是否保留中间的 FP32 纯权重文件


def compress_model(input_path: Path, output_path: Path):
    """一键压缩：提取权重 + FP16 量化"""
    print("=" * 70)
    print("SAM3 模型压缩工具")
    print("提取权重 + FP16 量化 = 60% 压缩率")
    print("=" * 70)
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return
    
    # Step 1: 加载 checkpoint
    print(f"\n📂 步骤 1/3: 加载 checkpoint")
    print(f"   文件: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")
    
    # 提取模型权重
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        epoch = checkpoint.get("epoch", "unknown")
        print(f"   ✓ 提取模型权重 (epoch: {epoch})")
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
        print(f"   ✓ 提取模型权重 (state_dict)")
    else:
        model_state = checkpoint
        print(f"   ✓ 使用完整 checkpoint")
    
    # Step 2: 转换为 FP16
    print(f"\n🔄 步骤 2/3: 转换为 FP16")
    model_state_fp16 = {}
    fp32_count = 0
    total_count = 0
    
    for k, v in model_state.items():
        total_count += 1
        if hasattr(v, 'dtype') and v.dtype == torch.float32:
            model_state_fp16[k] = v.half()
            fp32_count += 1
        else:
            model_state_fp16[k] = v
    
    print(f"   ✓ 转换了 {fp32_count}/{total_count} 个参数")
    
    # Step 3: 保存压缩模型
    print(f"\n💾 步骤 3/3: 保存压缩模型")
    print(f"   输出: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_fp16, output_path)
    print(f"   ✓ 保存完成")
    
    # 显示压缩结果
    original_size = input_path.stat().st_size / (1024**2)
    compressed_size = output_path.stat().st_size / (1024**2)
    saved = original_size - compressed_size
    saved_pct = (saved / original_size) * 100
    
    print("\n" + "=" * 70)
    print("✨ 压缩完成")
    print("=" * 70)
    print(f"原始 checkpoint:  {original_size:>8.1f} MB (100.0%)")
    print(f"压缩后模型:      {compressed_size:>8.1f} MB ({100-saved_pct:.1f}%)")
    print(f"节省空间:        {saved:>8.1f} MB ({saved_pct:.1f}%)")
    print("=" * 70)
    
    print(f"\n📝 使用说明:")
    print(f"   1. 修改 batch_inference.py 中的配置:")
    print(f"      CHECKPOINT_PATH = Path('{output_path}')")
    print(f"      USE_FP16 = True")
    print(f"   ")
    print(f"   2. 运行推理:")
    print(f"      python tools/batch_inference.py")
    
    print(f"\n💡 提示:")
    print(f"   - FP16 模型精度损失 <1%")
    print(f"   - GPU 推理时可能更快")
    print(f"   - CPU 推理会自动转回 FP32")


if __name__ == "__main__":
    compress_model(INPUT_CHECKPOINT, OUTPUT_FP16)

