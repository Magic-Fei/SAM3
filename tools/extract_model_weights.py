#!/usr/bin/env python
"""
从训练的 checkpoint 中提取纯模型权重，去除 optimizer、loss 等训练状态。
可以节省 20-30% 的存储空间。

使用方法:
    python tools/extract_model_weights.py
"""

import torch
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================
CHECKPOINT_PATH = Path(r"D:\qianpf\code\sam3-main\experiments\checkpoints\checkpoint_7.pt")
OUTPUT_PATH = Path(r"D:\qianpf\code\sam3-main\experiments\checkpoints\checkpoint_7_only.pt")


def extract_model_only(checkpoint_path: Path, output_path: Path):
    """从完整 checkpoint 提取纯模型权重"""
    print("=" * 70)
    print("模型权重提取工具")
    print("=" * 70)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        return
    
    print(f"\n📂 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 提取模型权重
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        print(f"✓ 找到模型权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
        print(f"✓ 找到模型权重 (state_dict)")
    else:
        model_state = checkpoint
        print(f"✓ 使用整个 checkpoint 作为模型权重")
    
    # 显示原始 checkpoint 包含的内容
    if isinstance(checkpoint, dict):
        print(f"\n原始 checkpoint 包含:")
        for key in checkpoint.keys():
            if key != "model":
                size = 0
                if isinstance(checkpoint[key], dict):
                    size = sum(v.numel() * v.element_size() 
                             for v in checkpoint[key].values() 
                             if hasattr(v, 'numel'))
                elif hasattr(checkpoint[key], 'numel'):
                    size = checkpoint[key].numel() * checkpoint[key].element_size()
                size_mb = size / (1024**2)
                print(f"  - {key}: {size_mb:.1f} MB")
    
    # 保存纯权重
    print(f"\n💾 保存纯模型权重到: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, output_path)
    
    # 显示大小对比
    original_size = checkpoint_path.stat().st_size / (1024**2)
    new_size = output_path.stat().st_size / (1024**2)
    saved = original_size - new_size
    saved_pct = (saved / original_size) * 100
    
    print("\n" + "=" * 70)
    print("✨ 压缩结果")
    print("=" * 70)
    print(f"原始 checkpoint:  {original_size:>8.1f} MB")
    print(f"纯模型权重:      {new_size:>8.1f} MB")
    print(f"节省空间:        {saved:>8.1f} MB ({saved_pct:.1f}%)")
    print("=" * 70)
    print(f"\n✓ 完成！可以使用 {output_path} 进行推理")


if __name__ == "__main__":
    extract_model_only(CHECKPOINT_PATH, OUTPUT_PATH)

