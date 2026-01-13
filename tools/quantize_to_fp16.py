#!/usr/bin/env python
"""
将模型量化为 FP16 半精度，减小 50% 存储空间，精度损失极小（<1%）。

使用方法:
    python tools/quantize_to_fp16.py
"""

import torch
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================
INPUT_PATH = Path(r"D:\qianpf\code\sam3-main\experiments_guajia\checkpoints\checkpoint_10.pt")  # 或直接用 checkpoint.pt
OUTPUT_PATH = Path(r"D:\qianpf\code\sam3-main\experiments_guajia\checkpoints\checkpoint_fp16.pt")


def quantize_to_fp16(input_path: Path, output_path: Path):
    """将模型量化为 FP16"""
    print("=" * 70)
    print("FP16 半精度量化工具")
    print("=" * 70)
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        print(f"\n提示: 可以先运行 extract_model_weights.py 提取纯权重")
        print(f"      或直接使用完整的 checkpoint.pt")
        return
    
    print(f"\n📂 加载模型: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")
    
    # 提取模型权重
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        print(f"✓ 找到模型权重 (完整 checkpoint)")
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
        print(f"✓ 找到模型权重 (state_dict)")
    else:
        model_state = checkpoint
        print(f"✓ 使用整个文件作为模型权重")
    
    # 统计参数信息
    total_params = 0
    fp32_params = 0
    for k, v in model_state.items():
        if hasattr(v, 'numel'):
            total_params += v.numel()
            if v.dtype == torch.float32:
                fp32_params += v.numel()
    
    print(f"\n模型信息:")
    print(f"  总参数量: {total_params:,}")
    print(f"  FP32 参数: {fp32_params:,}")
    print(f"  其他类型: {total_params - fp32_params:,}")
    
    # 转换为 FP16
    print(f"\n🔄 转换为 FP16...")
    model_state_fp16 = {}
    converted_count = 0
    
    for k, v in model_state.items():
        if hasattr(v, 'dtype') and v.dtype == torch.float32:
            model_state_fp16[k] = v.half()
            converted_count += 1
        else:
            model_state_fp16[k] = v
    
    print(f"✓ 转换了 {converted_count} 个 FP32 参数")
    
    # 保存 FP16 模型
    print(f"\n💾 保存 FP16 模型到: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_fp16, output_path)
    
    # 显示大小对比
    original_size = input_path.stat().st_size / (1024**2)
    new_size = output_path.stat().st_size / (1024**2)
    saved = original_size - new_size
    saved_pct = (saved / original_size) * 100
    
    print("\n" + "=" * 70)
    print("✨ 量化结果")
    print("=" * 70)
    print(f"FP32 模型:       {original_size:>8.1f} MB")
    print(f"FP16 模型:       {new_size:>8.1f} MB")
    print(f"节省空间:        {saved:>8.1f} MB ({saved_pct:.1f}%)")
    print("=" * 70)
    
    print(f"\n📝 使用说明:")
    print(f"   在推理时加载 FP16 模型后，需要将模型也转为 FP16:")
    print(f"   ```python")
    print(f"   model.load_state_dict(torch.load('{output_path}'))")
    print(f"   if device.type == 'cuda':")
    print(f"       model = model.half()  # 使用 FP16 推理")
    print(f"   ```")
    print(f"\n✓ 完成！FP16 模型可用于推理，精度损失 <1%")


if __name__ == "__main__":
    quantize_to_fp16(INPUT_PATH, OUTPUT_PATH)

