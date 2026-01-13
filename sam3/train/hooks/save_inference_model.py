"""
训练钩子：自动保存推理模型（FP16压缩版本）

在每次保存checkpoint时，自动提取模型权重并转换为FP16，
生成一个适合推理的轻量级模型文件。
"""

import os
import torch
from pathlib import Path


def save_inference_model_hook(trainer, epoch):
    """
    在保存checkpoint后调用，自动生成推理模型
    
    Args:
        trainer: 训练器实例
        epoch: 当前epoch
    """
    # 获取最新保存的checkpoint路径
    checkpoint_dir = Path(trainer.checkpoint_conf.save_dir)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    
    if not checkpoint_path.exists():
        return
    
    # 创建inference模型保存目录
    inference_dir = checkpoint_dir / "inference_models"
    inference_dir.mkdir(exist_ok=True)
    
    # 加载完整checkpoint
    print(f"\n🔄 正在生成推理模型 (epoch {epoch})...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 提取模型权重
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        print("❌ 无法找到模型权重，跳过推理模型生成")
        return
    
    # 转换为FP16
    model_state_fp16 = {
        k: v.half() if v.dtype == torch.float32 else v
        for k, v in model_state.items()
    }
    
    # 保存推理模型
    inference_path = inference_dir / f"model_epoch_{epoch}_fp16.pth"
    torch.save(model_state_fp16, inference_path)
    
    # 计算大小
    full_size = os.path.getsize(checkpoint_path) / (1024**3)
    inference_size = os.path.getsize(inference_path) / (1024**3)
    
    print(f"✅ 推理模型已保存:")
    print(f"   完整checkpoint: {full_size:.2f} GB")
    print(f"   推理模型(FP16): {inference_size:.2f} GB")
    print(f"   保存路径: {inference_path}")
    print(f"   压缩率: {(1 - inference_size/full_size)*100:.1f}%\n")
    
    # 可选：保存一个"最新"链接
    latest_path = inference_dir / "model_latest_fp16.pth"
    if latest_path.exists():
        latest_path.unlink()
    
    # 在Windows上创建副本，在Linux上创建符号链接
    try:
        os.symlink(inference_path, latest_path)
    except (OSError, NotImplementedError):
        # Windows或不支持符号链接的系统
        import shutil
        shutil.copy2(inference_path, latest_path)
    
    print(f"   最新模型: {latest_path}")


class InferenceModelSaver:
    """
    训练钩子类：在每次保存checkpoint时自动生成推理模型
    
    用法（在trainer中）：
        from sam3.train.hooks.save_inference_model import InferenceModelSaver
        trainer.register_hook(InferenceModelSaver())
    """
    
    def __init__(self, save_fp16=True, save_fp32=False):
        """
        Args:
            save_fp16: 是否保存FP16版本（推荐）
            save_fp32: 是否保存FP32版本（可选）
        """
        self.save_fp16 = save_fp16
        self.save_fp32 = save_fp32
    
    def after_save_checkpoint(self, trainer, epoch):
        """在保存checkpoint后自动调用"""
        save_inference_model_hook(trainer, epoch)

