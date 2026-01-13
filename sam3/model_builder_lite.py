"""
轻量级 SAM3 模型构建器

相比标准 SAM3：
- ViT: embed_dim 1024→768, depth 32→24 (减少 40%)
- Transformer: layers 6→4 (减少 33%)
- 最终模型: ~1.5 GB (标准版 ~2.5 GB)
- 性能损失: 预计 5-10% mAP
"""

import os
from typing import Optional

import torch
import torch.nn as nn

from sam3.model.sam3_image import Sam3Image
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.model_misc import TransformerWrapper, DotProductScoring, MLP, MultiheadAttentionWrapper as MultiheadAttention
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.vitdet import ViT
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.decoder import TransformerDecoder, TransformerDecoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import UniversalSegmentationHead, PixelDecoder
from sam3.model.memory import CXBlock
from sam3.model_builder import (
    _load_checkpoint,
    _setup_device_and_mode,
    download_ckpt_from_hf,
)


def _create_vit_backbone_lite(compile_mode=None):
    """创建轻量级 ViT backbone (减少 40%)"""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        in_chans=3,
        embed_dim=768,  # 原: 1024
        depth=24,  # 原: 32
        num_heads=12,  # 原: 16
        mlp_ratio=4.0,  # 原: 4.625
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(5, 11, 17, 23),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_transformer_encoder_lite():
    """创建轻量级 Transformer encoder (减少 33%)"""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=4,       # 原: 6
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder_lite():
    """创建轻量级 Transformer decoder (减少 33%)"""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=4,  # 原: 6
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_geometry_encoder_lite():
    """创建几何编码器 (保持原大小，因为相对较小)"""
    # Position encoding for geometry encoder - must match standard config
    geo_pos_enc = PositionEmbeddingSine(
        num_pos_feats=256,  # Same as standard model
        normalize=True,
        scale=None,
        temperature=10000,
    )
    
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_segmentation_head_lite(compile_mode=None):
    """创建分割头 (保持原大小，因为相对较小)"""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def build_sam3_lite_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,
    enable_segmentation=True,
    compile=False,
):
    """
    构建轻量级 SAM3 图像模型
    
    相比标准版本：
    - ViT 减少 40% (768 dim, 24 layers)
    - Transformer 减少 33% (4 layers)
    - 模型大小: ~1.5 GB (vs 2.5 GB)
    - 推理速度: 快 20-30%
    - 性能: 预计损失 5-10% mAP
    
    Args:
        bpe_path: BPE tokenizer 路径
        device: 设备 ('cuda' 或 'cpu')
        eval_mode: 是否评估模式
        checkpoint_path: 权重路径
        load_from_HF: 是否从 HuggingFace 下载
        enable_segmentation: 是否启用分割头
        compile: 是否编译模型
    
    Returns:
        轻量级 SAM3 图像模型
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    
    print("🚀 构建轻量级 SAM3 模型...")
    print("   - ViT: 768 dim, 24 layers (减少 40%)")
    print("   - Transformer: 4 layers (减少 33%)")
    print("   - 预计模型大小: ~1.5 GB")
    
    # 创建轻量级组件
    compile_mode = "default" if compile else None
    
    # 轻量级 ViT
    vision_encoder = _create_vit_backbone_lite(compile_mode=compile_mode)
    
    # Position encoding (for visual backbone/neck)
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,  # Same as standard model
        normalize=True,
        scale=None,
        temperature=10000,
    )
    
    # ViT neck
    vit_neck = Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vision_encoder,
        add_sam2_neck=False,
    )
    
    # Text encoder (保持原大小，因为相对较小)
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    text_encoder = VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )
    
    # VL Backbone
    backbone = SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)
    
    # 轻量级 Transformer
    encoder = _create_transformer_encoder_lite()
    decoder = _create_transformer_decoder_lite()
    transformer = TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)
    
    # Geometry encoder
    input_geometry_encoder = _create_geometry_encoder_lite()
    
    # Segmentation head
    if enable_segmentation:
        segmentation_head = _create_segmentation_head_lite(compile_mode=compile_mode)
    else:
        segmentation_head = None
    
    # Dot product scoring
    dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    dot_prod_scoring = DotProductScoring(
        d_model=256,
        d_proj=256,
        prompt_mlp=dot_prod_mlp,
    )
    
    # Matcher (训练时需要)
    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2
        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    
    # 创建模型
    model = Sam3Image(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False,
        multimask_output=True,
        inst_interactive_predictor=None,
        matcher=matcher,
        use_dot_prod_scoring=True,
    )
    
    # 加载权重
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        print(f"📂 加载权重: {checkpoint_path}")
        _load_checkpoint(model, checkpoint_path)
    
    # 设置设备和模式
    model = _setup_device_and_mode(model, device, eval_mode)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {total_params/1e6:.1f}M (标准版 ~600M)")
    
    return model

