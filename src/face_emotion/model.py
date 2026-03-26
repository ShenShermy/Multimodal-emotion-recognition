# src/face_emotion/model.py
# 基于 EfficientNet-B0 的人脸情绪分类模型
# ============================================================
# ✏️  【修改入口】
#   - 换用更大模型：将 "efficientnet_b0" 改为 "efficientnet_b2" / "b4"（精度↑，速度↓）
#   - 解冻策略：修改 unfreeze_from_layer 参数（blocks.0~7，数字越小解冻越多）
#   - 分类层结构：在 build_face_model 内替换 timm 的默认分类头为自定义 MLP
# ============================================================

import torch
import torch.nn as nn
import timm  # PyTorch Image Models 库，包含大量预训练模型
from config import FACE_NUM_CLS


def build_face_model(freeze_backbone=True):
    """
    构建人脸情绪分类模型。

    参数：
        freeze_backbone: True  = 只训练最后的分类层（训练快，初期稳定）
                         False = 微调全部参数（精度更高，训练更慢）
    返回：
        model: 准备好的 PyTorch 模型
    """
    # 加载 ImageNet 预训练的 EfficientNet-B0
    # pretrained=True 会自动从网上下载权重（约 20MB）
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=FACE_NUM_CLS  # 替换最后的分类层为 7 类
    )

    if freeze_backbone:
        # 冻结除最后分类层外的所有参数
        # 这样前几轮只训练分类头，避免大幅破坏预训练特征
        for name, param in model.named_parameters():
            if "classifier" not in name:  # classifier 是 EfficientNet 的分类层名
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    return model


def unfreeze_model(model, unfreeze_from_layer="blocks.5"):
    """
    解冻部分骨干网络层，用于第二阶段精细微调。
    通常在分类头收敛后（约10轮），再解冻深层进行全局优化。

    参数：
        unfreeze_from_layer: 从这一层开始解冻（EfficientNet 有 blocks.0~7）
    """
    for name, param in model.named_parameters():
        if unfreeze_from_layer in name or "classifier" in name:
            param.requires_grad = True
    print(f"Unfrozen layers from '{unfreeze_from_layer}' onwards.")
    return model
