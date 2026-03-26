# src/face_emotion/train.py
# 人脸情绪模型训练主脚本
# 包含：两阶段训练、学习率调度、早停、模型保存
# ============================================================
# ✏️  【修改入口】
#   - 早停耐心值：修改 PATIENCE（默认 7，越大训练越久）
#   - 第二阶段解冻时机：修改 epoch == 10 的判断条件
#   - 解冻层范围：修改 unfreeze_model(model, unfreeze_from_layer="blocks.5")
#   - 梯度裁剪阈值：修改 clip_grad_norm_ 的 max_norm 参数（默认 1.0）
#   - label_smoothing：修改 CrossEntropyLoss 的 label_smoothing 参数
# ============================================================

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from config import *
from src.face_emotion.preprocess import get_dataloaders
from src.face_emotion.model import build_face_model, unfreeze_model


def compute_class_weights(class_counts):
    """
    根据类别样本数计算权重，解决 FER-2013 类别不均衡问题。
    Disgust 类样本最少，权重最大，损失函数会更关注这类样本。
    """
    counts  = torch.FloatTensor(class_counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)  # 归一化
    return weights


def train_one_epoch(model, loader, optimizer, criterion, device):
    """单轮训练，返回平均损失和准确率"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播 + 梯度裁剪（防止梯度爆炸）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """验证集评估，返回损失和准确率"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


def train_face_model():
    """主训练函数，两阶段策略"""
    print("=" * 50)
    print("Training Face Emotion Model")
    print("=" * 50)

    # ── 数据加载 ──
    train_loader, test_loader, class_counts = get_dataloaders()

    # ── 模型初始化（第一阶段：冻结骨干）──
    model = build_face_model(freeze_backbone=True).to(DEVICE)

    # ── 损失函数：带类别权重的 CrossEntropy ──
    weights   = compute_class_weights(class_counts).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1  # Label Smoothing：让模型不要太自信，防过拟合
    )

    # ── 优化器和学习率调度 ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FACE_LR,
        weight_decay=1e-4  # L2 正则化
    )
    # CosineAnnealing：学习率从初始值余弦衰减到 0，比 StepLR 更平滑
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FACE_EPOCHS
    )

    # ── 训练循环 ──
    best_val_acc    = 0
    patience_counter = 0
    PATIENCE         = 7  # 早停耐心值：连续7轮无提升则停止
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, FACE_EPOCHS + 1):

        # ── 第二阶段解冻（第10轮开始）──
        if epoch == 10:
            model = unfreeze_model(model, unfreeze_from_layer="blocks.5")
            # 重置优化器，对解冻层用更小的学习率
            optimizer = torch.optim.AdamW([
                {"params": model.classifier.parameters(), "lr": FACE_LR},
                {"params": model.blocks[5:].parameters(), "lr": FACE_LR * 0.1},
                {"params": model.blocks[:5].parameters(), "lr": FACE_LR * 0.01},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=FACE_EPOCHS - 10
            )

        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        # 记录历史
        for k, v in zip(history.keys(), [tr_loss, val_loss, tr_acc, val_acc]):
            history[k].append(v)

        print(f"Epoch [{epoch:02d}/{FACE_EPOCHS}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ── 保存最优模型 ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "history":          history
            }, FACE_MODEL_PATH)
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    return model, history


if __name__ == "__main__":
    train_face_model()
