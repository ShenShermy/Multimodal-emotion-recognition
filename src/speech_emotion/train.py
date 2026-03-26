# src/speech_emotion/train.py
# 语音情绪模型训练主脚本
# ============================================================
# ✏️  【修改入口】
#   - 早停耐心值：修改 PATIENCE（默认 10）
#   - OneCycleLR 上升比例：修改 pct_start（默认 0.3，即前30%上升）
#   - 梯度裁剪：修改 clip_grad_norm_ 的 max_norm 参数
#   - label_smoothing：修改 CrossEntropyLoss 的参数
# ============================================================

import torch
import torch.nn as nn
from tqdm import tqdm
from config import *
from src.speech_emotion.preprocess import get_speech_dataloaders
from src.speech_emotion.model import SpeechEmotionCNN


def train_speech_model():
    """语音情绪模型训练流程"""
    print("=" * 50)
    print("Training Speech Emotion Model")
    print("=" * 50)

    train_loader, test_loader = get_speech_dataloaders()

    model = SpeechEmotionCNN().to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=SPEECH_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=SPEECH_LR,
        steps_per_epoch=len(train_loader),
        epochs=SPEECH_EPOCHS,
        pct_start=0.3  # 前30%时间学习率上升，后70%下降
    )

    best_val_acc     = 0
    patience_counter = 0
    PATIENCE         = 10
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, SPEECH_EPOCHS + 1):

        # ── 训练 ──
        model.train()
        tr_loss, correct, total = 0, 0, 0

        for mels, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            mels, labels = mels.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(mels)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR 每步都要 step

            tr_loss  += loss.item()
            correct  += (outputs.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        tr_acc = correct / total

        # ── 验证 ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for mels, labels in test_loader:
                mels, labels = mels.to(DEVICE), labels.to(DEVICE)
                outputs   = model(mels)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        for k, v in zip(history.keys(),
                        [tr_loss / len(train_loader), val_loss / len(test_loader),
                         tr_acc, val_acc]):
            history[k].append(v)

        print(f"Ep [{epoch:02d}/{SPEECH_EPOCHS}] "
              f"Train: {tr_acc:.3f} | Val: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "val_acc": val_acc, "history": history},
                       SPEECH_MODEL_PATH)
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
    train_speech_model()
