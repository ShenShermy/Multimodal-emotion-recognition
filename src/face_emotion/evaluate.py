# src/face_emotion/evaluate.py
# 模型评估：混淆矩阵、分类报告、训练曲线可视化

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import *
from src.face_emotion.preprocess import get_dataloaders
from src.face_emotion.model import build_face_model


def load_best_model():
    """加载已保存的最优模型"""
    model      = build_face_model(freeze_backbone=False).to(DEVICE)
    checkpoint = torch.load(FACE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")
    return model, checkpoint.get("history", {})


def get_predictions(model, loader):
    """在数据集上跑完整推理，收集真实标签和预测结果"""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix_face.png"):
    """绘制归一化混淆矩阵（颜色深浅代表比例）"""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=FACE_CLASSES, yticklabels=FACE_CLASSES)
    plt.title("Face Emotion - Confusion Matrix (Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(history, save_path="training_curves_face.png"):
    """绘制训练/验证的 Loss 和 Accuracy 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def evaluate_face_model():
    """完整评估流程"""
    _, test_loader, _ = get_dataloaders()
    model, history    = load_best_model()
    y_true, y_pred, y_prob = get_predictions(model, test_loader)

    # 打印分类报告（精确率、召回率、F1）
    print("\n" + "=" * 50)
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=FACE_CLASSES))

    # 可视化
    plot_confusion_matrix(y_true, y_pred)
    if history:
        plot_training_curves(history)


if __name__ == "__main__":
    evaluate_face_model()
