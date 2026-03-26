# src/speech_emotion/evaluate.py
# 语音情绪模型评估（与人脸模块结构一致）

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import *
from src.speech_emotion.preprocess import get_speech_dataloaders
from src.speech_emotion.model import SpeechEmotionCNN


def evaluate_speech_model():
    _, test_loader = get_speech_dataloaders()

    model      = SpeechEmotionCNN().to(DEVICE)
    checkpoint = torch.load(SPEECH_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for mels, labels in test_loader:
            mels  = mels.to(DEVICE)
            preds = model(mels).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=SPEECH_CLASSES))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=SPEECH_CLASSES, yticklabels=SPEECH_CLASSES)
    plt.title("Speech Emotion - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_speech.png", dpi=150)
    plt.show()

    # 训练曲线
    history = checkpoint.get("history", {})
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        keys_pairs = [("train_loss", "val_loss"), ("train_acc", "val_acc")]
        titles     = ["Loss", "Accuracy"]

        for ax, (k1, k2), title in zip(axes, keys_pairs, titles):
            ax.plot(history[k1], label=f"Train {title}")
            ax.plot(history[k2], label=f"Val {title}")
            ax.set_title(f"{title} Curve")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig("training_curves_speech.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    evaluate_speech_model()
