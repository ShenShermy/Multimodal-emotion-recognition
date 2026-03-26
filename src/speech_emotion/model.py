# src/speech_emotion/model.py
# CNN 语音情绪分类模型（将 Mel 频谱图当作图像处理）
# ============================================================
# ✏️  【修改入口】
#   - 加深网络：在 self.features 中继续添加 Conv2d Block
#   - 改用预训练模型：将 SpeechEmotionCNN 替换为 timm.create_model("resnet18", ...)，
#     并在第一层 Conv2d 的 in_channels 改为 1（单通道）
#   - Dropout 比例：修改 Dropout2d(0.25) / Dropout(0.5) 的参数
#   - 分类器结构：修改 self.classifier 中的 Linear 层维度
# ============================================================

import torch
import torch.nn as nn
from config import SPEECH_NUM_CLS


class SpeechEmotionCNN(nn.Module):
    """
    基于 CNN 的语音情绪分类器。
    输入：(batch, 1, 128, T) 的 Mel 频谱图
    输出：(batch, 8) 的情绪类别 logits

    架构思路：
    - 逐渐增加通道数（32→64→128→256），每层提取更高级的特征
    - MaxPool 逐步缩小空间尺寸，减少计算量
    - BatchNorm 加速训练，Dropout 防止过拟合
    - GlobalAveragePooling 替代全连接层，减少参数量
    """
    def __init__(self, num_classes=SPEECH_NUM_CLS):
        super().__init__()

        # ── 特征提取部分（卷积块）──
        self.features = nn.Sequential(
            # Block 1：捕捉低级频率特征
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 尺寸减半
            nn.Dropout2d(0.25),

            # Block 2：捕捉中级时频模式
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3：捕捉高级情绪相关特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # GlobalAveragePooling：对每个通道的特征图求均值
        # 优势：无论输入时间长度如何变化都能处理（比 Flatten 灵活）
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── 分类部分（全连接层）──
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)    # → (batch, 256, h, w)
        x = self.gap(x)         # → (batch, 256, 1, 1)
        x = self.classifier(x)  # → (batch, 8)
        return x
