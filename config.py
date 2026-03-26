# config.py
# 所有超参数和路径集中管理，修改一处全局生效
# ============================================================
# ✏️  【修改入口总览】
#   - 数据路径：FACE_DIR / SPEECH_DIR
#   - 人脸训练参数：FACE_BATCH / FACE_EPOCHS / FACE_LR / FACE_IMG_SIZE
#   - 语音训练参数：SPEECH_BATCH / SPEECH_EPOCHS / SPEECH_LR
#   - 语音特征参数：SPEECH_SR / SPEECH_DURATION / SPEECH_N_MELS
# ============================================================

import os
import torch

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR = "data"
FACE_DIR = os.path.join(DATA_DIR, "fer2013")    # FER-2013 解压后的路径
SPEECH_DIR = os.path.join(DATA_DIR, "ravdess")  # RAVDESS 解压后的路径
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 人脸情绪配置 ───────────────────────────────────────────
FACE_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
FACE_NUM_CLS = len(FACE_CLASSES)   # 7
FACE_IMG_SIZE = 224                # EfficientNet 输入尺寸
FACE_BATCH = 64                    # 批大小（显存不足可改为 32）
FACE_EPOCHS = 30                   # 训练轮数
FACE_LR = 1e-4                     # 初始学习率
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_model.pth")

# ── 语音情绪配置 ───────────────────────────────────────────
SPEECH_CLASSES = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
SPEECH_NUM_CLS = len(SPEECH_CLASSES)  # 8
SPEECH_SR = 22050                  # 采样率（RAVDESS 原始为 22050Hz）
SPEECH_DURATION = 3                # 截取/补齐到 3 秒
SPEECH_N_MELS = 128               # Mel 频谱图的频率 bins 数量
SPEECH_BATCH = 32                  # 批大小
SPEECH_EPOCHS = 50                 # 训练轮数
SPEECH_LR = 1e-3                   # 初始学习率
SPEECH_MODEL_PATH = os.path.join(MODEL_DIR, "speech_model.pth")

# ── 训练设备 ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
