# src/speech_emotion/preprocess.py
# RAVDESS 数据集解析、Mel 频谱图特征提取、DataLoader 构建
# ============================================================
# ✏️  【修改入口】
#   - 音频采样率 / 截取时长 / Mel bins：修改 config.py 中的 SPEECH_SR / SPEECH_DURATION / SPEECH_N_MELS
#   - 开启音频数据增强：取消注释 audio_to_melspectrogram 中 pitch_shift / time_stretch / 噪声 代码
#   - 调整 SpecAugment 强度：修改 _augment_spectrogram 中 freq_bins//6 和 time_steps//6
#   - 训练/测试集比例：修改 train_test_split 中的 test_size 参数（默认 0.2）
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import *


def parse_ravdess_filename(filepath):
    """
    解析 RAVDESS 文件名获取情绪标签。
    文件名格式：03-01-06-01-02-01-12.wav
    第3个数字（index=2）代表情绪：
        01=neutral, 02=calm, 03=happy, 04=sad,
        05=angry, 06=fearful, 07=disgust, 08=surprised
    """
    basename   = os.path.basename(filepath)
    parts      = basename.replace(".wav", "").split("-")
    emotion_id = int(parts[2]) - 1  # 转为 0-indexed (0~7)
    return emotion_id


def load_ravdess_metadata(data_dir):
    """
    扫描 RAVDESS 目录，构建包含 [路径, 标签] 的 DataFrame。
    RAVDESS 可能是平铺的，也可能按演员分文件夹，这里都能处理。
    """
    wav_files  = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
    wav_files += glob.glob(os.path.join(data_dir, "*.wav"))  # 平铺结构

    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")

    rows = []
    for f in wav_files:
        try:
            label = parse_ravdess_filename(f)
            rows.append({"path": f, "label": label})
        except Exception:
            continue  # 跳过命名不符合格式的文件

    df = pd.DataFrame(rows)
    print(f"Total samples: {len(df)}")
    print("Label distribution:\n", df["label"].value_counts().sort_index())
    return df


def audio_to_melspectrogram(audio_path, sr=SPEECH_SR, duration=SPEECH_DURATION,
                             n_mels=SPEECH_N_MELS):
    """
    将音频文件转换为 Mel 频谱图（2D 图像）。

    为什么用 Mel 频谱图而不是原始波形？
    - 人耳对频率的感知是对数尺度的（Mel 尺度模拟人耳）
    - 2D 图像可以直接用 CNN 处理，比 MFCC 保留更多信息
    - 是目前语音情绪识别的主流输入格式

    返回：
        mel_db: shape (1, n_mels, time_frames) 的 numpy 数组
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)

    # 固定长度：不足则补零，超出则截断
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    # ── 数据增强（仅训练时使用，取消注释即可启用）──
    # 音调偏移：模拟不同音调的人声
    # y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-3, 4))
    # 时间拉伸：模拟语速快慢
    # rate = np.random.uniform(0.8, 1.2)
    # y = librosa.effects.time_stretch(y, rate=rate)
    # 加高斯噪声：提升鲁棒性
    # y += np.random.randn(len(y)) * 0.005

    # 计算 Mel 频谱图并转为对数分贝尺度
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            fmax=8000,    # 最高关注频率
                                            hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # 转分贝

    # 归一化到 [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_db[np.newaxis, :, :]  # 增加 channel 维度 → (1, 128, T)


class SpeechDataset(Dataset):
    """
    语音情绪数据集。
    每个样本：音频路径 → Mel 频谱图 → 情绪标签
    """
    def __init__(self, df, augment=False):
        self.paths   = df["path"].values
        self.labels  = df["label"].values
        self.augment = augment  # 是否启用音频增强（训练时开启）

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = audio_to_melspectrogram(self.paths[idx])

        # 在线增强（每次读取时随机变换，增加多样性）
        if self.augment:
            mel = self._augment_spectrogram(mel)

        return torch.FloatTensor(mel), int(self.labels[idx])

    def _augment_spectrogram(self, mel):
        """
        频谱图层面的增强（比音频级增强更快）：
        - SpecAugment：随机遮住时间段和频率段（Google 提出的强增强方法）
        """
        _, freq_bins, time_steps = mel.shape

        # 随机遮住一段频率（Frequency Masking）
        f  = np.random.randint(0, freq_bins // 6)
        f0 = np.random.randint(0, freq_bins - f)
        mel[:, f0:f0 + f, :] = 0

        # 随机遮住一段时间（Time Masking）
        t  = np.random.randint(0, time_steps // 6)
        t0 = np.random.randint(0, time_steps - t)
        mel[:, :, t0:t0 + t] = 0

        return mel


def get_speech_dataloaders():
    """构建训练/测试 DataLoader"""
    df = load_ravdess_metadata(SPEECH_DIR)

    # 按类别分层采样划分训练/测试集
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    train_set = SpeechDataset(train_df, augment=True)
    test_set  = SpeechDataset(test_df,  augment=False)

    train_loader = DataLoader(train_set, batch_size=SPEECH_BATCH, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=SPEECH_BATCH, shuffle=False, num_workers=4)

    return train_loader, test_loader
