# app/app.py
# 双模态情绪识别 Gradio 应用
# 功能：实时摄像头人脸情绪 + 实时麦克风语音情绪，同时运行
#
# 运行方式：python app/app.py
# 本地访问：http://127.0.0.1:7860
# 公开链接：运行后会生成 gradio.live 临时链接（share=True）
# ============================================================
# ✏️  【修改入口】
#   - 推理帧率：修改 stream_every 参数（默认 0.1 秒/帧）
#   - 显示 Top-N 情绪：修改 gr.Label(num_top_classes=3) 的数值
#   - 界面主题：修改 gr.themes.Soft() 为 Default() / Monochrome() / Glass()
#   - 端口 / 公开链接：修改 demo.launch() 的 server_port / share 参数
# ============================================================

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 加入项目根路径

import cv2
import torch
import librosa
import numpy as np
from PIL import Image
import gradio as gr
from torchvision import transforms

from config import *
from src.face_emotion.model import build_face_model
from src.speech_emotion.model import SpeechEmotionCNN

# ═══════════════════════════════════════════════
# 1. 模型加载（启动时只加载一次）
# ═══════════════════════════════════════════════
print("Loading models...")

# ── 人脸模型 ──
face_model = build_face_model(freeze_backbone=False).to(DEVICE)
ckpt = torch.load(FACE_MODEL_PATH, map_location=DEVICE)
face_model.load_state_dict(ckpt["model_state_dict"])
face_model.eval()

# ── 语音模型 ──
speech_model = SpeechEmotionCNN().to(DEVICE)
ckpt2 = torch.load(SPEECH_MODEL_PATH, map_location=DEVICE)
speech_model.load_state_dict(ckpt2["model_state_dict"])
speech_model.eval()

# ── 人脸检测器（OpenCV Haar Cascade，轻量快速）──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("✅ All models loaded!")

# ═══════════════════════════════════════════════
# 2. 图像预处理（与训练时一致）
# ═══════════════════════════════════════════════
face_transform = transforms.Compose([
    transforms.Resize((FACE_IMG_SIZE, FACE_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── 情绪对应的 Emoji ───
FACE_EMOJI   = ['😠', '🤢', '😨', '😄', '😐', '😢', '😲']
SPEECH_EMOJI = ['😐', '😌', '😄', '😢', '😠', '😨', '🤢', '😲']

# ═══════════════════════════════════════════════
# 3. 推理函数
# ═══════════════════════════════════════════════
def predict_face_emotion(frame_np):
    """
    输入：摄像头捕获的 numpy 图像帧（RGB，H×W×3）
    输出：人脸情绪标签字典（用于 gr.Label 展示）

    流程：
    1. 用 OpenCV 检测人脸位置
    2. 裁剪人脸区域，转成 PIL Image
    3. 经过预处理送入模型
    4. Softmax → 返回各情绪概率
    """
    if frame_np is None:
        return {}

    # 转为 BGR（OpenCV 格式）用于人脸检测
    img_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # 每次缩放比例
        minNeighbors=5,    # 候选框最少邻居数（越大越严格）
        minSize=(30, 30)
    )

    if len(faces) == 0:
        # 没检测到人脸：直接用全图预测
        face_img = Image.fromarray(frame_np).convert("RGB")
    else:
        # 取面积最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop  = frame_np[y:y + h, x:x + w]
        face_img   = Image.fromarray(face_crop).convert("RGB")

    # 预处理 + 推理
    tensor = face_transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(face_model(tensor), dim=1)[0].cpu().numpy()

    # 构建结果字典：{Emoji+类名: 概率}，Gradio Label 组件会自动排序显示
    result = {
        f"{FACE_EMOJI[i]} {FACE_CLASSES[i]}": float(probs[i])
        for i in range(FACE_NUM_CLS)
    }
    return result


def predict_speech_emotion(audio):
    """
    输入：Gradio 传入的音频（tuple: (sample_rate, numpy_array) 或 filepath）
    输出：语音情绪标签字典

    流程：
    1. 解析音频数据（Gradio 可能传 filepath 或 tuple）
    2. 转为 Mel 频谱图
    3. 送入 CNN 模型推理
    """
    if audio is None:
        return {}

    # Gradio Audio 组件返回 (sample_rate, numpy_array) 的 tuple
    if isinstance(audio, tuple):
        sr, y = audio
        y = y.astype(np.float32)
        # 如果是立体声，取单声道
        if y.ndim > 1:
            y = y.mean(axis=1)
        # 重采样到目标采样率
        if sr != SPEECH_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SPEECH_SR)
    else:
        # filepath 格式
        y, sr = librosa.load(audio, sr=SPEECH_SR, duration=SPEECH_DURATION)

    # 固定长度
    target_len = SPEECH_SR * SPEECH_DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Mel 频谱图
    mel    = librosa.feature.melspectrogram(y=y, sr=SPEECH_SR, n_mels=SPEECH_N_MELS,
                                            fmax=8000, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # 推理
    tensor = torch.FloatTensor(mel_db[np.newaxis, np.newaxis, :, :]).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(speech_model(tensor), dim=1)[0].cpu().numpy()

    return {
        f"{SPEECH_EMOJI[i]} {SPEECH_CLASSES[i]}": float(probs[i])
        for i in range(SPEECH_NUM_CLS)
    }

# ═══════════════════════════════════════════════
# 4. Gradio UI 构建
# ═══════════════════════════════════════════════
with gr.Blocks(
    title="🎭 Multimodal Emotion Detector",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # 🎭 Multimodal Emotion Detector
    **双模态情绪识别系统** | 同时通过摄像头和麦克风实时检测你的情绪

    > 💡 **使用方法**：
    > - 左侧：点击摄像头开始实时识别人脸情绪（自动检测人脸区域）
    > - 右侧：录音后自动分析语音情绪
    """)

    with gr.Row():
        # ── 左栏：人脸情绪 ──
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Face Emotion（人脸情绪）")
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,   # 开启流式传输，实时帧推理
                label="Camera Input",
                height=300
            )
            face_output = gr.Label(
                num_top_classes=3,  # 只显示前3个最高概率
                label="Face Emotion Prediction"
            )

        # ── 右栏：语音情绪 ──
        with gr.Column(scale=1):
            gr.Markdown("### 🎙 Speech Emotion（语音情绪）")
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",          # 返回 (sr, array) tuple
                label="Microphone Input",
                streaming=False        # 录音结束后推理（比流式更稳定）
            )
            speech_output = gr.Label(
                num_top_classes=3,
                label="Speech Emotion Prediction"
            )

    # ── 绑定事件 ──
    # 摄像头：每帧触发人脸推理
    webcam_input.stream(
        fn=predict_face_emotion,
        inputs=webcam_input,
        outputs=face_output,
        stream_every=0.1  # 每 0.1 秒推理一次（约 10 FPS）
    )

    # 麦克风：录音完成后触发语音推理
    mic_input.change(
        fn=predict_speech_emotion,
        inputs=mic_input,
        outputs=speech_output
    )

    gr.Markdown("""
    ---
    📊 **模型信息：**
    人脸情绪：EfficientNet-B0 fine-tuned on FER-2013（7类）|
    语音情绪：CNN on Mel-Spectrogram trained on RAVDESS（8类）
    """)

# ═══════════════════════════════════════════════
# 5. 启动
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,
        share=True,             # True = 生成公开 gradio.live 临时链接（72小时有效）
        inbrowser=True          # 自动打开浏览器
    )
