# 🎭 Multimodal Emotion Recognition System

Real-time emotion detection from **facial expressions** and **speech**, running simultaneously via webcam and microphone.

## 📊 Model Performance

| Model | Dataset | Metric | Score |
|---|---|---|---|
| Face (EfficientNet-B0) | FER-2013 (35k imgs) | Val Accuracy | ~68% |
| Speech (CNN) | RAVDESS (1440 clips) | Val Accuracy | ~72% |

## 🛠 Tech Stack

`PyTorch` · `EfficientNet (timm)` · `librosa` · `Gradio` · `OpenCV`

---

## ▶️ Quick Start

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载数据集

**方法一：Kaggle CLI（推荐）**
```bash
pip install kaggle
# 把 kaggle.json（API Token）放到 ~/.kaggle/
kaggle datasets download -d msambare/fer2013
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
```

**方法二：浏览器下载**
- FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
- RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

解压后放到：
```
data/
├── fer2013/
│   ├── train/   ← 包含 7 个子文件夹
│   └── test/
└── ravdess/     ← 包含所有 .wav 文件
```

### 3. 训练模型
```bash
python run_training.py
```
> GPU 约 1-2 小时，CPU 约 4-6 小时

### 4. 评估模型
```bash
python run_evaluate.py
```
> 输出分类报告 + 混淆矩阵图 + 训练曲线图

### 5. 启动 Demo
```bash
python app/app.py
```
> 浏览器自动打开 http://127.0.0.1:7860

---

## ✏️ 修改指南（常见调参位置）

### 修改训练超参数
→ 编辑 **`config.py`**
- `FACE_BATCH / FACE_EPOCHS / FACE_LR`：人脸模型批大小、轮数、学习率
- `SPEECH_BATCH / SPEECH_EPOCHS / SPEECH_LR`：语音模型对应参数
- `SPEECH_SR / SPEECH_DURATION / SPEECH_N_MELS`：音频采样率、截取时长、Mel bins

### 修改数据路径
→ 编辑 **`config.py`** 中的 `FACE_DIR` / `SPEECH_DIR`

### 修改图像数据增强
→ 编辑 **`src/face_emotion/preprocess.py`** 的 `get_transforms()`

### 修改音频数据增强
→ 编辑 **`src/speech_emotion/preprocess.py`** 的 `audio_to_melspectrogram()`（取消注释增强代码）

### 更换骨干网络（人脸）
→ 编辑 **`src/face_emotion/model.py`** 中的 `timm.create_model("efficientnet_b0", ...)` 改为 `b2` / `b4`

### 修改 Demo 界面
→ 编辑 **`app/app.py`** 中的 Gradio Blocks 部分

---

## 📁 项目结构

```
emotion-recognition/
├── data/
│   ├── fer2013/          ← FER-2013 数据集
│   └── ravdess/          ← RAVDESS 数据集
├── src/
│   ├── face_emotion/
│   │   ├── preprocess.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── speech_emotion/
│       ├── preprocess.py
│       ├── model.py
│       ├── train.py
│       └── evaluate.py
├── models/               ← 训练后自动生成
├── app/
│   └── app.py
├── config.py             ← ⭐ 所有超参数在这里
├── run_training.py
├── run_evaluate.py
└── requirements.txt
```

## 🚀 GitHub 部署

```bash
git init
git add .
git commit -m "Initial commit: Multimodal Emotion Recognition"
git remote add origin https://github.com/你的用户名/emotion-recognition.git
git branch -M main
git push -u origin main
```

> 模型权重（.pth 文件）较大，建议用 Git LFS 或上传到 Google Drive 后在 README 放链接。
