# run_training.py
# 一键训练人脸 + 语音两个模型
# 运行方式：python run_training.py

from src.face_emotion.train import train_face_model
from src.speech_emotion.train import train_speech_model

print("Step 1/2: Training Face Emotion Model...")
train_face_model()

print("\nStep 2/2: Training Speech Emotion Model...")
train_speech_model()

print("\n✅ All training complete! Run: python app/app.py")
