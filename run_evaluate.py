# run_evaluate.py
# 一键评估两个模型，输出分类报告和可视化图表
# 运行方式：python run_evaluate.py（需先完成训练）

from src.face_emotion.evaluate import evaluate_face_model
from src.speech_emotion.evaluate import evaluate_speech_model

evaluate_face_model()
evaluate_speech_model()
