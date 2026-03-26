# src/face_emotion/preprocess.py
# FER-2013 数据加载、数据增强、DataLoader 构建
# ============================================================
# ✏️  【修改入口】
#   - 数据增强强度：修改 RandomRotation / ColorJitter / RandomAffine 参数
#   - 输入图像尺寸：修改 config.py 中的 FACE_IMG_SIZE
#   - 批大小：修改 config.py 中的 FACE_BATCH
#   - num_workers：Windows 用户需改为 0
# ============================================================

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import FACE_DIR, FACE_IMG_SIZE, FACE_BATCH


def get_transforms(mode="train"):
    """
    返回训练/验证的图像变换流水线。
    FER-2013 原始图像是 48x48 灰度图，这里：
    1. 转成 3 通道（EfficientNet 需要 RGB 输入）
    2. 放大到 224x224
    3. 训练时加入随机翻转、旋转等增强防止过拟合
    """
    # ImageNet 均值和标准差（迁移学习的标配）
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),         # 灰度→3通道
            transforms.Resize((FACE_IMG_SIZE, FACE_IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),              # 随机水平翻转
            transforms.RandomRotation(degrees=15),               # 随机旋转±15°
            transforms.ColorJitter(brightness=0.3,               # 亮度/对比度扰动
                                   contrast=0.3),
            transforms.RandomAffine(degrees=0,                   # 随机平移
                                    translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((FACE_IMG_SIZE, FACE_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_dataloaders():
    """
    构建训练集和测试集的 DataLoader。
    FER-2013 目录结构：
        fer2013/
            train/
                angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
            test/
                angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
    """
    train_dataset = datasets.ImageFolder(
        root=os.path.join(FACE_DIR, "train"),
        transform=get_transforms("train")
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(FACE_DIR, "test"),
        transform=get_transforms("test")
    )

    # 打印类别映射，确认顺序正确
    print("Class mapping:", train_dataset.class_to_idx)
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # 计算各类别样本数，用于后续加权损失函数
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    print("Class counts:", dict(zip(train_dataset.classes, class_counts)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=FACE_BATCH,
        shuffle=True,
        num_workers=4,   # 多进程加载，Windows 用户改为 0
        pin_memory=True  # GPU 训练时加速数据传输
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=FACE_BATCH,
        shuffle=False,
        num_workers=4
    )

    return train_loader, test_loader, class_counts
