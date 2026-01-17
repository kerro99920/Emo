"""
train.py - 模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import os

from dataset import EmotionDataset
from model import get_model, count_parameters
from utils import plot_training_history, save_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch

    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    pbar = tqdm(dataloader, desc='训练中', ncols=100)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    验证模型

    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='验证中', ncols=100)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(
    model_type='resnet18',
    num_epochs=20,
    batch_size=64,
    learning_rate=0.001,
    device='auto'
):
    """
    完整的训练流程

    Args:
        model_type: 模型类型 ('cnn', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet', 'efficientnet')
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 设备 ('auto', 'cuda', 'cpu')
    """

    # 设置设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print("=" * 70)
    print(f"开始训练 - 表情识别模型")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"模型类型: {model_type}")
    print(f"训练轮数: {num_epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print("=" * 70)

    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = EmotionDataset('../data/train', transform=train_transform)
    val_dataset = EmotionDataset('../data/val', transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 创建模型
    print("\n创建模型...")
    model = get_model(model_type, num_classes=5, pretrained=True)
    model = model.to(device)
    count_parameters(model)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    # 开始训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 70)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印结果
        print(f"\n结果:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f'../models/best_model_{model_type}.pth'
            save_checkpoint(
                model, optimizer, epoch+1, val_loss, val_acc, save_path
            )
            print(f"  ✓ 新的最佳模型！验证准确率: {val_acc:.2f}%")

    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"总训练时间: {total_time/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

    # 保存最终模型
    final_path = f'../models/final_model_{model_type}.pth'
    save_checkpoint(
        model, optimizer, num_epochs,
        history['val_loss'][-1], history['val_acc'][-1],
        final_path
    )

    # 绘制训练曲线
    plot_training_history(history, f'../results/training_history_{model_type}.png')

    return model, history


if __name__ == "__main__":
    # 训练配置
    config = {
        'model_type': 'efficientnet',  # 可选: cnn, resnet18, resnet34, resnet50, vgg16, mobilenet, efficientnet
        'num_epochs': 20,          # 训练轮数
        'batch_size': 64,          # 批次大小
        'learning_rate': 0.001,    # 学习率
        'device': 'auto'           # 'auto', 'cuda', 'cpu'
    }

    print("\n" + "=" * 70)
    print("可用模型:")
    print("  - cnn: 自定义CNN (从零训练)")
    print("  - resnet18: ResNet-18 (推荐)")
    print("  - resnet34: ResNet-34 (更深)")
    print("  - resnet50: ResNet-50 (最强)")
    print("  - vgg16: VGG-16 (经典)")
    print("  - mobilenet: MobileNetV2 (最快)")
    print("  - efficientnet: EfficientNet-B0 (最先进)")
    print("=" * 70)

    print("\n训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # 开始训练
    model, history = train_model(**config)

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print("模型已保存在 models/ 文件夹")
    print("训练曲线已保存在 results/ 文件夹")
    print("\n下一步:")
    print("  - 运行 evaluate.py 评估单个模型")
    print("  - 运行 train_multiple.py 训练所有模型并对比")