"""
utils.py - 工具函数
包含数据划分、可视化等功能
"""

import os
import shutil
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
from tqdm import tqdm


def split_dataset(raw_dir, train_dir, val_dir, split_ratio=0.8, seed=42, auto_clean=True):
    """
    将原始数据集划分为训练集和验证集

    Args:
        raw_dir: 原始数据目录
        train_dir: 训练集目录
        val_dir: 验证集目录
        split_ratio: 训练集比例
        seed: 随机种子
        auto_clean: 是否先清洗数据
    """

    # 如果启用自动清洗
    if auto_clean:
        print("\n先清洗数据再划分...")
        from dataset import quick_clean
        stats = quick_clean(raw_dir)

        print(f"\n清洗完成:")
        print(f"  总文件: {stats['total_files']}")
        print(f"  有效文件: {stats['valid_files']}")
        print(f"  问题文件: {stats['total_files'] - stats['valid_files']}")

    random.seed(seed)

    classes = ['anger', 'fear', 'happy', 'sad', 'surprise']

    print("=" * 60)
    print("开始划分数据集...")
    print("=" * 60)

    # 创建目标目录
    for split_dir in [train_dir, val_dir]:
        for class_name in classes:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    total_train = 0
    total_val = 0

    for class_name in classes:
        # 获取该类别所有图片
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  警告: {class_path} 不存在，跳过")
            continue

        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 打乱顺序
        random.shuffle(images)

        # 计算划分点
        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        val_images = images[split_point:]

        # 复制训练集图片（带进度条）
        print(f"\n正在处理 {class_name} 类别...")
        for img in tqdm(train_images, desc=f"  复制到训练集", ncols=80):
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        # 复制验证集图片（带进度条）
        for img in tqdm(val_images, desc=f"  复制到验证集", ncols=80):
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)

        total_train += len(train_images)
        total_val += len(val_images)

        print(f"  ✓ {class_name:10s}: {len(images):5d} 张 -> "
              f"训练集 {len(train_images):5d} 张, "
              f"验证集 {len(val_images):5d} 张")

    print("\n" + "-" * 60)
    print(f"总计: 训练集 {total_train} 张, 验证集 {total_val} 张")
    print("=" * 60)
    print("✅ 数据集划分完成！")


def plot_training_history(history, save_path='../results/training_history.png'):
    """
    绘制训练历史曲线

    Args:
        history: 包含训练历史的字典，格式：
                {'train_loss': [...], 'val_loss': [...],
                 'train_acc': [...], 'val_acc': [...]}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 绘制准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练历史曲线已保存至: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path='../results/confusion_matrix.png'):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存至: {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred, classes):
    """
    打印分类报告

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
    """
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        accuracy: 当前准确率
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"✅ 模型已保存至: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    加载模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
        device: 设备

    Returns:
        epoch: 轮次
        loss: 损失
        accuracy: 准确率
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"✅ 模型已加载: {filepath}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Loss: {checkpoint['loss']:.4f}")
    print(f"   Accuracy: {checkpoint['accuracy']:.2f}%")

    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


# 测试代码
if __name__ == "__main__":
    print("测试工具函数...")

    # 测试数据划分
    print("\n1. 测试数据集划分功能")
    raw_dir = '../data/raw'
    train_dir = '../data/train'
    val_dir = '../data/val'

    if os.path.exists(raw_dir):
        split_dataset(raw_dir, train_dir, val_dir, split_ratio=0.8)
    else:
        print(f"⚠️  {raw_dir} 不存在，跳过测试")

    # 测试绘图功能
    print("\n2. 测试绘图功能")
    dummy_history = {
        'train_loss': [2.0, 1.5, 1.2, 1.0, 0.8],
        'val_loss': [2.1, 1.6, 1.3, 1.1, 0.9],
        'train_acc': [30, 45, 55, 65, 75],
        'val_acc': [28, 43, 53, 63, 72]
    }
    plot_training_history(dummy_history)

    print("\n✅ 工具函数测试完成！")