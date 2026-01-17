"""
model.py - 表情识别CNN模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    """
    自定义卷积神经网络用于表情识别

    架构说明:
        - 3个卷积块，每块包含2个卷积层 + BatchNorm + ReLU + MaxPool + Dropout
        - 逐层增加特征图数量: 64 -> 128 -> 256
        - 使用MaxPool逐步降低空间维度: 48x48 -> 24x24 -> 12x12 -> 6x6
        - 最后用全连接层进行分类

    优点:
        - 结构简单，易于理解
        - 完全自定义，适合学习CNN原理
        - 参数量适中（约590万）

    缺点:
        - 需要从零训练，训练时间较长
        - 准确率相对预训练模型较低（约60-70%）

    推荐场景:
        - 学习深度学习和CNN原理
        - 理解卷积、池化、全连接层的作用

    输入: (batch_size, 3, 48, 48) RGB图像
    输出: (batch_size, 5) 5个类别的logits
    """
    def __init__(self, num_classes=5):
        super(EmotionCNN, self).__init__()

        # 第一个卷积块: 提取低级特征（边缘、纹理等）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入3通道(RGB) -> 输出64通道
            nn.BatchNorm2d(64),                          # 批归一化，加速训练
            nn.ReLU(inplace=True),                       # 激活函数，增加非线性
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 再次卷积，加深特征提取
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 池化，降维 48x48 -> 24x24
            nn.Dropout(0.25)                             # 随机丢弃25%神经元，防止过拟合
        )

        # 第二个卷积块: 提取中级特征（形状、局部模式等）
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64通道 -> 128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # 24x24 -> 12x12
            nn.Dropout(0.25)
        )

        # 第三个卷积块: 提取高级特征（复杂模式、语义信息等）
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 128通道 -> 256通道
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 12x12 -> 6x6
            nn.Dropout(0.25)
        )

        # 全连接层: 将特征映射到类别
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),  # 展平后: 9216维 -> 512维
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),              # 更高的dropout率，进一步防止过拟合
            nn.Linear(512, num_classes)   # 512维 -> 5个类别
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 (batch_size, 3, 48, 48)
        Returns:
            输出logits (batch_size, 5)
        """
        x = self.conv1(x)                # -> (batch_size, 64, 24, 24)
        x = self.conv2(x)                # -> (batch_size, 128, 12, 12)
        x = self.conv3(x)                # -> (batch_size, 256, 6, 6)
        x = x.view(x.size(0), -1)        # 展平 -> (batch_size, 9216)
        x = self.fc(x)                   # -> (batch_size, 5)
        return x


class EmotionResNet(nn.Module):
    """
    基于ResNet的迁移学习模型 - 残差网络

    ResNet介绍:
        - 2015年ImageNet冠军，革命性的"跳跃连接"设计
        - 解决了深层网络的梯度消失问题
        - 在ImageNet上预训练（识别1000种物体）

    支持版本:
        - ResNet18: 18层，参数量~1120万，训练最快
        - ResNet34: 34层，参数量~2150万，准确率更高
        - ResNet50: 50层，参数量~2350万，准确率最高

    迁移学习优势:
        - 利用预训练权重，大幅减少训练时间
        - 即使数据量不大也能达到高准确率
        - ResNet18推荐用于快速实验（准确率75-85%）
        - ResNet50推荐用于追求最高准确率（准确率80-90%）

    工作原理:
        1. 使用ImageNet预训练的特征提取器
        2. 只替换最后的分类层（1000类 -> 5类）
        3. 微调整个网络适应表情识别任务

    推荐场景:
        - 实际项目首选
        - 需要快速获得高准确率
        - 数据量中等的情况

    输入: (batch_size, 3, 48, 48) RGB图像
    输出: (batch_size, 5) 5个类别的logits
    """
    def __init__(self, num_classes=5, pretrained=True, resnet_type='resnet18'):
        super(EmotionResNet, self).__init__()

        from torchvision import models

        # 根据类型选择ResNet变体
        if resnet_type == 'resnet18':
            # ResNet18: 轻量级，训练最快，推荐初学者使用
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            # ResNet34: 中等规模，准确率和速度的良好平衡
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            # ResNet50: 最强大，准确率最高，但训练较慢
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的ResNet类型: {resnet_type}")

        # 替换最后的全连接层
        # 原始: fc(2048 或 512, 1000) -> 新: fc(2048 或 512, 5)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),              # Dropout防止过拟合
            nn.Linear(num_features, num_classes)  # 1000类 -> 5类
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 (batch_size, 3, 48, 48)
        Returns:
            输出logits (batch_size, 5)
        """
        return self.resnet(x)


class EmotionVGG(nn.Module):
    """
    基于VGG16的迁移学习模型 - 经典卷积神经网络

    VGG介绍:
        - 2014年ImageNet亚军，由牛津大学提出
        - 特点: 使用小卷积核(3x3)堆叠出深层网络
        - 架构简单统一，易于理解和实现

    VGG16架构:
        - 13个卷积层 + 3个全连接层 = 16层
        - 5个卷积块，逐层翻倍特征图: 64 -> 128 -> 256 -> 512 -> 512
        - 参数量约1.38亿（非常大）

    优点:
        - 经典架构，广泛应用
        - 预训练效果好
        - 特征提取能力强

    缺点:
        - 参数量巨大，训练和推理较慢
        - 显存占用高
        - 现代架构（如ResNet、EfficientNet）通常表现更好

    推荐场景:
        - 学习经典CNN架构
        - 对比不同架构的性能差异
        - 有充足计算资源的情况

    准确率预期: 75-85%
    训练时间: 较慢（比ResNet慢约2倍）

    输入: (batch_size, 3, 48, 48) RGB图像
    输出: (batch_size, 5) 5个类别的logits
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(EmotionVGG, self).__init__()

        from torchvision import models
        # 加载预训练的VGG16模型
        self.vgg = models.vgg16(pretrained=pretrained)

        # VGG的分类器是一个Sequential，包含3个全连接层
        # 原始最后一层: Linear(4096, 1000)
        # 我们只替换最后一层: Linear(4096, 5)
        num_features = self.vgg.classifier[6].in_features  # 获取最后一层的输入特征数(4096)
        self.vgg.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 (batch_size, 3, 48, 48)
        Returns:
            输出logits (batch_size, 5)
        """
        return self.vgg(x)


class EmotionMobileNet(nn.Module):
    """
    基于MobileNetV2的迁移学习模型 - 轻量级高效网络

    MobileNet介绍:
        - Google专为移动设备设计的轻量级网络
        - 核心技术: 深度可分离卷积（Depthwise Separable Convolution）
        - 用更少的参数和计算量达到接近大型网络的准确率

    MobileNetV2特点:
        - 参数量仅约220万（比ResNet18少80%）
        - 使用倒残差结构（Inverted Residuals）
        - 线性瓶颈层（Linear Bottlenecks）保留更多信息

    优点:
        - 训练速度最快（比ResNet18快约1.5-2倍）
        - 参数量小，部署容易
        - 非常适合移动端和嵌入式设备
        - 推理速度快

    缺点:
        - 准确率略低于ResNet和EfficientNet
        - 在大图片上表现不如大型网络

    推荐场景:
        - 需要快速看到训练结果
        - 计算资源有限（CPU训练）
        - 需要部署到移动设备
        - 实时应用（需要快速推理）

    准确率预期: 70-80%
    训练时间: 最快（约15-20分钟/20 epochs，GPU）

    适合人群:
        - 初学者（快速获得反馈）
        - 需要边缘部署的项目
        - CPU训练用户

    输入: (batch_size, 3, 48, 48) RGB图像
    输出: (batch_size, 5) 5个类别的logits
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(EmotionMobileNet, self).__init__()

        from torchvision import models
        # 加载预训练的MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # MobileNet的分类器结构: Sequential(Dropout, Linear)
        # 替换最后的Linear层: Linear(1280, 1000) -> Linear(1280, 5)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 (batch_size, 3, 48, 48)
        Returns:
            输出logits (batch_size, 5)
        """
        return self.mobilenet(x)


class EmotionEfficientNet(nn.Module):
    """
    基于EfficientNet-B0的迁移学习模型 - 最先进的高效网络

    EfficientNet介绍:
        - 2019年Google提出，ImageNet SOTA（当时最佳）
        - 核心思想: 统一缩放网络的深度、宽度和分辨率
        - 通过神经架构搜索(NAS)自动找到最优架构

    EfficientNet-B0特点:
        - 参数量约410万（介于MobileNet和ResNet之间）
        - B0是基础版本，还有B1-B7（越大越强但越慢）
        - 使用MBConv块（Mobile Inverted Bottleneck Conv）
        - 集成了Squeeze-and-Excitation注意力机制

    技术亮点:
        - 复合缩放策略（Compound Scaling）
        - 同时优化准确率、速度和模型大小
        - 代表了现代CNN的最佳实践

    优点:
        - 准确率/效率比最优（最佳平衡）
        - 参数量适中但准确率高
        - 训练速度快（比ResNet50快但比MobileNet慢）
        - 泛化能力强

    缺点:
        - 架构较新，资料相对较少
        - 比MobileNet稍微复杂一些

    推荐场景:
        - 追求最佳性能的实际项目
        - 需要平衡准确率和效率
        - 现代深度学习最佳实践
        - 学术研究和竞赛

    准确率预期: 78-88%（可能是所有模型中最高的）
    训练时间: 快（约25-35分钟/20 epochs，GPU）

    为什么选EfficientNet:
        - 如果你只能选一个模型，选它
        - 代表了2019-2020年CNN的巅峰
        - 在多个数据集上表现始终优异

    输入: (batch_size, 3, 48, 48) RGB图像
    输出: (batch_size, 5) 5个类别的logits
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(EmotionEfficientNet, self).__init__()

        from torchvision import models
        # 加载预训练的EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)

        # EfficientNet的分类器结构: Sequential(Dropout, Linear)
        # 替换最后的Linear层: Linear(1280, 1000) -> Linear(1280, 5)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 (batch_size, 3, 48, 48)
        Returns:
            输出logits (batch_size, 5)
        """
        return self.efficientnet(x)


def get_model(model_type='cnn', num_classes=5, pretrained=True):
    """
    获取模型的工厂函数

    Args:
        model_type: 模型类型
            - 'cnn': 自定义CNN (从零训练)
            - 'resnet18': ResNet-18 (推荐)
            - 'resnet34': ResNet-34 (更深更准确)
            - 'resnet50': ResNet-50 (最准确但较慢)
            - 'vgg16': VGG-16 (经典模型)
            - 'mobilenet': MobileNetV2 (轻量快速)
            - 'efficientnet': EfficientNet-B0 (最佳平衡)
        num_classes: 类别数量
        pretrained: 是否使用预训练权重

    Returns:
        model: 指定的模型
    """
    if model_type == 'cnn':
        model = EmotionCNN(num_classes=num_classes)
        print("✅ 创建自定义CNN模型")

    elif model_type in ['resnet18', 'resnet34', 'resnet50']:
        model = EmotionResNet(num_classes=num_classes, pretrained=pretrained,
                             resnet_type=model_type)
        print(f"✅ 创建{model_type.upper()}模型 (预训练: {pretrained})")

    elif model_type == 'vgg16':
        model = EmotionVGG(num_classes=num_classes, pretrained=pretrained)
        print(f"✅ 创建VGG16模型 (预训练: {pretrained})")

    elif model_type == 'mobilenet':
        model = EmotionMobileNet(num_classes=num_classes, pretrained=pretrained)
        print(f"✅ 创建MobileNetV2模型 (预训练: {pretrained})")

    elif model_type == 'efficientnet':
        model = EmotionEfficientNet(num_classes=num_classes, pretrained=pretrained)
        print(f"✅ 创建EfficientNet-B0模型 (预训练: {pretrained})")

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    return total_params, trainable_params


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("测试所有可用模型")
    print("=" * 70)

    models_to_test = ['cnn', 'resnet18', 'mobilenet', 'efficientnet']

    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"测试 {model_name.upper()} 模型")
        print('='*70)

        try:
            model = get_model(model_name, num_classes=5, pretrained=False)
            count_parameters(model)

            # 测试前向传播
            dummy_input = torch.randn(2, 3, 48, 48)
            output = model(dummy_input)
            print(f"\n输入形状: {dummy_input.shape}")
            print(f"输出形状: {output.shape}")
            print(f"✅ {model_name.upper()} 测试成功！")

        except Exception as e:
            print(f"❌ {model_name.upper()} 测试失败: {e}")

    print("\n" + "=" * 70)
    print("模型测试完成！")
    print("=" * 70)