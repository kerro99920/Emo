# 🤖 AI表情识别系统

基于深度学习的实时人脸表情分类系统，支持多种CNN架构（ResNet、VGG、MobileNet、EfficientNet等），提供模型训练、评估、优化（量化/剪枝/知识蒸馏）及Web演示功能。

## 📋 项目简介

本项目实现了一个完整的表情识别系统，能够识别5种基本表情：
- 😠 **愤怒** (Anger)
- 😨 **恐惧** (Fear)  
- 😊 **快乐** (Happy)
- 😢 **悲伤** (Sad)
- 😲 **惊讶** (Surprise)

## ✨ 主要特性

- **多模型支持**：自定义CNN、ResNet18/34/50、VGG16、MobileNetV2、EfficientNet-B0
- **迁移学习**：支持ImageNet预训练权重，大幅提升训练效率
- **模型优化**：支持动态量化、模型剪枝、知识蒸馏
- **数据清洗**：自动检测并处理损坏、重复、低质量图片
- **Web演示**：基于Flask的可视化界面，支持图片上传和摄像头拍照
- **批量训练**：一键训练多个模型并生成对比报告

## 📁 项目结构

```
emotion-recognition/
├── app.py                      # Flask Web服务入口
├── requirements.txt            # 训练环境依赖
├── requirements_web.txt        # Web部署依赖
├── .gitignore
│
├── src/                        # 核心代码
│   ├── model.py               # 模型定义（CNN、ResNet、VGG、MobileNet、EfficientNet）
│   ├── dataset.py             # 数据集加载与清洗
│   ├── train.py               # 单模型训练脚本
│   ├── train_multiple.py      # 批量训练与对比
│   ├── evaluate.py            # 模型评估工具
│   ├── optimize_distill.py    # 模型优化（量化/剪枝/蒸馏）
│   └── utils.py               # 工具函数（数据划分、可视化等）
│
├── templates/
│   └── index.html             # Web演示前端页面
│
├── data/                       # 数据目录（需自行准备）
│   ├── raw/                   # 原始数据
│   │   ├── anger/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   └── surprise/
│   ├── train/                 # 训练集（自动生成）
│   └── val/                   # 验证集（自动生成）
│
├── models/                     # 模型权重（训练后生成）
│   ├── best_model_*.pth
│   └── distilled_*.pth
│
└── results/                    # 训练结果（自动生成）
    ├── training_history_*.png
    ├── confusion_matrix_*.png
    └── models_comparison.csv
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd emotion-recognition

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

将表情图片按类别放入 `data/raw/` 目录：

```
data/raw/
├── anger/      # 愤怒表情图片
├── fear/       # 恐惧表情图片
├── happy/      # 快乐表情图片
├── sad/        # 悲伤表情图片
└── surprise/   # 惊讶表情图片
```

支持的图片格式：JPG、JPEG、PNG、BMP

### 3. 数据划分

```bash
cd src
python utils.py
```

这将自动清洗数据并按8:2比例划分为训练集和验证集。

### 4. 模型训练

**训练单个模型：**

```bash
python train.py
```

在 `train.py` 中修改配置：

```python
config = {
    'model_type': 'resnet18',  # 可选：cnn, resnet18/34/50, vgg16, mobilenet, efficientnet
    'num_epochs': 20,
    'batch_size': 64,
    'learning_rate': 0.001,
    'device': 'auto'
}
```

**批量训练多个模型：**

```bash
python train_multiple.py
```

支持交互式选择要训练的模型，并自动生成对比报告。

### 5. 模型评估

```bash
python evaluate.py
```

功能选项：
1. 评估单个模型
2. 评估所有模型
3. 对比优化效果
4. 预测单张图片

### 6. 启动Web演示

```bash
cd ..
pip install -r requirements_web.txt
python app.py
```

访问 http://localhost:5000 查看Web界面。

## 🔧 模型详解

### 可用模型对比

| 模型 | 参数量 | 预计准确率 | 训练速度 | 推荐场景 |
|------|--------|-----------|---------|---------|
| Custom CNN | ~590万 | 60-70% | 中 | 学习CNN原理 |
| ResNet18 | ~1120万 | 75-85% | 快 | **推荐首选** |
| ResNet34 | ~2150万 | 78-88% | 中 | 更高准确率 |
| ResNet50 | ~2350万 | 80-90% | 慢 | 最高准确率 |
| VGG16 | ~1.38亿 | 75-85% | 很慢 | 经典架构学习 |
| MobileNetV2 | ~220万 | 70-80% | **最快** | 移动端部署 |
| EfficientNet-B0 | ~410万 | 78-88% | 快 | **最佳平衡** |

### 模型选择建议

- **初学者/快速实验**：MobileNet（训练最快）
- **实际项目首选**：ResNet18 或 EfficientNet-B0
- **追求最高准确率**：ResNet50
- **移动端部署**：MobileNet + 量化
- **学习CNN原理**：Custom CNN

## ⚡ 模型优化

### 动态量化

将FP32模型转换为INT8，减小约75%体积：

```bash
python optimize_distill.py
# 选择 [1] 模型量化
```

### 模型剪枝

移除不重要的权重，减少计算量：

```bash
python optimize_distill.py
# 选择 [2] 模型剪枝
```

### 知识蒸馏

将大模型的知识迁移到小模型：

```bash
python optimize_distill.py
# 选择 [3] 知识蒸馏
```

推荐组合：
- ResNet50 → ResNet18
- EfficientNet → MobileNet
- ResNet18 → MobileNet

## 📊 API接口说明

### 获取可用模型

```http
GET /api/models
```

### 加载模型

```http
POST /api/load_model
Content-Type: application/json

{
    "model_filename": "best_model_resnet18.pth",
    "model_type": "resnet18"
}
```

### 预测表情

```http
POST /api/predict
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,..."
}
```

响应示例：

```json
{
    "predicted_class": "happy",
    "predicted_class_zh": "快乐",
    "emoji": "😊",
    "confidence": 0.953,
    "probabilities": {
        "anger": {"zh": "愤怒", "emoji": "😠", "probability": 0.01},
        "fear": {"zh": "恐惧", "emoji": "😨", "probability": 0.02},
        "happy": {"zh": "快乐", "emoji": "😊", "probability": 0.953},
        "sad": {"zh": "悲伤", "emoji": "😢", "probability": 0.01},
        "surprise": {"zh": "惊讶", "emoji": "😲", "probability": 0.007}
    }
}
```

### 服务状态

```http
GET /api/status
```

## 📈 训练技巧

### 数据增强

训练时默认启用以下增强：
- 随机水平翻转 (p=0.5)
- 随机旋转 (±10°)
- 颜色抖动 (亮度/对比度 ±0.2)

### 学习率调度

使用 `ReduceLROnPlateau`：当验证损失不再下降时自动降低学习率。

### 防止过拟合

- Dropout (Conv层: 0.25, FC层: 0.5)
- BatchNormalization
- 早停（保存最佳模型）

## 🔍 数据清洗

系统自动检测并处理以下问题：
- **损坏图片**：无法打开或读取
- **格式错误**：非标准图像格式
- **尺寸异常**：过小(<32px)或过大(>2000px)
- **质量过低**：全黑/全白或方差过小
- **重复图片**：基于MD5哈希去重

问题文件会被移动到 `data/quarantine/` 目录。

## ⚙️ 配置要求

### 最低配置

- Python 3.8+
- 8GB RAM
- 10GB 磁盘空间

### 推荐配置

- Python 3.9+
- 16GB RAM
- NVIDIA GPU (CUDA 11.0+)
- 20GB+ 磁盘空间

## 📝 注意事项

1. **GPU加速**：如有NVIDIA GPU，确保安装CUDA版本的PyTorch
2. **数据隐私**：训练数据不会上传到任何服务器
3. **模型文件**：`.pth` 文件较大，已在 `.gitignore` 中排除
4. **首次训练**：预训练模型权重会自动从网络下载

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至：[Kerro99920@gmail.com]

---

**如果这个项目对你有帮助，请给个 ⭐ Star！**
