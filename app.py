"""
app.py - Flask后端API，用于表情识别
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import get_model

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
model = None
device = None
transform = None
classes = ['anger', 'fear', 'happy', 'sad', 'surprise']

# 中文翻译
class_names_zh = {
    'anger': '愤怒',
    'fear': '恐惧',
    'happy': '快乐',
    'sad': '悲伤',
    'surprise': '惊讶'
}

# 表情emoji
class_emojis = {
    'anger': '😠',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲'
}


def load_model(model_path, model_type):
    """加载模型"""
    global model, device, transform

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = get_model(model_type, num_classes=5, pretrained=False)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)

    # 检查是否是量化模型
    if checkpoint.get('quantized', False):
        print("加载量化模型...")
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"✅ 模型加载成功: {model_type}")
    print(f"   训练时准确率: {checkpoint.get('accuracy', 'N/A')}")


def predict_image(image):
    """预测图片"""
    if model is None:
        return None

    # 预处理
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # 获取所有类别的概率
    probs = probabilities[0].cpu().numpy()

    result = {
        'predicted_class': classes[predicted.item()],
        'predicted_class_zh': class_names_zh[classes[predicted.item()]],
        'emoji': class_emojis[classes[predicted.item()]],
        'confidence': float(confidence.item()),
        'probabilities': {
            classes[i]: {
                'en': classes[i],
                'zh': class_names_zh[classes[i]],
                'emoji': class_emojis[classes[i]],
                'probability': float(probs[i])
            }
            for i in range(len(classes))
        }
    }

    return result


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """获取可用模型列表"""
    models_dir = 'models'

    if not os.path.exists(models_dir):
        return jsonify({'error': '模型文件夹不存在'}), 404

    models = []
    for f in os.listdir(models_dir):
        if f.endswith('.pth'):
            model_path = os.path.join(models_dir, f)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)

            # 推断模型类型
            if 'distilled' in f:
                parts = f.replace('distilled_', '').replace('.pth', '').split('_from_')
                model_type = parts[0]
            elif 'quantized' in f:
                model_type = f.replace('best_model_', '').replace('_quantized.pth', '')
            elif 'pruned' in f:
                model_type = f.replace('best_model_', '').split('_pruned_')[0]
            else:
                model_type = f.replace('best_model_', '').replace('final_model_', '').replace('.pth', '')

            models.append({
                'filename': f,
                'model_type': model_type,
                'size_mb': round(size_mb, 2),
                'is_quantized': 'quantized' in f,
                'is_pruned': 'pruned' in f,
                'is_distilled': 'distilled' in f
            })

    return jsonify({'models': models})


@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    """加载指定模型"""
    data = request.json
    model_filename = data.get('model_filename')
    model_type = data.get('model_type')

    if not model_filename or not model_type:
        return jsonify({'error': '缺少参数'}), 400

    model_path = os.path.join('models', model_filename)

    if not os.path.exists(model_path):
        return jsonify({'error': '模型文件不存在'}), 404

    try:
        load_model(model_path, model_type)
        return jsonify({
            'success': True,
            'message': f'模型加载成功: {model_type}',
            'device': str(device)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """预测接口"""
    if model is None:
        return jsonify({'error': '请先加载模型'}), 400

    try:
        # 获取图片数据
        if 'file' in request.files:
            # 文件上传
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        elif 'image' in request.json:
            # Base64编码的图片
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            return jsonify({'error': '未提供图片'}), 400

        # 预测
        result = predict_image(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """获取服务状态"""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'classes': classes,
        'classes_zh': class_names_zh
    })


if __name__ == '__main__':
    # 默认加载一个模型（如果存在）
    default_model = 'models/best_model_resnet18.pth'
    if os.path.exists(default_model):
        try:
            load_model(default_model, 'resnet18')
            print("✅ 默认模型加载成功")
        except Exception as e:
            print(f"⚠️  默认模型加载失败: {e}")
            print("   请通过API手动加载模型")

    print("\n" + "=" * 70)
    print("🚀 Flask服务器启动")
    print("=" * 70)
    print("访问地址: http://localhost:5000")
    print("API文档:")
    print("  GET  /api/models       - 获取可用模型列表")
    print("  POST /api/load_model   - 加载指定模型")
    print("  POST /api/predict      - 预测图片")
    print("  GET  /api/status       - 获取服务状态")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)