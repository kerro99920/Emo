"""
dataset.py - 表情识别数据集类（集成数据清洗功能）
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import hashlib
from tqdm import tqdm
import shutil
from collections import defaultdict


class EmotionDataset(Dataset):
    """
    表情识别数据集类（带自动清洗功能）
    """
    def __init__(self, root_dir, transform=None, auto_clean=False, clean_on_load=True):
        """
        初始化数据集

        Args:
            root_dir: 数据集根目录路径
            transform: 图像转换操作
            auto_clean: 是否在首次加载时自动清洗数据
            clean_on_load: 加载时是否验证每张图片
        """
        self.root_dir = root_dir
        self.transform = transform
        self.clean_on_load = clean_on_load

        # 定义5个表情类别
        self.classes = ['anger', 'fear', 'happy', 'sad', 'surprise']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 清洗统计
        self.clean_stats = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted': 0,
            'invalid_format': 0,
            'size_error': 0,
            'quality_error': 0
        }
        # 如果启用自动清洗
        if auto_clean:
            print("\n🧹 自动清洗模式已启用")
            self._auto_clean_dataset()

        # 加载所有图片路径和对应标签
        self.samples = []
        self._load_samples()

        print(f"数据集加载完成！共 {len(self.samples)} 张有效图片")
        self._print_statistics()

        # 如果有清洗统计，显示
        if self.clean_stats['total_files'] > 0:
            self._print_clean_stats()

    def _check_image_valid(self, img_path):
        """
        检查图片是否有效

        Returns:
            (is_valid, error_type)
        """
        try:
            # 尝试打开图片
            with Image.open(img_path) as img:
                # 检查格式
                if img.format not in ['JPEG', 'PNG', 'BMP']:
                    return False, 'invalid_format'

                # 检查尺寸
                width, height = img.size
                if width < 32 or height < 32:
                    return False, 'size_error'

                if width > 2000 or height > 2000:
                    return False, 'size_error'

                # 检查模式
                if img.mode not in ['RGB', 'L', 'RGBA']:
                    return False, 'invalid_format'

                # 验证数据完整性
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb)

                # 检查是否全黑或全白
                mean_val = img_array.mean()
                if mean_val < 5 or mean_val > 250:
                    return False, 'quality_error'

                # 检查方差
                var = img_array.var()
                if var < 10:
                    return False, 'quality_error'

            return True, None

        except Exception as e:
            return False, 'corrupted'

    def _calculate_file_hash(self, file_path):
        """计算文件MD5哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None

    def _auto_clean_dataset(self):
        """自动清洗数据集"""
        print("\n" + "=" * 70)
        print("🔍 开始自动清洗数据集...")
        print("=" * 70)

        quarantine_dir = os.path.join(os.path.dirname(self.root_dir), 'quarantine')
        os.makedirs(quarantine_dir, exist_ok=True)

        # 第1步：检测并移除损坏/异常的图片
        print("\n1️⃣  检测损坏和异常图片...")

        problem_files = []
        hash_dict = defaultdict(list)

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            files = [f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            print(f"\n检查 {class_name} 类别...")

            for filename in tqdm(files, desc=f"  扫描中", ncols=80):
                file_path = os.path.join(class_dir, filename)
                self.clean_stats['total_files'] += 1

                # 检查有效性
                is_valid, error_type = self._check_image_valid(file_path)

                if not is_valid:
                    problem_files.append((file_path, error_type, class_name, filename))
                    self.clean_stats[error_type] += 1
                else:
                    # 有效图片，计算哈希用于去重
                    file_hash = self._calculate_file_hash(file_path)
                    if file_hash:
                        hash_dict[file_hash].append(file_path)

        # 移除问题文件
        if problem_files:
            print(f"\n发现 {len(problem_files)} 个问题文件，正在移动到隔离区...")

            for file_path, error_type, class_name, filename in tqdm(problem_files, desc="移除中", ncols=80):
                rel_path = os.path.relpath(file_path, self.root_dir)
                dst_path = os.path.join(quarantine_dir, error_type, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                try:
                    shutil.move(file_path, dst_path)
                except:
                    pass

        # 第2步：检测并移除重复图片
        print("\n2️⃣  检测重复图片...")

        duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}

        if duplicates:
            dup_count = sum(len(v) - 1 for v in duplicates.values())
            print(f"\n发现 {len(duplicates)} 组重复（共 {dup_count} 个重复文件）")

            for hash_val, files in tqdm(duplicates.items(), desc="移除重复", ncols=80):
                # 保留第一个，移除其余
                for file_path in files[1:]:
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    dst_path = os.path.join(quarantine_dir, 'duplicates', rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    try:
                        shutil.move(file_path, dst_path)
                    except:
                        pass
        else:
            print("✓ 未发现重复图片")

        print("\n" + "=" * 70)
        print("✅ 自动清洗完成！")
        print("=" * 70)
        print(f"隔离目录: {quarantine_dir}")

    def _load_samples(self):
        """加载所有图片路径和标签（带验证）"""
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"⚠️  警告：文件夹 {class_dir} 不存在！")
                continue

            # 遍历该类别文件夹中的所有图片
            files = [f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            for img_name in files:
                img_path = os.path.join(class_dir, img_name)

                # 如果启用了加载时验证
                if self.clean_on_load:
                    is_valid, error_type = self._check_image_valid(img_path)

                    if not is_valid:
                        # 记录统计但不添加到样本中
                        if error_type:
                            self.clean_stats[error_type] = self.clean_stats.get(error_type, 0) + 1
                        continue

                label = self.class_to_idx[class_name]
                self.samples.append((img_path, label))
                self.clean_stats['valid_files'] += 1

    def _print_statistics(self):
        """打印数据集统计信息"""
        print("\n各类别图片数量：")
        for class_name in self.classes:
            count = sum(1 for _, label in self.samples
                       if label == self.class_to_idx[class_name])
            print(f"  {class_name}: {count} 张")

    def _print_clean_stats(self):
        """打印清洗统计信息"""
        print("\n" + "=" * 70)
        print("🧹 清洗统计")
        print("=" * 70)

        if self.clean_stats['total_files'] > 0:
            print(f"总扫描文件: {self.clean_stats['total_files']}")
            print(f"有效文件: {self.clean_stats['valid_files']}")

            problem_count = (self.clean_stats.get('corrupted', 0) +
                           self.clean_stats.get('invalid_format', 0) +
                           self.clean_stats.get('size_error', 0) +
                           self.clean_stats.get('quality_error', 0))

            if problem_count > 0:
                print(f"\n问题文件: {problem_count}")
                if self.clean_stats.get('corrupted', 0) > 0:
                    print(f"  - 损坏: {self.clean_stats['corrupted']}")
                if self.clean_stats.get('invalid_format', 0) > 0:
                    print(f"  - 格式错误: {self.clean_stats['invalid_format']}")
                if self.clean_stats.get('size_error', 0) > 0:
                    print(f"  - 尺寸错误: {self.clean_stats['size_error']}")
                if self.clean_stats.get('quality_error', 0) > 0:
                    print(f"  - 质量过低: {self.clean_stats['quality_error']}")
            else:
                print("✓ 所有文件都有效")

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本

        Args:
            idx: 样本索引

        Returns:
            image: 处理后的图像tensor
            label: 标签
        """
        img_path, label = self.samples[idx]

        # 读取图片并转换为RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图片失败: {img_path}, 错误: {e}")
            # 如果读取失败，返回一个黑色图片
            image = Image.new('RGB', (48, 48), (0, 0, 0))

        # 应用数据转换
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx):
        """根据索引获取类别名称"""
        return self.classes[idx]

    def get_class_distribution(self):
        """获取类别分布"""
        distribution = {class_name: 0 for class_name in self.classes}

        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1

        return distribution

    @staticmethod
    def clean_directory(data_dir, move_to_quarantine=True, remove_duplicates=True):
        """
        静态方法：独立清洗数据目录

        Args:
            data_dir: 数据目录
            move_to_quarantine: 是否移动到隔离区
            remove_duplicates: 是否移除重复

        Returns:
            清洗统计字典
        """
        print("\n" + "=" * 70)
        print("🧹 独立清洗模式")
        print("=" * 70)
        print(f"数据目录: {data_dir}")

        # 创建临时数据集对象进行清洗
        temp_dataset = EmotionDataset.__new__(EmotionDataset)
        temp_dataset.root_dir = data_dir
        temp_dataset.classes = ['anger', 'fear', 'happy', 'sad', 'surprise']
        temp_dataset.clean_stats = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted': 0,
            'invalid_format': 0,
            'size_error': 0,
            'quality_error': 0
        }

        # 执行清洗
        temp_dataset._auto_clean_dataset()

        return temp_dataset.clean_stats


# 便捷函数
def create_clean_dataset(root_dir, transform=None, auto_clean=True):
    """
    便捷函数：创建自动清洗的数据集

    Args:
        root_dir: 数据目录
        transform: 转换操作
        auto_clean: 是否自动清洗

    Returns:
        EmotionDataset对象
    """
    return EmotionDataset(root_dir, transform=transform, auto_clean=auto_clean)


def quick_clean(data_dir):
    """
    便捷函数：快速清洗数据目录（不创建数据集）

    Args:
        data_dir: 数据目录

    Returns:
        清洗统计
    """
    return EmotionDataset.clean_directory(data_dir)


# 测试代码
if __name__ == "__main__":
    from torchvision import transforms

    print("\n" + "=" * 70)
    print("🧪 数据集测试")
    print("=" * 70)

    print("\n选择测试模式:")
    print("  [1] 加载数据集（不清洗）")
    print("  [2] 加载数据集（自动清洗）")
    print("  [3] 仅清洗数据（不加载）")

    choice = input("\n请选择 (1/2/3): ").strip()

    # 定义简单的转换
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    if choice == '1':
        # 普通加载
        print("\n普通加载模式...")
        dataset = EmotionDataset(
            root_dir='../data/raw',
            transform=transform,
            auto_clean=False,
            clean_on_load=False
        )

    elif choice == '2':
        # 自动清洗加载
        print("\n自动清洗模式...")
        dataset = EmotionDataset(
            root_dir='../data/raw',
            transform=transform,
            auto_clean=True,
            clean_on_load=True
        )

    elif choice == '3':
        # 仅清洗
        print("\n仅清洗模式...")
        stats = quick_clean('../data/raw')

        print("\n清洗完成！统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        exit(0)

    else:
        print("无效选择")
        exit(1)

    # 测试获取样本
    if len(dataset) > 0:
        print(f"\n测试样本：")
        image, label = dataset[0]
        print(f"  图像形状: {image.shape}")
        print(f"  标签: {label} ({dataset.get_class_name(label)})")

        # 显示类别分布
        print(f"\n类别分布：")
        distribution = dataset.get_class_distribution()
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count} 张")
    else:
        print("\n⚠️  数据集为空！请检查data/raw文件夹中是否有图片。")

    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)