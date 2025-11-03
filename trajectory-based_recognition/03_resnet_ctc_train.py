import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import math
import json
import argparse
import editdistance
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


# ================== ResNet组件实现 ==================

class EnhancedResNetEncoder(nn.Module):
    """
    增强的ResNet编码器
    - 多尺度特征融合（layer3 + layer4）
    - 保留更多空间信息（高度方向2行）
    - 增强细粒度字母特征
    """

    def __init__(self, resnet_type='resnet50', feature_dim=512, freeze_backbone=False):
        super().__init__()

        self.resnet_type = resnet_type
        self.feature_dim = feature_dim

        # 加载ResNet，使用torchvision官方预训练权重
        if resnet_type == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone_out_channels = 512
            layer3_channels = 256
        elif resnet_type == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            backbone_out_channels = 512
            layer3_channels = 256
        elif resnet_type == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_out_channels = 2048
            layer3_channels = 1024
        else:
            raise ValueError(f"不支持的ResNet类型: {resnet_type}")

        # ResNet backbone
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 可选：冻结backbone
        if freeze_backbone:
            for param in [self.conv1, self.bn1, self.layer1,
                          self.layer2, self.layer3, self.layer4]:
                for p in param.parameters():
                    p.requires_grad = False

        # 多尺度特征融合
        self.layer3_proj = nn.Sequential(
            nn.Conv2d(layer3_channels, feature_dim // 2, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.layer4_proj = nn.Sequential(
            nn.Conv2d(backbone_out_channels, feature_dim // 2, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        # 保留高度信息：压缩到2行而非1行
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, None))

        # 最终投影：[feature_dim * 2] -> [feature_dim]
        self.final_proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x):
        """
        x: [B, 1, H, W] 灰度图
        return: [B, T, D] 序列特征
        """
        # 灰度图复制到3通道
        x = x.repeat(1, 3, 1, 1)

        # ResNet前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # 提取多尺度特征
        feat3 = self.layer3(x)  # [B, channels, H/8, W/8]
        feat4 = self.layer4(feat3)  # [B, channels, H/16, W/16]

        # 投影到统一维度
        feat3 = self.layer3_proj(feat3)  # [B, D/2, H/8, W/8]
        feat4 = self.layer4_proj(feat4)  # [B, D/2, H/16, W/16]

        # 上采样feat4到feat3的尺寸
        feat4_up = F.interpolate(
            feat4,
            size=feat3.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 拼接并融合
        fused = torch.cat([feat3, feat4_up], dim=1)  # [B, D, H/8, W/8]
        fused = self.fusion_conv(fused)

        # 保留高度信息
        features = self.adaptive_pool(fused)  # [B, D, 2, W/8]

        # 转换为序列格式
        B, D, H, W = features.shape
        features = features.permute(0, 3, 1, 2)  # [B, W/8, D, 2]
        features = features.reshape(B, W, D * H)  # [B, W/8, D*2]

        # 投影到目标维度
        features = self.final_proj(features)  # [B, W/8, feature_dim]

        return features


# ================== 数据加载模块 ==================

class HandwritingWordDataset(Dataset):
    """手写单词数据集（从JSON文件加载）"""

    def __init__(self, json_file: str, split: str = 'train',
                 img_height: int = 64, img_width: int = 256,
                 max_word_length: int = 15):
        """
        Args:
            json_file: JSON划分文件路径
            split: 'train' 或 'test'
            img_height: 图像高度
            img_width: 图像宽度
            max_word_length: 最大单词长度
        """
        self.json_file = json_file
        self.split = split
        self.max_word_length = max_word_length

        # 创建字符到索引的映射（CTC需要blank token）
        self.special_tokens = {
            '<blank>': 0,  # CTC blank token，必须是0
            '<unk>': 1,
        }

        # 添加字母
        self.char_to_idx = self.special_tokens.copy()
        for i in range(26):
            self.char_to_idx[chr(ord('a') + i)] = i + len(self.special_tokens)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 从JSON文件加载数据划分信息
        self.data_info = []
        self._load_from_json()

    def _load_from_json(self):
        """从JSON文件加载数据"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)

        # 获取对应split的数据
        file_list = split_data.get(self.split, [])

        for item in file_list:
            file_path = item['filepath']
            word = item['word'].lower()

            # 检查文件是否存在
            if not os.path.exists(file_path):
                continue

            # 过滤过长的单词
            if len(word) > self.max_word_length:
                continue

            # 检查单词是否只包含字母
            if not word.isalpha():
                continue

            # 转换为token序列（CTC不需要BOS/EOS）
            token_ids = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in word]

            self.data_info.append({
                'file_path': file_path,
                'word': word,
                'token_ids': token_ids
            })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        # 加载图像
        image = Image.open(info['file_path']).convert('L')
        image = self.transform(image)

        # 获取token序列
        token_ids = torch.LongTensor(info['token_ids'])

        return {
            'data': image,
            'token_ids': token_ids,
            'input_length': image.shape[1] // 8,  # 根据ResNet的下采样倍数估算
            'target_length': len(token_ids),
            'word': info['word']
        }


class TestDataset(Dataset):
    """测试数据集（从JSON文件加载）"""

    def __init__(self, json_file: str, char_to_idx: dict,
                 img_height: int = 64, img_width: int = 256):
        self.json_file = json_file
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 从JSON文件加载数据
        self.data_info = []
        self._load_from_json()

    def _load_from_json(self):
        """从JSON文件加载测试数据"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)

        # 获取test split的数据
        file_list = split_data.get('test', [])

        for item in file_list:
            file_path = item['filepath']
            word = item['word'].lower()
            filename = item['file']

            if not os.path.exists(file_path):
                continue

            self.data_info.append({
                'file_path': file_path,
                'filename': filename,
                'word': word
            })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        # 加载图像
        image = Image.open(info['file_path']).convert('L')
        image = self.transform(image)

        return {
            'data': image,
            'input_length': image.shape[1] // 8,
            'filename': info['filename'],
            'ground_truth': info['word']
        }


def collate_fn(batch):
    """处理变长序列的批处理函数（训练用）"""
    # 按输入长度排序
    batch = sorted(batch, key=lambda x: x['input_length'], reverse=True)

    # 分别收集各个字段
    data = torch.stack([item['data'] for item in batch])
    token_ids = [item['token_ids'] for item in batch]
    input_lengths = torch.LongTensor([item['input_length'] for item in batch])
    target_lengths = torch.LongTensor([item['target_length'] for item in batch])
    words = [item['word'] for item in batch]

    # 对于CTC，目标序列不需要padding，直接concatenate
    concatenated_targets = torch.cat(token_ids, dim=0)

    return {
        'data': data,
        'targets': concatenated_targets,  # CTC targets (concatenated)
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
        'words': words
    }


def test_collate_fn(batch):
    """测试数据的批处理函数"""
    # 按输入长度排序
    batch = sorted(batch, key=lambda x: x['input_length'], reverse=True)

    data = torch.stack([item['data'] for item in batch])
    input_lengths = torch.LongTensor([item['input_length'] for item in batch])
    filenames = [item['filename'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]

    return {
        'data': data,
        'input_lengths': input_lengths,
        'filenames': filenames,
        'ground_truths': ground_truths
    }


# ================== 模型定义模块 ==================

class HandwritingResNetCTC(nn.Module):
    """手写识别的ResNet + CTC模型"""

    def __init__(self, vocab_size: int, resnet_type: str = 'resnet50',
                 feature_dim: int = 512, freeze_backbone: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.resnet_type = resnet_type
        self.feature_dim = feature_dim

        # ResNet编码器
        self.encoder = EnhancedResNetEncoder(
            resnet_type=resnet_type,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )

        # CTC分类头
        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, vocab_size)
        )

        # CTC损失函数
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # 特殊token索引
        self.blank_idx = 0
        self.unk_idx = 1

    def count_parameters(self):
        """计算模型参数量"""
        def count_module_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        encoder_params = count_module_params(self.encoder)
        ctc_params = count_module_params(self.ctc_head)
        total_params = encoder_params + ctc_params

        return {
            'total_params': total_params,
            'encoder_params': encoder_params,
            'ctc_params': ctc_params
        }

    def print_model_info(self):
        """打印模型信息"""
        param_info = self.count_parameters()

        print("=" * 60)
        print("模型参数统计 (ResNet + CTC)")
        print("=" * 60)
        print(f"ResNet类型: {self.resnet_type}")
        print(f"总参数量: {param_info['total_params']:,} ({param_info['total_params'] / 1e6:.2f}M)")
        print(f"编码器参数: {param_info['encoder_params']:,} ({param_info['encoder_params'] / 1e6:.2f}M)")
        print(f"CTC层参数: {param_info['ctc_params']:,} ({param_info['ctc_params'] / 1e6:.2f}M)")
        print("=" * 60)

        return param_info

    def forward(self, x, target=None, input_lengths=None, target_lengths=None):
        """
        Args:
            x: (batch_size, 1, H, W)
            target: CTC目标序列 (concatenated targets)
            input_lengths: 输入序列长度
            target_lengths: 目标序列长度
        """
        # 编码
        encoded = self.encoder(x)  # (batch_size, seq_len, feature_dim)

        # CTC分类
        logits = self.ctc_head(encoded)  # (batch_size, seq_len, vocab_size)

        # 如果是训练模式且提供了目标，计算CTC损失
        if self.training and target is not None and target_lengths is not None:
            # CTC需要的格式：(seq_len, batch_size, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            # 获取实际的输出长度
            batch_size = x.size(0)
            output_lengths = torch.full((batch_size,), logits.size(1),
                                       dtype=torch.long, device=x.device)

            loss = self.ctc_loss(log_probs, target, output_lengths, target_lengths)
            return logits, loss, output_lengths
        else:
            # 推理模式
            batch_size = x.size(0)
            output_lengths = torch.full((batch_size,), logits.size(1),
                                       dtype=torch.long, device=x.device)
            return logits, output_lengths

    @torch.no_grad()
    def decode_predictions(self, logits, output_lengths, idx_to_char):
        """CTC贪心解码"""
        predictions = []

        for i, length in enumerate(output_lengths):
            # 获取当前样本的logits
            seq_logits = logits[i, :length]

            # 贪心解码
            pred_ids = torch.argmax(seq_logits, dim=-1)

            # CTC解码：去除重复和blank
            decoded_chars = []
            prev_id = -1

            for pred_id in pred_ids:
                pred_id = pred_id.item()
                # 跳过blank token和重复的token
                if pred_id != self.blank_idx and pred_id != prev_id:
                    if pred_id in idx_to_char:
                        char = idx_to_char[pred_id]
                        if char not in ['<blank>', '<unk>']:
                            decoded_chars.append(char)
                prev_id = pred_id

            predictions.append(''.join(decoded_chars))

        return predictions


# ================== 训练和推理模块 ==================

def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """计算准确率"""
    if len(predictions) != len(ground_truths):
        return 0.0
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / len(predictions)


def calculate_edit_distance_accuracy(predictions: List[str], ground_truths: List[str]) -> Tuple[float, float]:
    """计算基于编辑距离的准确率"""
    if len(predictions) != len(ground_truths):
        return 0.0, 0.0

    total_distance = 0
    total_chars = 0

    for pred, gt in zip(predictions, ground_truths):
        distance = editdistance.eval(pred, gt)
        total_distance += distance
        total_chars += len(gt)

    char_accuracy = max(0, 1 - total_distance / max(1, total_chars))
    avg_edit_distance = total_distance / len(predictions)

    return char_accuracy, avg_edit_distance


def train_epoch(model, dataloader, optimizer, device, epoch, idx_to_char, rank=0):
    """训练一个epoch（CTC版本）"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_ground_truths = []

    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        data = batch['data'].to(device)
        targets = batch['targets'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        optimizer.zero_grad()

        # 前向传播
        logits, loss, output_lengths = model(data, targets, input_lengths, target_lengths)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录损失
        total_loss += loss.item()

        # 生成预测结果用于计算准确率
        with torch.no_grad():
            if hasattr(model, 'module'):
                predictions = model.module.decode_predictions(logits, output_lengths, idx_to_char)
            else:
                predictions = model.decode_predictions(logits, output_lengths, idx_to_char)

            ground_truths = batch['words']

            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

        # 更新进度条
        if rank == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss / (batch_idx + 1):.4f}'
            })

    # 计算准确率
    avg_loss = total_loss / len(dataloader)
    accuracy = calculate_accuracy(all_predictions, all_ground_truths)
    char_acc, _ = calculate_edit_distance_accuracy(all_predictions, all_ground_truths)

    return avg_loss, accuracy, char_acc


def test_model(model, dataloader, device, idx_to_char, rank=0):
    """在测试集上评估模型"""
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc='Testing')
        else:
            pbar = dataloader

        for batch in pbar:
            data = batch['data'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            ground_truths = batch['ground_truths']

            # 生成预测
            if hasattr(model, 'module'):
                logits, output_lengths = model.module(data, input_lengths=input_lengths)
                predictions = model.module.decode_predictions(logits, output_lengths, idx_to_char)
            else:
                logits, output_lengths = model(data, input_lengths=input_lengths)
                predictions = model.decode_predictions(logits, output_lengths, idx_to_char)

            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

    # 计算准确率
    accuracy = calculate_accuracy(all_predictions, all_ground_truths)
    char_acc, avg_edit_dist = calculate_edit_distance_accuracy(all_predictions, all_ground_truths)

    return accuracy, char_acc, avg_edit_dist


def create_lm_prompt(predicted_word):
    """创建语言模型提示词"""
    prompt = f"""Correct the following possibly misspelled English word to its most likely correct common English word.
If it is already a correct English word, output it exactly as is.
Return only the corrected word, with no explanation or punctuation.
Input: {predicted_word}
Output:"""
    return prompt


def inference(model, dataloader, device, idx_to_char, output_file, rank=0, generate_lm_input=False):
    """批量推理并保存结果"""
    model.eval()
    results = []
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc='Inference')
        else:
            pbar = dataloader

        for batch in pbar:
            data = batch['data'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            filenames = batch['filenames']
            ground_truths = batch['ground_truths']

            # 生成预测
            if hasattr(model, 'module'):
                logits, output_lengths = model.module(data, input_lengths=input_lengths)
                predictions = model.module.decode_predictions(logits, output_lengths, idx_to_char)
            else:
                logits, output_lengths = model(data, input_lengths=input_lengths)
                predictions = model.decode_predictions(logits, output_lengths, idx_to_char)

            # 保存结果
            for filename, prediction, ground_truth in zip(filenames, predictions, ground_truths):
                result = {
                    'filename': filename,
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'correct': prediction == ground_truth,
                    'pred_length': len(prediction),
                    'true_length': len(ground_truth),
                    'model_type': 'resnet_ctc'
                }
                results.append(result)
                all_predictions.append(prediction)
                all_ground_truths.append(ground_truth)

    if rank == 0:
        # 计算整体准确率
        total_correct = sum(1 for r in results if r['correct'])
        accuracy = total_correct / len(results) if results else 0

        # 计算字符级准确率
        char_acc, avg_edit_dist = calculate_edit_distance_accuracy(all_predictions, all_ground_truths)

        # 添加统计信息
        summary = {
            'stage': 'original_ctc',
            'total_samples': len(results),
            'correct_samples': total_correct,
            'word_accuracy': accuracy,
            'character_accuracy': char_acc,
            'average_edit_distance': avg_edit_dist,
            'model_type': 'resnet_ctc'
        }

        # 保存结果到jsonl文件
        from pathlib import Path
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            # 先写入统计信息
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')

            # 再写入详细结果
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\n{'=' * 80}")
        print(f"原始CTC模型推理完成!")
        print(f"{'=' * 80}")
        print(f"模型类型: ResNet + CTC")
        print(f"测试样本总数: {len(results)}")
        print(f"正确预测数: {total_correct}")
        print(f"单词准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"字母准确率: {char_acc:.4f} ({char_acc * 100:.2f}%)")
        print(f"平均编辑距离: {avg_edit_dist:.4f}")
        print(f"结果已保存到: {output_file}")
        print(f"{'=' * 80}\n")

        # 如果需要生成LLM输入文件
        if generate_lm_input:
            llm_input_path = output_path.parent / f"llm_input_{output_path.stem}.jsonl"
            print(f"生成llm批量推理输入文件: {llm_input_path}")

            with open(llm_input_path, 'w', encoding='utf-8') as f:
                for idx, (pred, true) in enumerate(zip(all_predictions, all_ground_truths)):
                    llm_item = {
                        "custom_id": f"request-{idx}",
                        "prompt": create_lm_prompt(pred),
                        "original_prediction": pred,
                        "ground_truth": true
                    }
                    f.write(json.dumps(llm_item, ensure_ascii=False) + '\n')

            print(f"llm输入文件已保存: {llm_input_path}")
            print(f"总共 {len(all_predictions)} 条待纠正的预测\n")

            # 保存原始统计信息
            stats_path = output_path.parent / f"stats_original_{output_path.stem}.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"原始统计信息已保存到: {stats_path}")
            print(f"\n{'=' * 80}")
            print(f"接下来请运行llm批量推理脚本:")
            print(f"python trajectory-based_recognition/04_llm_correction.py \\")
            print(f"  --input {llm_input_path} \\")
            print(f"  --output {output_path.parent}/llm_output_{output_path.stem}.jsonl \\")
            print(f"  --model <your_llm_model_path>")
            print(f"{'=' * 80}\n")

        return results, accuracy, char_acc

    return results, 0.0, 0.0


def create_dataloaders(config: dict, char_to_idx: dict = None):
    """创建训练数据加载器（从JSON文件加载）"""
    # 创建训练数据集
    train_dataset = HandwritingWordDataset(
        config['split_json'],
        split='train',
        img_height=config['img_height'],
        img_width=config['img_width'],
        max_word_length=config['max_word_length']
    )

    # 如果提供了char_to_idx，使用它；否则使用数据集自己的
    if char_to_idx is None:
        char_to_idx = train_dataset.char_to_idx
        vocab_size = train_dataset.vocab_size
    else:
        vocab_size = len(char_to_idx)

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, char_to_idx, vocab_size


def create_test_dataloader(config: dict, char_to_idx: dict):
    """创建测试数据加载器（从JSON文件加载）"""
    test_dataset = TestDataset(
        config['split_json'],
        char_to_idx,
        img_height=config['img_height'],
        img_width=config['img_width']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return test_loader


def get_experiment_config(experiment_type, resnet_type):
    """获取实验配置"""
    base_config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'resnet_type': resnet_type,
        'feature_dim': 512,
        'freeze_backbone': False,
        'dropout': 0.1,
        'num_workers': 4,
        'img_height': 128,
        'img_width': 256,
        'max_word_length': 15,
        'warmup_epochs': 5,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'generate_lm_input': True  # 生成LLM输入文件
    }

    experiment_configs = {
        '200word': {
            **base_config,
            'split_json': 'trajectory-based_recognition/dataset_index/200-Word.json',
            'save_path': f'trajectory-based_recognition/output/mmwave_{resnet_type}_ctc_model.pth',
            'inference_output': f'trajectory-based_recognition/output/inference_results_{resnet_type}_ctc.jsonl'
        },
        'cross_user': {
            **base_config,
            'split_json': 'trajectory-based_recognition/dataset_index/user-split.json',
            'save_path': f'trajectory-based_recognition/output/mmwave_{resnet_type}_ctc_model_cross_user.pth',
            'inference_output': f'trajectory-based_recognition/output/inference_results_{resnet_type}_ctc_cross_user.jsonl'
        },
        'zero_shot': {
            **base_config,
            'split_json': 'trajectory-based_recognition/dataset_index/zero-shot.json',
            'save_path': f'trajectory-based_recognition/output/mmwave_{resnet_type}_ctc_model_zero_shot.pth',
            'inference_output': f'trajectory-based_recognition/output/inference_results_{resnet_type}_ctc_zero_shot.jsonl'
        }
    }

    if experiment_type not in experiment_configs:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                         f"Available: {list(experiment_configs.keys())}")

    return experiment_configs[experiment_type]


def run_training(rank, world_size, config):
    """在指定GPU上运行训练"""
    # 设置分布式训练环境
    if world_size > 1:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f'使用JSON文件: {config["split_json"]}')
        print(f'ResNet配置: {config["resnet_type"]}, feature_dim={config["feature_dim"]}')

    # 创建数据集和数据加载器
    if world_size > 1:
        # 分布式训练
        train_dataset = HandwritingWordDataset(
            config['split_json'],
            split='train',
            img_height=config['img_height'],
            img_width=config['img_width'],
            max_word_length=config['max_word_length']
        )
        char_to_idx = train_dataset.char_to_idx
        idx_to_char = train_dataset.idx_to_char
        vocab_size = train_dataset.vocab_size

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
    else:
        # 单机训练
        train_loader, char_to_idx, vocab_size = create_dataloaders(config)
        idx_to_char = {v: k for k, v in char_to_idx.items()}

    # 创建测试数据加载器
    test_loader = create_test_dataloader(config, char_to_idx)

    if rank == 0:
        print(f'训练集大小: {len(train_loader.dataset)}')
        print(f'测试集大小: {len(test_loader.dataset)}')
        print(f'词汇表大小: {vocab_size}')

    # 创建模型
    model = HandwritingResNetCTC(
        vocab_size=vocab_size,
        resnet_type=config['resnet_type'],
        feature_dim=config['feature_dim'],
        freeze_backbone=config['freeze_backbone'],
        dropout=config['dropout']
    ).to(device)

    # 打印模型参数信息（只在主进程）
    if rank == 0:
        model.print_model_info()

    # 使用DDP包装模型（如果是分布式训练）
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    warmup_epochs = config.get('warmup_epochs', 5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - warmup_epochs,
        eta_min=config['learning_rate'] * 0.01
    )

    # 训练循环
    best_test_acc = 0.0

    if rank == 0:
        print("\n开始训练...")

    for epoch in range(config['num_epochs']):
        # Warmup学习率
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate'] * lr_scale

        # 设置采样器的epoch（分布式训练）
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f'\n=== Epoch {epoch + 1}/{config["num_epochs"]} ===')

        # 训练
        train_loss, train_acc, train_char_acc = train_epoch(
            model, train_loader, optimizer, device, epoch + 1, idx_to_char, rank
        )

        # 调整学习率
        if epoch >= warmup_epochs:
            scheduler.step()

        if rank == 0:
            # 测试
            test_acc, test_char_acc, test_edit_dist = test_model(
                model, test_loader, device, idx_to_char, rank
            )

            print(f'训练 - 损失: {train_loss:.4f}, 单词准确率: {train_acc:.4f}, 字母准确率: {train_char_acc:.4f}')
            print(f'测试 - 编辑距离: {test_edit_dist:.4f}, 单词准确率: {test_acc:.4f}, 字母准确率: {test_char_acc:.4f}')
            print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')

            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc

                # 获取模型状态（处理DDP）
                if hasattr(model, 'module'):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_acc': best_test_acc,
                    'config': config,
                    'char_to_idx': char_to_idx
                }, config['save_path'])
                print(f'保存最佳模型，测试准确率: {best_test_acc:.4f}')

        # 同步所有进程（分布式训练）
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print('\n训练完成！')
        print(f'最佳测试准确率: {best_test_acc:.4f}')

        # 进行最终的详细推理
        print(f"\n开始最终详细推理...")

        # 加载最佳模型
        checkpoint = torch.load(config['save_path'], map_location=device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # 执行详细推理并保存结果（生成LLM输入文件）
        results, final_acc, final_char_acc = inference(
            model, test_loader, device, idx_to_char, config['inference_output'],
            rank, generate_lm_input=config.get('generate_lm_input', True)
        )

        print(f"\n最终结果总结:")
        print(f"最佳测试准确率: {best_test_acc:.4f}")
        print(f"最终推理 - 单词准确率: {final_acc:.4f}, 字母准确率: {final_char_acc:.4f}")

    # 清理分布式环境
    if world_size > 1:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='训练手写识别ResNet + CTC模型')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['200word', 'cross_user', 'zero_shot'],
                        help='实验类型: 200word, cross_user, zero_shot')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='ResNet编码器类型')
    parser.add_argument('--config', type=str, default=None, help='额外配置文件路径（可选）')
    parser.add_argument('--gpus', type=int, default=None, help='使用的GPU数量（可选，默认使用所有GPU）')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小（可选）')
    args = parser.parse_args()

    # 获取实验配置
    config = get_experiment_config(args.experiment, args.encoder)
    config['experiment_name'] = f"{args.experiment}_{args.encoder}"

    # 更新配置参数
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    # 如果提供了额外配置文件，更新配置
    if args.config:
        with open(args.config, 'r') as f:
            extra_config = json.load(f)
            config.update(extra_config)
            print(f"已加载额外配置: {args.config}")

    # 获取可用的GPU数量
    if args.gpus:
        world_size = min(args.gpus, torch.cuda.device_count())
    else:
        world_size = torch.cuda.device_count()

    if world_size == 0:
        print("警告：没有可用的GPU，将使用CPU训练")
        world_size = 1

    # 只在主进程打印配置信息
    if not ('LOCAL_RANK' in os.environ) or int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(f"=" * 60)
        print(f"实验类型: {args.experiment}")
        print(f"使用 {world_size} 张GPU进行训练")
        print(f"模型类型: ResNet + CTC")
        print(f"ResNet配置: {args.encoder}")
        print(f"特征维度: {config['feature_dim']}")
        print(f"损失函数: CTCLoss")
        print(f"数据来源: JSON文件 ({config['split_json']})")
        print(f"=" * 60)

    # 启动训练
    if world_size > 1:
        # 多进程分布式训练（适用于torchrun）
        # 检查是否在torchrun环境中
        if 'LOCAL_RANK' in os.environ:
            # 使用torchrun提供的环境变量
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            run_training(local_rank, world_size, config)
        else:
            # 使用torch.multiprocessing.spawn
            import torch.multiprocessing as mp
            mp.spawn(
                run_training,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
    else:
        # 单机训练
        run_training(0, 1, config)


if __name__ == '__main__':
    main()