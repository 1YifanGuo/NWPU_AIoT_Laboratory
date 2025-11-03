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
from torch.nn.utils.rnn import pad_sequence
import math
import json
import argparse
import editdistance
from tqdm import tqdm
from typing import List, Tuple


# ================== Conformer组件实现 ==================

class FeedForward(nn.Module):
    """Conformer的前馈网络模块"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(context)


class ConvolutionModule(nn.Module):
    """Conformer的卷积模块"""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # (batch, seq_len, d_model)


class ConformerBlock(nn.Module):
    """Conformer块：FF + MHSA + Conv + FF"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.conv_module = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Feed Forward 1
        residual = x
        x = self.norm1(x)
        x = self.ff1(x)
        x = self.dropout(x) * 0.5 + residual

        # Multi-Head Self Attention
        residual = x
        x = self.norm2(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x) + residual

        # Convolution Module
        residual = x
        x = self.norm3(x)
        x = self.conv_module(x)
        x = self.dropout(x) + residual

        # Feed Forward 2
        residual = x
        x = self.norm4(x)
        x = self.ff2(x)
        x = self.dropout(x) * 0.5 + residual

        return x


class ConformerEncoder(nn.Module):
    """Conformer编码器"""

    def __init__(self, input_dim: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, kernel_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ================== 数据加载模块 ==================

class MillimeterWaveDataset(Dataset):
    """毫米波数据集（从JSON文件加载）"""

    def __init__(self, json_file: str, split: str = 'train', max_length: int = 50):
        """
        Args:
            json_file: JSON划分文件路径
            split: 'train' 或 'test'
            max_length: 最大序列长度
        """
        self.json_file = json_file
        self.split = split
        self.max_length = max_length

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

            # 转换为token序列（CTC不需要BOS/EOS）
            token_ids = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in word]

            if len(token_ids) <= self.max_length:
                self.data_info.append({
                    'file_path': file_path,
                    'word': word,
                    'token_ids': token_ids
                })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        # 加载数据
        data = np.load(info['file_path'])  # shape: (seq_len, 180, 3)

        # 转换为tensor
        data = torch.FloatTensor(data)
        token_ids = torch.LongTensor(info['token_ids'])

        return {
            'data': data,
            'token_ids': token_ids,
            'input_length': data.shape[0],
            'target_length': len(token_ids),
            'word': info['word']
        }


class TestDataset(Dataset):
    """测试数据集（从JSON文件加载）"""

    def __init__(self, json_file: str, char_to_idx: dict):
        self.json_file = json_file
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}

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

            self.data_info.append({
                'file_path': file_path,
                'filename': filename,
                'word': word
            })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        # 加载数据
        data = np.load(info['file_path'])  # shape: (seq_len, 180, 3)
        data = torch.FloatTensor(data)

        return {
            'data': data,
            'input_length': data.shape[0],
            'filename': info['filename'],
            'ground_truth': info['word']
        }


def collate_fn(batch):
    """处理变长序列的批处理函数（训练用）"""
    # 按输入长度排序
    batch = sorted(batch, key=lambda x: x['input_length'], reverse=True)

    # 分别收集各个字段
    data = [item['data'] for item in batch]
    token_ids = [item['token_ids'] for item in batch]
    input_lengths = torch.LongTensor([item['input_length'] for item in batch])
    target_lengths = torch.LongTensor([item['target_length'] for item in batch])
    words = [item['word'] for item in batch]

    # Padding数据到相同长度
    max_input_len = max(d.shape[0] for d in data)

    # Pad input data
    padded_data = torch.zeros(len(batch), max_input_len, 180, 3)
    for i, d in enumerate(data):
        padded_data[i, :d.shape[0]] = d

    # 对于CTC，目标序列不需要padding，直接concatenate
    concatenated_targets = torch.cat(token_ids, dim=0)

    return {
        'data': padded_data,
        'targets': concatenated_targets,  # CTC targets (concatenated)
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
        'words': words
    }


def test_collate_fn(batch):
    """测试数据的批处理函数"""
    # 按输入长度排序
    batch = sorted(batch, key=lambda x: x['input_length'], reverse=True)

    data = [item['data'] for item in batch]
    input_lengths = torch.LongTensor([item['input_length'] for item in batch])
    filenames = [item['filename'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]

    # Padding数据到相同长度
    max_input_len = max(d.shape[0] for d in data)
    padded_data = torch.zeros(len(batch), max_input_len, 180, 3)
    for i, d in enumerate(data):
        padded_data[i, :d.shape[0]] = d

    return {
        'data': padded_data,
        'input_lengths': input_lengths,
        'filenames': filenames,
        'ground_truths': ground_truths
    }


# ================== 模型定义模块 ==================

class MillimeterWaveConformerCTC(nn.Module):
    """毫米波信号的Conformer + CTC模型"""

    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 2048, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 为每个通道创建独立的Conformer编码器
        self.distance_encoder = ConformerEncoder(180, d_model, n_layers // 2, n_heads, d_ff, kernel_size, dropout)
        self.azimuth_encoder = ConformerEncoder(180, d_model, n_layers // 2, n_heads, d_ff, kernel_size, dropout)
        self.elevation_encoder = ConformerEncoder(180, d_model, n_layers // 2, n_heads, d_ff, kernel_size, dropout)

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 最终的Conformer编码器
        self.final_encoder = ConformerEncoder(d_model, d_model, n_layers, n_heads, d_ff, kernel_size, dropout)

        # CTC分类头
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # CTC损失函数
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # 特殊token索引
        self.blank_idx = 0
        self.unk_idx = 1

    def create_mask(self, max_len, lengths):
        """创建序列掩码"""
        mask = torch.arange(max_len, device=lengths.device).expand(
            len(lengths), max_len
        ) < lengths.unsqueeze(1)
        return mask

    def count_parameters(self):
        """计算模型参数量"""

        def count_module_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        distance_params = count_module_params(self.distance_encoder)
        azimuth_params = count_module_params(self.azimuth_encoder)
        elevation_params = count_module_params(self.elevation_encoder)
        fusion_params = count_module_params(self.fusion_layer)
        final_params = count_module_params(self.final_encoder)
        ctc_params = count_module_params(self.ctc_head)

        encoder_params = distance_params + azimuth_params + elevation_params + final_params
        total_params = encoder_params + fusion_params + ctc_params

        return {
            'total_params': total_params,
            'encoder_params': encoder_params,
            'fusion_params': fusion_params,
            'ctc_params': ctc_params
        }

    def print_model_info(self):
        """打印模型信息"""
        param_info = self.count_parameters()

        print("=" * 60)
        print("模型参数统计 (Conformer + CTC)")
        print("=" * 60)
        print(f"总参数量: {param_info['total_params']:,} ({param_info['total_params'] / 1e6:.2f}M)")
        print(f"编码器参数: {param_info['encoder_params']:,} ({param_info['encoder_params'] / 1e6:.2f}M)")
        print(f"融合层参数: {param_info['fusion_params']:,} ({param_info['fusion_params'] / 1e6:.2f}M)")
        print(f"CTC层参数: {param_info['ctc_params']:,} ({param_info['ctc_params'] / 1e6:.2f}M)")
        print("=" * 60)

        return param_info

    def forward(self, x, target=None, input_lengths=None, target_lengths=None):
        """
        Args:
            x: (batch_size, seq_len, 180, 3)
            target: CTC目标序列 (concatenated targets)
            input_lengths: 输入序列长度
            target_lengths: 目标序列长度
        """
        batch_size, seq_len, freq_bins, channels = x.shape

        # 创建mask
        mask = None
        if input_lengths is not None:
            mask = self.create_mask(seq_len, input_lengths)

        # 分别提取各通道特征
        distance_features = self.distance_encoder(x[:, :, :, 0], mask)  # (batch_size, seq_len, d_model)
        azimuth_features = self.azimuth_encoder(x[:, :, :, 1], mask)  # (batch_size, seq_len, d_model)
        elevation_features = self.elevation_encoder(x[:, :, :, 2], mask)  # (batch_size, seq_len, d_model)

        # 特征融合
        fused_features = torch.cat([distance_features, azimuth_features, elevation_features], dim=-1)
        fused_features = self.fusion_layer(fused_features)

        # 最终编码
        encoded = self.final_encoder(fused_features, mask)

        # CTC分类
        logits = self.ctc_head(encoded)  # (batch_size, seq_len, vocab_size)

        # 如果是训练模式且提供了目标，计算CTC损失
        if self.training and target is not None and target_lengths is not None:
            # CTC需要的格式：(seq_len, batch_size, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            loss = self.ctc_loss(log_probs, target, input_lengths, target_lengths)
            return logits, loss, input_lengths
        else:
            return logits, input_lengths

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


def inference(model, dataloader, device, idx_to_char, output_file, rank=0):
    """批量推理并保存结果"""
    model.eval()
    results = []

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
                    'model_type': 'conformer_ctc'
                }
                results.append(result)

    if rank == 0:
        # 计算整体准确率
        total_correct = sum(1 for r in results if r['correct'])
        accuracy = total_correct / len(results) if results else 0

        # 计算字符级准确率
        all_preds = [r['prediction'] for r in results]
        all_gts = [r['ground_truth'] for r in results]
        char_acc, avg_edit_dist = calculate_edit_distance_accuracy(all_preds, all_gts)

        # 添加统计信息
        summary = {
            'total_samples': len(results),
            'correct_samples': total_correct,
            'word_accuracy': accuracy,
            'character_accuracy': char_acc,
            'average_edit_distance': avg_edit_dist,
            'model_type': 'conformer_ctc'
        }

        # 保存结果到jsonl文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 先写入统计信息
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')

            # 再写入详细结果
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"推理完成！结果已保存到 {output_file}")
        print(f"模型类型: Conformer + CTC")
        print(f"总样本数: {len(results)}")
        print(f"单词准确率: {accuracy:.4f} ({total_correct}/{len(results)})")
        print(f"字母准确率: {char_acc:.4f}")
        print(f"平均编辑距离: {avg_edit_dist:.4f}")

        return results, accuracy, char_acc

    return results, 0.0, 0.0


def create_dataloaders(config: dict, char_to_idx: dict = None):
    """创建训练数据加载器（从JSON文件加载）"""
    # 创建训练数据集
    train_dataset = MillimeterWaveDataset(config['split_json'], split='train')

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
    test_dataset = TestDataset(config['split_json'], char_to_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return test_loader


def get_experiment_config(experiment_type):
    """获取实验配置"""
    base_config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'd_model': 512,
        'n_layers': 8,
        'n_heads': 8,
        'd_ff': 1024,
        'kernel_size': 31,
        'dropout': 0.1,
        'num_workers': 4
    }

    experiment_configs = {
        '200word': {
            **base_config,
            'split_json': 'spectrogram-based_recognition/dataset_index/200-Word.json',
            'save_path': 'spectrogram-based_recognition/output/mmwave_conformer_ctc_model.pth',
            'inference_output': 'spectrogram-based_recognition/output/inference_results_conformer_ctc.jsonl'
        },
        'cross_user': {
            **base_config,
            'split_json': 'spectrogram-based_recognition/dataset_index/user-split.json',
            'save_path': 'spectrogram-based_recognition/output/mmwave_conformer_ctc_model_cross_user.pth',
            'inference_output': 'spectrogram-based_recognition/output/inference_results_conformer_ctc_cross_user.jsonl'
        },
        'zero_shot': {
            **base_config,
            'split_json': 'spectrogram-based_recognition/dataset_index/zero-shot.json',
            'save_path': 'spectrogram-based_recognition/output/mmwave_conformer_ctc_model_zero_shot.pth',
            'inference_output': 'spectrogram-based_recognition/output/inference_results_conformer_ctc_zero_shot.jsonl'
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
        print(f'Conformer配置: d_model={config["d_model"]}, n_layers={config["n_layers"]}')

    # 创建数据集和数据加载器
    if world_size > 1:
        # 分布式训练
        train_dataset = MillimeterWaveDataset(config['split_json'], split='train')
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
    model = MillimeterWaveConformerCTC(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout']
    ).to(device)

    # 打印模型参数信息（只在主进程）
    if rank == 0:
        model.print_model_info()

    # 使用DDP包装模型（如果是分布式训练）
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # 训练循环
    best_test_acc = 0.0

    if rank == 0:
        print("\n开始训练...")

    for epoch in range(config['num_epochs']):
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

        # 执行详细推理并保存结果
        results, final_acc, final_char_acc = inference(
            model, test_loader, device, idx_to_char, config['inference_output'], rank
        )

        print(f"\n最终结果总结:")
        print(f"最佳测试准确率: {best_test_acc:.4f}")
        print(f"最终推理 - 单词准确率: {final_acc:.4f}, 字母准确率: {final_char_acc:.4f}")

    # 清理分布式环境
    if world_size > 1:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='训练毫米波Conformer + CTC模型')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['200word', 'cross_user', 'zero_shot'],
                        help='实验类型: 200word, cross_user, zero_shot')
    parser.add_argument('--config', type=str, default=None, help='额外配置文件路径（可选）')
    parser.add_argument('--gpus', type=int, default=None, help='使用的GPU数量（可选，默认使用所有GPU）')
    parser.add_argument('--d_model', type=int, default=None, help='模型隐藏维度（可选）')
    parser.add_argument('--n_layers', type=int, default=None, help='Conformer层数（可选）')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小（可选）')
    args = parser.parse_args()

    # 获取实验配置
    config = get_experiment_config(args.experiment)
    config['experiment_name'] = args.experiment

    # 更新配置参数
    if args.d_model is not None:
        config['d_model'] = args.d_model
    if args.n_layers is not None:
        config['n_layers'] = args.n_layers
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
        print(f"模型类型: Conformer + CTC")
        print(f"模型配置: d_model={config['d_model']}, n_layers={config['n_layers']}, n_heads={config['n_heads']}")
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
            torch.multiprocessing.spawn(
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