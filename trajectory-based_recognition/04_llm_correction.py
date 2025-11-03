import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re
from vllm import LLM, SamplingParams


def extract_corrected_word(text):
    """从LLM输出中提取纠正后的单词"""
    # 首先尝试从双引号中提取
    # 匹配 "word" 或 \"word\" 格式
    quote_patterns = [
        r'\\"([a-zA-Z]+)\\"',  # 匹配 \"word\"
        r'"([a-zA-Z]+)"',  # 匹配 "word"
        r"'([a-zA-Z]+)'",  # 匹配 'word'
    ]

    for pattern in quote_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 返回第一个匹配的单词，转为小写
            return matches[0].lower()

    # 如果没有引号，尝试提取第一个纯字母单词
    text = text.strip()
    words = re.findall(r'[a-zA-Z]+', text)
    if words:
        return words[0].lower()

    return ""


def load_vllm_inputs(input_path):
    """加载vLLM输入文件"""
    inputs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            inputs.append(item)
    return inputs


def run_vllm_correction(input_path, output_path, model_path,
                        batch_size=256, max_tokens=10, temperature=0.0):
    """运行vLLM批量纠正"""
    print(f"\n{'=' * 80}")
    print(f"vLLM批量纠正任务")
    print(f"{'=' * 80}")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"模型路径: {model_path}")
    print(f"批次大小: {batch_size}")
    print(f"{'=' * 80}\n")

    # 加载输入数据
    print("加载输入数据...")
    inputs = load_vllm_inputs(input_path)
    print(f"加载了 {len(inputs)} 条待纠正的数据\n")

    # 初始化vLLM
    print(f"初始化vLLM模型: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=2048
    )
    print("模型加载完成\n")

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
        stop=["\n", ".", "Corrected", "Input"]
    )

    # 准备prompts
    prompts = [item['prompt'] for item in inputs]

    # 批量推理
    print("开始批量推理...")
    outputs = llm.generate(prompts, sampling_params)
    print("推理完成\n")

    # 处理结果
    print("处理结果...")
    results = []
    raw_outputs = []  # 保存原始输出用于调试

    for item, output in tqdm(zip(inputs, outputs), total=len(inputs), desc="处理输出"):
        generated_text = output.outputs[0].text
        corrected_word = extract_corrected_word(generated_text)

        result = {
            'custom_id': item['custom_id'],
            'original_prediction': item['original_prediction'],
            'ground_truth': item['ground_truth'],
            'lm_output': generated_text,
            'corrected_word': corrected_word,
            'original_correct': (item['original_prediction'] == item['ground_truth']),
            'corrected_correct': (corrected_word == item['ground_truth'])
        }
        results.append(result)

        # 保存原始输出用于调试
        raw_output = {
            'custom_id': item['custom_id'],
            'prompt': item['prompt'],
            'original_prediction': item['original_prediction'],
            'ground_truth': item['ground_truth'],
            'lm_full_output': generated_text,
            'lm_output_length': len(generated_text),
            'extracted_word': corrected_word,
            'finish_reason': output.outputs[0].finish_reason,
            'tokens': output.outputs[0].token_ids if hasattr(output.outputs[0], 'token_ids') else None
        }
        raw_outputs.append(raw_output)

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"结果已保存到: {output_path}\n")

    # 保存原始LLM输出用于调试
    raw_output_path = output_path.parent / f"raw_llm_output_{output_path.stem}.jsonl"
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        for raw in raw_outputs:
            f.write(json.dumps(raw, ensure_ascii=False) + '\n')

    print(f"原始LLM输出已保存到: {raw_output_path}")
    print(f"可以查看此文件来调试提取逻辑\n")

    # 计算统计信息
    total = len(results)
    original_correct = sum(1 for r in results if r['original_correct'])
    corrected_correct = sum(1 for r in results if r['corrected_correct'])
    improved = sum(1 for r in results if not r['original_correct'] and r['corrected_correct'])
    degraded = sum(1 for r in results if r['original_correct'] and not r['corrected_correct'])

    original_acc = original_correct / total
    corrected_acc = corrected_correct / total
    improvement = corrected_acc - original_acc

    print(f"\n{'=' * 80}")
    print(f"纠正结果统计")
    print(f"{'=' * 80}")
    print(f"总样本数: {total}")
    print(f"原始正确数: {original_correct} ({original_acc:.4f}, {original_acc * 100:.2f}%)")
    print(f"纠正后正确数: {corrected_correct} ({corrected_acc:.4f}, {corrected_acc * 100:.2f}%)")
    print(f"准确率提升: {improvement:+.4f} ({improvement * 100:+.2f}%)")
    print(f"改进样本数: {improved}")
    print(f"退化样本数: {degraded}")
    print(f"净提升: {improved - degraded}")
    print(f"{'=' * 80}\n")

    # 保存统计信息
    stats = {
        'total_samples': total,
        'original_correct': original_correct,
        'corrected_correct': corrected_correct,
        'original_accuracy': original_acc,
        'corrected_accuracy': corrected_acc,
        'accuracy_improvement': improvement,
        'improved_samples': improved,
        'degraded_samples': degraded,
        'net_improvement': improved - degraded,
        'model_path': model_path,
        'batch_size': batch_size,
        'temperature': temperature
    }

    stats_path = output_path.parent / f"stats_corrected_{output_path.stem}.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"统计信息已保存到: {stats_path}")

    # 展示一些示例
    print(f"\n{'=' * 80}")
    print("纠正示例（前20个）:")
    print(f"{'真实':<12} {'原始预测':<12} {'LM纠正':<12} {'原始':<6} {'纠正':<6}")
    print(f"{'-' * 60}")
    for i, result in enumerate(results[:20]):
        orig_mark = '✓' if result['original_correct'] else '✗'
        corr_mark = '✓' if result['corrected_correct'] else '✗'
        print(f"{result['ground_truth']:<12} "
              f"{result['original_prediction']:<12} "
              f"{result['corrected_word']:<12} "
              f"{orig_mark:<6} {corr_mark:<6}")
    print(f"{'=' * 80}\n")

    # 分析改进和退化的案例
    improved_cases = [r for r in results if not r['original_correct'] and r['corrected_correct']]
    degraded_cases = [r for r in results if r['original_correct'] and not r['corrected_correct']]

    if improved_cases:
        print("改进案例示例（最多10个）:")
        print(f"{'真实':<12} {'原始预测':<15} {'LM纠正':<12}")
        print(f"{'-' * 45}")
        for case in improved_cases[:10]:
            print(f"{case['ground_truth']:<12} "
                  f"{case['original_prediction']:<15} "
                  f"{case['corrected_word']:<12}")
        print()

    if degraded_cases:
        print("退化案例示例（最多10个）:")
        print(f"{'真实':<12} {'原始预测':<15} {'LM纠正':<12}")
        print(f"{'-' * 45}")
        for case in degraded_cases[:10]:
            print(f"{case['ground_truth']:<12} "
                  f"{case['original_prediction']:<15} "
                  f"{case['corrected_word']:<12}")
        print()

    return stats


def main():
    parser = argparse.ArgumentParser(description='vLLM批量纠正脚本')
    parser.add_argument('--input', type=str, required=True,
                        help='vLLM输入文件路径（jsonl格式）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出文件路径（jsonl格式）')
    parser.add_argument('--model', type=str, default='models/Qwen3-0.6B',
                        help='语言模型路径')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--max-tokens', type=int, default=10,
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='采样温度（0.0表示贪心解码）')

    args = parser.parse_args()

    run_vllm_correction(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )


if __name__ == '__main__':
    main()