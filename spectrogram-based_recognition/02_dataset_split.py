import os
import re
import json
import random
import numpy as np
from collections import defaultdict

# 根目录
root_dir = "mmPencil_dataset/mmWave_HotMap"

# 输出目录
output_json_200 = "spectrogram-based_recognition/dataset_index/200-Word.json"
output_json_zero = "spectrogram-based_recognition/dataset_index/zero-shot.json"
output_json_user = "spectrogram-based_recognition/dataset_index/user-split.json"

# 创建输出目录
for json_path in [output_json_200, output_json_zero, output_json_user]:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

# 文件名解析
pattern = re.compile(r'(?P<experiment>[^_]+)_(?P<subject>[^_]+)_(?P<word>[^_]+)_(?P<trial>w\d+)\.npy')

# 加载文件
files = [f for f in os.listdir(root_dir) if f.endswith(".npy")]
parsed = []
for f in files:
    m = pattern.match(f)
    if m:
        parsed.append({
            "file": f,
            "filepath": os.path.join(root_dir, f),  # 保存完整路径
            "experiment": m.group("experiment"),
            "subject": m.group("subject"),
            "word": m.group("word"),
            "trial": m.group("trial")
        })


# ------------------------------
# 划分方法1: 每类 word → 20 train + 4 test
# ------------------------------
def split_by_trials(parsed_files, output_json):
    word_to_files = defaultdict(list)
    for item in parsed_files:
        if item["experiment"] == "200-Word":
            word_to_files[item["word"]].append(item)

    train_files = []
    test_files = []

    for word, flist in word_to_files.items():
        assert len(flist) == 24, f"Word {word} does not have 24 trials, got {len(flist)}"
        random.shuffle(flist)
        train_files.extend(flist[:20])
        test_files.extend(flist[20:])

    # 保存为JSON
    split_data = {
        "split_method": "by_trials",
        "description": "每类单词划分为20个训练样本和4个测试样本",
        "train": train_files,
        "test": test_files,
        "statistics": {
            "train_count": len(train_files),
            "test_count": len(test_files),
            "total_words": len(word_to_files),
            "train_per_word": 20,
            "test_per_word": 4
        }
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    print(f"划分完成: {output_json}")
    print("训练集文件数:", len(train_files))
    print("测试集文件数:", len(test_files))


# ------------------------------
# 划分方法2: 150类 → train, 50类 → test
# ------------------------------
def split_by_words(parsed_files, output_json):
    word_to_files = defaultdict(list)
    for item in parsed_files:
        if item["experiment"] == "200-Word":
            word_to_files[item["word"]].append(item)

    all_words = list(word_to_files.keys())
    assert len(all_words) == 200, f"Expect 200 words, got {len(all_words)}"
    random.shuffle(all_words)
    train_words = all_words[:150]
    test_words = all_words[150:]

    train_files = []
    test_files = []

    for w in train_words:
        train_files.extend(word_to_files[w])
    for w in test_words:
        test_files.extend(word_to_files[w])

    # 保存为JSON
    split_data = {
        "split_method": "by_words",
        "description": "150类单词用于训练,50类单词用于测试(zero-shot)",
        "train": train_files,
        "test": test_files,
        "train_words": train_words,
        "test_words": test_words,
        "statistics": {
            "train_count": len(train_files),
            "test_count": len(test_files),
            "train_words_count": len(train_words),
            "test_words_count": len(test_words)
        }
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    print(f"划分完成: {output_json}")
    print("训练集文件数:", len(train_files))
    print("测试集文件数:", len(test_files))


# ------------------------------
# 划分方法3: 按用户划分
# ------------------------------
def split_by_users(parsed_files, output_json):
    """
    将 experiment=="200-Word" 的样本按 subject 字段划分;
    User-01, User-02 → train; User-03, User-04 → test; 其余用户忽略。
    保存到JSON文件中。
    """
    train_subjects = {"User-01", "User-02"}
    test_subjects = {"User-03", "User-04"}

    train_files = []
    test_files = []

    for item in parsed_files:
        if item["experiment"] != "200-Word":
            continue
        subj = item["subject"]
        if subj in train_subjects:
            train_files.append(item)
        elif subj in test_subjects:
            test_files.append(item)
        else:
            continue  # 其余用户暂时忽略

    # 保存为JSON
    split_data = {
        "split_method": "by_users",
        "description": "按用户划分: User-01/02用于训练, User-03/04用于测试",
        "train": train_files,
        "test": test_files,
        "train_subjects": list(train_subjects),
        "test_subjects": list(test_subjects),
        "statistics": {
            "train_count": len(train_files),
            "test_count": len(test_files),
            "train_users": len(train_subjects),
            "test_users": len(test_subjects)
        }
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    print(f"用户划分完成: {output_json}")
    print("训练集文件数:", len(train_files))
    print("测试集文件数:", len(test_files))


if __name__ == "__main__":
    random.seed(42)  # 固定随机种子,保证可复现

    print("=" * 50)
    print("执行划分策略...")
    print("=" * 50)

    # 执行划分策略
    print("\n1. 按试验次数划分:")
    split_by_trials(parsed, output_json_200)

    print("\n2. 按单词类别划分 (zero-shot):")
    split_by_words(parsed, output_json_zero)

    print("\n3. 按用户划分:")
    split_by_users(parsed, output_json_user)

    print("\n" + "=" * 50)
    print("所有划分策略执行完成!")
    print("=" * 50)