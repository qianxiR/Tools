#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 QA 问答内容的词云图（支持 CD / SEG 模式）

数据来源: {data_root}/{split}/qa_{split}.json（与 visualize_vqa.py 一致）

用法:
  python utils/generate_wordcloud.py --mode seg --data_root "E:/xzkjxm/dataes/CD_20" --split all
  python utils/generate_wordcloud.py --mode cd --data_root "E:/xzkjxm/dataes/CD-CD_20" --split test
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import jieba
from wordcloud import WordCloud


# ═══════════════════════════════════════════════════════════════
#  数据加载与文本提取
# ═══════════════════════════════════════════════════════════════

def load_qa_json(qa_path: Path) -> list:
    """
    入参: qa_path — qa_{split}.json 的路径
    方法: 加载 JSON，过滤有效样本（有 qa 字段且无 error）
    出参: 有效样本列表
    """
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [s for s in data if s.get("qa") and not s.get("error")]


def extract_text(samples: list) -> str:
    """
    入参: samples — load_qa_json 返回的有效样本列表
    方法: 遍历每个样本的 qa.qa 列表，提取 question + answer + mcq options 文本
    出参: 合并后的原始文本字符串
    """
    texts = []
    for sample in samples:
        qa_record = sample["qa"]
        for item in qa_record.get("qa", []):
            question = item.get("question", "")
            answer = item.get("answer", "")
            if question:
                texts.append(question)
            if answer:
                texts.append(answer)
            # MCQ: 提取选项文本作为额外语料
            for opt_val in item.get("options", {}).values():
                if opt_val:
                    texts.append(str(opt_val))
    return " ".join(texts)


# ═══════════════════════════════════════════════════════════════
#  分词与词云生成
# ═══════════════════════════════════════════════════════════════

def segment_text(text: str) -> str:
    """
    入参: text — 原始文本
    方法: jieba 分词，过滤停用词和长度 < 2 的词
    出参: 空格分隔的分词结果
    """
    stopwords = {
        "的", "了", "是", "在", "有", "和", "与", "于", "对", "为", "到", "从", "以", "及",
        "这", "那", "上", "下", "中", "内", "外", "前", "后", "左", "右", "里", "间", "个",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿",
        "什么", "怎么", "如何", "哪里", "哪个", "哪些", "多少", "几", "吗", "呢", "吧", "啊",
        "可以", "能", "会", "没有", "不", "也", "还", "就", "都", "而", "被", "把",
        "by", "and", "the", "to", "of", "in", "is", "it", "are", "was", "were", "be",
        "a", "an", "or", "but", "not", "this", "that", "these", "those", "have", "has",
        "based", "comparison", "temporal", "image", "pair", "detectable", "involving",
        "within", "observed", "area", "specific", "land", "cover", "class", "transformed",
        "original", "classification", "identified", "post", "change", "regional", "spatial",
        "extent", "surface", "coverage", "net", "expansion", "reduction", "quantitative",
        "percentage", "proportion", "represent", "types", "magnitude", "comprehensive",
        "description", "remote", "sensing", "confirm", "whether", "there", "been", "any",
        "you", "provided", "since", "appeared", "earlier", "their", "what", "when",
    }
    words = jieba.cut(text)
    filtered = [w for w in words if len(w) >= 2 and w.lower() not in stopwords]
    return " ".join(filtered)


def generate_wordcloud(text: str, output_path: Path):
    """
    入参: text — 分词后文本, output_path — 输出图片路径
    方法: 生成词云并保存，打印词频 Top 20
    出参: None
    """
    if not text.strip():
        print("  警告: 无有效文本，跳过词云生成")
        return

    wc = WordCloud(
        font_path="C:/Windows/Fonts/msyh.ttc",
        width=1200,
        height=800,
        background_color="white",
        max_words=200,
        max_font_size=150,
        random_state=42,
    ).generate(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(output_path))
    print(f"  词云已保存: {output_path}")

    # 词频统计
    counts = Counter(text.split())
    print("  Top 20 词频:")
    for word, cnt in counts.most_common(20):
        print(f"    {word}: {cnt}")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def process_split(data_root: Path, split: str, mode: str, output: Path):
    """
    入参:
        data_root — 数据集根目录
        split — 单个划分名 (train/val/test)
        mode — seg 或 cd
        output — 词云输出路径
    方法: 加载 qa_{split}.json → 提取文本 → 分词 → 生成词云
    出参: None
    """
    qa_path = data_root / split / f"qa_{split}.json"
    if not qa_path.exists():
        print(f"  跳过 {split}: {qa_path} 不存在")
        return

    samples = load_qa_json(qa_path)
    if not samples:
        print(f"  跳过 {split}: 无有效样本")
        return

    print(f"  [{split}] 加载 {len(samples)} 个样本")

    raw_text = extract_text(samples)
    print(f"  [{split}] 提取 {len(raw_text)} 字符")

    segmented = segment_text(raw_text)
    print(f"  [{split}] 分词 {len(segmented.split())} 个词")

    generate_wordcloud(segmented, output)


def main():
    """
    入参: 命令行参数 (mode, data_root, split, output)
    方法: 解析参数，支持 split=all 遍历 train/val/test，每个 split 生成独立词云
    出参: None
    """
    parser = argparse.ArgumentParser(description="生成 QA 词云图 (SEG/CD)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["seg", "cd"],
                        help="模式: seg=语义分割, cd=变化检测")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录")
    parser.add_argument("--split", type=str, default="test",
                        help="数据划分 (train/val/test/all, default: test)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认: {data_root}/{split}/wordcloud.png）")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    # split=all 展开为三个标准划分
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for sp in splits:
        output = Path(args.output) if args.output else data_root / sp / "wordcloud.png"
        print(f"\n{'='*50}")
        print(f"模式: {args.mode.upper()}, 划分: {sp}")
        print(f"{'='*50}")
        process_split(data_root, sp, args.mode, output)


if __name__ == "__main__":
    main()
