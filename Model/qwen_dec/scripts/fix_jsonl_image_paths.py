# -*- coding: utf-8 -*-
"""批量修复管网数据集 JSONL 的 image 绝对路径前缀：F:\\数据集\\GW\\ -> F:\\管网\\。

为什么: 数据迁移到 F:/管网 后，JSONL 内 image 仍是旧盘绝对路径（F:\数据集\GW\...），
       训练/评估读图会 FileNotFoundError；仅替换前缀，其余字段与字节不变。
用法: python scripts/fix_jsonl_image_paths.py
"""
from pathlib import Path

OLD = "F:\\\\数据集\\\\GW\\\\"
NEW = "F:\\\\管网\\\\"
DATASETS = (
    Path(r"F:/管网/数据_筛选3000"),
    Path(r"F:/管网/数据_筛选500"),
)
JSONL_NAMES = ("pipe_qwenvl.jsonl", "pipe_qwenvl_train.jsonl",
               "pipe_qwenvl_val.jsonl", "pipe_qwenvl_test.jsonl")


def fix_file(path: Path) -> int:
    """入参: JSONL 文件路径。方法: 读文本替换旧前缀，写回；返回替换次数（0 表示无变化）。"""
    text = path.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count:
        path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return count


def main() -> None:
    """入参: 无。方法: 遍历两个数据集的全部 JSONL 做前缀替换并打印统计；校验首行可解析且图片存在。"""
    total = 0
    for ds in DATASETS:
        for name in JSONL_NAMES:
            path = ds / name
            if not path.is_file():
                continue
            count = fix_file(path)
            total += count
            print(f"{path}: {count} 处替换")
    # 校验：每个 train JSONL 首行 image 路径真实存在
    import json
    for ds in DATASETS:
        first = json.loads((ds / "pipe_qwenvl_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        ok = Path(first["image"]).is_file()
        print(f"校验 {ds.name} 首图存在: {ok}")
        if not ok:
            raise SystemExit("image 路径仍不可达，请检查")
    print(f"总计替换 {total} 处")


if __name__ == "__main__":
    main()