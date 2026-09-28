"""Auto-scan the deploy package for recognized inputs and outputs.

入参(全局):
- 模块函数均以部署包根目录为基准做约定结构扫描。

方法:
- 一键识别：SAM3 权重、Qwen 权重目录、文本 embedding 缓存、提示词文件、
  配对数据目录（含 image+label）、切片数据集目录（含 train.jsonl）、
  可视化目录、大图输入目录、最近一次训练输出（含 best_model 的时间戳目录）。

出参:
- scan_deploy(deploy_root) -> dict；各项均为 Path 或 None（未发现时不设置）。
"""

from __future__ import annotations

from pathlib import Path


def scan_deploy(deploy_root: Path) -> dict:
    """入参: 部署包根目录；方法: 按约定结构扫描并返回可识别项；出参: dict[str, Path|None]。"""
    root = Path(deploy_root)
    resources = root / "resources"
    datasets = root / "datasets"
    found: dict = {
        "deploy_root": root,
        "sam3": next((resources / name for name in ("sam3.pt", "sam3.pth") if (resources / name).is_file()), None),
        "qwen_dir": next((path for path in resources.iterdir() if path.is_dir() and (path / "config.json").is_file()) if resources.is_dir() else (), None),
        "embedding": next((path for path in resources.glob("*embedding*.pt")), None),
        "prompt_file": next((path for path in (resources / "prompts").glob("*.txt")), None) if (resources / "prompts").is_dir() else None,
        "pair_dir": None,
        "big_image_dir": next((datasets / name for name in ("大图输入", "big_images") if (datasets / name).is_dir()), None),
        "vis_dir": datasets / "VIS",
        "dataset_dir": None,
        "latest_run": None,
    }
    if datasets.is_dir():
        for child in sorted(datasets.iterdir()):
            if child.is_dir() and (child / "train.jsonl").is_file():
                found["dataset_dir"] = child
                break
        for child in sorted(datasets.iterdir()):
            if child.is_dir() and (child / "image").is_dir() and (child / "label").is_dir():
                found["pair_dir"] = child
                break
    exp_root = root / "runs" / "exp_calss_edge"
    if exp_root.is_dir():
        candidates = [path for path in exp_root.iterdir() if path.is_dir() and (path / "best_model.torch").is_file()]
        if candidates:
            found["latest_run"] = max(candidates, key=lambda path: path.stat().st_mtime)
    return found


def resolve_weight(value: str | Path, prefer: str | None = None) -> Path:
    """
    入参:
    - value: 权重文件路径或文件夹路径。
    - prefer: 文件夹内优先匹配的文件名子串（如 "sam3"、"embedding"）。

    方法:
    - 参数指向文件时直接校验存在并返回；
    - 指向文件夹时自动扫描其中 .pt/.pth：优先 prefer 子串匹配，
      无匹配则取第一个权重文件；找不到时报清晰错误。

    出参:
    - Path: 解析后的权重文件路径。
    """
    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth"))
        if prefer:
            matched = [candidate for candidate in candidates if prefer.lower() in candidate.name.lower()]
            if matched:
                return matched[0]
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"未在文件夹中找到权重文件（.pt/.pth）: {path}")
    raise FileNotFoundError(f"权重路径不存在: {path}（文件或文件夹均可）")
