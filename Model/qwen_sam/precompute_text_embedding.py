"""Precompute the Qwen3-VL text embedding cache for the classification prompt.

入参(全局):
- --qwen-path: Qwen3-VL-4B-Instruct 权重目录。
- --prompt: 提示词文本（或 --prompt-file 从 UTF-8 文件读取）。
- --out: 缓存输出路径（.pt）。

方法:
- 用 Qwen3VLForConditionalGeneration 对提示词做纯文本前向（output_hidden_states），
  取语言侧最后层 hidden[-1] [1,T,D_t] 与 attention_mask 一并缓存；
  网络端 text_proj 将 D_t(=2560) 投影到解码器维度，训练/推理不再加载 Qwen 权重。

出参:
- 无；缓存写 --out（含 prompt 原文便于复核）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def _guard_getpass() -> None:
    """
    入参: 无。

    方法:
    - Windows 下若 LOGNAME/USER/LNAME/USERNAME 全部缺失，getpass.getuser()
      会走 POSIX 分支 import pwd 直接崩溃；torch._dynamo 初始化会调用它，
      故在此兜底 hook，保证任何环境变量组合下返回可用用户名。

    出参: 无（仅修饰 getpass 模块）。
    """
    try:
        import getpass

        if getattr(getpass, "_qwensam_patched", False):
            return
        original = getpass.getuser

        def safe_getuser():
            try:
                return original()
            except Exception:
                return os.environ.get("USERNAME") or os.environ.get("USER") or "user"

        getpass.getuser = safe_getuser
        getpass._qwensam_patched = True
    except Exception:
        pass


def encode_prompt(qwen_path: str, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    入参:
    - qwen_path: Qwen 权重目录。
    - prompt_text: 提示词文本。

    方法:
    - 加载 Qwen3VLForConditionalGeneration（fp16, eval），tokenizer 直接编码文本
      （不加 chat 模板，与运行期一致），前向取 hidden_states[-1] 与 attention_mask。

    出参:
    - tuple: (hidden [1,T,D] cpu, mask [1,T] cpu bool, hidden_size int)。
    """
    _guard_getpass()
    qwen_dir = Path(qwen_path)
    if not qwen_dir.is_dir() or not (qwen_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"[缺失] 未找到 Qwen 权重目录: {qwen_dir}\n"
            f"请把 Qwen3-VL-4B-Instruct 权重放入该目录（需含 config.json 与 *.safetensors），"
            f"或修改 --qwen-path 指向实际权重目录。"
        )
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        qwen_path, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    with torch.no_grad():
        toks = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        toks = {key: value.to(device) for key, value in toks.items()}
        output = model(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        sequence = output.hidden_states[-1].cpu()
        mask = toks["attention_mask"].bool().cpu()
    hidden_size = int(sequence.shape[-1])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sequence, mask, hidden_size


def main() -> None:
    """入参: CLI；方法: 编码提示词并写缓存；出参: 无。"""
    parser = argparse.ArgumentParser(description="Precompute Qwen3-VL text embedding cache")
    parser.add_argument("--qwen-path", default=r"E:\rqx\pre\Qwen3-VL-4B-Instruct", help="Qwen3-VL 权重目录")
    parser.add_argument("--prompt", default=None, help="提示词文本；与 --prompt-file 二选一")
    parser.add_argument("--prompt-file", default=None, help="UTF-8 提示词文件")
    parser.add_argument("--out", required=True, help="缓存输出路径 .pt")
    args = parser.parse_args()
    if args.prompt and args.prompt_file:
        parser.error("--prompt and --prompt-file are mutually exclusive")
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt_text = args.prompt
    else:
        parser.error("either --prompt or --prompt-file is required")
    hidden, mask, hidden_size = encode_prompt(args.qwen_path, prompt_text)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": hidden, "mask": mask, "hidden_size": hidden_size, "prompt": prompt_text}, out_path)
    print(f"[TextEmbedding] tokens={hidden.shape[1]}, hidden_size={hidden_size} -> {out_path}")


if __name__ == "__main__":
    main()
