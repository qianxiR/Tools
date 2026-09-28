"""
自训练 TransformerDecoder 检测管线的训练入口（仿 Change3D train_CC_raw.py）。

做什么:
    - 直接读现有 pipe_qwenvl_{train,val,test}.jsonl（磁盘仍 JSON），解析 <obj>[...]</obj>
      → 目标列表 → 自建词表编码为 token id 序列。不重新生成 JSONL。
    - 数据增强复用 model.data_augment.flip_bbox + _parse_assistant（格式无关，操作 bbox 整数）。
    - 损失 = CrossEntropyLoss(ignore_index=<pad>=0)，shifted CE（去 <start>），pack_padded_sequence 对齐。
    - val 评估回调：每 VAL_EVAL_INTERVAL epoch 对 val 子集逐图贪心解码 → 解析 → IoU 匹配 → 按 F1 存 best。

为什么不复用 trl/transformers Trainer:
    自训练 decoder 的 forward 签名（memory + caps + caplens + pack_padded_sequence）与 HF Trainer
    的 (input_ids, labels) 范式不兼容，且 pack_padded_sequence 要求按长度排序。原生循环更清晰。

环境变量（与 train_qwen3vl_lora.py 风格一致，全部可覆盖）:
    SIGLIP_MODEL_PATH / DATASET_ROOT / TRAIN_JSONL / VAL_JSONL / PROMPT_JSON / WORDMAP_JSON
    OUTPUT_DIR / EMBED_DIM / N_HEAD / N_LAYER / DROPOUT / LEARNING_RATE / WEIGHT_DECAY
    BATCH_SIZE / NUM_WORKERS / NUM_TRAIN_EPOCHS / GRAD_CLIP / AUGMENT_TRAIN
    VAL_EVAL_INTERVAL / VAL_EVAL_SIZE / F1_EARLY_STOPPING_PATIENCE
    SEED / DEVICE / DTYPE

启动（遵循全局规则，脱离会话）:
    cd F:/xzrsagent/VLLM/QwenDec
    nohup python train_custom_decoder.py >> runs/logs/custom_decoder.out 2>&1 &
    echo "nohup launched, pid=$!"
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence

from model.custom_decoder import SiglipDecoderDetector
from model.custom_vocab import (
    END_INDEX,
    PAD_INDEX,
    START_INDEX,
    TOKENS_PER_TARGET,
    build_word_map,
    encode_caption,
    load_word_map,
    parse_token_sequence,
    rev_word_map,
    save_word_map,
    tokens_to_text,
)
from model.data_augment import _parse_assistant, flip_bbox
from model.prompt_loader import load_label_codes
from detection_metrics import box_iou, match_image, precision_recall_f1

# ----------------------------------------------------------------------------
# 路径与超参（env 覆盖）
# ----------------------------------------------------------------------------
SIGLIP_MODEL_PATH = os.environ.get(
    "SIGLIP_MODEL_PATH",
    r"C:\Users\Administrator\.cache\modelscope\hub\models\google\siglip2-base-patch16-224",
)
DATASET_ROOT = os.environ.get("DATASET_ROOT", r"F:\数据集\GW\数据_筛选3000")
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", os.path.join(DATASET_ROOT, "pipe_qwenvl_train.jsonl"))
VAL_JSONL = os.environ.get("VAL_JSONL", os.path.join(DATASET_ROOT, "pipe_qwenvl_val.jsonl"))
PROMPT_JSON = os.environ.get("PROMPT_JSON", "data/pipe_defect_3000_prompt_rawint.json")
WORDMAP_JSON = os.environ.get("WORDMAP_JSON", "data/pipe_defect_wordmap.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/pipe_defect_custom_decoder")

EMBED_DIM = int(os.environ.get("EMBED_DIM", "256"))
N_HEAD = int(os.environ.get("N_HEAD", "8"))
N_LAYER = int(os.environ.get("N_LAYER", "3"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))  # Windows 友好
NUM_TRAIN_EPOCHS = int(os.environ.get("NUM_TRAIN_EPOCHS", "100"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "5.0"))
AUGMENT_TRAIN = int(os.environ.get("AUGMENT_TRAIN", "1"))  # 1=开启水平翻转
VAL_EVAL_INTERVAL = int(os.environ.get("VAL_EVAL_INTERVAL", "3"))
VAL_EVAL_SIZE = int(os.environ.get("VAL_EVAL_SIZE", "300"))  # <=0 表示全量
F1_EARLY_STOPPING_PATIENCE = int(os.environ.get("F1_EARLY_STOPPING_PATIENCE", "0"))  # 0=关闭
SEED = int(os.environ.get("SEED", "42"))
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE_NAME = os.environ.get("DTYPE", "bf16")
MAX_LEN = 1 + 13 * TOKENS_PER_TARGET + 1  # 67；留余量在 encode_caption 内部截断保护
VAL_IOU_THRESHOLD = 0.5

DTYPE = torch.bfloat16 if DTYPE_NAME == "bf16" else torch.float32


# ----------------------------------------------------------------------------
# 工具：读取 JSONL、解析 assistant、坐标 0-1000 → 像素（评估用）
# ----------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """入参: jsonl 路径。出参: record 列表。"""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_targets(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """入参: record。方法: 解析 conversations[1]["value"] 的 <obj>[...]</obj> → 目标列表。
    出参: [{"bbox_2d":[...],"label":...}]。"""
    return _parse_assistant(record["conversations"][1]["value"])


def get_image(record: Dict[str, Any]) -> Any:
    """入参: record。出参: 图像字段（路径 str 或 PIL，augment 后是 PIL）。"""
    return record["image"]


def ensure_or_build_wordmap(label_codes: List[str]) -> Dict[str, int]:
    """入参: 类别码列表。方法: 若 WORDMAP_JSON 存在则加载，否则构造并保存。
    出参: word_map dict。"""
    p = Path(WORDMAP_JSON)
    if p.exists():
        return load_word_map(p)
    wm = build_word_map(label_codes)
    save_word_map(wm, p)
    print(f"[vocab] built and saved to {p} (size={len(wm)})")
    return wm


# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------
class PipeDefectTokenDataset(Dataset):
    """读 JSONL → 解析目标 → 自建词表编码为定长 token id 序列。
    返回 (image, caption_ids, caplen) 三元组；image 为路径 str 或 PIL。"""

    def __init__(
        self,
        jsonl_path: str,
        word_map: Dict[str, int],
        augment: bool = False,
        augment_p: float = 0.5,
        max_len: int = MAX_LEN,
    ):
        self.records = load_jsonl(jsonl_path)
        self.word_map = word_map
        self.augment = augment
        self.augment_p = augment_p
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        # 数据增强：水平翻转（与现有 Qwen 链一致，操作 0-1000 bbox 整数，格式无关）
        if self.augment and random.random() < self.augment_p:
            record = self._hflip_record(record)
        targets = get_targets(record)
        ids, caplen = encode_caption(self.word_map, targets, max_len=self.max_len)
        image = get_image(record)
        caption = torch.tensor(ids, dtype=torch.long)
        caplen_t = torch.tensor([caplen], dtype=torch.long)
        return image, caption, caplen_t

    @staticmethod
    def _hflip_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """入参: record。方法: 读图水平翻转 + 同步翻转 bbox（复用 flip_bbox），
              返回深拷贝并替换 image(PIL) 与 conversations[1]["value"]（保持与现链一致）。
        出参: 翻转后的 record。"""
        img_field = record["image"]
        if isinstance(img_field, Image.Image):
            pil = img_field.convert("RGB")
        else:
            pil = Image.open(img_field).convert("RGB")
        flipped = pil.transpose(Image.FLIP_LEFT_RIGHT)
        targets = get_targets(record)
        flipped_targets = flip_bbox(targets)
        # 重新序列化为 <obj>[...]</obj>（保持 record 完整性；编码阶段会再解析）
        body = json.dumps(flipped_targets, ensure_ascii=False, separators=(",", ":"))
        new_record = copy.deepcopy(record)
        new_record["image"] = flipped
        new_record["conversations"][1]["value"] = f"<obj>{body}</obj>"
        return new_record


def collate_fn(batch):
    """入参: list of (image, caption, caplen)。
    方法: image 保持 list（Siglip extractor 接受 list）；caption/caplen stack。
    出参: (images_list, captions [B,L], caplens [B,1])。"""
    images = [b[0] for b in batch]
    captions = torch.stack([b[1] for b in batch], dim=0)
    caplens = torch.stack([b[2] for b in batch], dim=0)
    return images, captions, caplens


# ----------------------------------------------------------------------------
# 推理：贪心解码（训练 val 回调与 inference_custom 共用）
# ----------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(
    model: SiglipDecoderDetector,
    images,
    max_len: int = MAX_LEN,
    device: str = DEVICE,
) -> List[List[int]]:
    """入参: model, images (list), max_len。
    方法: 以 <start> 起步逐 token argmax，遇 <end> 该样本停止，max_len 截断。
    出参: list[List[int]]，每样本解码出的 token id 序列（含 <start>，到 <end> 截止）。"""
    model.eval()
    memory = model.encode_memory(images)  # (S, B, C)
    B = memory.size(1)
    decoder = model.decoder

    tgt = torch.full((B, max_len), PAD_INDEX, dtype=torch.long, device=device)
    tgt[:, 0] = START_INDEX
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    last_step = 1
    for step in range(1, max_len):
        tgt_in = tgt[:, : step + 1].permute(1, 0)  # (step+1, B)
        L = tgt_in.size(0)
        mask = (torch.triu(torch.ones(L, L)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device).to(memory.dtype)
        tgt_emb = decoder.position_encoding(decoder.vocab_embedding(tgt_in))
        pred = decoder.transformer(tgt_emb, memory, tgt_mask=mask)  # (L, B, V)
        logits = decoder.wdc(pred)[-1]  # (B, V)
        next_tok = logits.argmax(-1)  # (B,)
        tgt[~finished, step] = next_tok[~finished]
        finished = finished | (next_tok == END_INDEX)
        last_step = step + 1
        if finished.all():
            break
    # 截到每样本真实长度（到 <end> 或 last_step）
    out: List[List[int]] = []
    for i in range(B):
        seq = tgt[i].tolist()
        # 去尾部 pad
        try:
            end_pos = seq.index(END_INDEX, 1)
            seq = seq[: end_pos + 1]
        except ValueError:
            seq = seq[:last_step]
        out.append(seq)
    return out


def predict_targets(
    model: SiglipDecoderDetector,
    image,
    word_map: Dict[str, int],
    max_len: int = MAX_LEN,
    device: str = DEVICE,
) -> List[Dict[str, Any]]:
    """入参: model, 单图（路径或 PIL）, word_map。
    出参: [{"bbox_2d":[...],"label":...}]（0-1000 坐标）。"""
    images = [image] if not isinstance(image, list) else image
    seqs = greedy_decode(model, images, max_len=max_len, device=device)
    rev = rev_word_map(word_map)
    return parse_token_sequence(seqs[0], rev, word_map)


def denormalize_box(box: List[int], width: int, height: int) -> List[int]:
    """入参: box=[x1,y1,x2,y2] (0-1000)，原图宽高。出参: 像素 xyxy（与 inference.py 同实现）。"""
    x1 = max(0, min(width, int(round(box[0] / 1000.0 * width))))
    y1 = max(0, min(height, int(round(box[1] / 1000.0 * height))))
    x2 = max(0, min(width, int(round(box[2] / 1000.0 * width))))
    y2 = max(0, min(height, int(round(box[3] / 1000.0 * height))))
    return [x1, y1, x2, y2]


# ----------------------------------------------------------------------------
# val F1 评估
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_val_f1(
    model: SiglipDecoderDetector,
    val_records: List[Dict[str, Any]],
    word_map: Dict[str, int],
    iou_threshold: float = VAL_IOU_THRESHOLD,
    max_eval: int = VAL_EVAL_SIZE,
    device: str = DEVICE,
) -> Dict[str, float]:
    """入参: model, val records, word_map。
    方法: 对（最多 max_eval 张）val 图逐图贪心解码 → 解析 → 0-1000 坐标还原像素 → IoU 匹配。
    出参: {"precision","recall","f1","tp","fp","fn"}。"""
    model.eval()
    rev = rev_word_map(word_map)
    n = len(val_records) if max_eval <= 0 else min(max_eval, len(val_records))
    tp = fp = fn = 0
    for i in range(n):
        rec = val_records[i]
        image = get_image(rec)
        gt_targets = get_targets(rec)
        # 推理时 image 必须是 PIL（extractor 接受路径，但 augment 后可能存 PIL）
        if not isinstance(image, Image.Image):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")
        W, H = pil.size
        seq = greedy_decode(model, [pil], device=device)[0]
        pred_norm = parse_token_sequence(seq, rev, word_map)
        pred_px = [denormalize_box(t["bbox_2d"], W, H) for t in pred_norm]
        pred_labels = [t["label"] for t in pred_norm]
        gt_px = [denormalize_box(t["bbox_2d"], W, H) for t in gt_targets]
        gt_labels = [t["label"] for t in gt_targets]
        m = match_image(pred_px, gt_px, pred_labels, gt_labels, iou_threshold)
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
    prf = precision_recall_f1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, **prf}


# ----------------------------------------------------------------------------
# 训练循环
# ----------------------------------------------------------------------------
def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    """入参: optimizer, grad_clip。方法: nn.utils.clip_grad_norm_。出参: 无。"""
    if grad_clip and grad_clip > 0:
        nn.utils.clip_grad_norm_([p for grp in optimizer.param_groups for p in grp["params"]], grad_clip)


def train_one_epoch(
    epoch: int,
    train_loader: DataLoader,
    model: SiglipDecoderDetector,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: str,
    print_freq: int = 20,
) -> float:
    """入参: epoch, train_loader, model, criterion, optimizer。
    方法: 单 epoch 训练——forward → shifted CE（去 <start>）→ pack_padded_sequence 对齐 → backward。
    出参: 平均 loss。"""
    model.train()
    running_loss = 0.0
    n_steps = 0
    t0 = time.time()
    for i, (images, caps, caplens) in enumerate(train_loader):
        caps = caps.to(device)
        caplens = caplens.to(device)

        pred, caps_sorted, decode_lengths, _sort_ind = model(images, caps, caplens)
        # pred: (B, L, V)；targets 去掉 <start>：caps_sorted[:, 1:]
        targets = caps_sorted[:, 1:]

        # pack_padded_sequence 要求每条 decode_length <= L-1
        decode_lengths = [max(1, min(dl, pred.size(1))) for dl in decode_lengths]

        scores = pack_padded_sequence(pred, decode_lengths, batch_first=True).data
        tgt_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = criterion(scores, tgt_packed)

        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, grad_clip)
        optimizer.step()

        running_loss += loss.item()
        n_steps += 1
        if (i % print_freq) == 0:
            print(
                f"[epoch {epoch}] step {i}/{len(train_loader)} "
                f"loss={loss.item():.4f} avg={running_loss/n_steps:.4f} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    return running_loss / max(1, n_steps)


def save_checkpoint(
    model: SiglipDecoderDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_f1: float,
    word_map: Dict[str, int],
    path: str,
) -> None:
    """入参: 各训练状态。方法: 落盘 checkpoint（含 model state、optimizer、epoch、best_f1、word_map）。
    出参: 无。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "word_map": word_map,
            "embed_dim": EMBED_DIM,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT,
        },
        path,
    )


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 1) 词表
    label_codes = load_label_codes(Path(PROMPT_JSON))
    word_map = ensure_or_build_wordmap(label_codes)
    vocab_size = len(word_map)
    print(f"[vocab] size={vocab_size}, label_codes={label_codes}")

    # 2) 数据
    train_dataset = PipeDefectTokenDataset(
        TRAIN_JSONL, word_map, augment=bool(AUGMENT_TRAIN), max_len=MAX_LEN,
    )
    val_records = load_jsonl(VAL_JSONL)
    print(f"[data] train={len(train_dataset)} val={len(val_records)}")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=False,
    )

    # 3) 模型
    model = SiglipDecoderDetector(
        siglip_path=SIGLIP_MODEL_PATH, vocab_size=vocab_size,
        embed_dim=EMBED_DIM, n_head=N_HEAD, n_layer=N_LAYER, dropout=DROPOUT,
        device=DEVICE, dtype=DTYPE,
    ).to(device=DEVICE, dtype=DTYPE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_train/1e6:.2f}M")

    # 4) 优化器 + 损失
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    # 5) 训练循环 + val F1 回调
    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"[train] start: epochs={NUM_TRAIN_EPOCHS} batch={BATCH_SIZE} lr={LEARNING_RATE}")

    for epoch in range(1, NUM_TRAIN_EPOCHS + 1):
        avg_loss = train_one_epoch(
            epoch, train_loader, model, criterion, optimizer, GRAD_CLIP, DEVICE,
        )
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}", flush=True)

        # val 评估（降频）
        do_eval = (epoch % VAL_EVAL_INTERVAL == 0) or (epoch == NUM_TRAIN_EPOCHS)
        if do_eval:
            metrics = evaluate_val_f1(model, val_records, word_map, device=DEVICE)
            cur_f1 = metrics["f1"]
            print(
                f"[val @epoch {epoch}] P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} F1={cur_f1:.4f} "
                f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']}",
                flush=True,
            )
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(
                    model, optimizer, epoch, best_f1, word_map,
                    os.path.join(OUTPUT_DIR, "best_detector.pt"),
                )
                print(f"[checkpoint] new best F1={best_f1:.4f} saved (epoch {epoch})", flush=True)
            else:
                no_improve += 1
                print(
                    f"[val] no improve ({no_improve}/{F1_EARLY_STOPPING_PATIENCE}); "
                    f"best F1={best_f1:.4f} @epoch {best_epoch}",
                    flush=True,
                )
            # 周期 checkpoint（每 10 epoch 或最后一轮）
            if epoch % 10 == 0 or epoch == NUM_TRAIN_EPOCHS:
                save_checkpoint(
                    model, optimizer, epoch, best_f1, word_map,
                    os.path.join(OUTPUT_DIR, f"checkpoint-epoch{epoch}.pt"),
                )

            if F1_EARLY_STOPPING_PATIENCE > 0 and no_improve >= F1_EARLY_STOPPING_PATIENCE:
                print(
                    f"[early-stop] {F1_EARLY_STOPPING_PATIENCE} evals without improvement. "
                    f"Stopping at epoch {epoch}.",
                    flush=True,
                )
                break

    print(
        f"[done] best F1={best_f1:.4f} @epoch {best_epoch}, "
        f"best ckpt={os.path.join(OUTPUT_DIR, 'best_detector.pt')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
