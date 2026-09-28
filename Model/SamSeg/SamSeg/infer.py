"""
SegEarth-OV3 通用底层模块
提供 SAM 3 模型加载、推理、后处理、FPN 特征提取等公共能力
供 infer_seg.py / infer_cd.py 等上层脚本调用

入参:
    各函数见 docstring

出参:
    可复用的工具函数和模型加载器
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label as scipy_label

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ═══════════════════════════════════════════════════════════════
#  统一类别 & 颜色定义 — 全项目唯一真值源
#  其他文件应 from infer import DEFAULT_CLASSES, NAME_COLOR
# ═══════════════════════════════════════════════════════════════

# 默认类别列表：每行一类，逗号分隔同义词；第 1 行 background 为 class 0（不发给模型）
DEFAULT_CLASSES = [
    "background",
    "building,house,roof,structure",
    "road,highway,street,pavement,expressway",
    "water,lake,river,pond,pool,sea",
    "bareland,barren,soil,ground,dirt,sand",
    "vegetation,forest,tree,grass,lawn,meadow,shrub,woodland",
    "farmland,agricultural,crop,farm,field,cropland,plantation",
]

# 公共类别颜色映射（语义分割 & 变化检测共用）
# background→白(255,255,255), building→红(255,0,0), road→紫(128,0,128),
# water→蓝(0,0,255), barren→褐(139,90,43), forest→绿(0,128,0), agricultural→深绿(0,100,0)
NAME_COLOR = {
    # building 类 — 红色
    "building":     (255, 0, 0),
    "house":        (255, 0, 0),
    "roof":         (255, 0, 0),
    "structure":    (255, 0, 0),
    # road 类 — 紫色
    "road":         (128, 0, 128),
    "highway":      (128, 0, 128),
    "street":       (128, 0, 128),
    "pavement":     (128, 0, 128),
    "expressway":   (128, 0, 128),
    # water 类 — 蓝色
    "water":        (0, 0, 255),
    "lake":         (0, 0, 255),
    "river":        (0, 0, 255),
    "pond":         (0, 0, 255),
    "pool":         (0, 0, 255),
    "sea":          (0, 0, 255),
    # bareland 类 — 褐色
    "bareland":     (139, 90, 43),
    "barren":       (139, 90, 43),
    "soil":         (139, 90, 43),
    "ground":       (139, 90, 43),
    "dirt":         (139, 90, 43),
    "sand":         (139, 90, 43),
    # vegetation 类 — 绿色
    "vegetation":   (0, 128, 0),
    "forest":       (0, 128, 0),
    "tree":         (0, 128, 0),
    "grass":        (0, 128, 0),
    "lawn":         (0, 128, 0),
    "meadow":       (0, 128, 0),
    "shrub":        (0, 128, 0),
    "woodland":     (0, 128, 0),
    "vegeation":    (0, 128, 0),     # 拼写容错
    # agricultural 类 — 深绿色
    "farmland":     (0, 100, 0),
    "agricultural": (0, 100, 0),
    "crop":         (0, 100, 0),
    "farm":         (0, 100, 0),
    "field":        (0, 100, 0),
    "cropland":     (0, 100, 0),
    "plantation":   (0, 100, 0),
}

# 默认备用调色板（未匹配到 NAME_COLOR 的类别使用）
DEFAULT_PALETTE = [
    [128, 128, 128], [0, 128, 128], [128, 0, 0], [0, 128, 0],
    [128, 128, 0],   [0, 0, 128],   [64, 0, 0],  [192, 0, 0],
    [64, 128, 0],    [192, 128, 0], [0, 64, 0],  [128, 64, 0],
    [0, 192, 0],     [128, 192, 0],
]


def build_palette(num_cls, cls_name_map):
    """
    入参: num_cls, cls_name_map (cls_id → [synonyms])
    方法: 按 NAME_COLOR 匹配类别颜色，未匹配的用 DEFAULT_PALETTE
    出参: palette (num_cls, 3) uint8
    """
    palette = np.zeros((num_cls, 3), dtype=np.uint8)
    default_ci = 0
    for cls_id in range(num_cls):
        if cls_id == 0:
            continue
        synonyms = cls_name_map.get(cls_id, [str(cls_id)])
        matched = False
        for syn in synonyms:
            if syn in NAME_COLOR:
                palette[cls_id] = NAME_COLOR[syn]
                matched = True
                break
        if not matched:
            palette[cls_id] = DEFAULT_PALETTE[default_ci % len(DEFAULT_PALETTE)]
            default_ci += 1
    return palette


def load_model(ckpt="sam3/sam3.pt", bpe="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
               conf=0.5, device=None):
    """
    入参:
        ckpt: SAM 3 权重路径
        bpe: BPE 词表路径
        conf: 置信度阈值
        device: 计算设备，None 则自动选择
    方法: 构建 SAM 3 模型并包装为 Sam3Processor，float32 模式
    出参: (processor, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam3_image_model(
        bpe_path=bpe, checkpoint_path=ckpt, device=device, load_from_HF=False,
    )
    model = model.to(device).float()
    processor = Sam3Processor(model, confidence_threshold=conf, device=device)
    return processor, device


def load_classes(path):
    """
    入参: path - 类别文件路径，每行一类，同义词逗号分隔
    方法: 逐行解析，展开同义词为独立 query；跳过第 1 行 (class 0 = background)
    出参: (query_words, query_idx, num_cls, num_queries)
    注意: class 0 (background) 不生成 query，argmax 时自动作为默认值
    """
    with open(path, "r") as f:
        lines = f.readlines()
    query_words, query_idx = [], []
    for idx, line in enumerate(lines):
        if idx == 0:
            # class 0 = background，不发文本提示
            continue
        synonyms = [w.strip() for w in line.split(",")]
        query_words.extend(synonyms)
        query_idx.extend([idx] * len(synonyms))
    num_cls = len(lines)  # 总类别数（含 background）
    num_queries = len(query_words)
    return query_words, query_idx, num_cls, num_queries


def inference_single_view(processor, query_words, num_queries, device, image):
    """
    入参:
        processor: Sam3Processor 实例
        query_words: 文本提示列表
        num_queries: 提示总数
        device: 计算设备
        image: PIL.Image
    方法: 双头融合推理（实例头+语义头+存在性过滤）
    出参: seg_logits - (num_queries, H, W)
    """
    w, h = image.size
    seg_logits = torch.zeros((num_queries, h, w), device=device)

    with torch.no_grad():
        state = processor.set_image(image)
        for qi, word in enumerate(query_words):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=word, state=state)

            if state["masks_logits"].shape[0] > 0:
                for inst_id in range(state["masks_logits"].shape[0]):
                    inst_logit = state["masks_logits"][inst_id].squeeze()
                    inst_score = state["object_score"][inst_id]
                    if inst_logit.shape != (h, w):
                        inst_logit = F.interpolate(
                            inst_logit.view(1, 1, *inst_logit.shape),
                            size=(h, w), mode="bilinear", align_corners=False,
                        ).squeeze()
                    seg_logits[qi] = torch.max(seg_logits[qi], inst_logit * inst_score)

            sem_logit = state["semantic_mask_logits"]
            if sem_logit.shape != (h, w):
                sem_logit = F.interpolate(
                    sem_logit, size=(h, w), mode="bilinear", align_corners=False,
                ).squeeze()
            seg_logits[qi] = torch.max(seg_logits[qi], sem_logit)
            seg_logits[qi] = seg_logits[qi] * state["presence_score"]

    return seg_logits


def _gaussian_weight_2d(h, w, sigma_scale=0.125):
    """
    生成 2D 高斯权重矩阵，中心权重最高、边缘衰减。
    sigma_scale: sigma = min(h, w) * sigma_scale
    出参: (h, w) tensor，值域 (0, 1]
    """
    y = torch.arange(h, dtype=torch.float32) - (h - 1) / 2.0
    x = torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sigma = min(h, w) * sigma_scale
    gauss = torch.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    return gauss


def slide_inference(processor, query_words, num_queries, device, image, stride, crop_size,
                    padding=32):
    """
    入参: 同 inference_single_view + stride, crop_size, padding
    方法:
        基于 Remote SAMsing 优化的滑动窗口推理:
        1. Contextual Padding — 每个 tile 向外扩展 padding 像素，确保边界对象被完整分割
        2. Gaussian Weight Blending — 重叠区域使用高斯权重融合，中心像素权重高，避免拼接缝
    出参: preds - (num_queries, H, W)
    """
    w_img, h_img = image.size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size

    preds = torch.zeros((num_queries, h_img, w_img), device=device)
    weight_mat = torch.zeros((1, h_img, w_img), device=device)
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    total = h_grids * w_grids

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Contextual Padding: 向外扩展 padding 像素
            pad_y1 = max(y1 - padding, 0)
            pad_x1 = max(x1 - padding, 0)
            pad_y2 = min(y2 + padding, h_img)
            pad_x2 = min(x2 + padding, w_img)

            # 推理使用 padded 区域
            pad_img = image.crop((pad_x1, pad_y1, pad_x2, pad_y2))
            pad_logits = inference_single_view(
                processor, query_words, num_queries, device, pad_img)

            # 裁剪回核心区域（丢弃 padding）
            inner_y1 = y1 - pad_y1
            inner_x1 = x1 - pad_x1
            inner_y2 = inner_y1 + (y2 - y1)
            inner_x2 = inner_x1 + (x2 - x1)
            crop_logits = pad_logits[:, inner_y1:inner_y2, inner_x1:inner_x2]

            # 高斯加权融合
            tile_h, tile_w = crop_logits.shape[1], crop_logits.shape[2]
            gauss = _gaussian_weight_2d(tile_h, tile_w).to(device)
            preds[:, y1:y2, x1:x2] += crop_logits * gauss.unsqueeze(0)
            weight_mat[:, y1:y2, x1:x2] += gauss.unsqueeze(0)

            done = h_idx * w_grids + w_idx + 1
            print(f"\r      滑动窗口 [{done}/{total}] ({x1},{y1})-({x2},{y2}) pad=({pad_x1},{pad_y1})-({pad_x2},{pad_y2})", end="", flush=True)
    print()  # 换行

    assert (weight_mat == 0).sum() == 0, "滑动窗口覆盖不完整"
    return preds / weight_mat


def run_inference(processor, query_words, num_queries, device, image,
                  slide=0, stride=512, tile_size=0, tile_overlap=0, padding=32):
    """
    入参:
        processor, 类别信息, device, image, 滑动窗口参数
        stride: 滑动窗口步长
        slide: 滑动窗口裁剪尺寸，0=自动（大图自动启用滑动窗口）
        padding: 上下文填充像素数（Remote SAMsing 优化）
    方法:
        slide > 0 或图像超过 1024 → 滑动窗口推理（含 contextual padding + 高斯融合）
        否则 → 单图推理
    出参: seg_logits - (num_queries, H, W)
    """
    w, h = image.size

    # 大图自动启用滑动窗口
    crop_size = slide if slide > 0 else stride
    if w > crop_size or h > crop_size:
        print(f"      滑动窗口模式: stride={stride}, crop={crop_size}, padding={padding}, 图像 {w}x{h}")
        return slide_inference(processor, query_words, num_queries, device,
                               image, stride, crop_size, padding=padding)

    # 单图推理
    print(f"      单图推理: 图像 {w}x{h}")
    return inference_single_view(processor, query_words, num_queries, device, image)


def multipass_inference(processor, query_words, query_idx_tensor, num_cls, num_queries,
                        device, image, prob=0.1, max_passes=10, coverage_target=0.95):
    """
    多轮迭代推理（基于 Remote SAMsing 策略）

    每轮:
      1. 把已分割区域涂黑（场景简化，残差区域更突出）
      2. 对简化后的图片做 SAM3 推理
      3. 只接受高置信度的新预测，填充未覆盖区域
      4. 阈值逐轮衰减（严格→宽松）

    入参:
        processor, query_words, query_idx_tensor, num_cls, num_queries, device, image
        prob: 初始置信度阈值
        max_passes: 最大迭代轮数
        coverage_target: 目标覆盖率（0-1）
    出参: seg_pred_np - (H, W) numpy int64, 像素值=类别ID（0=背景）
    """
    w, h = image.size
    img_np = np.array(image).copy()

    # 累积结果
    final_seg = np.zeros((h, w), dtype=np.int64)  # 最终类别图
    covered = np.zeros((h, w), dtype=bool)          # 已覆盖像素

    for p in range(max_passes):
        # 把已分割像素涂黑（场景简化）
        masked_np = img_np.copy()
        masked_np[covered] = 0
        masked_img = Image.fromarray(masked_np)

        # 单轮推理
        print(f"      --- Pass {p+1}/{max_passes} ---")
        seg_logits = inference_single_view(processor, query_words, num_queries, device, masked_img)

        # 聚合同义词 → 类别级 logits
        agg = aggregate_logits(seg_logits, query_idx_tensor, num_cls, num_queries)

        # argmax + 自适应阈值（首轮 0.7，Pass2/3 按 0.7 倍递减，之后锁定最低阈值）
        pred = agg.argmax(0).cpu().numpy()
        max_vals = agg.max(0)[0].cpu().numpy()
        if p == 0:
            thd = 0.7
        elif p <= 2:
            thd = 0.7 * (0.7 ** p)
        else:
            thd = 0.7 * (0.7 ** 2)  # 锁定 Pass3 的阈值，不再继续降
        confident = max_vals > thd

        # 只填充尚未覆盖的区域
        new_mask = confident & ~covered
        final_seg[new_mask] = pred[new_mask]
        covered |= new_mask

        cov = covered.sum() / covered.size
        new_pct = new_mask.sum() / covered.size
        print(f"      Pass {p+1} 完成: 覆盖率={cov:.1%}, 新增={new_pct:.1%}, 阈值={thd:.3f}")

        if cov >= coverage_target:
            print(f"      ✓ 覆盖率达标 {cov:.1%}，停止迭代")
            break
        if new_pct < 0.001:
            print(f"      ✓ 收敛（新增 < 0.1%），停止迭代")
            break

    return final_seg


def aggregate_logits(seg_logits, query_idx_tensor, num_cls, num_queries):
    """
    入参: seg_logits (num_queries, H, W), query_idx_tensor, num_cls, num_queries
    方法: 同义词聚合，同一类别的多个 query logits 取 max（逐类别处理，节省显存）
    出参: agg_logits - (num_cls, H, W)
    """
    if num_cls == num_queries:
        return seg_logits
    H, W = seg_logits.shape[1], seg_logits.shape[2]
    agg = torch.zeros(num_cls, H, W, device=seg_logits.device, dtype=seg_logits.dtype)
    for c in range(num_cls):
        mask = (query_idx_tensor == c)
        if mask.any():
            agg[c] = seg_logits[mask].max(0)[0]
    return agg


def postprocess(seg_np, num_cls, min_area=64, max_hole=256, smooth_kernel=3):
    """
    入参: seg_np - 分割图 numpy, num_cls, 过滤参数
    方法:
        1. 形态学平滑（闭运算+开运算）— 填补缝隙、连接碎片、去除毛刺
        2. 连通域过滤（去小碎片）
        3. 填洞（内部小空洞，排除边界空洞）
        4. 实例内部平滑插值
    出参: 处理后的 seg_np
    """
    # 构造圆形结构元素（比方形更自然）
    if smooth_kernel > 1:
        _r = smooth_kernel
        _y, _x = np.ogrid[-_r:_r+1, -_r:_r+1]
        _struct = (_x*_x + _y*_y <= _r*_r).astype(np.uint8)
    else:
        _struct = np.ones((3, 3), dtype=np.uint8)

    for cls_id in range(1, num_cls):
        cls_mask = (seg_np == cls_id)
        if cls_mask.sum() == 0:
            continue

        # Step 1: 形态学平滑 — 闭运算（膨胀→腐蚀）+ 开运算（腐蚀→膨胀）
        if smooth_kernel > 0:
            cls_mask = binary_dilation(cls_mask, structure=_struct)
            cls_mask = binary_erosion(cls_mask, structure=_struct)
            cls_mask = binary_erosion(cls_mask, structure=_struct)
            cls_mask = binary_dilation(cls_mask, structure=_struct)

        # Step 2: 连通域过滤 — 去除小碎片
        if min_area > 0:
            labeled, n = scipy_label(cls_mask)
            for cid in range(1, n + 1):
                if (labeled == cid).sum() < min_area:
                    cls_mask[labeled == cid] = False

        # Step 3: 填洞 — 内部封闭空洞，排除边界连通区域
        if max_hole > 0:
            inv = ~cls_mask
            h_labeled, h_n = scipy_label(inv)
            border = set()
            border.update(h_labeled[0, :].tolist())
            border.update(h_labeled[-1, :].tolist())
            border.update(h_labeled[:, 0].tolist())
            border.update(h_labeled[:, -1].tolist())
            border.discard(0)
            for hid in range(1, h_n + 1):
                h_area = (h_labeled == hid).sum()
                if hid in border or h_area > max_hole:
                    continue
                cls_mask[h_labeled == hid] = True

        seg_np[cls_mask] = cls_id

    # Step 4: 实例内部平滑插值 — 填充连通域内部残留的背景/未分类像素
    seg_np = _fill_instance_interior(seg_np, num_cls)
    return seg_np


def _fill_instance_interior(seg_np, num_cls):
    """
    对前景区域的每个连通域内部，用周围已知像素平滑插值填充未分类像素。

    方法:
        1. 找到所有前景像素（>0）构成的连通域（不区分类别）
        2. 对每个实例，找出其内部尚未分类的像素（class 0）
        3. 迭代膨胀：每轮将边界已知像素向外扩展1层，填充相邻的未分类像素
        4. 直到实例内部没有未分类像素

    入参: seg_np (H, W) int64, num_cls
    出参: 填充后的 seg_np
    """
    foreground = (seg_np > 0)
    if foreground.sum() == 0:
        return seg_np

    labeled_instances, n_instances = scipy_label(foreground)

    for iid in range(1, n_instances + 1):
        instance_mask = (labeled_instances == iid)

        # 找到实例内部的未分类像素
        interior_bg = instance_mask & (seg_np == 0)
        if interior_bg.sum() == 0:
            continue  # 该实例内部没有空洞，跳过

        # 迭代膨胀填充：从已知标签边界向外扩展，直到填满内部
        filled = seg_np.copy()
        structure = np.ones((3, 3), dtype=np.uint8)  # 8-邻域

        for _ in range(max(filled.shape)):  # 上限 = 图像最大尺寸
            # 当前实例内的已知像素
            known = instance_mask & (filled > 0)
            if known.sum() == 0:
                break

            # 膨胀已知像素，获取候选填充位置
            dilated = binary_dilation(known, structure=structure)

            # 只填充实例内部且尚未分类的像素
            to_fill = dilated & interior_bg & (filled == 0)
            if to_fill.sum() == 0:
                break

            # 对每个待填充像素，取其 8 邻域内已知像素的多数类
            h, w = filled.shape
            ys, xs = np.where(to_fill)
            for y, x in zip(ys, xs):
                y_lo, y_hi = max(0, y - 1), min(h, y + 2)
                x_lo, x_hi = max(0, x - 1), min(w, x + 2)
                neighbors = filled[y_lo:y_hi, x_lo:x_hi]
                known_neighbors = neighbors[neighbors > 0]
                if len(known_neighbors) > 0:
                    counts = np.bincount(known_neighbors)
                    filled[y, x] = counts.argmax()

            # 更新 interior_bg（剩余未填充部分）
            interior_bg = instance_mask & (filled == 0)

        seg_np[:] = filled

    return seg_np


def extract_edge(mask_np, kernel_size=3):
    """
    入参: mask_np - 二值 mask (0/255), kernel_size
    方法: dilated - eroded 提取边缘
    出参: edge_mask - (0/255) uint8
    """
    mask_bool = mask_np > 0
    structure = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = binary_dilation(mask_bool, structure=structure)
    eroded = binary_erosion(mask_bool, structure=structure)
    return (dilated.astype(np.uint8) - eroded.astype(np.uint8)) * 255


def extract_fpn_features(processor, image):
    """
    入参: processor, image(PIL)
    方法: 提取 SAM3 backbone FPN 特征
    出参: fpn_features, original_size
    """
    with torch.no_grad():
        state = processor.set_image(image)
        fpn_features = state["backbone_out"]["backbone_fpn"]
        original_size = (image.size[1], image.size[0])
    return fpn_features, original_size


def compute_fpn_similarity(fpn_t1, fpn_t2, target_size):
    """
    入参: 两组 FPN 特征, target_size
    方法: 上采样后计算余弦相似度，归一化到 [0,1]
    出参: similarity_map - (1, H, W)
    """
    def upsample(fpn_features, size):
        upsampled = []
        for feat in fpn_features[-1:]:
            B, C, H, W = feat.shape
            if C > 320:
                p1 = F.interpolate(feat[:, :320], size=size, mode="bilinear", align_corners=False)
                p2 = F.interpolate(feat[:, 320:], size=size, mode="bilinear", align_corners=False)
                feat = torch.cat([p1, p2], dim=1)
            else:
                feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            upsampled.append(feat)
        return torch.cat(upsampled, dim=1)

    feat_t1 = upsample(fpn_t1, target_size)
    feat_t2 = upsample(fpn_t2, target_size)
    sim = F.cosine_similarity(feat_t1, feat_t2, dim=1)
    return (sim * 0.5 + 0.5).squeeze(1)


# ═══════════════════════════════════════════════════════════════
#  实例级变化检测（IoU 匹配）
# ═══════════════════════════════════════════════════════════════

def _extract_instances(mask, num_cls, min_area=4):
    """
    从分割 mask 中提取所有实例（每类每连通域为一个实例）

    入参:
        mask: (H, W) int64, 像素值=类别 ID（0=背景）
        num_cls: 类别总数（含背景）
        min_area: 最小实例面积，低于此值的连通域被丢弃
    出参:
        List[Dict]: 每个实例 {"cls_id", "bbox", "area"}
        bbox 格式: [x1, y1, x2, y2]
    """
    instances = []
    for cls_id in range(1, num_cls):
        binary = (mask == cls_id)
        if binary.sum() == 0:
            continue
        labeled, n = scipy_label(binary)
        for comp_id in range(1, n + 1):
            comp_mask = (labeled == comp_id)
            area = int(comp_mask.sum())
            if area < min_area:
                continue
            coords = np.where(comp_mask)
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())
            instances.append({
                "cls_id": cls_id,
                "bbox": [x_min, y_min, x_max + 1, y_max + 1],
                "area": area,
                "comp_mask": comp_mask,
                "matched": False,
            })
    return instances



def compute_instance_change_map(t1_mask, t2_mask, num_cls,
                                iou_threshold=0.0, min_area=4):
    """
    实例级变化检测（像素掩码 IoU 匹配）

    流程:
        1. 分别从 t1_mask / t2_mask 提取实例（连通域）
        2. 计算所有 T1×T2 实例对的像素掩码 IoU（非 bbox）
        3. 贪心匹配：按 IoU 降序逐一配对（每实例最多配一个）
        4. 分类判定:
           - 未匹配的 T1 实例 → 消失 → 标记 cls_id
           - 未匹配的 T2 实例 → 新增 → 标记 cls_id
           - 匹配但 cls_id 不同 → 类别变化 → 标记 T2 的 cls_id
           - 匹配且 cls_id 相同 → 无变化 → 不标记
        5. 构建 change_mask

    入参:
        t1_mask: (H, W) int64, T1 分割结果
        t2_mask: (H, W) int64, T2 分割结果
        num_cls: 类别总数（含背景）
        iou_threshold: 掩码 IoU 阈值，默认 0（有重叠即匹配）
        min_area: 最小实例面积
    出参:
        change_mask: (H, W) int64, 像素值=cls_id（0=背景/无变化）
    """
    h, w = t1_mask.shape[:2]
    change_mask = np.zeros((h, w), dtype=np.int64)

    insts_t1 = _extract_instances(t1_mask, num_cls, min_area)
    insts_t2 = _extract_instances(t2_mask, num_cls, min_area)

    n1, n2 = len(insts_t1), len(insts_t2)

    # 无实例的边界情况
    if n1 == 0 and n2 == 0:
        return change_mask
    if n1 == 0:
        for inst in insts_t2:
            change_mask[inst["comp_mask"]] = inst["cls_id"]
        return change_mask
    if n2 == 0:
        for inst in insts_t1:
            change_mask[inst["comp_mask"]] = inst["cls_id"]
        return change_mask

    # 计算所有 (T1_i, T2_j) 的像素掩码 IoU
    pairs = []
    for i, a in enumerate(insts_t1):
        for j, b in enumerate(insts_t2):
            # bbox 快速预筛：bbox 不重叠则掩码必定不重叠
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
                continue
            # 像素掩码 IoU
            inter = int(np.logical_and(a["comp_mask"], b["comp_mask"]).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(a["comp_mask"], b["comp_mask"]).sum())
            iou = inter / union
            if iou > iou_threshold:
                pairs.append((iou, i, j))

    # 贪心匹配：按 IoU 降序，每实例最多配一个
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_t1 = set()
    matched_t2 = set()
    for iou_val, i, j in pairs:
        if i in matched_t1 or j in matched_t2:
            continue
        insts_t1[i]["matched"] = True
        insts_t2[j]["matched"] = True
        matched_t1.add(i)
        matched_t2.add(j)

    # 构建变化图
    # 未匹配的 T1 实例 → 消失
    for inst in insts_t1:
        if not inst["matched"]:
            change_mask[inst["comp_mask"]] = inst["cls_id"]

    # 未匹配的 T2 实例 → 新增
    for inst in insts_t2:
        if not inst["matched"]:
            change_mask[inst["comp_mask"]] = inst["cls_id"]

    # 匹配但类别不同的 → 类别变化（标记 T2 的新类别）
    for i in matched_t1:
        # 找到对应的 T2 j
        for iou_val, ii, jj in pairs:
            if ii == i and jj in matched_t2:
                if insts_t1[ii]["cls_id"] != insts_t2[jj]["cls_id"]:
                    # 标记 T2 实例区域为新类别
                    change_mask[insts_t2[jj]["comp_mask"]] = insts_t2[jj]["cls_id"]
                break

    n_t1_only = sum(1 for inst in insts_t1 if not inst["matched"])
    n_t2_only = sum(1 for inst in insts_t2 if not inst["matched"])
    n_cls_chg = sum(
        1 for iou_val, i, j in pairs
        if i in matched_t1 and j in matched_t2
        and insts_t1[i]["cls_id"] != insts_t2[j]["cls_id"]
    )
    print(f"      实例统计: T1={n1}, T2={n2}, 消失={n_t1_only}, "
          f"新增={n_t2_only}, 类别变化={n_cls_chg}")

    return change_mask
