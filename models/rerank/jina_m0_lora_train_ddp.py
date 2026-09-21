# jina_m0_lora_train_ddp.py
# 多卡分布式训练版本，使用 Accelerate 库
# 启动方式：accelerate launch --multi_gpu --num_processes=N jina_m0_lora_train_ddp.py [args...]
# 或使用配置文件：accelerate launch --config_file accelerate_config.yaml jina_m0_lora_train_ddp.py [args...]

import os
import sys
import math
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from PIL import Image
import wandb
import clip
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model

# Accelerate 分布式训练
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataset import MSRVTT_Dataset
from models.rerank.jina_modeling import JinaVLForRanking, formatting_prompts_func

CONFIG = {
    "data_path": "./imgflip_data/msrvtt",
    "image_path": "./imgflip_data/images",
    "train_csv": "train_ids.csv",
    "train_json": "train_data.json",
    "train_emotion_json": "train_emotion.json",
    "batch_size": 128,
    "lr": 1e-5,
    "epochs": 5,
    "save_dir": "./checkpoints_rerank",
    "image_max_side": 256,
    "jina_micro_batch": 64,
    "qwen_min_side": 32,
    "max_text_len": 2048,
}


class QueryDataset(Dataset):
    """将 query 索引封装成 Dataset，用于分布式采样"""
    def __init__(self, num_queries, caption_to_indices, captions, image_paths, upvotes, 
                 text_matrix, image_matrix, sim, topk_base, topk_cached, topk_base_cached,
                 extra_negs, candidate_mode, label_mode):
        self.num_queries = num_queries
        self.caption_to_indices = caption_to_indices
        self.captions = captions
        self.image_paths = image_paths
        self.upvotes = upvotes
        self.text_matrix = text_matrix
        self.image_matrix = image_matrix
        self.sim = sim
        self.topk_base = topk_base
        self.topk_cached = topk_cached
        self.topk_base_cached = topk_base_cached
        self.extra_negs = extra_negs
        self.candidate_mode = candidate_mode
        self.label_mode = label_mode
        self.num_images = min(image_matrix.shape[0], len(image_paths))
    
    def __len__(self):
        return self.num_queries
    
    def __getitem__(self, q_idx):
        """返回一个 query 的训练数据"""
        # 获取 top-k 候选
        if self.topk_cached is not None and (self.topk_base_cached == self.topk_base):
            base_topk = np.array(self.topk_cached[q_idx], dtype=np.int64)
        else:
            if self.sim is None:
                k_base = min(self.topk_base, self.num_images)
                t = self.text_matrix[q_idx].cpu().numpy()
                sims_q = (t[None, :] @ self.image_matrix.cpu().numpy().T).squeeze(0)
                base_topk = np.argpartition(-sims_q, range(k_base))[:k_base]
            else:
                k_base = min(self.topk_base, self.sim.shape[1])
                base_topk = np.argpartition(-self.sim[q_idx], range(k_base))[:k_base]
        
        # 构建候选集
        pos_all = np.array(self.caption_to_indices[self.captions[q_idx]], dtype=np.int64)
        if self.candidate_mode == "topk":
            core_set = base_topk
        elif self.candidate_mode == "pos_all":
            core_set = pos_all
        else:
            core_set = np.unique(np.concatenate([base_topk, pos_all]))
        
        # 添加额外负样本
        all_indices = np.arange(self.num_images)
        mask = np.ones_like(all_indices, dtype=bool)
        mask[core_set] = False
        candidates_neg = all_indices[mask]
        if self.extra_negs > 0 and candidates_neg.size > 0:
            sel = np.random.choice(candidates_neg, size=min(self.extra_negs, candidates_neg.size), replace=False)
            train_indices = np.concatenate([core_set, sel])
        else:
            train_indices = core_set
        
        if train_indices.size == 0:
            return None
        
        train_indices = train_indices[(train_indices >= 0) & (train_indices < len(self.image_paths))]
        if train_indices.size == 0:
            return None
        
        # 构建标签
        gt = set(self.caption_to_indices[self.captions[q_idx]])
        pos_list = [(i, self.upvotes[i]) for i in train_indices 
                    if i in gt and (isinstance(self.upvotes, (list, np.ndarray)) and i < len(self.upvotes)) and self.upvotes[i] > 0]
        
        if len(pos_list) == 0:
            return None
        
        labels = np.zeros(train_indices.shape[0], dtype=np.float32)
        if self.label_mode == "inv_rank":
            sorted_pos = sorted(pos_list, key=lambda x: x[1], reverse=True)
            for rank, (i, _) in enumerate(sorted_pos, start=1):
                w = 1.0 / float(rank)
                idx_in_arr = np.where(train_indices == i)[0][0]
                labels[idx_in_arr] = float(w)
        elif self.label_mode == "raw_log":
            vals = np.array([u for _, u in pos_list], dtype=np.float32)
            vals = np.log1p(vals)
            for (i, _), s in zip(pos_list, vals):
                idx_in_arr = np.where(train_indices == i)[0][0]
                labels[idx_in_arr] = float(s)
        else:
            vals = np.array([u for _, u in pos_list], dtype=np.float32)
            vals = np.log1p(vals)
            exps = np.exp(vals - vals.max())
            denom = exps.sum()
            sm = exps / (denom if denom > 0 else 1.0)
            for (i, _), s in zip(pos_list, sm):
                idx_in_arr = np.where(train_indices == i)[0][0]
                labels[idx_in_arr] = float(s)
        
        return {
            "q_idx": q_idx,
            "train_indices": train_indices,
            "labels": labels,
            "caption": self.captions[q_idx]
        }


def collate_fn(batch):
    """过滤掉 None 的样本"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch


def compute_features(clip_model, dataloader, device, accelerator=None):
    """计算 CLIP 特征（支持分布式）"""
    text_feats = []
    image_feats = []
    upvotes = []
    captions = []
    clip_model.eval()
    
    try:
        clip_dev = next(clip_model.parameters()).device
    except (StopIteration, AttributeError):
        clip_dev = torch.device("cpu")
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(clip_dev)
            query_ids = batch["query_ids"].to(clip_dev)
            img = clip_model.encode_image(images)
            txt = clip_model.encode_text(query_ids)
            img = img / img.norm(dim=1, keepdim=True).clamp(min=1e-6)
            txt = txt / txt.norm(dim=1, keepdim=True).clamp(min=1e-6)
            image_feats.append(img.cpu())
            text_feats.append(txt.cpu())
            upvotes.extend(batch["upvote"].cpu().numpy().tolist())
            captions.extend(batch["query_text"])
    
    text_matrix = torch.cat(text_feats, dim=0)
    image_matrix = torch.cat(image_feats, dim=0)
    return text_matrix, image_matrix, np.array(upvotes, dtype=np.float32), captions


def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    s = scores.unsqueeze(0)
    y = labels.unsqueeze(0)
    diff = s.transpose(0, 1) - s
    rel = (y.transpose(0, 1) - y) > 0
    pos = torch.nonzero(rel, as_tuple=False)
    if pos.size(0) == 0:
        return scores.sum() * 0.0
    d = diff[pos[:, 0], pos[:, 1]]
    return torch.nn.functional.softplus(-d).mean()


def lambdarank_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(scores.device)
    n = scores.size(0)
    if n == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype, requires_grad=True)
    if (labels > 0).sum() == 0:
        return scores.sum() * 0.0
    
    labels_clamped = torch.clamp(labels, max=10.0)
    gains = torch.pow(2.0, labels_clamped) - 1.0
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order, dtype=torch.long)
    ranks[order] = torch.arange(n, device=scores.device, dtype=torch.long)
    discounts = 1.0 / torch.log2(ranks.float() + 2.0)
    ideal_order = torch.argsort(labels, descending=True)
    ideal_ranks = torch.empty_like(ideal_order, dtype=torch.long)
    ideal_ranks[ideal_order] = torch.arange(n, device=scores.device, dtype=torch.long)
    ideal_discounts = 1.0 / torch.log2(ideal_ranks.float() + 2.0)
    idcg = (gains[ideal_order] * ideal_discounts[ideal_order]).sum()
    idcg = torch.clamp(idcg, min=1e-8)
    
    s = scores.unsqueeze(0)
    g = gains.unsqueeze(0)
    d = discounts.unsqueeze(0)
    sd = s.transpose(0, 1) - s
    rel = (labels.unsqueeze(0).transpose(0, 1) - labels.unsqueeze(0)) > 0
    pos = torch.nonzero(rel, as_tuple=False)
    if pos.size(0) == 0:
        return scores.sum() * 0.0
    
    gi = g[:, pos[:, 0]].squeeze(0)
    gj = g[:, pos[:, 1]].squeeze(0)
    di = d[:, pos[:, 0]].squeeze(0)
    dj = d[:, pos[:, 1]].squeeze(0)
    delta_ndcg = torch.abs((gi - gj) * (di - dj)) / idcg
    delta_ndcg = torch.clamp(delta_ndcg, max=100.0)
    sdiff = sd[pos[:, 0], pos[:, 1]]
    
    if torch.isinf(delta_ndcg).any() or torch.isnan(delta_ndcg).any():
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype, requires_grad=True)
    
    loss_per_pair = delta_ndcg * torch.nn.functional.softplus(-sdiff)
    loss_val = loss_per_pair.mean()
    
    if torch.isnan(loss_val) or torch.isinf(loss_val):
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype, requires_grad=True)
    return loss_val


def build_image_paths(dataset, image_root):
    paths = []
    for item in dataset.data:
        vid = item["video_id"]
        name = vid if "." in vid else f"{vid}.jpg"
        paths.append(os.path.join(image_root, name))
    return paths


def _build_image_path(image_dir, vid):
    """构建图片路径（与推理脚本对齐）"""
    return os.path.join(image_dir, vid if "." in vid else f"{vid}.jpg")

def _safe_clip_indices(arr, max_len):
    """安全裁剪索引（与推理脚本对齐）"""
    try:
        a = np.array(arr, dtype=np.int64)
    except Exception:
        return arr
    if max_len is None or int(max_len) <= 0:
        return a
    mask = (a >= 0) & (a < int(max_len))
    return a[mask]


def score_batch(model, processor, texts, images, device, requires_grad=False):
    """计算模型分数（支持分布式训练）"""
    msgs = []
    imgs_list = []
    from PIL import Image as _I
    max_side_cfg = int(CONFIG.get("image_max_side", 448))
    min_side_cfg = int(CONFIG.get("qwen_min_side", 28))
    mb = int(CONFIG.get("jina_micro_batch", 0))
    
    for t, img in zip(texts, images):
        if img is None:
            img = _I.new("RGB", (224, 224), (0, 0, 0))
        else:
            w, h = img.size
            if min(w, h) < min_side_cfg:
                scale_up = float(min_side_cfg) / float(min(w, h))
                w = int(round(w * scale_up))
                h = int(round(h * scale_up))
                img = img.resize((w, h), Image.BICUBIC)
            max_side = max_side_cfg
            if max(w, h) > max_side:
                scale_down = float(max_side) / float(max(w, h))
                w = int(round(w * scale_down))
                h = int(round(h * scale_down))
                img = img.resize((w, h), Image.BICUBIC)
            w, h = img.size
            if min(w, h) < min_side_cfg:
                w2 = max(w, min_side_cfg)
                h2 = max(h, min_side_cfg)
                canvas = _I.new("RGB", (w2, h2), (0, 0, 0))
                canvas.paste(img, ((w2 - w) // 2, (h2 - h) // 2))
                img = canvas
        imgs_list.append(img)
    
    # 获取模型设备（处理 DDP 包装）
    # 注意：在 DDP 训练时，避免在 forward 之外访问 model.parameters()，这会导致 autograd hooks 问题
    # 优先使用传入的 device 参数
    if device is not None:
        mdl_dev = device
    else:
        # 如果 device 未提供，才从模型获取（但这不是推荐做法）
        try:
            if hasattr(model, 'module'):
                mdl_dev = next(model.module.parameters()).device
            else:
                mdl_dev = next(model.parameters()).device
        except (StopIteration, RuntimeError):
            # 如果无法获取，使用默认设备
            mdl_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    outs = []
    if mb and mb > 0:
        for i in range(0, len(imgs_list), mb):
            batch_texts = []
            batch_images = []
            for j in range(i, min(i + mb, len(texts))):
                prompt_text = formatting_prompts_func(
                    query=texts[j],
                    doc="",
                    query_type='text',
                    doc_type='image'
                )
                batch_texts.append(prompt_text)
                batch_images.append(imgs_list[j])
            
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", 
                             padding=True, truncation=True, max_length=int(CONFIG.get("max_text_len", 512)))
            inputs = {k: v.to(mdl_dev) for k, v in inputs.items()}
            
            # 安全地获取 score_token_id，避免在 DDP 训练时触发 autograd hooks
            try:
                # 使用 try-except 包裹，避免访问模型属性时触发 DDP hooks
                tok_id = 100  # 默认值
                try:
                    # 尝试从模型获取，但使用最安全的方式
                    if hasattr(model, 'module') and hasattr(model.module, "score_token_id"):
                        tok_id = int(model.module.score_token_id)
                    elif hasattr(model, "score_token_id"):
                        tok_id = int(model.score_token_id)
                except (AttributeError, RuntimeError, TypeError):
                    pass  # 使用默认值
                
                bsz = inputs["input_ids"].size(0)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], 
                    torch.full((bsz, 1), tok_id, device=inputs["input_ids"].device)], dim=1)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], 
                    torch.ones((bsz, 1), device=inputs["attention_mask"].device)], dim=1)
            except Exception:
                pass
            
            if requires_grad:
                if mdl_dev.type == "cuda":
                    from torch import amp as _amp
                    with _amp.autocast("cuda", dtype=torch.bfloat16):  # 使用 bfloat16 与模型 dtype 一致
                        with torch.set_grad_enabled(True):
                            logits = model(**inputs)
                else:
                    with torch.set_grad_enabled(True):
                        logits = model(**inputs)
            else:
                with torch.inference_mode():
                    if mdl_dev.type == "cuda":
                        from torch import amp as _amp
                        with _amp.autocast("cuda", dtype=torch.bfloat16):  # 使用 bfloat16 与模型 dtype 一致
                            logits = model(**inputs)
                    else:
                        logits = model(**inputs)
            
            outs.append(logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, device=mdl_dev))
            
            if not requires_grad:
                del inputs, logits
                if mdl_dev.type == "cuda":
                    torch.cuda.empty_cache()
        
        out = torch.cat(outs, dim=0)
    else:
        batch_texts = []
        for j in range(len(texts)):
            prompt_text = formatting_prompts_func(
                query=texts[j],
                doc="",
                query_type='text',
                doc_type='image'
            )
            batch_texts.append(prompt_text)
        
        inputs = processor(text=batch_texts, images=imgs_list, return_tensors="pt", 
                         padding=True, truncation=True, max_length=int(CONFIG.get("max_text_len", 512)))
        inputs = {k: v.to(mdl_dev) for k, v in inputs.items()}
        
        # 安全地获取 score_token_id，避免在 DDP 训练时触发 autograd hooks
        try:
            tok_id = 100  # 默认值
            try:
                # 尝试从模型获取，但使用最安全的方式
                if hasattr(model, 'module') and hasattr(model.module, "score_token_id"):
                    tok_id = int(model.module.score_token_id)
                elif hasattr(model, "score_token_id"):
                    tok_id = int(model.score_token_id)
            except (AttributeError, RuntimeError, TypeError):
                pass  # 使用默认值
            
            bsz = inputs["input_ids"].size(0)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], 
                torch.full((bsz, 1), tok_id, device=inputs["input_ids"].device)], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], 
                torch.ones((bsz, 1), device=inputs["attention_mask"].device)], dim=1)
        except Exception:
            pass
        
        if requires_grad:
            if mdl_dev.type == "cuda":
                from torch import amp as _amp
                with _amp.autocast("cuda", dtype=torch.bfloat16):  # 使用 bfloat16 与模型 dtype 一致
                    with torch.set_grad_enabled(True):
                        out = model(**inputs)
            else:
                with torch.set_grad_enabled(True):
                    out = model(**inputs)
        else:
            with torch.inference_mode():
                if mdl_dev.type == "cuda":
                    from torch import amp as _amp
                    with _amp.autocast("cuda", dtype=torch.bfloat16):  # 使用 bfloat16 与模型 dtype 一致
                        out = model(**inputs)
                else:
                    out = model(**inputs)
        
        if not requires_grad:
            del inputs
            if mdl_dev.type == "cuda":
                torch.cuda.empty_cache()
    
    if requires_grad:
        return out if isinstance(out, torch.Tensor) else torch.as_tensor(out, device=mdl_dev)
    else:
        return out.detach().cpu().view(-1) if isinstance(out, torch.Tensor) else torch.as_tensor(out, device="cpu").view(-1)


def format_eval_metrics(eval_metrics, epoch=None, prefix=""):
    """格式化评估指标，使其更易读
    
    Args:
        eval_metrics: 包含 R@k 和 P@k 指标的字典
        epoch: 当前epoch（可选）
        prefix: 前缀字符串（如 "Initial Eval"）
    """
    ks = [1, 3, 5, 10, 50, 100]
    
    # 构建表格格式的输出
    lines = []
    if prefix:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"{prefix} Evaluation Results" + (f" (Epoch {epoch})" if epoch is not None else ""))
        lines.append(f"{'=' * 80}")
    else:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"Evaluation Results" + (f" (Epoch {epoch})" if epoch is not None else ""))
        lines.append(f"{'=' * 80}")
    
    # 表头
    lines.append(f"{'Metric':<12} | {'R@1':>8} | {'R@3':>8} | {'R@5':>8} | {'R@10':>9} | {'R@50':>9} | {'R@100':>10}")
    lines.append("-" * 80)
    
    # Recall 行
    recall_values = [f"{eval_metrics.get(f'R@{k}', 0.0):>7.2f}%" for k in ks]
    lines.append(f"{'Recall':<12} | {recall_values[0]:>8} | {recall_values[1]:>8} | {recall_values[2]:>8} | {recall_values[3]:>9} | {recall_values[4]:>9} | {recall_values[5]:>10}")
    
    # Precision 行
    precision_values = [f"{eval_metrics.get(f'P@{k}', 0.0):>7.2f}%" for k in ks]
    lines.append(f"{'Precision':<12} | {precision_values[0]:>8} | {precision_values[1]:>8} | {precision_values[2]:>8} | {precision_values[3]:>9} | {precision_values[4]:>9} | {precision_values[5]:>10}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

def evaluate_recalls_jina(captions, image_paths, model, processor, topk_base, precomputed_topk, 
                         train_text_matrix, train_image_matrix, image_captions=None, accelerator=None):
    """评估函数（仅在主进程执行，与单卡版本对齐）"""
    # 确保模型处于 eval 模式
    model.eval()
    
    from collections import defaultdict
    # caption_to_indices 应该基于 image pool 的 captions 构建，索引是 images 的索引
    # 如果没有提供 image_captions，尝试从 image_paths 推断（假设每个 image 对应一个 caption）
    if image_captions is None:
        # 如果没有提供，假设 image_captions 与 captions 相同（这种情况应该不会发生，但为了兼容性）
        if accelerator is None or accelerator.is_main_process:
            print("[Eval] Warning: image_captions not provided, using captions as fallback (may cause incorrect evaluation)")
        image_captions = captions
    
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(image_captions):
        caption_to_indices[cap].append(idx)
    
    total = len(captions)
    need_sim = precomputed_topk is None
    if need_sim:
        sim = train_text_matrix.cpu().numpy() @ train_image_matrix.cpu().numpy().T
    
    # 获取设备
    # 注意：在 DDP 训练时，避免在 forward 之外访问 model.parameters()，这会导致 autograd hooks 问题
    # 优先使用 accelerator.device（如果提供）
    if accelerator is not None and hasattr(accelerator, 'device'):
        device = accelerator.device
    else:
        # 如果 accelerator 未提供，尝试从模型获取（但这不是推荐做法）
        try:
            if hasattr(model, 'module'):
                device = next(model.module.parameters()).device
            else:
                device = next(model.parameters()).device
        except (StopIteration, RuntimeError):
            # 如果无法获取，使用默认设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # k值列表与推理脚本对齐（包含3）
    ks = [1, 3, 5, 10, 50, 100]
    # 优化：改变循环顺序，每个query只处理一次，然后计算所有k值的指标
    # 初始化所有k值的统计
    recalls = {f"R@{k}": 0.0 for k in ks}
    precisions = {f"P@{k}": 0.0 for k in ks}
    hits_dict = {k: 0 for k in ks}
    acc_dict = {k: 0.0 for k in ks}
    
    # 只在主进程显示进度条
    iterator = range(total)
    if accelerator is None or accelerator.is_main_process:
        iterator = tqdm(iterator, desc="Evaluating")
    
    for q_idx in iterator:
        # 正常流程：动态选择候选图像
        if need_sim:
            k_base = min(topk_base, train_image_matrix.shape[0])
            base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
        else:
            base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
            k_base = base_topk.shape[0]
        # 使用_safe_clip_indices进行安全检查（与推理脚本对齐）
        base_topk = _safe_clip_indices(base_topk, len(image_paths))
        if len(base_topk) == 0:
            continue
        
        # 使用_build_image_path构建图片路径（与推理脚本对齐）
        imgs = []
        for idx_i in base_topk.tolist():
            if idx_i < len(image_paths):
                p = image_paths[idx_i]
            else:
                # 如果image_paths是路径列表，直接使用；否则需要从dataset构建
                p = image_paths[idx_i] if isinstance(image_paths, list) and idx_i < len(image_paths) else None
            if p and os.path.exists(p):
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (224, 224)))
            else:
                imgs.append(Image.new("RGB", (224, 224)))
        
        texts = [captions[q_idx]] * len(imgs)
        # 使用score_batch（评估模式，不需要梯度）
        # 确保在评估模式下，模型不会更新参数
        with torch.no_grad():
            scores = score_batch(model, processor, texts, imgs, device, requires_grad=False)
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()
        # 及时清理图片对象（与推理脚本对齐）
        del imgs
        
        order = np.argsort(-scores)
        reranked_topk = base_topk[order]
        
        # 正常流程：使用 caption_to_indices 构建 ground truth
        if caption_to_indices is None:
            if accelerator is None or accelerator.is_main_process:
                print(f"[Eval] Warning: caption_to_indices not available for query {q_idx}, skipping")
            continue
        gt = caption_to_indices[captions[q_idx]]
        # 对每个k值计算指标（只处理一次，计算所有k值）
        for k in ks:
            k_eff = min(k, len(base_topk))
            if k_eff == 0:
                continue
            topk_used = reranked_topk[:k_eff]
            if any(i in gt for i in topk_used):
                hits_dict[k] += 1
            correct = sum(1 for i in topk_used if i in gt)
            acc_dict[k] += correct / float(k_eff)
        # 每处理一定数量的查询后清理一次缓存（与推理脚本对齐）
        if (q_idx + 1) % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    # 计算最终的recall和precision
    for k in ks:
        recalls[f"R@{k}"] = hits_dict[k] / total * 100.0
        precisions[f"P@{k}"] = acc_dict[k] / total * 100.0
    return {**recalls, **precisions}


def train(args=None):
    # ============ 初始化 Accelerator ============
    # 检查是否使用 gradient checkpointing
    use_grad_checkpointing = bool(getattr(args, "grad_checkpointing", False)) if args else False
    
    # DDP 参数配置
    # 重要：如果使用 gradient checkpointing 并设置静态图，必须设置 find_unused_parameters=False
    # 因为静态图假设所有参数都会被使用，与 find_unused_parameters=True 冲突
    # 如果确实有未使用的参数，可以设置 find_unused_parameters=True，但不设置静态图
    if use_grad_checkpointing:
        # 使用 gradient checkpointing 时，设置静态图需要 find_unused_parameters=False
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,  # 静态图要求所有参数都被使用
            broadcast_buffers=True
        )
        # 注意：此时 accelerator 还未初始化，使用普通 print
        print("[DDP] Using find_unused_parameters=False for gradient checkpointing compatibility")
    else:
        # 不使用 gradient checkpointing 时，可以设置 find_unused_parameters=True（LoRA 可能有未使用的参数）
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,  # LoRA 可能有未使用的参数
            broadcast_buffers=True
        )
    
    # 梯度累积步数
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1) if args else 1
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16" if (args and getattr(args, "mixed_precision", "fp16") == "fp16") else "no",
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb" if (args and args.wandb_project) else None,
    )
    
    # 设置随机种子（确保所有进程一致）
    if args and hasattr(args, "seed"):
        set_seed(args.seed)
    
    # 日志辅助函数
    def print_main(*msg):
        if accelerator.is_main_process:
            print(*msg)
    
    print_main(f"[Accelerate] Distributed training initialized")
    print_main(f"  - Num processes: {accelerator.num_processes}")
    print_main(f"  - Process index: {accelerator.process_index}")
    print_main(f"  - Mixed precision: {accelerator.mixed_precision}")
    print_main(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # ============ 配置更新 ============
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
    
    try:
        ep = str(os.environ.get("HF_ENDPOINT", "")).strip()
        if not (ep.startswith("http://") or ep.startswith("https://")):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    except Exception:
        pass
    
    # CLIP 模型放在 CPU 上（仅用于特征提取）
    device_clip = torch.device("cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device_clip)
    
    if args is not None:
        CONFIG.update({
            "data_path": args.data_path,
            "image_path": args.image_path,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "save_dir": args.save_dir,
        })
        topk_base = getattr(args, "topk_base", 100)
        extra_negs = max(0, getattr(args, "extra_negatives", 0))
        candidate_mode = getattr(args, "candidate_mode", "topk_plus_pos")
        model_name = getattr(args, "model_name", "jinaai/jina-reranker-m0")
        lora_r = int(getattr(args, "lora_r", 8))
        lora_alpha = int(getattr(args, "lora_alpha", 16))
        lora_dropout = float(getattr(args, "lora_dropout", 0.05))
        loss_type = getattr(args, "loss_type", "lambdarank")
        label_mode = getattr(args, "label_mode", "inv_rank")
        train_sample_limit = int(getattr(args, "train_sample_limit", 0))
        train_sample_mode = str(getattr(args, "train_sample_mode", "first")).lower()
        train_sample_seed = int(getattr(args, "train_sample_seed", 0))
        CONFIG["image_max_side"] = int(getattr(args, "image_max_side", CONFIG.get("image_max_side", 256)))
        CONFIG["jina_micro_batch"] = int(getattr(args, "jina_micro_batch", CONFIG.get("jina_micro_batch", 4)))
    else:
        topk_base = 100
        extra_negs = 0
        candidate_mode = "topk_plus_pos"
        model_name = "jinaai/jina-reranker-m0"
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.05
        loss_type = "lambdarank"
        label_mode = "inv_rank"
        train_sample_limit = 0
    
    # ============ 数据集加载 ============
    train_csv_path = os.path.join(CONFIG["data_path"], CONFIG["train_csv"])
    train_json_path = os.path.join(CONFIG["data_path"], CONFIG["train_json"])
    train_emotion_json_path = os.path.join(CONFIG["data_path"], CONFIG["train_emotion_json"])
    
    train_dataset = MSRVTT_Dataset(
        csv_path=train_csv_path,
        json_path=train_json_path,
        features_path=CONFIG["image_path"],
        emotion_json_path=train_emotion_json_path,
        clip_preprocess=preprocess,
        bert_tokenizer=None,
        max_words=32,
        is_train=True,
        load_image=True,
    )
    
    # 训练数据采样索引
    train_sample_indices = None
    train_dataset_full = train_dataset
    
    if train_sample_limit and train_sample_limit > 0:
        try:
            total_train = len(getattr(train_dataset, "data"))
            n_lim = min(int(train_sample_limit), total_train)
            print_main(f"[Train] Sampling queries: mode={train_sample_mode}, seed={train_sample_seed}, n={n_lim}/{total_train}")
            if train_sample_mode == "random":
                rng = np.random.RandomState(train_sample_seed)
                indices = rng.choice(total_train, size=n_lim, replace=False)
                indices = sorted(indices.tolist())
                train_sample_indices = np.array(indices, dtype=np.int64)
            else:
                train_sample_indices = np.arange(n_lim, dtype=np.int64)
        except Exception as e:
            print_main(f"[Train] Warning: Failed to prepare sampling: {e}")
    
    # ============ 特征缓存 ============
    cache_train_path = os.path.join(CONFIG["save_dir"], "train_cache.pt")
    
    # 只在主进程创建缓存目录
    if accelerator.is_main_process:
        os.makedirs(CONFIG["save_dir"], exist_ok=True)
    accelerator.wait_for_everyone()
    
    if os.path.exists(cache_train_path):
        print_main(f"[Cache] Loading training cache from: {cache_train_path}")
        obj = torch.load(cache_train_path, map_location="cpu", weights_only=False)
        text_matrix_full = obj["text"].cpu()
        image_matrix_full = obj["image"].cpu()
        upvotes_full = obj.get("upvotes", None)
        captions_full = obj["captions"]
        topk_cached = obj.get("topk_indices", None)
        topk_base_cached = obj.get("topk_base", None)
        
        sim = None
        if train_sample_limit and train_sample_limit > 0:
            topk_cached = None
            topk_base_cached = None
        elif topk_cached is not None and topk_base_cached == topk_base:
            sim = None
        else:
            print_main(f"[Cache] Precomputing similarity matrix")
            sim = (text_matrix_full.cpu().numpy() @ image_matrix_full.cpu().numpy().T)
        
        if train_sample_limit and train_sample_limit > 0 and train_sample_indices is not None:
            try:
                max_idx = train_sample_indices.max()
                cache_size = text_matrix_full.shape[0]
                if max_idx < cache_size:
                    text_matrix = text_matrix_full[train_sample_indices]
                    captions = [captions_full[int(i)] for i in train_sample_indices.tolist()]
                    upvotes = upvotes_full
                    image_matrix = image_matrix_full
                    print_main(f"[Train] Precomputing similarity matrix for sampled queries...")
                    sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
                else:
                    n_q = min(int(train_sample_limit), text_matrix_full.shape[0])
                    text_matrix = text_matrix_full[:n_q]
                    captions = captions_full[:n_q]
                    upvotes = upvotes_full
                    image_matrix = image_matrix_full
                    sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
            except Exception as e:
                print_main(f"[Train] Warning: Failed to apply sampling: {e}")
                n_q = min(int(train_sample_limit), text_matrix_full.shape[0])
                text_matrix = text_matrix_full[:n_q]
                captions = captions_full[:n_q]
                upvotes = upvotes_full if not isinstance(upvotes_full, np.ndarray) else upvotes_full[:n_q]
                image_matrix = image_matrix_full
        else:
            text_matrix = text_matrix_full
            captions = captions_full
            image_matrix = image_matrix_full
            upvotes = upvotes_full
    else:
        # 只在主进程计算并保存缓存
        if accelerator.is_main_process:
            print_main(f"[Cache] Computing training features")
            loader_full = DataLoader(train_dataset_full, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
            text_matrix_full, image_matrix_full, upvotes_full, captions_full = compute_features(
                clip_model, loader_full, torch.device("cpu"))
            topk_cached = None
            topk_base_cached = None
            sim = (text_matrix_full.cpu().numpy() @ image_matrix_full.cpu().numpy().T)
            
            try:
                k_base = min(topk_base, sim.shape[1])
                topk_list = [np.argpartition(-sim[q], range(k_base))[:k_base] for q in range(sim.shape[0])]
                topk_py = [arr.tolist() for arr in topk_list]
            except Exception:
                topk_py = None
            
            cache_data = {
                "text": text_matrix_full.cpu(),
                "image": image_matrix_full.cpu(),
                "upvotes": upvotes_full,
                "captions": captions_full,
                "topk_base": topk_base,
                "topk_indices": topk_py
            }
            torch.save(cache_data, cache_train_path)
            print_main(f"[Cache] Saved training cache to: {cache_train_path}")
        
        accelerator.wait_for_everyone()
        
        # 所有进程重新加载缓存
        obj = torch.load(cache_train_path, map_location="cpu", weights_only=False)
        text_matrix_full = obj["text"].cpu()
        image_matrix_full = obj["image"].cpu()
        upvotes_full = obj.get("upvotes", None)
        captions_full = obj["captions"]
        topk_cached = obj.get("topk_indices", None)
        topk_base_cached = obj.get("topk_base", None)
        
        if train_sample_limit and train_sample_limit > 0 and train_sample_indices is not None:
            max_idx = train_sample_indices.max()
            cache_size = text_matrix_full.shape[0]
            if max_idx < cache_size:
                text_matrix = text_matrix_full[train_sample_indices]
                captions = [captions_full[int(i)] for i in train_sample_indices.tolist()]
                upvotes = upvotes_full
                image_matrix = image_matrix_full
                sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
            else:
                n_q = min(int(train_sample_limit), text_matrix_full.shape[0])
                text_matrix = text_matrix_full[:n_q]
                captions = captions_full[:n_q]
                upvotes = upvotes_full
                image_matrix = image_matrix_full
                sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
        else:
            text_matrix = text_matrix_full
            captions = captions_full
            image_matrix = image_matrix_full
            upvotes = upvotes_full
            sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)
    
    image_paths_train = build_image_paths(train_dataset_full, CONFIG["image_path"])
    print_main(f"[Train] Image pool: {len(image_paths_train)} images")
    
    # ============ 模型初始化 ============
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    try:
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
    except Exception:
        pass
    
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # 在 DDP 训练中，每个进程都会加载模型
    # 关键优化：
    # 1. 使用 low_cpu_mem_usage=True 减少 CPU 内存占用
    # 2. 先加载到 CPU，避免在加载时就在 GPU 上分配显存
    # 3. Accelerate 会在 prepare() 时将其移到正确的 GPU
    print_main(f"[Model] Loading model (process {accelerator.process_index}/{accelerator.num_processes})")
    print_main(f"[Model] Note: Each process loads a full model copy in DDP mode")
    
    # 在加载前清理显存（如果有残留）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    base_model = JinaVLForRanking.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 与单卡版本一致，数值范围与 float32 相同，训练更稳定
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # 多卡时减少内存
        # 注意：不要使用 device_map，让 Accelerate 管理设备分配
    )
    
    # 确保模型在 CPU 上（Accelerate 会将其移到正确的 GPU）
    # 这样可以避免在加载时就在所有 GPU 上分配显存
    base_model = base_model.cpu()
    
    # 再次清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_main(f"[Model] Model loaded on CPU, will be moved to GPU {accelerator.process_index} by Accelerate")
    
    train_lora = bool(getattr(args, "train_lora", True)) if args is not None else True
    
    def _count_trainable(m):
        return sum(int(p.requires_grad) for p in m.parameters())
    
    def _count_trainable_by_name(m):
        stats = {"lora": 0, "score": 0, "other": 0}
        for name, p in m.named_parameters():
            if p.requires_grad:
                if "lora" in name.lower():
                    stats["lora"] += p.numel()
                elif "score" in name.lower():
                    stats["score"] += p.numel()
                else:
                    stats["other"] += p.numel()
        return stats
    
    if train_lora:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        peft_cfg = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, 
            task_type=TaskType.CAUSAL_LM, 
            target_modules=target_modules
        )
        model = get_peft_model(base_model, peft_cfg)
        print_main("[Model] LoRA enabled")
        
        lora_cnt = _count_trainable(model)
        if lora_cnt == 0:
            print_main("[Warning] No trainable LoRA params, retrying with auto target_modules")
            peft_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, 
                task_type=TaskType.CAUSAL_LM, target_modules=None
            )
            model = get_peft_model(base_model, peft_cfg)
    else:
        model = base_model
        for param in model.parameters():
            param.requires_grad = False
        print_main("[Model] LoRA disabled: only training score head")
    
    # Score 层始终可训练
    if train_lora and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model
    else:
        base = model
    
    if hasattr(base, "score"):
        for param in base.score.parameters():
            param.requires_grad = True
        print_main("[Model] Score head is trainable")
    
    trainable_cnt = _count_trainable(model)
    trainable_stats = _count_trainable_by_name(model)
    print_main(f"[Model] Trainable params: {trainable_cnt:,} (LoRA: {trainable_stats['lora']:,}, score: {trainable_stats['score']:,})")
    
    if trainable_cnt == 0:
        raise RuntimeError("No trainable params found")
    
    # Gradient checkpointing
    try:
        if args is not None and getattr(args, "grad_checkpointing", False):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                try:
                    setattr(model.config, "use_cache", False)
                except Exception:
                    pass
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
    except Exception:
        pass
    
    # 可选：从 checkpoint 恢复（支持 LoRA 和 score-only 两种格式）
    try:
        if args is not None and getattr(args, "resume_path", ""):
            rp = str(getattr(args, "resume_path"))
            if rp and os.path.exists(rp):
                print_main(f"[Checkpoint] Loading checkpoint from: {rp}")
                obj_resume = torch.load(rp, map_location="cpu", weights_only=False)
                ps = obj_resume.get("peft", None)
                model_name_resume = obj_resume.get("model_name", "unknown")
                train_lora_resume = obj_resume.get("train_lora", True)  # 默认 True（兼容旧 checkpoint）
                
                if ps:
                    if train_lora and train_lora_resume:
                        # 当前使用 LoRA，checkpoint 也有 LoRA：直接加载
                        model.load_state_dict(ps, strict=False)
                        print_main(f"[Checkpoint] Loaded LoRA + score checkpoint")
                    elif not train_lora and not train_lora_resume:
                        # 当前不使用 LoRA，checkpoint 也没有 LoRA：直接加载 score
                        model_state = model.state_dict()
                        loaded_count = 0
                        for name, param in ps.items():
                            if name in model_state:
                                model_state[name].copy_(param)
                                loaded_count += 1
                        model.load_state_dict(model_state, strict=False)
                        print_main(f"[Checkpoint] Loaded score-only checkpoint ({loaded_count} params)")
                    elif train_lora and not train_lora_resume:
                        # 当前使用 LoRA，但 checkpoint 只有 score：只加载 score 部分
                        model_state = model.state_dict()
                        loaded_count = 0
                        for name, param in ps.items():
                            # score 参数在 PEFT 模型中的路径是 base_model.model.score.*
                            peft_name = f"base_model.model.{name}" if not name.startswith("base_model") else name
                            if peft_name in model_state:
                                model_state[peft_name].copy_(param)
                                loaded_count += 1
                            elif name in model_state:
                                model_state[name].copy_(param)
                                loaded_count += 1
                        model.load_state_dict(model_state, strict=False)
                        print_main(f"[Checkpoint] Loaded score from score-only checkpoint into LoRA model ({loaded_count} params)")
                    else:
                        # 当前不使用 LoRA，但 checkpoint 有 LoRA：只加载 score 部分
                        model_state = model.state_dict()
                        loaded_count = 0
                        for name, param in ps.items():
                            # 从 PEFT 路径提取原始名称
                            orig_name = name.replace("base_model.model.", "") if "base_model.model." in name else name
                            if orig_name in model_state and "score" in orig_name:
                                model_state[orig_name].copy_(param)
                                loaded_count += 1
                        model.load_state_dict(model_state, strict=False)
                        print_main(f"[Checkpoint] Loaded score from LoRA checkpoint ({loaded_count} params)")
                    
                    print_main(f"[Checkpoint] Successfully loaded checkpoint:")
                    print_main(f"  - Path: {rp}")
                    print_main(f"  - Model name: {model_name_resume}")
                    print_main(f"  - Checkpoint train_lora: {train_lora_resume}")
                    print_main(f"  - Current train_lora: {train_lora}")
                    total_params = sum(p.numel() for p in ps.values())
                    print_main(f"  - Total parameters in checkpoint: {total_params:,}")
                else:
                    print_main(f"[Checkpoint] Warning: No 'peft' key found in checkpoint {rp}")
            else:
                print_main(f"[Checkpoint] Warning: Checkpoint path does not exist: {rp}")
    except Exception as e:
        print_main(f"[Checkpoint] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    # ============ 优化器和调度器 ============
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=CONFIG["lr"])
    
    sch_type = getattr(args, "scheduler", "cosine_warmup") if args is not None else "cosine_warmup"
    scheduler = None
    total_queries = text_matrix.shape[0]
    # 考虑梯度累积和多卡
    effective_batch_size = gradient_accumulation_steps * accelerator.num_processes
    total_steps = CONFIG["epochs"] * (total_queries // effective_batch_size + 1)
    
    if sch_type == "cosine_warmup":
        warmup = int(getattr(args, "warmup_steps", max(1, int(0.05 * total_steps)))) if args else max(1, int(0.05 * total_steps))
        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif sch_type == "cosine":
        tmax = CONFIG["epochs"]
        min_lr = float(getattr(args, "min_lr", 0.0)) if args else 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=min_lr)
    
    # ============ 使用 Accelerator 准备模型和优化器 ============
    # 在 prepare 之前，确保所有进程都完成了模型加载
    accelerator.wait_for_everyone()
    
    print_main(f"[Accelerate] Preparing model for GPU {accelerator.process_index}...")
    model, opt, scheduler = accelerator.prepare(model, opt, scheduler)
    
    # 准备后清理一次显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_main(f"[Accelerate] Model prepared and moved to GPU {accelerator.process_index}")
    
    # ============ 如果使用 gradient checkpointing，设置静态图以支持 DDP ============
    if use_grad_checkpointing:
        # 尝试在模型上设置静态图（支持 DDP 与 gradient checkpointing 的兼容性）
        # Accelerate 包装后的模型结构可能不同，需要尝试多种方式
        success = False
        ddp_model = None
        
        # 方法1: 直接检查 model 是否是 DDP
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            ddp_model = model
        # 方法2: 检查 model.module 是否是 DDP（Accelerate 可能这样包装）
        elif hasattr(model, 'module') and isinstance(model.module, torch.nn.parallel.DistributedDataParallel):
            ddp_model = model.module
        # 方法3: 检查 model 是否有 _ddp_wrapped_model 属性（Accelerate 的内部结构）
        elif hasattr(model, '_ddp_wrapped_model'):
            ddp_model = model._ddp_wrapped_model
        # 方法4: 尝试通过 Accelerate 的内部结构访问
        elif hasattr(accelerator, 'state') and hasattr(accelerator.state, 'models'):
            for m in accelerator.state.models:
                if isinstance(m, torch.nn.parallel.DistributedDataParallel):
                    ddp_model = m
                    break
        
        if ddp_model is not None:
            try:
                ddp_model._set_static_graph()
                print_main("[DDP] Successfully set static graph for gradient checkpointing compatibility")
                success = True
            except Exception as e:
                print_main(f"[DDP] Failed to set static graph: {e}")
        
        if not success:
            print_main("[DDP] Warning: Could not set static graph. Gradient checkpointing with DDP may cause errors.")
            print_main("[DDP] Consider disabling gradient checkpointing (--grad_checkpointing=False) if errors occur.")
            print_main("[DDP] Or try setting find_unused_parameters=True (but this may conflict with static graph)")
    
    # ============ 构建 Query Dataset ============
    from collections import defaultdict
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(captions):
        caption_to_indices[cap].append(idx)
    
    query_dataset = QueryDataset(
        num_queries=total_queries,
        caption_to_indices=caption_to_indices,
        captions=captions,
        image_paths=image_paths_train,
        upvotes=upvotes,
        text_matrix=text_matrix,
        image_matrix=image_matrix,
        sim=sim,
        topk_base=topk_base,
        topk_cached=topk_cached,
        topk_base_cached=topk_base_cached,
        extra_negs=extra_negs,
        candidate_mode=candidate_mode,
        label_mode=label_mode
    )
    
    # 使用分布式采样器
    query_loader = DataLoader(
        query_dataset,
        batch_size=1,  # 每个 query 作为一个样本
        shuffle=False,
        sampler=DistributedSampler(
            query_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True
        ),
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # ============ WandB 初始化（仅主进程）============
    best_r1 = -1.0
    best_r5 = -1.0
    best_r10 = -1.0
    
    if accelerator.is_main_process and args and args.wandb_project:
        wandb.init(
            project=args.wandb_project, 
            entity=getattr(args, 'wandb_entity', None), 
            config=vars(args), 
            mode=getattr(args, 'wandb_mode', 'disabled')
        )
        wandb.watch(accelerator.unwrap_model(model), log=None)
    
    # ============ 训练前初始评估（可选）============
    eval_before_train = bool(getattr(args, "eval_before_train", False)) if args is not None else False
    if eval_before_train and accelerator.is_main_process:
        print_main("=" * 80)
        print_main("[Initial Eval] Running evaluation with original (untrained) model...")
        print_main("=" * 80)
        
        # 准备测试数据（与训练后评估使用相同的逻辑）
        test_json_path = os.path.join(CONFIG["data_path"], "test_data.json")
        test_emotion_json_path = os.path.join(CONFIG["data_path"], "test_emotion.json")
        test_dataset = MSRVTT_Dataset(
            csv_path=test_json_path,
            json_path=test_json_path,
            features_path=CONFIG["image_path"],
            emotion_json_path=test_emotion_json_path,
            clip_preprocess=preprocess,
            bert_tokenizer=None,
            max_words=32,
            is_train=False,
            load_image=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
        image_paths_test = build_image_paths(test_dataset, CONFIG["image_path"])
        cache_test_path = os.path.join(CONFIG["save_dir"], "test_cache.pt")
        
        if os.path.exists(cache_test_path):
            print_main(f"[Initial Eval Cache] Loading test cache from: {cache_test_path}")
            obj_t = torch.load(cache_test_path, map_location="cpu", weights_only=False)
            t_text_full = obj_t["text"].cpu()
            t_image_full = obj_t["image"].cpu()
            t_caps_full = obj_t["captions"]
            topk_test = obj_t.get("topk_indices", None)
            print_main(f"[Initial Eval Cache] Loaded test cache: text_matrix shape={t_text_full.shape}, image_matrix shape={t_image_full.shape}, "
                  f"captions={len(t_caps_full)}, topk_cached={'Yes' if topk_test is not None else 'No'}")
            # 如果使用了采样，从完整缓存中截取
            if args is not None and getattr(args, "eval_sample_limit", 0):
                total_eval = len(t_caps_full)
                n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                seed = int(getattr(args, "eval_sample_seed", 0))
                print_main(f"[Initial Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                if mode == "random":
                    rng = np.random.RandomState(seed)
                    indices = rng.choice(total_eval, size=n_eval, replace=False)
                    print_main(f"[Initial Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                else:
                    indices = np.arange(n_eval, dtype=np.int64)
                    print_main(f"[Initial Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                if isinstance(t_text_full, torch.Tensor):
                    t_text = t_text_full[indices]
                else:
                    t_text = t_text_full[indices]
                t_caps = [t_caps_full[int(i)] for i in indices.tolist()]
                t_image = t_image_full  # image pool 保持完整
                topk_test = None
                print_main(f"[Initial Eval] Applied sampling to cached features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
            else:
                # 没有采样限制，使用完整特征
                t_text = t_text_full
                t_caps = t_caps_full
                t_image = t_image_full
        else:
            # 缓存不存在，使用完整dataset计算完整特征并保存
            print_main(f"[Initial Eval Cache] Computing test features (cache not found at {cache_test_path})")
            print_main(f"[Initial Eval Cache] Computing full test features (will save complete cache for future use)")
            t_text_full, t_image_full, _, t_caps_full = compute_features(clip_model, test_loader, torch.device("cpu"))
            topk_test = None
            # 保存完整测试缓存（不采样）
            try:
                os.makedirs(CONFIG["save_dir"], exist_ok=True)
                cache_data = {
                    "text": t_text_full.cpu(),
                    "image": t_image_full.cpu(),
                    "captions": t_caps_full
                }
                torch.save(cache_data, cache_test_path)
                print_main(f"[Initial Eval Cache] Saved complete test cache to: {cache_test_path}")
                print_main(f"[Initial Eval Cache] Cache contents: text_matrix shape={t_text_full.shape}, image_matrix shape={t_image_full.shape}, "
                      f"captions={len(t_caps_full)}")
            except Exception as e:
                print_main(f"[Initial Eval Cache] Warning: Failed to save test cache: {e}")
                import traceback
                traceback.print_exc()
            # 如果使用了采样，从完整特征中截取
            if args is not None and getattr(args, "eval_sample_limit", 0):
                total_eval = len(t_caps_full)
                n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                seed = int(getattr(args, "eval_sample_seed", 0))
                print_main(f"[Initial Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                if mode == "random":
                    rng = np.random.RandomState(seed)
                    indices = rng.choice(total_eval, size=n_eval, replace=False)
                    print_main(f"[Initial Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                else:
                    indices = np.arange(n_eval, dtype=np.int64)
                    print_main(f"[Initial Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                if isinstance(t_text_full, torch.Tensor):
                    t_text = t_text_full[indices]
                else:
                    t_text = t_text_full[indices]
                t_caps = [t_caps_full[int(i)] for i in indices.tolist()]
                t_image = t_image_full  # image pool 保持完整
                print_main(f"[Initial Eval] Applied sampling to computed features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
            else:
                # 没有采样限制，使用完整特征
                t_text = t_text_full
                t_caps = t_caps_full
                t_image = t_image_full
        
        subset_pool = bool(getattr(args, "eval_subset_image_pool", False)) if args else False
        if subset_pool:
            print_main(f"[Initial Eval] Subsetting image pool to match selected queries")
            caps_full = [item["query"] for item in test_dataset.data]
            selected = set(t_caps)
            keep_idx = [i for i, c in enumerate(caps_full) if c in selected]
            if isinstance(t_image, torch.Tensor):
                t_image = t_image[keep_idx]
            if isinstance(image_paths_test, list):
                image_paths_test = [image_paths_test[i] for i in keep_idx]
            print_main(f"[Initial Eval] Image pool: {len(keep_idx)}/{len(caps_full)} images kept")
        else:
            print_main(f"[Initial Eval] Image pool: keeping all {len(image_paths_test) if isinstance(image_paths_test, list) else t_image.shape[0]} images (full pool)")
        
        # 执行初始评估
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        
        # 获取 image pool 的 captions（用于构建 gt 索引）
        image_captions_test = [item["query"] for item in test_dataset.data]
        
        initial_eval_metrics = evaluate_recalls_jina(
            t_caps, image_paths_test, unwrapped_model, processor, 
            topk_base=topk_base, precomputed_topk=topk_test, 
            train_text_matrix=t_text, train_image_matrix=t_image, 
            image_captions=image_captions_test,
            accelerator=accelerator
        )
        
        # 使用格式化函数输出更易读的结果
        print_main(format_eval_metrics(initial_eval_metrics, epoch=0, prefix="[Initial Eval]"))
        
        if wandb.run is not None:
            wandb.log({
                "initial_R@1": initial_eval_metrics["R@1"], "initial_P@1": initial_eval_metrics["P@1"],
                "initial_R@5": initial_eval_metrics["R@5"], "initial_P@5": initial_eval_metrics["P@5"],
                "initial_R@10": initial_eval_metrics["R@10"], "initial_P@10": initial_eval_metrics["P@10"],
                "initial_R@50": initial_eval_metrics["R@50"], "initial_P@50": initial_eval_metrics["P@50"],
                "initial_R@100": initial_eval_metrics["R@100"], "initial_P@100": initial_eval_metrics["P@100"],
                "epoch": 0
            })
    
    accelerator.wait_for_everyone()
    
    # ============ 训练循环 ============
    global_step = 0
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        count = 0
        
        # 设置 epoch（用于 sampler 的 shuffle）
        if hasattr(query_loader.sampler, "set_epoch"):
            query_loader.sampler.set_epoch(epoch)
        
        # 进度条仅在主进程显示
        if accelerator.is_main_process:
            pbar = tqdm(query_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        else:
            pbar = query_loader
        
        for batch in pbar:
            if batch is None or len(batch) == 0:
                continue
            
            # 处理 batch 中的每个 query
            for sample in batch:
                train_indices = sample["train_indices"]
                labels = sample["labels"]
                caption = sample["caption"]
                
                # 加载图像
                imgs = []
                for i in train_indices:
                    if i < len(image_paths_train) and os.path.exists(image_paths_train[i]):
                        try:
                            imgs.append(Image.open(image_paths_train[i]).convert("RGB"))
                        except Exception:
                            imgs.append(Image.new("RGB", (224, 224)))
                    else:
                        imgs.append(Image.new("RGB", (224, 224)))
                
                texts = [caption] * len(imgs)
                target_t = torch.from_numpy(labels).to(accelerator.device).float()
                
                # 使用 accumulate 上下文管理器
                with accelerator.accumulate(model):
                    preds = score_batch(model, processor, texts, imgs, accelerator.device, requires_grad=True)
                    
                    # 检查正负样本对
                    has_positive_pair = (target_t > 0).any() and (target_t == 0).any()
                    if not has_positive_pair:
                        continue
                    
                    if loss_type == "ranknet":
                        loss = ranknet_loss(preds, target_t)
                    elif loss_type == "lambdarank":
                        loss = lambdarank_loss(preds, target_t)
                    else:
                        loss = torch.nn.functional.mse_loss(preds, target_t)
                    
                    # 检查损失有效性
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
                        continue
                    
                    accelerator.backward(loss)
                    
                    # 梯度裁剪
                    # 注意：在 DDP 训练时，使用 accelerator.clip_grad_norm_ 时应该传入模型本身
                    # 或者使用 torch.nn.utils.clip_grad_norm_ 并传入可训练参数
                    if accelerator.sync_gradients:
                        # 方法1: 使用 accelerator 的方法（推荐）
                        try:
                            accelerator.clip_grad_norm_(model, max_norm=1.0)
                        except (TypeError, AttributeError):
                            # 如果 accelerator.clip_grad_norm_ 不支持直接传入模型，使用标准方法
                            # 获取解包的模型的可训练参数
                            unwrapped_model = accelerator.unwrap_model(model)
                            trainable_params = [p for p in unwrapped_model.parameters() if p.requires_grad]
                            if len(trainable_params) > 0:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()
                    opt.zero_grad()
                    
                    global_step += 1
                
                loss_val = float(loss.item())
                if not (math.isnan(loss_val) or math.isinf(loss_val)):
                    total_loss += loss_val
                    count += 1
                
                # 更新进度条
                if accelerator.is_main_process and count % 10 == 0:
                    avg_loss = total_loss / max(count, 1)
                    if hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "samples": count})
        
        # 同步所有进程的统计信息
        total_loss_tensor = torch.tensor([total_loss], device=accelerator.device)
        count_tensor = torch.tensor([count], device=accelerator.device)
        
        total_loss_tensor = accelerator.reduce(total_loss_tensor, reduction="sum")
        count_tensor = accelerator.reduce(count_tensor, reduction="sum")
        
        avg_loss = total_loss_tensor.item() / max(count_tensor.item(), 1)
        
        print_main(f"Epoch {epoch+1} completed. Samples: {int(count_tensor.item())}, Avg loss: {avg_loss:.6f}")
        
        if accelerator.is_main_process and wandb.run is not None:
            wandb.log({"train_loss": avg_loss, "epoch": epoch+1, "lr": opt.param_groups[0]["lr"]})
        
        # ============ 保存 Checkpoint（仅主进程）============
        if accelerator.is_main_process:
            try:
                pre_suffix = f"_jina_m0_lora_{loss_type}_{label_mode}"
                checkpoint_path = os.path.join(CONFIG["save_dir"], f"jina_m0_lora_epoch_{epoch+1}_pre_eval{pre_suffix}.pt")
                
                # 解包模型以保存
                unwrapped_model = accelerator.unwrap_model(model)
                # 只保存可训练的参数（score 层 + 可选的 LoRA）
                if train_lora:
                    # 使用 PEFT 时，state_dict 已经只包含 LoRA + score
                    trainable_state = unwrapped_model.state_dict()
                else:
                    # 不使用 PEFT 时，只保存 score 层参数
                    trainable_state = {}
                    for name, param in unwrapped_model.named_parameters():
                        if param.requires_grad:
                            trainable_state[name] = param.data.clone()
                
                checkpoint_data = {
                    "peft": trainable_state,
                    "model_name": model_name,
                    "train_lora": train_lora,  # 记录是否使用了 LoRA
                }
                torch.save(checkpoint_data, checkpoint_path)
                
                total_params = sum(p.numel() for p in checkpoint_data["peft"].values())
                print_main(f"[Checkpoint] Saved: {checkpoint_path}")
                print_main(f"  - Parameters: {total_params:,}")
            except Exception as e:
                print_main(f"[Checkpoint] Error: {e}")
        
        accelerator.wait_for_everyone()
        
        # ============ 评估（仅主进程）============
        if accelerator.is_main_process:
            test_json_path = os.path.join(CONFIG["data_path"], "test_data.json")
            test_emotion_json_path = os.path.join(CONFIG["data_path"], "test_emotion.json")
            
            test_dataset = MSRVTT_Dataset(
                csv_path=test_json_path,
                json_path=test_json_path,
                features_path=CONFIG["image_path"],
                emotion_json_path=test_emotion_json_path,
                clip_preprocess=preprocess,
                bert_tokenizer=None,
                max_words=32,
                is_train=False,
                load_image=True,
            )
            
            test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
            image_paths_test = build_image_paths(test_dataset, CONFIG["image_path"])
            
            cache_test_path = os.path.join(CONFIG["save_dir"], "test_cache.pt")
            
            if os.path.exists(cache_test_path):
                obj_t = torch.load(cache_test_path, map_location="cpu", weights_only=False)
                t_text = obj_t["text"].cpu()
                t_image = obj_t["image"].cpu()
                t_caps = obj_t["captions"]
                topk_test = obj_t.get("topk_indices", None)
            else:
                # 缓存不存在，使用完整dataset计算完整特征并保存
                print_main(f"[Cache] Computing test features (cache not found at {cache_test_path})")
                print_main(f"[Cache] Computing full test features (will save complete cache for future use)")
                t_text_full, t_image_full, _, t_caps_full = compute_features(clip_model, test_loader, torch.device("cpu"))
                topk_test = None
                # 保存完整测试缓存（不采样）
                try:
                    os.makedirs(CONFIG["save_dir"], exist_ok=True)
                    cache_data = {
                        "text": t_text_full.cpu(),
                        "image": t_image_full.cpu(),
                        "captions": t_caps_full
                    }
                    torch.save(cache_data, cache_test_path)
                    print_main(f"[Cache] Saved complete test cache to: {cache_test_path}")
                    print_main(f"[Cache] Cache contents: text_matrix shape={t_text_full.shape}, image_matrix shape={t_image_full.shape}, "
                          f"captions={len(t_caps_full)}")
                except Exception as e:
                    print_main(f"[Cache] Warning: Failed to save test cache: {e}")
                    import traceback
                    traceback.print_exc()
                # 如果使用了采样，从完整特征中截取
                if args is not None and getattr(args, "eval_sample_limit", 0):
                    total_eval = len(t_caps_full)
                    n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                    mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                    seed = int(getattr(args, "eval_sample_seed", 0))
                    print_main(f"[Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                    if mode == "random":
                        rng = np.random.RandomState(seed)
                        indices = rng.choice(total_eval, size=n_eval, replace=False)
                        print_main(f"[Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                    else:
                        indices = np.arange(n_eval, dtype=np.int64)
                        print_main(f"[Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                    if isinstance(t_text_full, torch.Tensor):
                        t_text = t_text_full[indices]
                    else:
                        t_text = t_text_full[indices]
                    t_caps = [t_caps_full[int(i)] for i in indices.tolist()]
                    t_image = t_image_full  # image pool 保持完整
                    print_main(f"[Eval] Applied sampling to computed features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
                else:
                    # 没有采样限制，使用完整特征
                    t_text = t_text_full
                    t_caps = t_caps_full
                    t_image = t_image_full
            
            # 如果使用了采样，从完整缓存中截取
            if args is not None and getattr(args, "eval_sample_limit", 0):
                total_eval = len(t_caps)
                n_eval = min(int(getattr(args, "eval_sample_limit", 0)), total_eval)
                mode = str(getattr(args, "eval_sample_mode", "first")).lower()
                seed = int(getattr(args, "eval_sample_seed", 0))
                print_main(f"[Eval] Sampling queries: mode={mode}, seed={seed}, n={n_eval}/{total_eval}")
                if mode == "random":
                    rng = np.random.RandomState(seed)
                    indices = rng.choice(total_eval, size=n_eval, replace=False)
                    print_main(f"[Eval] Random sampling: selected {n_eval} queries from {total_eval} total queries")
                else:
                    indices = np.arange(n_eval, dtype=np.int64)
                    print_main(f"[Eval] First-N sampling: selected first {n_eval} queries from {total_eval} total queries")
                if isinstance(t_text, torch.Tensor):
                    t_text = t_text[indices]
                else:
                    t_text = t_text[indices]
                t_caps = [t_caps[int(i)] for i in indices.tolist()]
                t_image = t_image  # image pool 保持完整
                topk_test = None
                print_main(f"[Eval] Applied sampling to cached features: {len(indices)}/{total_eval} queries selected, image pool={t_image.shape[0]} (full)")
            else:
                # 没有采样限制，使用完整特征
                pass  # t_text, t_caps, t_image 已经是完整的
            
            subset_pool = bool(getattr(args, "eval_subset_image_pool", False)) if args else False
            if subset_pool:
                print_main(f"[Eval] Subsetting image pool to match selected queries")
                caps_full = [item["query"] for item in test_dataset.data]
                selected = set(t_caps)
                keep_idx = [i for i, c in enumerate(caps_full) if c in selected]
                if isinstance(t_image, torch.Tensor):
                    t_image = t_image[keep_idx]
                if isinstance(image_paths_test, list):
                    image_paths_test = [image_paths_test[i] for i in keep_idx]
                print_main(f"[Eval] Image pool: {len(keep_idx)}/{len(caps_full)} images kept")
            else:
                print_main(f"[Eval] Image pool: keeping all {len(image_paths_test) if isinstance(image_paths_test, list) else t_image.shape[0]} images (full pool)")
            
            # 评估前确保模型处于eval模式，并检查模型参数是否有效
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()
            
            # 检查模型参数是否有inf/nan
            has_invalid_params = False
            for p in unwrapped_model.parameters():
                if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                    has_invalid_params = True
                    print_main(f"Warning: Model parameters contain inf/nan before evaluation. Evaluation results may be invalid.")
                    break
            if has_invalid_params:
                print_main(f"Skipping evaluation due to invalid model parameters.")
                eval_metrics = {f"R@{k}": 0.0 for k in [1, 3, 5, 10, 50, 100]}
                eval_metrics.update({f"P@{k}": 0.0 for k in [1, 3, 5, 10, 50, 100]})
            else:
                # 获取 image pool 的 captions（用于构建 gt 索引）
                image_captions_test = [item["query"] for item in test_dataset.data]
                
                eval_metrics = evaluate_recalls_jina(
                    t_caps, image_paths_test, unwrapped_model, processor, 
                    topk_base=topk_base, precomputed_topk=topk_test, 
                    train_text_matrix=t_text, train_image_matrix=t_image, 
                    image_captions=image_captions_test,
                    accelerator=accelerator
                )
            
            # 使用格式化函数输出更易读的结果
            print_main(format_eval_metrics(eval_metrics, epoch=epoch+1))
            
            if wandb.run is not None:
                wandb.log({
                    "R@1": eval_metrics["R@1"], "P@1": eval_metrics["P@1"],
                    "R@5": eval_metrics["R@5"], "P@5": eval_metrics["P@5"],
                    "R@10": eval_metrics["R@10"], "P@10": eval_metrics["P@10"],
                    "R@50": eval_metrics["R@50"], "P@50": eval_metrics["P@50"],
                    "R@100": eval_metrics["R@100"], "P@100": eval_metrics["P@100"],
                    "epoch": epoch+1
                })
            
            # 保存最佳模型
            suffix = f"_jina_m0_lora_{loss_type}_{label_mode}"
            keep_all_best = bool(getattr(args, "keep_all_best", False)) if args else False
            
            if eval_metrics["R@1"] > best_r1:
                best_r1 = eval_metrics["R@1"]
                # 只保存可训练的参数
                if train_lora:
                    best_state = unwrapped_model.state_dict()
                else:
                    best_state = {}
                    for name, param in unwrapped_model.named_parameters():
                        if param.requires_grad:
                            best_state[name] = param.data.clone()
                torch.save(
                    {"peft": best_state, "model_name": model_name, "train_lora": train_lora},
                    os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@1{suffix}.pt")
                )
                if keep_all_best:
                    torch.save(
                        {"peft": best_state, "model_name": model_name, "train_lora": train_lora},
                        os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@1_epoch_{epoch+1}{suffix}.pt")
                    )
                print_main(f"[Best] New best R@1: {best_r1:.2f}%")
                if wandb.run is not None:
                    wandb.log({"best_R@1": best_r1})
            
            if eval_metrics["R@5"] > best_r5:
                best_r5 = eval_metrics["R@5"]
                if train_lora:
                    best_state = unwrapped_model.state_dict()
                else:
                    best_state = {}
                    for name, param in unwrapped_model.named_parameters():
                        if param.requires_grad:
                            best_state[name] = param.data.clone()
                torch.save(
                    {"peft": best_state, "model_name": model_name, "train_lora": train_lora},
                    os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@5{suffix}.pt")
                )
                if keep_all_best:
                    torch.save(
                        {"peft": best_state, "model_name": model_name, "train_lora": train_lora},
                        os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@5_epoch_{epoch+1}{suffix}.pt")
                    )
                if wandb.run is not None:
                    wandb.log({"best_R@5": best_r5})
            
            if eval_metrics["R@10"] > best_r10:
                best_r10 = eval_metrics["R@10"]
                if train_lora:
                    best_state = unwrapped_model.state_dict()
                else:
                    best_state = {}
                    for name, param in unwrapped_model.named_parameters():
                        if param.requires_grad:
                            best_state[name] = param.data.clone()
                torch.save(
                    {"peft": best_state, "model_name": model_name, "train_lora": train_lora},
                    os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@10{suffix}.pt")
                )
                if keep_all_best:
                    torch.save(
                        {"peft": best_state, "model_name": model_name, "train_lora": train_lora},
                        os.path.join(CONFIG["save_dir"], f"jina_m0_lora_best_R@10_epoch_{epoch+1}{suffix}.pt")
                    )
                if wandb.run is not None:
                    wandb.log({"best_R@10": best_r10})
        
        accelerator.wait_for_everyone()
    
    # 训练结束
    if accelerator.is_main_process:
        print_main(f"\n[Training] Completed!")
        print_main(f"  Best R@1: {best_r1:.2f}%")
        print_main(f"  Best R@5: {best_r5:.2f}%")
        print_main(f"  Best R@10: {best_r10:.2f}%")
        
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # 数据和训练配置
    parser.add_argument("--data_path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--image_path", type=str, default=CONFIG["image_path"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--save_dir", type=str, default=CONFIG["save_dir"])
    
    # 分布式训练配置
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                       help="Gradient accumulation steps for larger effective batch size")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                       help="Mixed precision training mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # WandB 配置
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--wandb_log_interval", type=int, default=0)
    
    # 模型配置
    parser.add_argument("--topk_base", type=int, default=100)
    parser.add_argument("--extra_negatives", type=int, default=0)
    parser.add_argument("--candidate_mode", type=str, default="topk_plus_pos")
    parser.add_argument("--keep_all_best", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine_warmup")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default="lambdarank")
    parser.add_argument("--label_mode", type=str, default="inv_rank")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-reranker-m0")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_lora", type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)
    
    # 采样配置
    parser.add_argument("--train_sample_limit", type=int, default=0)
    parser.add_argument("--train_sample_mode", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--train_sample_seed", type=int, default=0)
    parser.add_argument("--eval_sample_limit", type=int, default=200)
    parser.add_argument("--eval_sample_mode", type=str, default="random", choices=["first", "random"])
    parser.add_argument("--eval_sample_seed", type=int, default=114)
    parser.add_argument("--eval_subset_image_pool", action="store_true")
    
    # 其他配置
    parser.add_argument("--image_max_side", type=int, default=256)
    parser.add_argument("--jina_micro_batch", type=int, default=64)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--qwen_min_side", type=int, default=32)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--eval_before_train", action="store_true",
                        help="Run evaluation with original (untrained) model before training starts. "
                             "Useful to see baseline performance before training.")
    
    args = parser.parse_args()
    train(args)

