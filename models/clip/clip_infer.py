# clip_infer.py
import os
import sys
import torch
import gc
import numpy as np
from torch.utils.data import DataLoader
import clip
import torch.nn as nn
from transformers import BertModel
import requests
import json
import base64
import time
import threading
import collections
import concurrent.futures
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if project_root not in sys.path:
    sys.path.append(project_root)

from models.rerank.jina_modeling import formatting_prompts_func

from dataset import MSRVTT_Dataset
from models.rerank.rerank_model import RerankLinear, RerankMLP
from models.rerank.jina_modeling import JinaVLForRanking

def compute_recalls(similarity_matrix, all_captions_meta, ks=(1, 5, 10, 50, 100)):
    total = similarity_matrix.shape[0]
    caption_to_indices = {}
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices.setdefault(cap, []).append(idx)
    recalls = {}
    for k in ks:
        k_eff = min(k, similarity_matrix.shape[1])
        hits = 0
        for q_idx in range(total):
            topk = np.argpartition(-similarity_matrix[q_idx], range(k_eff))[:k_eff]
            gt = caption_to_indices[all_captions_meta[q_idx]]
            if any(i in gt for i in topk):
                hits += 1
        recalls[f"R@{k}"] = hits / total * 100.0
    return recalls

def compute_precisions(similarity_matrix, all_captions_meta, ks=(1, 5, 10, 50, 100)):
    total = similarity_matrix.shape[0]
    caption_to_indices = {}
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices.setdefault(cap, []).append(idx)
    precisions = {}
    for k in ks:
        k_eff = min(k, similarity_matrix.shape[1])
        acc = 0.0
        for q_idx in range(total):
            topk = np.argpartition(-similarity_matrix[q_idx], range(k_eff))[:k_eff]
            gt = caption_to_indices[all_captions_meta[q_idx]]
            correct = sum(1 for i in topk if i in gt)
            acc += correct / float(k_eff)
        precisions[f"P@{k}"] = acc / total * 100.0
    return precisions

def compute_recalls_precisions(similarity_matrix, query_captions, image_captions, ks=(1, 5, 10, 50, 100), limit_k=None):
    total = similarity_matrix.shape[0]
    caption_to_indices = {}
    for idx, cap in enumerate(image_captions):
        caption_to_indices.setdefault(cap, []).append(idx)
    recalls = {f"R@{k}": 0.0 for k in ks}
    precisions = {f"P@{k}": 0.0 for k in ks}
    k_max = min(limit_k or max(ks), similarity_matrix.shape[1])
    for q_idx in range(total):
        topk_max = np.argpartition(-similarity_matrix[q_idx], range(k_max))[:k_max]
        gt = caption_to_indices.get(query_captions[q_idx], [])
        for k in ks:
            k_eff = min(k, k_max)
            topk_used = topk_max[:k_eff]
            if any(i in gt for i in topk_used):
                recalls[f"R@{k}"] += 1.0
            correct = sum(1 for i in topk_used if i in gt)
            precisions[f"P@{k}"] += correct / float(k_eff)
    for k in ks:
        recalls[f"R@{k}"] = recalls[f"R@{k}"] / total * 100.0
        precisions[f"P@{k}"] = precisions[f"P@{k}"] / total * 100.0
    return recalls, precisions

def _build_caption_index_map(all_captions_meta):
    caption_to_indices = {}
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices.setdefault(cap, []).append(idx)
    return caption_to_indices

def compute_main_post_recalls(ranked_indices_list, query_video_ids, image_pool_video_ids, ks=(1, 3, 5, 10, 50, 100)):
    """
    计算主贴图片的召回指标。
    每个query对应一个主贴图片，找到它在召回结果中的排名位置。
    
    Args:
        ranked_indices_list: 每个query的排序后的图片索引列表（list of arrays）
        query_video_ids: 每个query对应的主贴图片ID列表（长度等于query数量）
        image_pool_video_ids: 图片池中所有图片的video_id列表（长度等于图片池大小）
        ks: 要计算的k值列表
    
    Returns:
        recalls: 主贴图片的召回率字典，格式为 {f"main_R@{k}": value}
    """
    total = len(ranked_indices_list)
    if total == 0:
        return {f"main_R@{k}": 0.0 for k in ks}
    
    if len(query_video_ids) != total:
        print(f"[Warning] query_video_ids length ({len(query_video_ids)}) != ranked_indices_list length ({total})")
    
    # 构建video_id到索引的映射（图片池中的索引）
    vid_to_idx = {}
    for idx, vid in enumerate(image_pool_video_ids):
        # 处理可能的格式差异（带或不带.jpg后缀）
        vid_key = str(vid).strip()
        vid_key_no_ext = vid_key.replace('.jpg', '').strip()
        # 存储原始格式
        vid_to_idx[vid_key] = idx
        # 如果格式不同，也存储无后缀版本
        if vid_key != vid_key_no_ext:
            vid_to_idx[vid_key_no_ext] = idx
        # 也存储带后缀版本（如果原始没有）
        if '.' not in vid_key:
            vid_to_idx[f"{vid_key}.jpg"] = idx
    
    recalls = {f"main_R@{k}": 0.0 for k in ks}
    not_found_count = 0
    
    for q_idx in range(total):
        if q_idx >= len(query_video_ids):
            continue
            
        query_main_vid = str(query_video_ids[q_idx]).strip()
        query_main_vid_no_ext = query_main_vid.replace('.jpg', '').strip()
        
        # 找到主贴图片在图片池中的索引
        main_post_idx = None
        # 尝试多种匹配方式
        if query_main_vid in vid_to_idx:
            main_post_idx = vid_to_idx[query_main_vid]
        elif query_main_vid_no_ext in vid_to_idx:
            main_post_idx = vid_to_idx[query_main_vid_no_ext]
        elif f"{query_main_vid_no_ext}.jpg" in vid_to_idx:
            main_post_idx = vid_to_idx[f"{query_main_vid_no_ext}.jpg"]
        
        if main_post_idx is None:
            # 如果找不到主贴图片，跳过这个query
            not_found_count += 1
            continue
        
        ranked_indices = ranked_indices_list[q_idx]
        if not isinstance(ranked_indices, np.ndarray):
            ranked_indices = np.array(ranked_indices, dtype=np.int64)
        
        # 找到主贴图片在排序结果中的位置（从1开始）
        try:
            position = np.where(ranked_indices == main_post_idx)[0]
            if len(position) > 0:
                rank = int(position[0]) + 1  # 排名从1开始
            else:
                # 如果不在topk中，排名为总长度+1（表示未召回）
                rank = len(ranked_indices) + 1
        except Exception as e:
            rank = len(ranked_indices) + 1
        
        # 计算各个k值的召回
        for k in ks:
            if rank <= k:
                recalls[f"main_R@{k}"] += 1.0
    
    # 转换为百分比
    valid_total = total - not_found_count
    if valid_total > 0:
        for k in ks:
            recalls[f"main_R@{k}"] = recalls[f"main_R@{k}"] / valid_total * 100.0
    else:
        for k in ks:
            recalls[f"main_R@{k}"] = 0.0
    
    if not_found_count > 0:
        print(f"[Main Post Recall] {not_found_count}/{total} queries的主贴图片未在图片池中找到")
    
    return recalls


def rerank_topk_by_model(q_idx, base_topk_indices, text_matrix, image_matrix, model, device):
    t = text_matrix[q_idx]
    ims = image_matrix[base_topk_indices]
    t_rep = np.repeat(t[None, :], ims.shape[0], axis=0)
    pair = np.concatenate([t_rep, ims], axis=1)
    with torch.no_grad():
        pair_t = torch.from_numpy(pair).to(device).float()
        scores = model(pair_t).detach().cpu().numpy()
    order = np.argsort(-scores)
    return base_topk_indices[order]

def rerank_topk_by_model_torch(q_idx, base_topk_indices, text_matrix_t, image_matrix_t, model, device):
    with torch.no_grad():
        idx_t = torch.as_tensor(base_topk_indices, dtype=torch.long, device=device)
        t = text_matrix_t[q_idx].to(device)
        ims = image_matrix_t.index_select(0, idx_t)
        t_rep = t.unsqueeze(0).repeat(ims.shape[0], 1)
        pair_t = torch.cat([t_rep, ims], dim=1).float()
        scores = model(pair_t).detach().view(-1)
        order = torch.argsort(scores, descending=True)
    return base_topk_indices[order.cpu().numpy()]
class EmotionAdapter(nn.Module):
    def __init__(self, emo_dim=768, clip_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.proj = nn.Sequential(
            nn.Linear(emo_dim, clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(clip_dim * 2, 1),
            nn.Sigmoid()
        )
    def emotion_proj(self, emotion_ids, emotion_mask):
        emo_feat = self.bert(emotion_ids, attention_mask=emotion_mask).pooler_output
        return self.proj(emo_feat)
    def fuse(self, base_feat, emo_proj):
        g = self.gate(torch.cat([base_feat, emo_proj], dim=-1))
        return base_feat + g * emo_proj

class JinaEmotionAdapter(nn.Module):
    """Jina 专用的情感适配器，用于将情感标签融合到 Jina 模型特征中"""
    def __init__(self, emo_dim=768, jina_dim=1536):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.proj = nn.Sequential(
            nn.Linear(emo_dim, jina_dim),
            nn.ReLU(),
            nn.Linear(jina_dim, jina_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(jina_dim * 2, 1),
            nn.Sigmoid()
        )
        # 冻结 BERT 参数（只训练投影层和 gate）
        for p in self.bert.parameters():
            p.requires_grad = False
    
    def emotion_proj(self, emotion_ids, emotion_mask):
        """将情感文本编码并投影到 jina 特征空间"""
        with torch.no_grad():
            emo_feat = self.bert(emotion_ids, attention_mask=emotion_mask).pooler_output
        return self.proj(emo_feat)
    
    def fuse(self, base_feat, emo_proj):
        """融合基础特征和情感特征"""
        # 确保 base_feat 和 emo_proj 的 dtype 一致（PyTorch 会自动提升，但显式转换更安全）
        if base_feat.dtype != emo_proj.dtype:
            emo_proj = emo_proj.to(dtype=base_feat.dtype)
        g = self.gate(torch.cat([base_feat, emo_proj], dim=-1))
        return base_feat + g * emo_proj

def _load_jina_lora(model_name, lora_r, lora_alpha, lora_dropout, device, lora_path):
    """
    加载官方结构的 JinaVLForRanking，本地/LoRA 与 API 对齐：
    - 无 LoRA 时：直接使用预训练 JinaVLForRanking；
    - 有 LoRA 时：在 JinaVLForRanking 上注入 LoRA，并只加载 \"peft\" 权重。
    """
    print(f"[Jina] Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    base = JinaVLForRanking.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch.bfloat16,  # 使用模型原生的 bfloat16
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    base = base.to(device)
    print(f"[Jina] Base model loaded on device: {device}, dtype: bfloat16")
    # 如果lora_path为空或不存在，直接使用原版模型（不应用LoRA）
    if not lora_path or not os.path.exists(lora_path):
        model = base
        model.eval()
        print(f"[Jina] Using base model (no LoRA)")
        return processor, model
    
    # 加载 checkpoint，检查是否使用了 LoRA
    print(f"[Jina] Loading checkpoint from: {lora_path}")
    obj = torch.load(lora_path, map_location=device)
    if not isinstance(obj, dict):
        print(f"[Jina] Warning: Checkpoint is not a dict, using base model")
        model = base
        model.eval()
        return processor, model
    
    peft_state = obj.get("peft") or obj
    model_name_ckpt = obj.get("model_name", "unknown")
    train_lora_ckpt = obj.get("train_lora", True)  # 默认 True（兼容旧 checkpoint）
    
    if train_lora_ckpt:
        # checkpoint 包含 LoRA 参数，需要注入 LoRA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        cfg_lora = LoraConfig(r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout), task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
        model = get_peft_model(base, cfg_lora).to(device)
        model.eval()
        if peft_state:
            model.load_state_dict(peft_state, strict=False)
            # 确保score head是float32（与训练时保持一致）
            if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "score"):
                model.base_model.model.score = model.base_model.model.score.float()
            elif hasattr(model, "score"):
                model.score = model.score.float()
        total_params = sum(p.numel() for p in peft_state.values()) if isinstance(peft_state, dict) else 0
        score_head_dtype = "N/A"
        if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "score"):
            score_head_dtype = str(next(model.base_model.model.score.parameters()).dtype)
        elif hasattr(model, "score"):
            score_head_dtype = str(next(model.score.parameters()).dtype)
        print(f"[Jina] LoRA checkpoint loaded:")
        print(f"  - Path: {lora_path}")
        print(f"  - Model name in checkpoint: {model_name_ckpt}")
        print(f"  - LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"  - Score head dtype: {score_head_dtype}")
        print(f"  - Total parameters: {total_params:,}")
    else:
        # checkpoint 只包含 score 层参数（没有 LoRA）
        model = base
        model.eval()
        if peft_state:
            # 只加载 score 层参数
            # 先过滤出score相关的参数，排除visual和其他不应该保存的参数
            score_state = {k: v for k, v in peft_state.items() if "score" in k.lower() and "visual" not in k.lower()}
            
            # 如果checkpoint中有visual相关的参数，给出警告
            visual_keys = [k for k in peft_state.keys() if "visual" in k.lower()]
            if len(visual_keys) > 0:
                print(f"[Jina] Warning: Checkpoint contains {len(visual_keys)} visual-related parameters (will be ignored): {visual_keys[:3]}...")
            
            if score_state:
                # 使用load_state_dict加载，会自动处理参数名称映射
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(score_state, strict=False)
                    if missing_keys:
                        print(f"[Jina] Warning: Some score parameters not found in model: {missing_keys[:5]}...")
                    if unexpected_keys:
                        print(f"[Jina] Warning: Some unexpected parameters: {unexpected_keys[:5]}...")
                except Exception as e:
                    print(f"[Jina] Warning: load_state_dict failed, trying manual copy: {e}")
                    # 降级到手动copy
                    model_state = model.state_dict()
                    loaded_count = 0
                    for name, param in score_state.items():
                        if name in model_state:
                            # score head参数：确保是float32
                            model_state[name].copy_(param.to(torch.float32))
                            loaded_count += 1
                    print(f"[Jina] Manually loaded {loaded_count} score parameters")
            else:
                print(f"[Jina] Warning: No score parameters found in checkpoint after filtering!")
                print(f"[Jina] Available keys in checkpoint: {list(peft_state.keys())[:10]}...")
            
            # 确保score head是float32（与训练时保持一致）
            if hasattr(model, "score"):
                model.score = model.score.float()
            
            # 统计加载的参数
            score_keys_in_checkpoint = [k for k in peft_state.keys() if "score" in k.lower() and "visual" not in k.lower()]
            loaded_count = sum(1 for k in score_keys_in_checkpoint if k in model.state_dict())
            print(f"[Jina] Score-only checkpoint loaded (no LoRA):")
            print(f"  - Path: {lora_path}")
            print(f"  - Model name in checkpoint: {model_name_ckpt}")
            print(f"  - Score parameters in checkpoint: {len(score_keys_in_checkpoint)}")
            print(f"  - Loaded {loaded_count} score parameters")
            if len(score_keys_in_checkpoint) > 0:
                print(f"  - Score parameter keys: {score_keys_in_checkpoint[:5]}...")
            print(f"  - Score head dtype: {next(model.score.parameters()).dtype if hasattr(model, 'score') else 'N/A'}")
            # 只统计score相关的参数数量
            score_params_count = sum(p.numel() for k, p in peft_state.items() if "score" in k.lower() and "visual" not in k.lower()) if isinstance(peft_state, dict) else 0
            total_params = sum(p.numel() for p in peft_state.values()) if isinstance(peft_state, dict) else 0
            print(f"  - Score parameters in checkpoint: {score_params_count:,}")
            if total_params != score_params_count:
                print(f"  - Total parameters in checkpoint: {total_params:,} (includes non-score parameters that will be ignored)")
    
    return processor, model

def _forward_with_emotion_fusion_jina(model, inputs, emotion_adapter=None, emotion_features=None, device=None):
    """
    带情感融合的前向传播函数（用于推理）
    在不修改模型类的情况下，拦截 hidden_states 并融合情感特征
    """
    # 获取模型的基础类（可能是 PEFT 包装的）
    base_model = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_model = model.base_model.model
    elif hasattr(model, "module"):
        base_model = model.module
        if hasattr(base_model, "base_model") and hasattr(base_model.base_model, "model"):
            base_model = base_model.base_model.model
    
    # 调用基础模型的 forward，获取 hidden_states
    kwargs = inputs.copy()
    kwargs.pop("output_hidden_states", None)
    kwargs.pop("use_cache", None)
    
    with torch.inference_mode():
        outputs = base_model.forward(
            **kwargs,
            use_cache=False,
            output_hidden_states=True,
        )
    
    # 获取最后一个 token 的 hidden state
    hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
    pooled_features = hidden_states[:, -1]  # [batch_size, hidden_size]
    
    # 如果提供了情感特征，进行融合
    if emotion_features is not None and emotion_adapter is not None:
        # emotion_features 应该是 [batch_size, hidden_size]
        if emotion_features.shape[0] == pooled_features.shape[0]:
            pooled_features = emotion_adapter.fuse(pooled_features, emotion_features)
    
    # 通过 score head 得到最终分数
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        score_head = model.base_model.model.score
    elif hasattr(model, "module") and hasattr(model.module, "score"):
        score_head = model.module.score
    elif hasattr(model, "score"):
        score_head = model.score
    else:
        # 如果找不到 score head，使用基础模型的
        score_head = base_model.score
    
    pooled_logits = score_head(pooled_features)
    return pooled_logits.squeeze(-1)

def _score_batch_jina(model, processor, texts, images, device, image_max_side=256, qwen_min_side=32, jina_micro_batch=0, max_text_len=2048, emotion_adapter=None, emotion_ids=None, emotion_mask=None, use_emotion_fusion=False):
    msgs = []
    imgs_list = []
    from PIL import Image as _I
    max_side_cfg = int(image_max_side)
    min_side_cfg = int(qwen_min_side)
    mb = int(jina_micro_batch) if jina_micro_batch else 0
    for t, img in zip(texts, images):
        if img is None:
            img = _I.new("RGB", (224, 224), (0, 0, 0))
        else:
            w, h = img.size
            # First ensure minimum side
            if min(w, h) < min_side_cfg:
                scale_up = float(min_side_cfg) / float(min(w, h))
                w = int(round(w * scale_up))
                h = int(round(h * scale_up))
                img = img.resize((w, h), Image.BICUBIC)
            # Then clamp maximum side
            max_side = max_side_cfg
            if max(w, h) > max_side:
                scale_down = float(max_side) / float(max(w, h))
                w = int(round(w * scale_down))
                h = int(round(h * scale_down))
                img = img.resize((w, h), Image.BICUBIC)
            # Final pad if rounding still violates min side
            w, h = img.size
            if min(w, h) < min_side_cfg:
                w2 = max(w, min_side_cfg)
                h2 = max(h, min_side_cfg)
                canvas = _I.new("RGB", (w2, h2), (0, 0, 0))
                canvas.paste(img, ((w2 - w) // 2, (h2 - h) // 2))
                img = canvas
        imgs_list.append(img)
    mdl_dev = next(model.parameters()).device
    outs = []
    if mb and mb > 0:
        # 使用micro batch分批处理以节省内存
        for i in range(0, len(imgs_list), mb):
            # 使用官方的 formatting_prompts_func 格式化 prompt（与官方实现对齐）
            # 对于 reranking：query 是文本，doc 是图像
            batch_texts = []
            batch_images = []
            for j in range(i, min(i + mb, len(texts))):
                # 对于每个 query-document 对，使用官方格式
                # 注意：这里 texts[j] 是同一个 query，imgs_list[j] 是不同的 document
                prompt_text = formatting_prompts_func(
                    query=texts[j],
                    doc="",  # doc 是图像，在 prompt 中用占位符
                    query_type='text',
                    doc_type='image'
                )
                batch_texts.append(prompt_text)
                batch_images.append(imgs_list[j])
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True, truncation=True, max_length=int(max_text_len))
            inputs = {k: v.to(mdl_dev) for k, v in inputs.items()}
            try:
                tok_id = int(getattr(model, "score_token_id", 100))
                bsz = inputs["input_ids"].size(0)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((bsz, 1), tok_id, device=inputs["input_ids"].device)], dim=1)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((bsz, 1), device=inputs["attention_mask"].device)], dim=1)
            except Exception:
                pass
            
            # 处理情感特征（如果启用）
            emotion_features_batch = None
            if use_emotion_fusion and emotion_adapter is not None and emotion_ids is not None and emotion_mask is not None:
                # 提取当前 batch 的情感数据
                batch_start = i
                batch_end = min(i + mb, len(texts))
                if isinstance(emotion_ids, list):
                    emotion_ids_batch = emotion_ids[batch_start:batch_end]
                    emotion_mask_batch = emotion_mask[batch_start:batch_end]
                else:
                    emotion_ids_batch = emotion_ids[batch_start:batch_end]
                    emotion_mask_batch = emotion_mask[batch_start:batch_end]
                
                # 确保是 tensor 格式
                if not isinstance(emotion_ids_batch, torch.Tensor):
                    emotion_ids_batch = torch.tensor(emotion_ids_batch, device=mdl_dev)
                if not isinstance(emotion_mask_batch, torch.Tensor):
                    emotion_mask_batch = torch.tensor(emotion_mask_batch, device=mdl_dev)
                
                # 如果 emotion_ids_batch 是单个样本，需要扩展到 batch size
                if emotion_ids_batch.dim() == 1:
                    emotion_ids_batch = emotion_ids_batch.unsqueeze(0).expand(bsz, -1)
                    emotion_mask_batch = emotion_mask_batch.unsqueeze(0).expand(bsz, -1)
                
                # 提取情感特征
                emotion_features_batch = emotion_adapter.emotion_proj(emotion_ids_batch, emotion_mask_batch)
            
            # 使用带情感融合的前向传播（如果启用）
            with torch.inference_mode():
                if mdl_dev.type == "cuda":
                    from torch import amp as _amp
                    with _amp.autocast("cuda", dtype=torch.bfloat16):
                        if use_emotion_fusion and emotion_adapter is not None:
                            outputs = _forward_with_emotion_fusion_jina(
                                model, inputs, emotion_adapter=emotion_adapter,
                                emotion_features=emotion_features_batch,
                                device=mdl_dev
                            )
                        else:
                            outputs = model(**inputs)
                else:
                    if use_emotion_fusion and emotion_adapter is not None:
                        outputs = _forward_with_emotion_fusion_jina(
                            model, inputs, emotion_adapter=emotion_adapter,
                            emotion_features=emotion_features_batch,
                            device=mdl_dev
                        )
                    else:
                        outputs = model(**inputs)
            outs.append(outputs if isinstance(outputs, torch.Tensor) else torch.as_tensor(outputs, device=mdl_dev))
            # 清理中间变量（不频繁调用 empty_cache，避免拖慢推理速度）
            del inputs, outputs
        out = torch.cat(outs, dim=0)
    else:
        # 使用官方的 formatting_prompts_func 格式化 prompt（与官方实现对齐）
        batch_texts = []
        for j in range(len(texts)):
            prompt_text = formatting_prompts_func(
                query=texts[j],
                doc="",  # doc 是图像，在 prompt 中用占位符
                query_type='text',
                doc_type='image'
            )
            batch_texts.append(prompt_text)
        inputs = processor(text=batch_texts, images=imgs_list, return_tensors="pt", padding=True, truncation=True, max_length=int(max_text_len))
        inputs = {k: v.to(mdl_dev) for k, v in inputs.items()}
        try:
            tok_id = int(getattr(model, "score_token_id", 100))
            bsz = inputs["input_ids"].size(0)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((bsz, 1), tok_id, device=inputs["input_ids"].device)], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((bsz, 1), device=inputs["attention_mask"].device)], dim=1)
        except Exception:
            pass
        
        # 处理情感特征（如果启用）
        emotion_features_batch = None
        if use_emotion_fusion and emotion_adapter is not None and emotion_ids is not None and emotion_mask is not None:
            if isinstance(emotion_ids, list):
                emotion_ids_tensor = torch.tensor(emotion_ids, device=mdl_dev)
                emotion_mask_tensor = torch.tensor(emotion_mask, device=mdl_dev)
            else:
                emotion_ids_tensor = emotion_ids.to(mdl_dev) if hasattr(emotion_ids, 'to') else torch.tensor(emotion_ids, device=mdl_dev)
                emotion_mask_tensor = emotion_mask.to(mdl_dev) if hasattr(emotion_mask, 'to') else torch.tensor(emotion_mask, device=mdl_dev)
            
            # 如果 emotion_ids_tensor 是单个样本，需要扩展到 batch size
            if emotion_ids_tensor.dim() == 1:
                emotion_ids_tensor = emotion_ids_tensor.unsqueeze(0).expand(bsz, -1)
                emotion_mask_tensor = emotion_mask_tensor.unsqueeze(0).expand(bsz, -1)
            
            emotion_features_batch = emotion_adapter.emotion_proj(emotion_ids_tensor, emotion_mask_tensor)
        
        # 使用带情感融合的前向传播（如果启用）
        with torch.inference_mode():
            if mdl_dev.type == "cuda":
                from torch import amp as _amp
                with _amp.autocast("cuda", dtype=torch.bfloat16):
                    if use_emotion_fusion and emotion_adapter is not None:
                        out = _forward_with_emotion_fusion_jina(
                            model, inputs, emotion_adapter=emotion_adapter,
                            emotion_features=emotion_features_batch,
                            device=mdl_dev
                        )
                    else:
                        out = model(**inputs)
            else:
                if use_emotion_fusion and emotion_adapter is not None:
                    out = _forward_with_emotion_fusion_jina(
                        model, inputs, emotion_adapter=emotion_adapter,
                        emotion_features=emotion_features_batch,
                        device=mdl_dev
                    )
                else:
                    out = model(**inputs)
        # 清理中间变量（不调用 empty_cache，避免拖慢推理速度）
        del inputs
    # 转换为 float32，因为 numpy 不支持 bfloat16
    return out.detach().cpu().float().view(-1)

def collect_features(clip_model, dataloader, device, emotion_adapter=None, use_image_emotion_fusion=False, use_text_emotion_fusion=False):
    text_feats = []
    image_feats = []
    clip_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            query_ids = batch["query_ids"].to(device)
            img = clip_model.encode_image(images)
            txt = clip_model.encode_text(query_ids)
            if use_image_emotion_fusion or use_text_emotion_fusion:
                emotion_ids = batch["emotion_input_ids"].to(device)
                emotion_mask = batch["emotion_attention_mask"].to(device)
                emo_proj = emotion_adapter.emotion_proj(emotion_ids, emotion_mask)
                if use_image_emotion_fusion:
                    img = emotion_adapter.fuse(img, emo_proj)
                if use_text_emotion_fusion:
                    txt = emotion_adapter.fuse(txt, emo_proj)
            img = img / img.norm(dim=1, keepdim=True).clamp(min=1e-6)
            txt = txt / txt.norm(dim=1, keepdim=True).clamp(min=1e-6)
            image_feats.append(img.cpu().numpy())
            text_feats.append(txt.cpu().numpy())
    text_matrix = np.concatenate(text_feats, axis=0)
    image_matrix = np.concatenate(image_feats, axis=0)
    return text_matrix, image_matrix

def apply_image_projection(image_matrix, image_projection_path, projection_mix=1.0):
    if image_projection_path is None or image_projection_path == "":
        return image_matrix
    if not os.path.exists(image_projection_path):
        print(f"[Projection] WARNING: projection file not found: {image_projection_path}; skipping projection")
        return image_matrix
    print(f"[Projection] Loading image projection from: {image_projection_path} (mix={projection_mix})")
    proj_state = torch.load(image_projection_path, map_location="cpu", weights_only=False)
    W = proj_state["W"] if isinstance(proj_state, dict) and "W" in proj_state else proj_state
    if isinstance(W, torch.Tensor):
        W = W.cpu().numpy()
    orig_shape = image_matrix.shape
    if image_matrix.ndim == 1:
        image_matrix = image_matrix.reshape(1, -1)
    original_matrix = image_matrix
    projected_matrix = image_matrix @ W.T
    mix = float(projection_mix) if projection_mix is not None else 1.0
    if mix <= 0.0:
        image_matrix = original_matrix
    elif mix >= 1.0:
        image_matrix = projected_matrix
    else:
        image_matrix = (1.0 - mix) * original_matrix + mix * projected_matrix
    norms = np.linalg.norm(image_matrix, axis=1, keepdims=True)
    image_matrix = image_matrix / (norms + 1e-8)
    if len(orig_shape) == 1:
        image_matrix = image_matrix.reshape(-1)
    print(f"[Projection] Applied {W.shape} projection; output shape={image_matrix.shape}")
    return image_matrix

def load_or_compute_test_cache(data_path, image_path, batch_size, device, clip_model, preprocess, topk_base, save_dir, use_image_emotion_fusion=False, use_text_emotion_fusion=False, emotion_adapter=None, zero_shot=False, sample_limit=0, sample_mode="first", sample_seed=0, subset_image_pool=False, image_projection_path=None, projection_mix=1.0):
    from torch.utils.data import DataLoader
    if use_image_emotion_fusion or use_text_emotion_fusion:
        cache_test_path = os.path.join(save_dir, f"test_cache_emofuse_{int(use_image_emotion_fusion)}_{int(use_text_emotion_fusion)}.pt")
        # cache_test_path = os.path.join(save_dir, "test_cache.pt")
    elif zero_shot:
        cache_test_path = os.path.join(save_dir, "test_cache_zeroshot.pt")
    else:
        cache_test_path = os.path.join(save_dir, "test_cache.pt")
    test_json_path = os.path.join(data_path, "test_data.json")
    test_emotion_json_path = os.path.join(data_path, "test_emotion.json")
    bert_tokenizer = None
    if use_image_emotion_fusion or use_text_emotion_fusion:
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    if os.path.exists(cache_test_path):
        print(f"[Cache] Loading test cache from: {cache_test_path}")
        obj = torch.load(cache_test_path, map_location=device, weights_only=False)
        t_text = obj["text"].cpu().numpy()
        t_image = obj["image"].cpu().numpy()
        t_image_orig = t_image.copy()
        t_image = apply_image_projection(t_image, image_projection_path, projection_mix=projection_mix)
        captions = obj["captions"]
        captions_full = list(captions)
        topk = obj.get("topk_indices", None)
        if image_projection_path:
            topk = None  # cached top-k indices are based on unprojected features
        vids = obj.get("video_ids", None)
        print(f"[Cache] Loaded: text_matrix shape={t_text.shape}, image_matrix shape={t_image.shape}, "
              f"captions={len(captions)}, topk_cached={'Yes' if topk is not None else 'No'}")
        query_video_ids = obj.get("query_video_ids", None)  # query对应的主贴图片ID列表
        if vids is None:
            test_dataset_tmp = MSRVTT_Dataset(
                csv_path=test_json_path,
                json_path=test_json_path,
                features_path=image_path,
                emotion_json_path=test_emotion_json_path,
                clip_preprocess=preprocess,
                bert_tokenizer=bert_tokenizer,
                max_words=32,
                is_train=False,
                load_image=False,
            )
            vids = [item["video_id"] for item in test_dataset_tmp.data]
            if query_video_ids is None:
                query_video_ids = [item["video_id"] for item in test_dataset_tmp.data]
        if query_video_ids is None:
            # 如果没有保存query_video_ids，从vids推断（假设vids的前len(captions)个是query对应的）
            query_video_ids = vids[:len(captions)] if len(vids) >= len(captions) else vids
        if sample_limit and int(sample_limit) > 0:
            total = len(captions)
            n = min(int(sample_limit), total)
            if str(sample_mode).lower() == "random":
                rng = np.random.RandomState(int(sample_seed) if sample_seed is not None else 0)
                indices = rng.choice(total, size=n, replace=False)
                print(f"[Cache] Sampling queries: mode=random, seed={int(sample_seed) if sample_seed is not None else 0}, n={n}/{total}")
            else:
                indices = np.arange(n, dtype=np.int64)
                print(f"[Cache] Sampling queries: mode=first, n={n}/{total}")
            t_text = t_text[indices]
            captions = [captions[int(i)] for i in indices.tolist()]
            if query_video_ids is not None:
                query_video_ids = [query_video_ids[int(i)] for i in indices.tolist()]
            topk = None
            if subset_image_pool:
                selected = set(captions)
                keep_idx = [i for i, c in enumerate(captions_full) if c in selected]
                t_image = t_image[keep_idx]
                t_image_orig = t_image_orig[keep_idx]
                if isinstance(vids, list):
                    vids = [vids[i] for i in keep_idx]
                captions_full = [captions_full[i] for i in keep_idx]
        return t_text, t_image, t_image_orig, captions, topk, vids, captions_full, query_video_ids
        
    print(f"[Cache] Computing test features (cache not found at {cache_test_path})")
    test_dataset = MSRVTT_Dataset(
        csv_path=test_json_path,
        json_path=test_json_path,
        features_path=image_path,
        emotion_json_path=test_emotion_json_path,
        clip_preprocess=preprocess,
        bert_tokenizer=bert_tokenizer,
        max_words=32,
        is_train=False,
        load_image=True,
    )
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    text_matrix, image_matrix = collect_features(
        clip_model, dataloader, device,
        emotion_adapter=emotion_adapter,
        use_image_emotion_fusion=use_image_emotion_fusion,
        use_text_emotion_fusion=use_text_emotion_fusion,
    )
    image_matrix_orig = image_matrix.copy()
    image_matrix = apply_image_projection(image_matrix, image_projection_path, projection_mix=projection_mix)
    sim = np.dot(text_matrix, image_matrix.T)
    k_base = min(topk_base, sim.shape[1])
    topk = [np.argpartition(-sim[q], range(k_base))[:k_base] for q in range(sim.shape[0])]
    topk_py = [arr.tolist() for arr in topk]
    # 始终保存完整特征缓存（即使使用了采样），这样下次可以快速截取
    os.makedirs(save_dir, exist_ok=True)
    query_video_ids_list = [item["video_id"] for item in test_dataset.data]
    torch.save({
        "text": torch.from_numpy(text_matrix),
        "image": torch.from_numpy(image_matrix_orig),
        "captions": [item["query"] for item in test_dataset.data],
        "topk_base": topk_base,
        "topk_indices": topk_py,
        "video_ids": [item["video_id"] for item in test_dataset.data],
        "query_video_ids": query_video_ids_list,  # 保存query对应的主贴图片ID列表
    }, cache_test_path)
    print(f"[Cache] Saved test cache to: {cache_test_path}")
    print(f"[Cache] Cache contents: text_matrix shape={text_matrix.shape}, image_matrix shape={image_matrix.shape}, "
          f"captions={len([item['query'] for item in test_dataset.data])}, topk_base={topk_base}")
    if sample_limit and int(sample_limit) > 0:
        total = len(test_dataset)
        n = min(int(sample_limit), total)
        if str(sample_mode).lower() == "random":
            rng = np.random.RandomState(int(sample_seed) if sample_seed is not None else 0)
            indices = rng.choice(total, size=n, replace=False)
            print(f"[Cache] Sampling queries: mode=random, seed={int(sample_seed) if sample_seed is not None else 0}, n={n}/{total}")
        else:
            indices = np.arange(n, dtype=np.int64)
            print(f"[Cache] Sampling queries: mode=first, n={n}/{total}")
        t_text_subset = text_matrix[indices]
        captions_all = [item["query"] for item in test_dataset.data][:total]
        captions_subset = [captions_all[int(i)] for i in indices.tolist()]
        vids_all = [item["video_id"] for item in test_dataset.data][:total]
        vids_subset = [vids_all[int(i)] for i in indices.tolist()]
        query_video_ids_subset = [vids_all[int(i)] for i in indices.tolist()]  # query对应的主贴图片ID
        topk_subset = None
        captions_image_pool = [item["query"] for item in test_dataset.data]
        if subset_image_pool:
            selected = set(captions_subset)
            keep_idx = [i for i, c in enumerate(captions_image_pool) if c in selected]
            image_matrix = image_matrix[keep_idx]
            image_matrix_orig = image_matrix_orig[keep_idx]
            vids_all = [vids_all[i] for i in keep_idx]
            captions_image_pool = [captions_image_pool[i] for i in keep_idx]
            print(f"[Cache] Subset image pool: {len(keep_idx)}/{total} images kept")
        vids_out = vids_subset if subset_image_pool else vids_all
        query_video_ids_out = query_video_ids_subset if subset_image_pool else query_video_ids_list
        return t_text_subset, image_matrix, image_matrix_orig, captions_subset, topk_subset, vids_out, captions_image_pool, query_video_ids_out
    query_video_ids_list = [item["video_id"] for item in test_dataset.data]
    return text_matrix, image_matrix, image_matrix_orig, [item["query"] for item in test_dataset.data], topk, [item["video_id"] for item in test_dataset.data], [item["query"] for item in test_dataset.data], query_video_ids_list

def _encode_image_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _build_image_path(image_dir, vid):
    return os.path.join(image_dir, vid if "." in vid else f"{vid}.jpg")

def _safe_clip_indices(arr, max_len):
    try:
        a = np.array(arr, dtype=np.int64)
    except Exception:
        return arr
    if max_len is None or int(max_len) <= 0:
        return a
    mask = (a >= 0) & (a < int(max_len))
    return a[mask]

def output_query_results(queries, ranked_indices_list, all_captions, image_caps_full, video_ids, 
                        image_path, output_dir, top_k=10, base_ranked_indices_list=None, 
                        base_scores_list=None, rerank_scores_list=None, test_dataset=None):
    """
    将评测结果输出到文件夹，包含 markdown 文件和图片。
    只输出 top k 结果中有成功召回的 query。
    
    Args:
        queries: query 索引列表
        ranked_indices_list: 每个 query 的排序后的图片索引列表（rerank 后的结果）
        all_captions: query 的 caption 列表
        image_caps_full: 图片池中所有图片的 caption 列表
        video_ids: 图片池中所有图片的 video_id 列表
        image_path: 图片文件路径
        output_dir: 输出目录
        top_k: 输出前 k 个结果（默认 10）
        base_ranked_indices_list: base 模型的排序结果（可选）
        base_scores_list: base 模型的分数列表（可选）
        rerank_scores_list: rerank 模型的分数列表（可选）
        test_dataset: 测试数据集对象，用于获取图片的表情、文字等信息（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 构建 caption 到索引的映射（用于判断是否正确）
    caption_to_indices = {}
    for idx, cap in enumerate(image_caps_full):
        caption_to_indices.setdefault(cap, []).append(idx)
    
    # 构建 video_id 到数据集索引的映射（用于获取图片信息）
    vid_to_dataset_idx = {}
    if test_dataset is not None:
        for idx, item in enumerate(test_dataset.data):
            vid = item.get("video_id", "")
            if vid:
                # 存储多种格式的映射（带或不带 .jpg 后缀）
                vid_key = str(vid).strip()
                vid_key_no_ext = vid_key.replace('.jpg', '').replace('.JPG', '').strip()
                # 存储原始格式
                vid_to_dataset_idx[vid_key] = idx
                # 存储无后缀版本
                if vid_key != vid_key_no_ext:
                    vid_to_dataset_idx[vid_key_no_ext] = idx
                # 存储带 .jpg 后缀版本（如果原始没有）
                if '.' not in vid_key:
                    vid_to_dataset_idx[f"{vid_key}.jpg"] = idx
                    vid_to_dataset_idx[f"{vid_key}.JPG"] = idx
        print(f"[Output] Built video_id mapping: {len(vid_to_dataset_idx)} entries from {len(test_dataset.data)} dataset items")
        # 调试：显示前几个 video_id 示例
        if len(test_dataset.data) > 0:
            sample_vids = [item.get("video_id", "") for item in test_dataset.data[:3]]
            print(f"[Output] Sample video_ids from dataset: {sample_vids}")
    else:
        print(f"[Output] Warning: test_dataset is None, cannot get image info")
    
    # 调试：显示 video_ids 列表的前几个示例
    if video_ids is not None and len(video_ids) > 0:
        sample_vids_list = video_ids[:3] if len(video_ids) >= 3 else video_ids
        print(f"[Output] Sample video_ids from video_ids list: {sample_vids_list}")
    
    # 过滤：只保留 top k 中有成功召回的 query
    filtered_queries = []
    for q_idx in queries:
        if q_idx >= len(all_captions) or q_idx >= len(ranked_indices_list):
            continue
        
        query_text = all_captions[q_idx]
        ranked_indices = ranked_indices_list[q_idx]
        gt_indices = set(caption_to_indices.get(query_text, []))
        
        # 检查 top k 中是否有成功召回
        top_k_indices = ranked_indices[:top_k] if len(ranked_indices) >= top_k else ranked_indices
        has_correct = any(idx_i in gt_indices for idx_i in top_k_indices)
        
        if has_correct:
            filtered_queries.append(q_idx)
    
    markdown_lines = []
    markdown_lines.append("# 评测结果可视化\n\n")
    markdown_lines.append(f"共输出 {len(filtered_queries)} 个 query 的结果（top {top_k} 中有成功召回的），每个 query 显示 top {top_k} 个结果。\n\n")
    if test_dataset is not None:
        markdown_lines.append(f"数据集信息：共 {len(test_dataset.data)} 条数据，video_id 映射：{len(vid_to_dataset_idx)} 条。\n\n")
    markdown_lines.append("---\n\n")
    
    # 统计匹配情况（用于调试）
    match_count = 0
    total_count = 0
    
    for q_idx in filtered_queries:
        if q_idx >= len(all_captions) or q_idx >= len(ranked_indices_list):
            continue
        
        query_text = all_captions[q_idx]
        ranked_indices = ranked_indices_list[q_idx]
        
        # 获取 ground truth 索引
        gt_indices = set(caption_to_indices.get(query_text, []))
        
        # 限制到 top_k
        top_k_indices = ranked_indices[:top_k] if len(ranked_indices) >= top_k else ranked_indices
        
        markdown_lines.append(f"## Query {q_idx + 1}: {query_text}\n\n")
        markdown_lines.append(f"**Query 文本**: `{query_text}`\n\n")
        
        # 判断是否有 rerank 结果
        has_rerank = rerank_scores_list is not None and len(rerank_scores_list) > q_idx
        
        # 辅助函数：获取图片信息
        def get_image_info(vid, debug=False):
            """获取图片的表情、文字等信息"""
            nonlocal match_count, total_count
            emotions_str = "N/A"
            texts_str = "N/A"
            total_count += 1
            if test_dataset is not None and vid_to_dataset_idx:
                vid_key = str(vid).strip()
                vid_key_no_ext = vid_key.replace('.jpg', '').replace('.JPG', '').strip()
                # 尝试多种格式匹配
                dataset_idx = None
                if vid_key in vid_to_dataset_idx:
                    dataset_idx = vid_to_dataset_idx[vid_key]
                elif vid_key_no_ext in vid_to_dataset_idx:
                    dataset_idx = vid_to_dataset_idx[vid_key_no_ext]
                elif f"{vid_key_no_ext}.jpg" in vid_to_dataset_idx:
                    dataset_idx = vid_to_dataset_idx[f"{vid_key_no_ext}.jpg"]
                elif f"{vid_key_no_ext}.JPG" in vid_to_dataset_idx:
                    dataset_idx = vid_to_dataset_idx[f"{vid_key_no_ext}.JPG"]
                
                if dataset_idx is not None and dataset_idx < len(test_dataset.data):
                    match_count += 1
                    item = test_dataset.data[dataset_idx]
                    # 获取表情信息
                    if "emotions" in item and item["emotions"]:
                        emotions_list = [str(e) for e in item["emotions"] if e]
                        if emotions_list:
                            emotions_str = ", ".join(emotions_list)
                    # 获取文字信息（候选文本）
                    if "candidate_texts" in item and item["candidate_texts"]:
                        if isinstance(item["candidate_texts"], list) and len(item["candidate_texts"]) > 0:
                            texts_str = str(item["candidate_texts"][0])
                        elif isinstance(item["candidate_texts"], str):
                            texts_str = item["candidate_texts"]
            return emotions_str, texts_str
        
        # 如果有 base 模型结果且也有 rerank 结果，显示 base 模型作为对比
        if base_ranked_indices_list is not None and q_idx < len(base_ranked_indices_list) and len(base_ranked_indices_list[q_idx]) > 0 and has_rerank:
            base_top_k = base_ranked_indices_list[q_idx][:top_k]
            markdown_lines.append("### Base 模型结果（对比）\n\n")
            markdown_lines.append("| 排名 | 图片 | Video ID | 是否正确 | 相似度 | 表情 | 文字 |\n")
            markdown_lines.append("|------|------|----------|----------|--------|------|------|\n")
            
            for rank, idx_i in enumerate(base_top_k, start=1):
                vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                is_correct = "✅ 正确" if idx_i in gt_indices else "❌ 错误"
                score = base_scores_list[q_idx][rank-1] if base_scores_list and q_idx < len(base_scores_list) and rank-1 < len(base_scores_list[q_idx]) else "N/A"
                # 格式化分数
                score_str = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
                # 获取图片信息
                emotions_str, texts_str = get_image_info(vid)
                # 限制文字长度
                texts_str = texts_str[:50] + "..." if len(texts_str) > 50 else texts_str
                
                # 复制图片到输出目录
                img_filename = f"query_{q_idx}_base_rank_{rank}_{vid.replace('/', '_').replace('.jpg', '')}.jpg"
                img_output_path = os.path.join(images_dir, img_filename)
                img_source_path = _build_image_path(image_path, vid)
                
                if os.path.exists(img_source_path):
                    try:
                        from shutil import copyfile
                        copyfile(img_source_path, img_output_path)
                        img_rel_path = f"images/{img_filename}"
                        markdown_lines.append(f"| {rank} | ![img]({img_rel_path}) | `{vid}` | {is_correct} | {score_str} | {emotions_str} | {texts_str} |\n")
                    except Exception as e:
                        markdown_lines.append(f"| {rank} | 图片加载失败 | `{vid}` | {is_correct} | {score_str} | {emotions_str} | {texts_str} |\n")
                else:
                    markdown_lines.append(f"| {rank} | 图片不存在 | `{vid}` | {is_correct} | {score_str} | {emotions_str} | {texts_str} |\n")
            
            markdown_lines.append("\n")
        
        # Rerank 模型结果（如果有）或 Base 模型结果（如果没有 rerank）
        if has_rerank:
            markdown_lines.append("### Rerank 模型结果\n\n")
        else:
            markdown_lines.append("### Base 模型结果\n\n")
        markdown_lines.append("| 排名 | 图片 | Video ID | 是否正确 | 分数 | 表情 | 文字 |\n")
        markdown_lines.append("|------|------|----------|----------|------|------|------|\n")
        
        for rank, idx_i in enumerate(top_k_indices, start=1):
            vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
            is_correct = "✅ 正确" if idx_i in gt_indices else "❌ 错误"
            if has_rerank:
                score = rerank_scores_list[q_idx][rank-1] if rerank_scores_list and q_idx < len(rerank_scores_list) and rank-1 < len(rerank_scores_list[q_idx]) else "N/A"
            else:
                score = base_scores_list[q_idx][rank-1] if base_scores_list and q_idx < len(base_scores_list) and rank-1 < len(base_scores_list[q_idx]) else "N/A"
            # 格式化分数
            score_str = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
            # 获取图片信息
            emotions_str, texts_str = get_image_info(vid)
            # 限制文字长度
            texts_str = texts_str[:50] + "..." if len(texts_str) > 50 else texts_str
            
            # 复制图片到输出目录
            img_filename = f"query_{q_idx}_rank_{rank}_{vid.replace('/', '_').replace('.jpg', '')}.jpg"
            img_output_path = os.path.join(images_dir, img_filename)
            img_source_path = _build_image_path(image_path, vid)
            
            if os.path.exists(img_source_path):
                try:
                    from shutil import copyfile
                    copyfile(img_source_path, img_output_path)
                    img_rel_path = f"images/{img_filename}"
                    markdown_lines.append(f"| {rank} | ![img]({img_rel_path}) | `{vid}` | {is_correct} | {score_str} | {emotions_str} | {texts_str} |\n")
                except Exception as e:
                    markdown_lines.append(f"| {rank} | 图片加载失败 | `{vid}` | {is_correct} | {score_str} | {emotions_str} | {texts_str} |\n")
            else:
                markdown_lines.append(f"| {rank} | 图片不存在 | `{vid}` | {is_correct} | {score_str} | {emotions_str} | {texts_str} |\n")
        
        markdown_lines.append("\n---\n\n")
    
    # 添加统计信息
    if total_count > 0:
        match_rate = match_count / total_count * 100.0
        markdown_lines.append(f"\n## 统计信息\n\n")
        markdown_lines.append(f"- 共处理 {total_count} 个图片结果\n")
        markdown_lines.append(f"- 成功匹配到数据集信息：{match_count} 个 ({match_rate:.1f}%)\n")
        markdown_lines.append(f"- 未匹配：{total_count - match_count} 个\n\n")
    
    # 写入 markdown 文件
    markdown_path = os.path.join(output_dir, "results.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.writelines(markdown_lines)
    
    print(f"[Output] 评测结果已保存到: {markdown_path}")
    print(f"[Output] 图片已保存到: {images_dir}")
    if total_count > 0:
        print(f"[Output] 图片信息匹配统计: {match_count}/{total_count} ({match_count/total_count*100:.1f}%)")

class RateLimiter:
    def __init__(self, rate_per_minute=50):
        self.rate = int(rate_per_minute)
        self._dq = collections.deque()
        self._lock = threading.Lock()
    def acquire(self):
        while True:
            now = time.time()
            with self._lock:
                while self._dq and self._dq[0] <= now - 60.0:
                    self._dq.popleft()
                if len(self._dq) < self.rate:
                    self._dq.append(now)
                    return
                wait_t = (self._dq[0] + 60.0) - now
            if wait_t > 0:
                time.sleep(min(wait_t, 0.1))

_RERANK_CACHE = {}
_RERANK_CACHE_LOCK = threading.Lock()

def _load_rerank_cache(path):
    try:
        if path and os.path.exists(path):
            print(f"[Cache] Loading rerank cache from: {path}")
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict):
                with _RERANK_CACHE_LOCK:
                    _RERANK_CACHE.clear()
                    for q, mp in data.items():
                        if isinstance(mp, dict):
                            _RERANK_CACHE[q] = dict(mp)
                print(f"[Cache] Rerank cache loaded: {len(_RERANK_CACHE)} queries")
        else:
            print(f"[Cache] Rerank cache not found at: {path}")
    except Exception as e:
        print(f"[Cache] Warning: Failed to load rerank cache: {e}")

def _save_rerank_cache(path):
    try:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with _RERANK_CACHE_LOCK:
                data = {q: dict(mp) for q, mp in _RERANK_CACHE.items()}
            torch.save(data, path)
            print(f"[Cache] Saved rerank cache to: {path}")
            print(f"[Cache] Rerank cache contains: {len(_RERANK_CACHE)} queries")
    except Exception as e:
        print(f"[Cache] Warning: Failed to save rerank cache: {e}")

def rerank_topk_by_api(query_text, base_topk_indices, video_ids, image_dir, model_name, api_key, limiter=None, return_scores=False, timeout_s=None, max_docs_per_request=32, max_retries=2):
    if video_ids is None:
        return (base_topk_indices, [None] * len(base_topk_indices)) if return_scores else base_topk_indices
    max_len = len(video_ids) if isinstance(video_ids, list) else None
    base_topk_indices = _safe_clip_indices(base_topk_indices, max_len)
    vids = [video_ids[int(i)] for i in base_topk_indices]
    with _RERANK_CACHE_LOCK:
        cache_q = _RERANK_CACHE.get(query_text) or {}
    cached_scores = []
    missing_vids = []
    for vid in vids:
        if vid in cache_q:
            cached_scores.append(cache_q[vid])
        else:
            cached_scores.append(None)
            missing_vids.append(vid)
    if missing_vids:
        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else f"Bearer {os.environ.get('JINA_API_KEY','')}"
        }
        scores_by_vid = {}
        n = len(missing_vids)
        start = 0
        while start < n:
            end = min(start + int(max_docs_per_request), n)
            chunk_vids = missing_vids[start:end]
            docs = []
            for vid in chunk_vids:
                ip = _build_image_path(image_dir, vid)
                if os.path.exists(ip):
                    try:
                        docs.append({"image": _encode_image_b64(ip)})
                    except Exception:
                        docs.append({"text": vid})
                else:
                    docs.append({"text": vid})
            payload = {
                "model": model_name,
                "query": query_text,
                "documents": docs,
                "return_documents": False
            }
            tries = 0
            while True:
                try:
                    if limiter is not None:
                        limiter.acquire()
                    if timeout_s is None or timeout_s <= 0:
                        r = requests.post(url, headers=headers, data=json.dumps(payload))
                    else:
                        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=float(timeout_s))
                    
                    # 检查 HTTP 状态码
                    if r.status_code == 200:
                        # 成功响应，解析结果
                        try:
                            j = r.json()
                        except Exception as e:
                            print(f"[Jina] Failed to parse JSON response: {e}")
                            tries += 1
                            if tries > int(max_retries):
                                print(f"[Jina] Max retries reached, skipping this chunk")
                                break
                            time.sleep(min(1.0 * tries, 5.0))
                            continue
                        
                        res = j.get("results") or j.get("data") or j.get("items") or []
                        msg = j.get("detail")
                        if isinstance(msg, (dict, list)):
                            try:
                                msg = json.dumps(msg)[:200]
                            except Exception:
                                msg = str(msg)[:200]
                        elif isinstance(msg, str):
                            msg = msg[:200]
                        print(f"[Jina] status={r.status_code} keys={list(j.keys())} docs={len(docs)} res_len={len(res)} detail={msg}")
                        for item in res:
                            if isinstance(item, dict):
                                idx = None
                                if "index" in item:
                                    idx = int(item["index"])
                                elif "document_index" in item:
                                    idx = int(item["document_index"])
                                sc = None
                                if "relevance_score" in item:
                                    sc = item["relevance_score"]
                                elif "score" in item:
                                    sc = item["score"]
                                if idx is not None and 0 <= idx < len(chunk_vids):
                                    scores_by_vid[chunk_vids[idx]] = sc
                        break
                    elif r.status_code == 429:
                        # 请求过多，需要等待更长时间后重试
                        tries += 1
                        if tries > int(max_retries):
                            print(f"[Jina] Rate limited (429), max retries reached, skipping this chunk")
                            break
                        # 尝试从响应头中获取 Retry-After 信息
                        retry_after = r.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                            except Exception:
                                wait_time = min(60.0 * tries, 300.0)  # 最多等待5分钟
                        else:
                            # 指数退避：429 时等待更长时间
                            wait_time = min(60.0 * tries, 300.0)  # 最多等待5分钟
                        print(f"[Jina] Rate limited (429), waiting {wait_time:.1f}s before retry {tries}/{max_retries}")
                        time.sleep(wait_time)
                    elif r.status_code >= 500:
                        # 服务器错误，可以重试
                        tries += 1
                        if tries > int(max_retries):
                            print(f"[Jina] Server error ({r.status_code}), max retries reached, skipping this chunk")
                            break
                        wait_time = min(10.0 * tries, 60.0)  # 最多等待1分钟
                        print(f"[Jina] Server error ({r.status_code}), waiting {wait_time:.1f}s before retry {tries}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        # 其他客户端错误（4xx，除了429），通常不应该重试
                        try:
                            error_msg = r.text[:200]
                        except Exception:
                            error_msg = "Unknown error"
                        print(f"[Jina] Client error ({r.status_code}): {error_msg}")
                        # 对于某些4xx错误（如401认证失败），不应该重试
                        if r.status_code in [401, 403]:
                            print(f"[Jina] Authentication/authorization error, stopping retries")
                            break
                        tries += 1
                        if tries > int(max_retries):
                            print(f"[Jina] Max retries reached for status {r.status_code}")
                            break
                        time.sleep(min(1.0 * tries, 5.0))
                except requests.exceptions.Timeout:
                    # 超时异常
                    tries += 1
                    if tries > int(max_retries):
                        print(f"[Jina] Timeout exception, max retries reached, skipping this chunk")
                        break
                    wait_time = min(5.0 * tries, 30.0)
                    print(f"[Jina] Timeout, waiting {wait_time:.1f}s before retry {tries}/{max_retries}")
                    time.sleep(wait_time)
                except requests.exceptions.RequestException as e:
                    # 其他网络异常
                    tries += 1
                    if tries > int(max_retries):
                        print(f"[Jina] Request exception ({type(e).__name__}): {e}, max retries reached, skipping this chunk")
                        break
                    wait_time = min(1.0 * tries, 5.0)
                    print(f"[Jina] Request exception ({type(e).__name__}): {e}, waiting {wait_time:.1f}s before retry {tries}/{max_retries}")
                    time.sleep(wait_time)
                except Exception as e:
                    # 其他未知异常
                    tries += 1
                    if tries > int(max_retries):
                        print(f"[Jina] Unexpected exception ({type(e).__name__}): {e}, max retries reached, skipping this chunk")
                        break
                    wait_time = min(1.0 * tries, 5.0)
                    print(f"[Jina] Unexpected exception ({type(e).__name__}): {e}, waiting {wait_time:.1f}s before retry {tries}/{max_retries}")
                    time.sleep(wait_time)
            start = end
        with _RERANK_CACHE_LOCK:
            cache_q = _RERANK_CACHE.get(query_text)
            if cache_q is None:
                cache_q = {}
                _RERANK_CACHE[query_text] = cache_q
            for vid, sc in scores_by_vid.items():
                cache_q[vid] = sc
        with _RERANK_CACHE_LOCK:
            cache_q = _RERANK_CACHE.get(query_text) or {}
        for i, vid in enumerate(vids):
            if cached_scores[i] is None and vid in cache_q:
                cached_scores[i] = cache_q[vid]
    try:
        arr = np.array([float(x) if x is not None else -1e20 for x in cached_scores], dtype=np.float32)
        order = np.argsort(-arr)
        ordered = base_topk_indices[order]
        if return_scores:
            ordered_scores = [cached_scores[i] for i in order.tolist()]
            return ordered, ordered_scores
        return ordered
    except Exception:
        return (base_topk_indices, [None] * len(base_topk_indices)) if return_scores else base_topk_indices

def infer(checkpoint_path="./checkpoints_clip/clip_best_model.pt",
          data_path="./imgflip_data/msrvtt",
          image_path="./imgflip_data/images",
          test_json="test_data.json",
          test_emotion_json="test_emotion.json",
          batch_size=128,
          enable_rerank=True,
          rerank_model_path="",
          rerank_model_type="linear",
          rerank_mlp_hidden_dim=512,
          rerank_mlp_layers=1,
          rerank_mlp_dropout=0.1,
          save_dir="./checkpoints_rerank",
          topk_base=100,
          device=None,
          use_image_emotion_fusion=False,
          use_text_emotion_fusion=False,
          zero_shot=False,
          use_rerank_api=False,
          jina_model_name="jinaai/jina-reranker-m0",
          jina_api_key="",
          use_jina_lora=False,
          jina_lora_path=None,
          jina_lora_r=8,
          jina_lora_alpha=16,
          jina_lora_dropout=0.05,
          num_samples=0,
          sample_mode="first",
          sample_seed=0,
          subset_image_pool=False,
          query_expansion_alpha=0.03,
          query_expansion_alpha_image=0.027,
          query_expansion_k=2,
          query_expansion_k_image=1,
          query_expansion_k_schedule="3,2,2",
          query_expansion_k_image_schedule="2,3,1",
          query_expansion_iters=3,
          image_projection_path=None,
          projection_mix=0.9,
          image_projection_sim_fusion=0.5,
          debug_samples=0,
          rerank_cache_path=None,
          jina_timeout=60,
          jina_chunk_size=32,
          jina_max_retries=2,
          jina_image_max_side=256,
          jina_qwen_min_side=32,
          jina_micro_batch=0,
          jina_max_text_len=2048,
          use_jina_cache=False,
          use_jina_emotion_fusion=False,
          output_top_n_queries=0,
          output_results_dir=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CLIP] Loading CLIP model: ViT-B/32 on device: {device}")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    emotion_adapter = None
    if not zero_shot and os.path.exists(checkpoint_path):
        print(f"[CLIP] Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and ("clip_state_dict" in state or "adapter_state_dict" in state):
            if "clip_state_dict" in state:
                clip_model.load_state_dict(state["clip_state_dict"], strict=False)
                print(f"[CLIP] Loaded clip_state_dict from checkpoint")
            else:
                try:
                    clip_model.load_state_dict(state, strict=False)
                except Exception:
                    clip_model.load_state_dict(state, strict=False)
                print(f"[CLIP] Loaded state_dict from checkpoint")
            if use_image_emotion_fusion or use_text_emotion_fusion:
                emotion_adapter = EmotionAdapter(emo_dim=768, clip_dim=512).to(device)
                if "adapter_state_dict" in state:
                    emotion_adapter.load_state_dict(state["adapter_state_dict"], strict=False)
                    print(f"[CLIP] Loaded adapter_state_dict for emotion fusion")
        else:
            try:
                clip_model.load_state_dict(state)
            except Exception:
                clip_model.load_state_dict(state, strict=False)
            print(f"[CLIP] Loaded state_dict directly from checkpoint")
            if use_image_emotion_fusion or use_text_emotion_fusion:
                emotion_adapter = EmotionAdapter(emo_dim=768, clip_dim=512).to(device)
        print(f"[CLIP] Checkpoint loaded successfully")
        if isinstance(state, dict):
            print(f"  - Keys in checkpoint: {list(state.keys())[:5]}{'...' if len(state.keys()) > 5 else ''}")
    else:
        if zero_shot:
            print(f"[CLIP] Using zero-shot mode (no checkpoint)")
        else:
            print(f"[CLIP] Warning: Checkpoint not found at {checkpoint_path}, using pretrained weights")
        use_image_emotion_fusion = False
        use_text_emotion_fusion = False
    text_matrix, image_matrix, image_matrix_orig, all_caps, precomputed_topk, video_ids, image_caps_full, query_video_ids = load_or_compute_test_cache(
        data_path=data_path,
        image_path=image_path,
        batch_size=batch_size,
        device=device,
        clip_model=clip_model,
        preprocess=preprocess,
        topk_base=topk_base,
        save_dir=save_dir,
        use_image_emotion_fusion=use_image_emotion_fusion,
        use_text_emotion_fusion=use_text_emotion_fusion,
        emotion_adapter=emotion_adapter,
        zero_shot=zero_shot,
        sample_limit=num_samples,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        subset_image_pool=subset_image_pool,
        image_projection_path=image_projection_path,
        projection_mix=projection_mix,
    )
    
    # 加载测试数据集（用于输出可视化结果时获取图片信息）
    test_dataset_for_output = None
    if output_top_n_queries > 0 and output_results_dir:
        test_json_path = os.path.join(data_path, test_json)
        test_emotion_json_path = os.path.join(data_path, test_emotion_json)
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        try:
            test_dataset_for_output = MSRVTT_Dataset(
                csv_path=test_json_path,
                json_path=test_json_path,
                features_path=image_path,
                emotion_json_path=test_emotion_json_path,
                clip_preprocess=None,  # 不需要图像预处理
                bert_tokenizer=bert_tokenizer,
                max_words=32,
                is_train=False,
                load_image=False,  # 不需要加载图像
            )
            print(f"[Output] Loaded test dataset for output visualization: {len(test_dataset_for_output.data)} samples")
        except Exception as e:
            print(f"[Output] Warning: Failed to load test dataset for output visualization: {e}")
            test_dataset_for_output = None
    
    del clip_model
    if emotion_adapter is not None:
        del emotion_adapter
    gc.collect()
    torch.cuda.empty_cache() # 强制清理显存碎片
    sim = np.dot(text_matrix, image_matrix.T)
    if image_projection_sim_fusion > 0.0 and image_matrix_orig is not None:
        sim_orig = np.dot(text_matrix, image_matrix_orig.T)
        beta = float(image_projection_sim_fusion)
        sim = (1.0 - beta) * sim + beta * sim_orig
        print(f"[Projection] Fused projected/original similarities (beta={beta})")
    if query_expansion_alpha > 0.0 and query_expansion_k > 0:
        image_expansion_alpha = query_expansion_alpha_image if query_expansion_alpha_image > 0.0 else query_expansion_alpha
        k_schedule = [int(x) for x in str(query_expansion_k_schedule).split(",") if x.strip()] if query_expansion_k_schedule else []
        k_image_schedule = [int(x) for x in str(query_expansion_k_image_schedule).split(",") if x.strip()] if query_expansion_k_image_schedule else []
        for it in range(max(1, int(query_expansion_iters))):
            k_exp = min(k_schedule[min(it, len(k_schedule) - 1)] if k_schedule else query_expansion_k, sim.shape[1])
            k_exp_i = min(k_image_schedule[min(it, len(k_image_schedule) - 1)] if k_image_schedule else query_expansion_k_image, sim.shape[0])
            # Expand queries with top-k images
            expanded_text = []
            for q_idx in range(sim.shape[0]):
                topk_idx = np.argpartition(-sim[q_idx], range(k_exp))[:k_exp]
                weights = sim[q_idx][topk_idx]
                weights = np.maximum(weights, 0.0)
                weights = weights / (weights.sum() + 1e-8)
                weighted_img = (image_matrix[topk_idx] * weights[:, None]).sum(axis=0)
                expanded = (1.0 - query_expansion_alpha) * text_matrix[q_idx] + query_expansion_alpha * weighted_img
                expanded_text.append(expanded)
            text_matrix = np.stack(expanded_text, axis=0)
            text_matrix = text_matrix / (np.linalg.norm(text_matrix, axis=1, keepdims=True) + 1e-8)
            # Expand images with top-k queries (symmetric pseudo-relevance feedback)
            sim_t = np.dot(text_matrix, image_matrix.T)
            expanded_image = []
            for i_idx in range(sim_t.shape[1]):
                topk_q = np.argpartition(-sim_t[:, i_idx], range(k_exp_i))[:k_exp_i]
                weights = sim_t[topk_q, i_idx]
                weights = np.maximum(weights, 0.0)
                weights = weights / (weights.sum() + 1e-8)
                weighted_txt = (text_matrix[topk_q] * weights[:, None]).sum(axis=0)
                expanded = (1.0 - image_expansion_alpha) * image_matrix[i_idx] + image_expansion_alpha * weighted_txt
                expanded_image.append(expanded)
            image_matrix = np.stack(expanded_image, axis=0)
            image_matrix = image_matrix / (np.linalg.norm(image_matrix, axis=1, keepdims=True) + 1e-8)
            sim = np.dot(text_matrix, image_matrix.T)
    ks = (1, 3, 5, 10, 50, 100)
    base_metrics, base_precisions = compute_recalls_precisions(sim, all_caps, image_caps_full, ks=ks, limit_k=max(ks))
    
    # 计算base模型的主贴图片召回
    k_base = min(topk_base, sim.shape[1])
    base_ranked_indices_list = []
    for q_idx in range(sim.shape[0]):
        if precomputed_topk is not None:
            base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
            if base_topk.shape[0] > k_base:
                base_topk = base_topk[:k_base]
        else:
            base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
        base_topk = _safe_clip_indices(base_topk, sim.shape[1])
        sim_q = sim[q_idx][base_topk]
        order_base = np.argsort(-sim_q)
        base_sorted = base_topk[order_base]
        base_ranked_indices_list.append(base_sorted)
    base_main_recalls = compute_main_post_recalls(base_ranked_indices_list, query_video_ids, video_ids, ks=ks)
    if use_rerank_api:
        if rerank_cache_path is None:
            os.makedirs(save_dir, exist_ok=True)
            rerank_cache_path = os.path.join(save_dir, f"rerank_cache_{jina_model_name}.pt")
        _load_rerank_cache(rerank_cache_path)
    if debug_samples and debug_samples > 0:
        total = sim.shape[0]
        n = min(debug_samples, total)
        k_base = min(topk_base, sim.shape[1])
        jina_model_objs = None
        
        # ============ 加载评估阶段的 Jina 特征缓存（可选，用于debug模式）============
        eval_feature_map = None
        eval_candidate_map = None
        if use_jina_cache and use_jina_lora and not use_rerank_api:
            jina_eval_cache_path = os.path.join(save_dir, "jina_features_eval_cache.pt")
            if os.path.exists(jina_eval_cache_path):
                print(f"[Jina Cache] Loading evaluation cache for debug mode from: {jina_eval_cache_path}")
                try:
                    jina_eval_cache = torch.load(jina_eval_cache_path, map_location="cpu", weights_only=False)
                    if "features" in jina_eval_cache and "query_indices" in jina_eval_cache:
                        eval_feature_map = {}
                        eval_candidate_map = {}
                        candidate_indices_list = jina_eval_cache.get("candidate_indices", None)
                        for idx, (feat_list, q_idx_list) in enumerate(zip(
                            jina_eval_cache["features"],
                            jina_eval_cache["query_indices"]
                        )):
                            q_idx_val = int(q_idx_list[0].item() if isinstance(q_idx_list, torch.Tensor) else q_idx_list[0])
                            if q_idx_val not in eval_feature_map:
                                eval_feature_map[q_idx_val] = feat_list
                                if candidate_indices_list is not None and idx < len(candidate_indices_list):
                                    eval_candidate_map[q_idx_val] = candidate_indices_list[idx]
                        print(f"[Jina Cache] Loaded {len(eval_feature_map)} cached query features for debug mode")
                    else:
                        print(f"[Jina Cache] Warning: Evaluation cache format incorrect for debug mode")
                        eval_feature_map = None
                        eval_candidate_map = None
                except Exception as e:
                    print(f"[Jina Cache] Error loading evaluation cache for debug mode: {e}")
                    eval_feature_map = None
                    eval_candidate_map = None
        
        if use_jina_lora and not use_rerank_api:
            print("Loading Jina LoRA model for reranking...")
            jina_model_objs = _load_jina_lora(jina_model_name, jina_lora_r, jina_lora_alpha, jina_lora_dropout, device, jina_lora_path)
        for q_idx in range(n):
            if precomputed_topk is not None:
                base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
                if base_topk.shape[0] > k_base:
                    base_topk = base_topk[:k_base]
            else:
                base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
            base_topk = _safe_clip_indices(base_topk, (len(video_ids) if video_ids is not None and isinstance(video_ids, list) else sim.shape[1]))
            sim_q = sim[q_idx][base_topk]
            order_base = np.argsort(-sim_q)
            base_sorted = base_topk[order_base]
            base_scores_sorted = sim_q[order_base]
            print(f"Query[{q_idx}] {all_caps[q_idx]}")
            for rank, (idx_i, sc) in enumerate(zip(base_sorted.tolist(), base_scores_sorted.tolist()), start=1):
                vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                print(f"CLIP #{rank} id={vid} sim={sc:.6f}")
            if use_rerank_api:
                ordered, api_scores = rerank_topk_by_api(all_caps[q_idx], base_topk, video_ids, image_path, jina_model_name, jina_api_key, return_scores=True, timeout_s=jina_timeout, max_docs_per_request=jina_chunk_size, max_retries=jina_max_retries)
                for rank, idx_i in enumerate(ordered.tolist(), start=1):
                    vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                    sc = api_scores[rank-1] if api_scores and len(api_scores) >= rank else None
                    if sc is None:
                        print(f"API  #{rank} id={vid} score=N/A")
                    else:
                        try:
                            print(f"API  #{rank} id={vid} score={float(sc):.6f}")
                        except Exception:
                            print(f"API  #{rank} id={vid} score={sc}")
            elif rerank_model_path and os.path.exists(rerank_model_path):
                d_text = text_matrix.shape[1]
                d_img = image_matrix.shape[1]
                if rerank_model_type == "mlp":
                    model = RerankMLP(d_text + d_img, hidden_dim=rerank_mlp_hidden_dim, dropout=rerank_mlp_dropout, num_layers=rerank_mlp_layers).to(device)
                else:
                    model = RerankLinear(d_text + d_img).to(device)
                state = torch.load(rerank_model_path, map_location=device)
                model.load_state_dict(state)
                t = text_matrix[q_idx]
                ims = image_matrix[base_topk]
                t_rep = np.repeat(t[None, :], ims.shape[0], axis=0)
                pair = np.concatenate([t_rep, ims], axis=1)
                with torch.no_grad():
                    pair_t = torch.from_numpy(pair).to(device).float()
                    scores = model(pair_t).detach().cpu().numpy()
                order = np.argsort(-scores)
                ordered = base_topk[order]
                for rank, idx_i in enumerate(ordered.tolist(), start=1):
                    vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                    sc = scores[order[rank-1]]
                    print(f"MLP  #{rank} id={vid} score={float(sc):.6f}")
            else:
                if use_jina_lora:
                    if not jina_model_objs:
                        # 如果没有预加载，则在这里加载
                        print("Loading Jina model for reranking...")
                        jina_model_objs = _load_jina_lora(jina_model_name, jina_lora_r, jina_lora_alpha, jina_lora_dropout, device, jina_lora_path)
                    
                    # 如果使用评估缓存，直接通过 score head 计算分数
                    if use_jina_cache and eval_feature_map is not None and q_idx in eval_feature_map:
                        cached_features = eval_feature_map[q_idx]
                        # 如果缓存中有候选索引，使用缓存的候选索引
                        if eval_candidate_map is not None and q_idx in eval_candidate_map:
                            base_topk = eval_candidate_map[q_idx].cpu().numpy()
                            base_topk = _safe_clip_indices(base_topk, (len(video_ids) if video_ids is not None and isinstance(video_ids, list) else sim.shape[1]))
                            # 对齐特征数量
                            if cached_features.shape[0] != len(base_topk):
                                min_count = min(cached_features.shape[0], len(base_topk))
                                cached_features = cached_features[:min_count]
                                base_topk = base_topk[:min_count]
                        
                        processor, jina_model = jina_model_objs
                        # 获取 score head
                        if hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "model"):
                            score_head = jina_model.base_model.model.score
                        elif hasattr(jina_model, "score"):
                            score_head = jina_model.score
                        else:
                            raise RuntimeError("Score head not found in model")
                        
                        # 确保 score head 在正确的设备上
                        if next(score_head.parameters()).device.type != "cuda" and device.type == "cuda":
                            score_head = score_head.to(device)
                        
                        # 获取 score head 的 dtype，确保特征与模型 dtype 匹配
                        score_head_dtype = next(score_head.parameters()).dtype
                        
                        # 通过 score head 计算分数
                        with torch.no_grad():
                            features_input = cached_features.to(device).to(dtype=score_head_dtype)
                            scores_t = score_head(features_input).squeeze(-1)
                            # numpy 不支持 bfloat16，需要先转换为 float32
                            scores_t = scores_t.cpu().float().numpy()
                        print(f"[Jina Cache] Using cached features for query {q_idx}")
                    else:
                        # 正常流程：加载图像并通过模型前向传播
                        imgs = []
                        for idx_i in base_topk.tolist():
                            p = _build_image_path(image_path, video_ids[idx_i] if video_ids is not None else str(idx_i))
                            if os.path.exists(p):
                                try:
                                    from PIL import Image as _I
                                    imgs.append(_I.open(p).convert("RGB"))
                                except Exception:
                                    imgs.append(None)
                            else:
                                imgs.append(None)
                        processor, jina_model = jina_model_objs
                        texts = [all_caps[q_idx]] * len(imgs)
                        scores_t = _score_batch_jina(jina_model, processor, texts, imgs, device, 
                                                      image_max_side=jina_image_max_side, 
                                                      qwen_min_side=jina_qwen_min_side, 
                                                      jina_micro_batch=jina_micro_batch, 
                                                      max_text_len=jina_max_text_len).numpy()
                        # 及时清理图片对象
                        del imgs
                    
                    order = np.argsort(-scores_t)
                    ordered = base_topk[order]
                    model_type = "Base" if (not jina_lora_path or not os.path.exists(jina_lora_path)) else "LoRA"
                    cache_tag = " [Cache]" if (use_jina_cache and eval_feature_map is not None and q_idx in eval_feature_map) else ""
                    for rank, idx_i in enumerate(ordered.tolist(), start=1):
                        vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                        sc = scores_t[order[rank-1]]
                        print(f"{model_type}{cache_tag} #{rank} id={vid} score={float(sc):.6f}")
                else:
                    print("No reranker enabled")
        if use_rerank_api:
            _save_rerank_cache(rerank_cache_path)
        sys.exit(0)
    if rerank_model_path and os.path.exists(rerank_model_path):
        d_text = text_matrix.shape[1]
        d_img = image_matrix.shape[1]
        print(f"[Rerank] Loading rerank model from: {rerank_model_path}")
        if rerank_model_type == "mlp":
            model = RerankMLP(d_text + d_img, hidden_dim=rerank_mlp_hidden_dim, dropout=rerank_mlp_dropout, num_layers=rerank_mlp_layers).to(device)
            print(f"[Rerank] Model type: MLP (hidden_dim={rerank_mlp_hidden_dim}, layers={rerank_mlp_layers}, dropout={rerank_mlp_dropout})")
        else:
            model = RerankLinear(d_text + d_img).to(device)
            print(f"[Rerank] Model type: Linear")
        state = torch.load(rerank_model_path, map_location=device)
        model.load_state_dict(state)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Rerank] Model loaded successfully, total parameters: {total_params:,}")
        total = sim.shape[0]
        caption_to_indices = _build_caption_index_map(image_caps_full)
        recalls = {f"R@{k}": 0.0 for k in ks}
        precisions = {f"P@{k}": 0.0 for k in ks}
        k_base = min(topk_base, sim.shape[1])
        text_matrix_t = torch.from_numpy(text_matrix).to(device).float()
        image_matrix_t = torch.from_numpy(image_matrix).to(device).float()
        # 收集 rerank 结果和分数用于可视化
        rerank_ranked_indices_list = []
        rerank_scores_list = []
        base_scores_list = []
        for q_idx in tqdm(range(total), desc="Reranking (MLP/Linear)"):
            if precomputed_topk is not None:
                base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
                if base_topk.shape[0] > k_base:
                    base_topk = base_topk[:k_base]
            else:
                base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
            base_topk = _safe_clip_indices(base_topk, sim.shape[1])
            # 收集 base 模型的分数
            base_scores = sim[q_idx][base_topk]
            base_order = np.argsort(-base_scores)
            base_sorted_scores = base_scores[base_order]
            base_scores_list.append(base_sorted_scores.tolist())
            # 计算 rerank 分数
            t = text_matrix[q_idx]
            ims = image_matrix[base_topk]
            t_rep = np.repeat(t[None, :], ims.shape[0], axis=0)
            pair = np.concatenate([t_rep, ims], axis=1)
            with torch.no_grad():
                pair_t = torch.from_numpy(pair).to(device).float()
                scores = model(pair_t).detach().cpu().numpy().flatten()
            order = np.argsort(-scores)
            reranked_topk = base_topk[order]
            reranked_scores = scores[order]
            rerank_ranked_indices_list.append(reranked_topk)
            rerank_scores_list.append(reranked_scores.tolist())
            gt = caption_to_indices[all_caps[q_idx]]
            for k in ks:
                k_eff = min(k, k_base)
                topk_used = reranked_topk[:k_eff]
                if any(i in gt for i in topk_used):
                    recalls[f"R@{k}"] += 1.0
                correct = sum(1 for i in topk_used if i in gt)
                precisions[f"P@{k}"] += correct / float(k_eff)
        for k in ks:
            recalls[f"R@{k}"] = recalls[f"R@{k}"] / total * 100.0
            precisions[f"P@{k}"] = precisions[f"P@{k}"] / total * 100.0
        rerank_metrics = recalls
        rerank_precisions = precisions
        # 计算rerank后的主贴图片召回（使用已计算的reranked结果）
        rerank_main_recalls = compute_main_post_recalls(rerank_ranked_indices_list, query_video_ids, video_ids, ks=ks)
        
        # 输出可视化结果
        if output_top_n_queries > 0 and output_results_dir:
            n_queries = min(output_top_n_queries, total)
            query_indices = list(range(n_queries))
            output_query_results(
                queries=query_indices,
                ranked_indices_list=rerank_ranked_indices_list,
                all_captions=all_caps,
                image_caps_full=image_caps_full,
                video_ids=video_ids,
                image_path=image_path,
                output_dir=output_results_dir,
                top_k=10,
                base_ranked_indices_list=base_ranked_indices_list,
                base_scores_list=base_scores_list,
                rerank_scores_list=rerank_scores_list,
                test_dataset=test_dataset_for_output
            )
        return {
            **{f"base_{k}": v for k, v in base_metrics.items()},
            **{f"base_{k}": v for k, v in base_precisions.items()},
            **{f"rerank_{k}": v for k, v in rerank_metrics.items()},
            **{f"rerank_{k}": v for k, v in rerank_precisions.items()},
            **{f"base_{k}": v for k, v in base_main_recalls.items()},
            **{f"rerank_{k}": v for k, v in rerank_main_recalls.items()},
        }
    if use_rerank_api:
        total = sim.shape[0]
        caption_to_indices = _build_caption_index_map(image_caps_full)
        recalls = {f"R@{k}": 0.0 for k in ks}
        precisions = {f"P@{k}": 0.0 for k in ks}
        k_base = min(topk_base, sim.shape[1])
        base_topk_list = []
        for q_idx in range(total):
            if precomputed_topk is not None:
                base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
                if base_topk.shape[0] > k_base:
                    base_topk = base_topk[:k_base]
            else:
                base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
            base_topk = _safe_clip_indices(base_topk, (len(video_ids) if video_ids is not None and isinstance(video_ids, list) else sim.shape[1]))
            base_topk_list.append(base_topk)
        
        # 检查哪些 query 已经有缓存（支持增量推理）
        reranked_all = [None] * total
        cached_query_indices = set()
        with _RERANK_CACHE_LOCK:
            for q_idx in range(total):
                query_text = all_caps[q_idx]
                cache_q = _RERANK_CACHE.get(query_text, {})
                # 检查这个 query 的所有候选图像是否都有缓存
                base_topk = base_topk_list[q_idx]
                all_cached = True
                for idx_i in base_topk:
                    vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                    if vid not in cache_q:
                        all_cached = False
                        break
                if all_cached and len(base_topk) > 0:
                    # 这个 query 的所有候选都有缓存，可以直接从缓存构建排序结果
                    cached_query_indices.add(q_idx)
                    # 从缓存中获取分数并排序
                    cached_scores = []
                    cached_indices = []
                    for idx_i in base_topk:
                        vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                        sc = cache_q.get(vid)
                        if sc is not None:
                            cached_scores.append(float(sc))
                            cached_indices.append(idx_i)
                    if len(cached_scores) > 0:
                        order = np.argsort(-np.array(cached_scores))
                        reranked_all[q_idx] = np.array(cached_indices)[order]
        
        if len(cached_query_indices) > 0:
            print(f"[Incremental] Found {len(cached_query_indices)}/{total} queries already cached, skipping API calls")
        
        # 只对没有缓存的 query 进行 API 调用
        uncached_indices = [q_idx for q_idx in range(total) if q_idx not in cached_query_indices]
        if len(uncached_indices) > 0:
            print(f"[Incremental] Processing {len(uncached_indices)}/{total} new queries via API")
            limiter = RateLimiter(rate_per_minute=20)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                future_to_idx = {}
                for q_idx in uncached_indices:
                    fut = ex.submit(
                        rerank_topk_by_api,
                        all_caps[q_idx],
                        base_topk_list[q_idx],
                        video_ids,
                        image_path,
                        jina_model_name,
                        jina_api_key,
                        limiter,
                        False,
                        jina_timeout,
                        jina_chunk_size,
                        jina_max_retries
                    )
                    future_to_idx[fut] = q_idx
                for fut in tqdm(concurrent.futures.as_completed(list(future_to_idx.keys())), total=len(uncached_indices), desc="Reranking (API)"):
                    idx = future_to_idx[fut]
                    reranked_all[idx] = fut.result()
        else:
            print(f"[Incremental] All queries are cached, skipping API calls")
        
        # 收集 base 和 rerank 的分数用于可视化
        base_scores_list = []
        rerank_scores_list = []
        for q_idx in range(total):
            base_topk = base_topk_list[q_idx]
            base_scores = sim[q_idx][base_topk]
            base_order = np.argsort(-base_scores)
            base_sorted_scores = base_scores[base_order]
            base_scores_list.append(base_sorted_scores.tolist())
            
            reranked_topk = reranked_all[q_idx]
            if reranked_topk is None:
                # 如果某个 query 没有结果，使用 base_topk 作为后备
                reranked_topk = base_topk_list[q_idx]
                rerank_scores_list.append(base_sorted_scores.tolist())
            else:
                # 从缓存中获取 rerank 分数
                query_text = all_caps[q_idx]
                cached_scores = []
                with _RERANK_CACHE_LOCK:
                    cache_q = _RERANK_CACHE.get(query_text, {})
                    for idx_i in reranked_topk:
                        vid = video_ids[idx_i] if video_ids is not None else str(idx_i)
                        sc = cache_q.get(vid)
                        if sc is not None:
                            cached_scores.append(float(sc))
                        else:
                            cached_scores.append(0.0)
                rerank_scores_list.append(cached_scores)
            
            gt = caption_to_indices[all_caps[q_idx]]
            for k in ks:
                k_eff = min(k, k_base)
                topk_used = reranked_topk[:k_eff]
                if any(i in gt for i in topk_used):
                    recalls[f"R@{k}"] += 1.0
                correct = sum(1 for i in topk_used if i in gt)
                precisions[f"P@{k}"] += correct / float(k_eff)
        for k in ks:
            recalls[f"R@{k}"] = recalls[f"R@{k}"] / total * 100.0
            precisions[f"P@{k}"] = precisions[f"P@{k}"] / total * 100.0
        # 计算rerank后的主贴图片召回
        rerank_main_recalls = compute_main_post_recalls(reranked_all, query_video_ids, video_ids, ks=ks)
        _save_rerank_cache(rerank_cache_path)
        
        # 输出可视化结果
        if output_top_n_queries > 0 and output_results_dir:
            n_queries = min(output_top_n_queries, total)
            query_indices = list(range(n_queries))
            output_query_results(
                queries=query_indices,
                ranked_indices_list=reranked_all,
                all_captions=all_caps,
                image_caps_full=image_caps_full,
                video_ids=video_ids,
                image_path=image_path,
                output_dir=output_results_dir,
                top_k=10,
                base_ranked_indices_list=base_ranked_indices_list,
                base_scores_list=base_scores_list,
                rerank_scores_list=rerank_scores_list,
                test_dataset=test_dataset_for_output
            )
        return {
            **{f"base_{k}": v for k, v in base_metrics.items()},
            **{f"base_{k}": v for k, v in base_precisions.items()},
            **{f"rerank_{k}": v for k, v in recalls.items()},
            **{f"rerank_{k}": v for k, v in precisions.items()},
            **{f"base_{k}": v for k, v in base_main_recalls.items()},
            **{f"rerank_{k}": v for k, v in rerank_main_recalls.items()},
        }
    if use_jina_lora:
        total = sim.shape[0]
        caption_to_indices = _build_caption_index_map(image_caps_full)
        recalls = {f"R@{k}": 0.0 for k in ks}
        precisions = {f"P@{k}": 0.0 for k in ks}
        k_base = min(topk_base, sim.shape[1])
        
        # ============ 加载测试数据集（如果需要情感融合）============
        test_dataset_for_emotion = None
        if use_jina_emotion_fusion:
            test_json_path = os.path.join(data_path, test_json)
            test_emotion_json_path = os.path.join(data_path, test_emotion_json)
            from transformers import BertTokenizer
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            test_dataset_for_emotion = MSRVTT_Dataset(
                csv_path=test_json_path,
                json_path=test_json_path,
                features_path=image_path,
                emotion_json_path=test_emotion_json_path,
                clip_preprocess=None,  # 不需要图像预处理
                bert_tokenizer=bert_tokenizer,
                max_words=32,
                is_train=False,
                load_image=False,  # 不需要加载图像
            )
            print(f"[Jina Emotion] Loaded test dataset for emotion fusion: {len(test_dataset_for_emotion.data)} samples")
        
        # ============ 加载评估阶段的 Jina 特征缓存（可选）============
        # 注意：缓存存储的是 score head 之前的特征（融合情感之前），
        # 情感特征每次动态计算并融合，因此缓存文件不需要区分是否使用情感融合
        jina_eval_cache = None
        eval_feature_map = None
        eval_candidate_map = None
        if use_jina_cache:
            jina_eval_cache_path = os.path.join(save_dir, "jina_features_eval_cache.pt")
            if os.path.exists(jina_eval_cache_path):
                print(f"[Jina Cache] Loading evaluation cache from: {jina_eval_cache_path}")
                try:
                    jina_eval_cache = torch.load(jina_eval_cache_path, map_location="cpu", weights_only=False)
                    if "features" in jina_eval_cache and "query_indices" in jina_eval_cache:
                        # 构建评估缓存的特征映射和候选图像索引映射
                        eval_feature_map = {}
                        eval_candidate_map = {}
                        candidate_indices_list = jina_eval_cache.get("candidate_indices", None)
                        for idx, (feat_list, q_idx_list) in enumerate(zip(
                            jina_eval_cache["features"],
                            jina_eval_cache["query_indices"]
                        )):
                            q_idx_val = int(q_idx_list[0].item() if isinstance(q_idx_list, torch.Tensor) else q_idx_list[0])
                            if q_idx_val not in eval_feature_map:
                                eval_feature_map[q_idx_val] = feat_list
                                if candidate_indices_list is not None and idx < len(candidate_indices_list):
                                    eval_candidate_map[q_idx_val] = candidate_indices_list[idx]
                        print(f"[Jina Cache] Loaded {len(eval_feature_map)} cached query features for evaluation")
                        if eval_candidate_map:
                            print(f"[Jina Cache] Loaded candidate indices for {len(eval_candidate_map)} queries")
                        print(f"[Jina Cache] Using cached features, skipping model forward pass (much faster!)")
                    else:
                        print(f"[Jina Cache] Warning: Evaluation cache format incorrect, falling back to model inference")
                        jina_eval_cache = None
                        eval_feature_map = None
                        eval_candidate_map = None
                except Exception as e:
                    print(f"[Jina Cache] Error loading evaluation cache: {e}, falling back to model inference")
                    jina_eval_cache = None
                    eval_feature_map = None
                    eval_candidate_map = None
            else:
                print(f"[Jina Cache] Evaluation cache not found: {jina_eval_cache_path}")
                print(f"[Jina Cache] To use evaluation cache, run precompute script for test set")
                print(f"[Jina Cache] Falling back to model inference")
        
        processor, jina_model = _load_jina_lora(jina_model_name, jina_lora_r, jina_lora_alpha, jina_lora_dropout, device, jina_lora_path)
        
        # ============ 初始化 Jina 情感适配器（在模型加载后，以便从 checkpoint 加载权重）============
        if use_jina_emotion_fusion:
            # 获取模型的 hidden_size（用于情感特征投影，与训练代码对齐）
            jina_dim = 1536  # 默认值
            if hasattr(jina_model, "config"):
                jina_dim = jina_model.config.hidden_size
            elif hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "model") and hasattr(jina_model.base_model.model, "config"):
                jina_dim = jina_model.base_model.model.config.hidden_size
            else:
                # 尝试从 checkpoint 中读取 emotion_adapter 的维度
                if jina_lora_path and os.path.exists(jina_lora_path):
                    try:
                        obj = torch.load(jina_lora_path, map_location="cpu", weights_only=False)
                        if isinstance(obj, dict) and "emotion_adapter" in obj:
                            emotion_state = obj["emotion_adapter"]
                            # 从 proj.0.weight 的形状推断 jina_dim: [jina_dim, 768]
                            if "proj.0.weight" in emotion_state:
                                jina_dim = emotion_state["proj.0.weight"].shape[0]
                                print(f"[Jina Emotion] Inferred jina_dim={jina_dim} from checkpoint emotion_adapter")
                    except Exception:
                        pass
                print(f"[Jina Emotion] Warning: Could not determine hidden_size from model config, using inferred/default {jina_dim}")
            
            jina_emotion_adapter = JinaEmotionAdapter(emo_dim=768, jina_dim=jina_dim).to(device)
            jina_emotion_adapter.eval()  # 确保处于评估模式
            # 确保 emotion_adapter 的 dtype 与模型一致（bfloat16）
            # 注意：BERT 部分保持 float32，但 proj 和 gate 层需要与模型 dtype 一致
            # 实际上，emotion_adapter 的 proj 和 gate 层应该使用 float32（因为它们是可训练的），
            # 但在推理时，输出会被转换为与 features_input 相同的 dtype
            print(f"[Jina Emotion] Initialized with jina_dim={jina_dim}")
            
            # 从 checkpoint 中加载 emotion_adapter 权重（如果存在）
            if jina_lora_path and os.path.exists(jina_lora_path):
                try:
                    obj = torch.load(jina_lora_path, map_location=device, weights_only=False)
                    if isinstance(obj, dict) and "emotion_adapter" in obj:
                        jina_emotion_adapter.load_state_dict(obj["emotion_adapter"], strict=False)
                        print(f"[Jina Emotion] Loaded emotion_adapter from checkpoint: {jina_lora_path}")
                    else:
                        print(f"[Jina Emotion] Warning: emotion_adapter not found in checkpoint: {jina_lora_path}")
                        print(f"[Jina Emotion] Available keys: {list(obj.keys()) if isinstance(obj, dict) else 'N/A'}")
                except Exception as e:
                    print(f"[Jina Emotion] Warning: Failed to load emotion_adapter from checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[Jina Emotion] Warning: jina_lora_path not provided or does not exist, emotion_adapter will use random weights!")
                print(f"[Jina Emotion] This will likely cause poor performance. Please provide a checkpoint with trained emotion_adapter.")
            print(f"[Jina Emotion] Emotion fusion enabled for Jina reranking")
        else:
            jina_emotion_adapter = None
            print(f"[Jina Emotion] Emotion fusion disabled for Jina reranking")
        
        # 如果使用评估缓存，优化模型设备：只需要 score head 在 GPU 上
        if use_jina_cache and eval_feature_map is not None:
            # 检查模型是否在 GPU 上
            model_device = next(jina_model.parameters()).device
            if model_device.type == "cuda":
                # 如果模型在 GPU 上，可以选择移回 CPU 以节省显存（但保持 score head 在 GPU）
                # 这里我们保持模型在 GPU 上，因为 score head 需要访问
                pass
            # 确保 score head 在 GPU 上
            if hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "model"):
                score_head = jina_model.base_model.model.score
            elif hasattr(jina_model, "score"):
                score_head = jina_model.score
            else:
                score_head = None
            if score_head is not None:
                if next(score_head.parameters()).device.type != "cuda" and device.type == "cuda":
                    score_head = score_head.to(device)
                    print(f"[Jina Cache] Moved score head to {device} for cached inference")
        
        from PIL import Image as _I
        rerank_ranked_indices_list = []
        rerank_scores_list = []
        base_scores_list = []
        
        # 检查哪些 query 已经有缓存（支持增量推理）
        cached_query_indices = set()
        if use_jina_cache and eval_feature_map is not None:
            for q_idx in range(total):
                if q_idx in eval_feature_map:
                    # 检查候选索引是否也有缓存
                    if eval_candidate_map is None or q_idx in eval_candidate_map:
                        cached_query_indices.add(q_idx)
        
        if len(cached_query_indices) > 0:
            print(f"[Incremental] Found {len(cached_query_indices)}/{total} queries already cached in Jina eval cache (will use cached features)")
        
        # 只对没有缓存的 query 进行推理
        uncached_indices = [q_idx for q_idx in range(total) if q_idx not in cached_query_indices]
        if len(uncached_indices) > 0:
            print(f"[Incremental] Processing {len(uncached_indices)}/{total} new queries via Jina model (will compute features)")
        else:
            print(f"[Incremental] All queries are cached, using cached features for all queries")
        
        for q_idx in tqdm(range(total), desc="Reranking (Jina LoRA)"):
            # 如果使用评估缓存，使用缓存中存储的候选图像索引
            if use_jina_cache and eval_candidate_map is not None and q_idx in eval_candidate_map:
                # 使用缓存中的候选图像索引
                base_topk = eval_candidate_map[q_idx].cpu().numpy()
                base_topk = _safe_clip_indices(base_topk, (len(video_ids) if video_ids is not None and isinstance(video_ids, list) else sim.shape[1]))
                if len(base_topk) == 0:
                    continue
            else:
                # 正常流程：动态选择候选图像
                if precomputed_topk is not None:
                    base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
                    if base_topk.shape[0] > k_base:
                        base_topk = base_topk[:k_base]
                else:
                    base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
                base_topk = _safe_clip_indices(base_topk, (len(video_ids) if video_ids is not None and isinstance(video_ids, list) else sim.shape[1]))
            
            # 收集 base 模型的分数
            base_scores = sim[q_idx][base_topk]
            base_order = np.argsort(-base_scores)
            base_sorted_scores = base_scores[base_order]
            base_scores_list.append(base_sorted_scores.tolist())
            
            # 如果使用评估缓存，不需要加载图像和计算特征
            if use_jina_cache and eval_feature_map is not None and q_idx in eval_feature_map:
                # 使用缓存的特征，直接通过 score head 计算分数
                cached_features = eval_feature_map[q_idx]  # [N, hidden_dim]
                
                # 验证特征数量与候选图像数量一致
                if cached_features.shape[0] != len(base_topk):
                    # 如果数量不一致，使用较小的数量
                    min_count = min(cached_features.shape[0], len(base_topk))
                    cached_features = cached_features[:min_count]
                    base_topk = base_topk[:min_count]
                    if cached_features.shape[0] != len(base_topk):
                        print(f"[Jina Cache] Warning: Feature count mismatch for query {q_idx}, using {min_count} candidates")
                
                # 获取 score head
                if hasattr(jina_model, "base_model") and hasattr(jina_model.base_model, "model"):
                    score_head = jina_model.base_model.model.score
                elif hasattr(jina_model, "score"):
                    score_head = jina_model.score
                else:
                    raise RuntimeError("Score head not found in model")
                
                # 确保 score head 在正确的设备上
                if next(score_head.parameters()).device.type != "cuda" and device.type == "cuda":
                    score_head = score_head.to(device)
                
                # 获取 score head 的 dtype，确保特征与模型 dtype 匹配
                score_head_dtype = next(score_head.parameters()).dtype
                
                # 通过 score head 计算分数
                with torch.no_grad():
                    features_input = cached_features.to(device).to(dtype=score_head_dtype)
                    
                    # 如果启用情感融合，动态计算情感特征并融合
                    # 注意：缓存中存储的是融合前的特征，情感特征每次重新计算
                    if use_jina_emotion_fusion and jina_emotion_adapter is not None and test_dataset_for_emotion is not None:
                        try:
                            # 从测试数据集中获取当前 query 的情感标签
                            if q_idx < len(test_dataset_for_emotion.data):
                                item = test_dataset_for_emotion.data[q_idx]
                                if "emotion_input_ids" in item and "emotion_attention_mask" in item:
                                    emotion_ids_tensor = torch.tensor(item["emotion_input_ids"], device=device).unsqueeze(0)
                                    emotion_mask_tensor = torch.tensor(item["emotion_attention_mask"], device=device).unsqueeze(0)
                                    # 扩展到 batch size（特征数量）
                                    batch_size = features_input.shape[0]
                                    emotion_ids_tensor = emotion_ids_tensor.expand(batch_size, -1)
                                    emotion_mask_tensor = emotion_mask_tensor.expand(batch_size, -1)
                                    # 动态提取情感特征（每次重新计算，不缓存）
                                    # 确保 emotion_features 的 dtype 与 features_input 一致
                                    emotion_features = jina_emotion_adapter.emotion_proj(emotion_ids_tensor, emotion_mask_tensor)
                                    # 调试信息（仅在第一个 query 时打印）
                                    if q_idx == 0:
                                        print(f"[Jina Emotion Debug] Query {q_idx}: features_input.shape={features_input.shape}, features_input.dtype={features_input.dtype}")
                                        print(f"[Jina Emotion Debug] Query {q_idx}: emotion_features.shape={emotion_features.shape}, emotion_features.dtype={emotion_features.dtype}")
                                    emotion_features = emotion_features.to(dtype=score_head_dtype)
                                    # 融合特征
                                    features_input = jina_emotion_adapter.fuse(features_input, emotion_features)
                                    if q_idx == 0:
                                        print(f"[Jina Emotion Debug] Query {q_idx}: After fuse, features_input.shape={features_input.shape}, features_input.dtype={features_input.dtype}")
                        except Exception as e:
                            # 如果融合失败，继续使用原始特征
                            print(f"[Jina Emotion] Warning: Failed to fuse emotion features for query {q_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    scores_t = score_head(features_input).squeeze(-1)
                    # numpy 不支持 bfloat16，需要先转换为 float32
                    scores_t = scores_t.cpu().float().numpy()
            else:
                # 正常流程：加载图像并通过模型前向传播
                imgs = []
                for idx_i in base_topk.tolist():
                    p = _build_image_path(image_path, video_ids[idx_i] if video_ids is not None else str(idx_i))
                    if os.path.exists(p):
                        try:
                            imgs.append(_I.open(p).convert("RGB"))
                        except Exception:
                            imgs.append(None)
                    else:
                        imgs.append(None)
                texts = [all_caps[q_idx]] * len(imgs)
                # 获取情感标签（如果启用情感融合）
                emotion_ids_batch = None
                emotion_mask_batch = None
                if use_jina_emotion_fusion and jina_emotion_adapter is not None and test_dataset_for_emotion is not None:
                    try:
                        # 从测试数据集中获取当前 query 的情感标签
                        if q_idx < len(test_dataset_for_emotion.data):
                            item = test_dataset_for_emotion.data[q_idx]
                            if "emotion_input_ids" in item and "emotion_attention_mask" in item:
                                emotion_ids_batch = [item["emotion_input_ids"]] * len(imgs)
                                emotion_mask_batch = [item["emotion_attention_mask"]] * len(imgs)
                    except Exception as e:
                        # 如果获取失败，继续不使用情感融合
                        emotion_ids_batch = None
                        emotion_mask_batch = None
                scores_t = _score_batch_jina(jina_model, processor, texts, imgs, device,
                                             image_max_side=jina_image_max_side,
                                             qwen_min_side=jina_qwen_min_side,
                                             jina_micro_batch=jina_micro_batch,
                                             max_text_len=jina_max_text_len,
                                             emotion_adapter=jina_emotion_adapter,
                                             emotion_ids=emotion_ids_batch,
                                             emotion_mask=emotion_mask_batch,
                                             use_emotion_fusion=use_jina_emotion_fusion).numpy()
                # 及时清理图片对象
                del imgs
            
            order = np.argsort(-scores_t)
            reranked_topk = base_topk[order]
            reranked_scores = scores_t[order]
            rerank_ranked_indices_list.append(reranked_topk)
            rerank_scores_list.append(reranked_scores.tolist())
            gt = caption_to_indices[all_caps[q_idx]]
            for k in ks:
                k_eff = min(k, k_base)
                topk_used = reranked_topk[:k_eff]
                if any(i in gt for i in topk_used):
                    recalls[f"R@{k}"] += 1.0
                correct = sum(1 for i in topk_used if i in gt)
                precisions[f"P@{k}"] += correct / float(k_eff)
            # 每处理一定数量的查询后清理一次缓存（间隔不要太小，否则会拖慢推理）
            if (q_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
        for k in ks:
            recalls[f"R@{k}"] = recalls[f"R@{k}"] / total * 100.0
            precisions[f"P@{k}"] = precisions[f"P@{k}"] / total * 100.0
        # 计算rerank后的主贴图片召回
        rerank_main_recalls = compute_main_post_recalls(rerank_ranked_indices_list, query_video_ids, video_ids, ks=ks)
        
        # 输出可视化结果
        if output_top_n_queries > 0 and output_results_dir:
            n_queries = min(output_top_n_queries, total)
            query_indices = list(range(n_queries))
            output_query_results(
                queries=query_indices,
                ranked_indices_list=rerank_ranked_indices_list,
                all_captions=all_caps,
                image_caps_full=image_caps_full,
                video_ids=video_ids,
                image_path=image_path,
                output_dir=output_results_dir,
                top_k=10,
                base_ranked_indices_list=base_ranked_indices_list,
                base_scores_list=base_scores_list,
                rerank_scores_list=rerank_scores_list,
                test_dataset=test_dataset_for_output
            )
        
        return {
            **{f"base_{k}": v for k, v in base_metrics.items()},
            **{f"base_{k}": v for k, v in base_precisions.items()},
            **{f"rerank_{k}": v for k, v in recalls.items()},
            **{f"rerank_{k}": v for k, v in precisions.items()},
            **{f"base_{k}": v for k, v in base_main_recalls.items()},
            **{f"rerank_{k}": v for k, v in rerank_main_recalls.items()},
        }
    
    # 如果没有使用 rerank，只输出 base 模型的可视化结果
    if output_top_n_queries > 0 and output_results_dir:
        total = sim.shape[0]
        n_queries = min(output_top_n_queries, total)
        query_indices = list(range(n_queries))
        # 收集 base 模型的分数
        base_scores_list = []
        for q_idx in query_indices:
            k_base = min(topk_base, sim.shape[1])
            if precomputed_topk is not None:
                base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
                if base_topk.shape[0] > k_base:
                    base_topk = base_topk[:k_base]
            else:
                base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
            base_topk = _safe_clip_indices(base_topk, sim.shape[1])
            base_scores = sim[q_idx][base_topk]
            base_order = np.argsort(-base_scores)
            base_sorted_scores = base_scores[base_order]
            base_scores_list.append(base_sorted_scores.tolist())
        
        output_query_results(
            queries=query_indices,
            ranked_indices_list=base_ranked_indices_list,
            all_captions=all_caps,
            image_caps_full=image_caps_full,
            video_ids=video_ids,
            image_path=image_path,
            output_dir=output_results_dir,
            top_k=10,
            base_ranked_indices_list=None,  # 没有 rerank，所以不显示 base 对比
            base_scores_list=base_scores_list,
            rerank_scores_list=None,  # 没有 rerank
            test_dataset=test_dataset_for_output
        )
    
    return {
        **{f"base_{k}": v for k, v in base_metrics.items()},
        **{f"base_{k}": v for k, v in base_precisions.items()},
        **{f"rerank_{k}": v for k, v in base_metrics.items()},
        **{f"rerank_{k}": v for k, v in base_precisions.items()},
        **{f"base_{k}": v for k, v in base_main_recalls.items()},
        **{f"rerank_{k}": v for k, v in base_main_recalls.items()},
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_clip/clip_best_model.pt")
    parser.add_argument("--data_path", type=str, default="./imgflip_data/msrvtt")
    parser.add_argument("--image_path", type=str, default="./imgflip_data/images")
    parser.add_argument("--test_json", type=str, default="test_data.json")
    parser.add_argument("--test_emotion_json", type=str, default="test_emotion.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--enable_rerank", action="store_true")
    parser.add_argument("--rerank_model_path", type=str, default=None)
    parser.add_argument("--rerank_model_type", type=str, default=None)
    parser.add_argument("--rerank_mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--rerank_mlp_layers", type=int, default=1)
    parser.add_argument("--rerank_mlp_dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_rerank")
    parser.add_argument("--topk_base", type=int, default=100)
    parser.add_argument("--use_image_emotion_fusion", action="store_true")
    parser.add_argument("--use_text_emotion_fusion", action="store_true")
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--use_rerank_api", action="store_true")
    parser.add_argument("--jina_model_name", type=str, default="jinaai/jina-reranker-m0")
    parser.add_argument("--jina_api_key", type=str, default="")
    parser.add_argument("--use_jina_lora", action="store_true")
    parser.add_argument("--jina_lora_path", type=str, default=None)
    parser.add_argument("--jina_lora_r", type=int, default=8)
    parser.add_argument("--jina_lora_alpha", type=int, default=16)
    parser.add_argument("--jina_lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_samples", type=int, default=0, help="number of queries to evaluate (sampling only applies to queries; image pool stays full)")
    parser.add_argument("--num_queries", type=int, default=None, help="alias of --num_samples; number of queries to evaluate")
    parser.add_argument("--sample_mode", type=str, default="first", choices=["first","random"])
    parser.add_argument("--subset_image_pool", action="store_true")
    parser.add_argument("--query_expansion_alpha", type=float, default=0.03, help="Weight for text-side pseudo-relevance feedback expansion (0 disables)")
    parser.add_argument("--query_expansion_alpha_image", type=float, default=0.027, help="Weight for image-side pseudo-relevance feedback expansion")
    parser.add_argument("--query_expansion_k", type=int, default=2, help="Number of top retrieved images to use for text-side expansion")
    parser.add_argument("--query_expansion_k_image", type=int, default=1, help="Number of top retrieved queries to use for image-side expansion")
    parser.add_argument("--query_expansion_k_schedule", type=str, default="3,2,2", help="Comma-separated per-iteration k values for text-side expansion")
    parser.add_argument("--query_expansion_k_image_schedule", type=str, default="2,3,1", help="Comma-separated per-iteration k values for image-side expansion")
    parser.add_argument("--query_expansion_iters", type=int, default=3, help="Number of alternating query-image expansion iterations")
    parser.add_argument("--image_projection_path", type=str, default=None, help="Path to a learned image-to-text projection matrix (saved torch dict with key 'W')")
    parser.add_argument("--projection_mix", type=float, default=0.9, help="Interpolation factor between original and projected image features (0=original, 1=projection)")
    parser.add_argument("--image_projection_sim_fusion", type=float, default=0.5, help="Weight for fusing original-image similarity into projected similarity (0=disabled)")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--debug_samples", type=int, default=0)
    parser.add_argument("--rerank_cache_path", type=str, default=None)
    parser.add_argument("--jina_timeout", type=float, default=120)
    parser.add_argument("--jina_chunk_size", type=int, default=25)
    parser.add_argument("--jina_max_retries", type=int, default=2)
    parser.add_argument("--jina_image_max_side", type=int, default=256, help="Maximum side length for image resizing (aligned with training)")
    parser.add_argument("--jina_qwen_min_side", type=int, default=32, help="Minimum side length for image resizing (aligned with training)")
    parser.add_argument("--jina_micro_batch", type=int, default=0, help="Micro batch size for Jina model inference (0 means no micro batching, aligned with training)")
    parser.add_argument("--jina_max_text_len", type=int, default=2048, help="Maximum text length for Jina model (aligned with training)")
    parser.add_argument("--use_jina_cache", action="store_true",
                        help="Use pre-extracted Jina features cache for inference (much faster). "
                             "Cache should be generated by precompute_jina_features.py first. "
                             "When enabled, only score head is used for inference, skipping model forward pass.")
    parser.add_argument("--use_jina_emotion_fusion", action="store_true",
                        help="Enable emotion fusion for Jina reranking. "
                             "Requires emotion_adapter in checkpoint and test_emotion.json file. "
                             "Compatible with --use_jina_cache option.")
    parser.add_argument("--output_top_n_queries", type=int, default=0,
                        help="Output visualization results for top N queries (0 means disabled). "
                             "Results will be saved as markdown file and images.")
    parser.add_argument("--output_results_dir", type=str, default=None,
                        help="Directory to save visualization results (markdown and images). "
                             "Required if --output_top_n_queries > 0.")
    args = parser.parse_args()
    if getattr(args, "num_queries", None) is not None:
        args.num_samples = args.num_queries
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = infer(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        image_path=args.image_path,
        test_json=args.test_json,
        test_emotion_json=args.test_emotion_json,
        batch_size=args.batch_size,
        enable_rerank=args.enable_rerank,
        rerank_model_path=args.rerank_model_path,
        rerank_model_type=args.rerank_model_type,
        rerank_mlp_hidden_dim=args.rerank_mlp_hidden_dim,
        rerank_mlp_layers=args.rerank_mlp_layers,
        rerank_mlp_dropout=args.rerank_mlp_dropout,
        save_dir=args.save_dir,
        topk_base=args.topk_base,
        device=device,
        use_image_emotion_fusion=args.use_image_emotion_fusion,
        use_text_emotion_fusion=args.use_text_emotion_fusion,
        zero_shot=args.zero_shot,
        use_rerank_api=args.use_rerank_api,
        jina_model_name=args.jina_model_name,
        jina_api_key=args.jina_api_key,
        use_jina_lora=args.use_jina_lora,
        jina_lora_path=args.jina_lora_path,
        jina_lora_r=args.jina_lora_r,
        jina_lora_alpha=args.jina_lora_alpha,
        jina_lora_dropout=args.jina_lora_dropout,
        num_samples=args.num_samples,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        subset_image_pool=args.subset_image_pool,
        query_expansion_alpha=args.query_expansion_alpha,
        query_expansion_alpha_image=args.query_expansion_alpha_image,
        query_expansion_k=args.query_expansion_k,
        query_expansion_k_image=args.query_expansion_k_image,
        query_expansion_k_schedule=args.query_expansion_k_schedule,
        query_expansion_k_image_schedule=args.query_expansion_k_image_schedule,
        query_expansion_iters=args.query_expansion_iters,
        image_projection_path=args.image_projection_path,
        projection_mix=args.projection_mix,
        image_projection_sim_fusion=args.image_projection_sim_fusion,
        debug_samples=args.debug_samples,
        rerank_cache_path=args.rerank_cache_path,
        jina_timeout=args.jina_timeout,
        jina_chunk_size=args.jina_chunk_size,
        jina_max_retries=args.jina_max_retries,
        jina_image_max_side=args.jina_image_max_side,
        jina_qwen_min_side=args.jina_qwen_min_side,
        jina_micro_batch=args.jina_micro_batch,
        jina_max_text_len=args.jina_max_text_len,
        use_jina_cache=args.use_jina_cache,
        use_jina_emotion_fusion=args.use_jina_emotion_fusion,
        output_top_n_queries=args.output_top_n_queries,
        output_results_dir=args.output_results_dir,
    )
    if "base_R@1" in metrics:
        print("\n=== 标准召回指标 ===")
        for k in [1, 3, 5, 10, 50, 100]:
            br = metrics.get(f"base_R@{k}", 0.0)
            bp = metrics.get(f"base_P@{k}", 0.0)
            rr = metrics.get(f"rerank_R@{k}", 0.0)
            rp = metrics.get(f"rerank_P@{k}", 0.0)
            print(f"base R@{k}: {br:.2f}% | base P@{k}: {bp:.2f}% | rerank R@{k}: {rr:.2f}% | rerank P@{k}: {rp:.2f}%")
        print("\n=== 主贴图片召回指标 ===")
        for k in [1, 3, 5, 10, 50, 100]:
            bmr = metrics.get(f"base_main_R@{k}", 0.0)
            rmr = metrics.get(f"rerank_main_R@{k}", 0.0)
            print(f"base main R@{k}: {bmr:.2f}% | rerank main R@{k}: {rmr:.2f}%")
    else:
        for k in [1, 3, 5, 10, 50, 100]:
            r = metrics.get(f"R@{k}", 0.0)
            p = metrics.get(f"P@{k}", 0.0)
            print(f"R@{k}: {r:.2f}% | P@{k}: {p:.2f}%")

if __name__ == "__main__":
    main()
