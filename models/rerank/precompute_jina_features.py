# precompute_jina_features_single_gpu.py
# 单卡版本的 Jina 特征预提取脚本
# 适用于只有一张 GPU 或想使用单卡的情况

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import clip
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

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
    "test_json": "test_data.json",
    "test_emotion_json": "test_emotion.json",
    "save_dir": "./checkpoints_rerank_score_only",
    "image_max_side": 256,
    "jina_micro_batch": 50,
    "qwen_min_side": 32,
    "max_text_len": 2048,
}


def preprocess_image(img, max_side_cfg, min_side_cfg):
    """预处理单张图像"""
    from PIL import Image as _I
    if img is None:
        return _I.new("RGB", (224, 224), (0, 0, 0))
    
    w, h = img.size
    if min(w, h) < min_side_cfg:
        scale_up = float(min_side_cfg) / float(min(w, h))
        w = int(round(w * scale_up))
        h = int(round(h * scale_up))
        img = img.resize((w, h), Image.BICUBIC)
    if max(w, h) > max_side_cfg:
        scale_down = float(max_side_cfg) / float(max(w, h))
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
    return img


def extract_jina_features(model, processor, texts, images, device, micro_batch=32):
    """提取 Jina 主干网络的特征（在 score token 位置之前）"""
    max_side_cfg = int(CONFIG.get("image_max_side", 256))
    min_side_cfg = int(CONFIG.get("qwen_min_side", 32))
    
    # 图像预处理（可以并行化）
    imgs_list = [preprocess_image(img, max_side_cfg, min_side_cfg) for img in images]
    
    mdl_dev = next(model.parameters()).device
    all_features = []
    
    # 批处理提取特征
    # 优化：如果图像数量小于micro_batch，直接处理，避免不必要的循环
    if len(imgs_list) <= micro_batch:
        # 小batch直接处理，避免循环开销
        micro_batch = len(imgs_list)
    
    for i in range(0, len(imgs_list), micro_batch):
        batch_texts = []
        batch_images = imgs_list[i:i+micro_batch]
        batch_text_strs = texts[i:i+micro_batch]
        
        # 批量生成prompt
        batch_texts = [
            formatting_prompts_func(
                query=text,
                doc="",
                query_type='text',
                doc_type='image'
            )
            for text in batch_text_strs
        ]
        
        inputs = processor(
            text=batch_texts, 
            images=batch_images, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=int(CONFIG.get("max_text_len", 512))
        )
        inputs = {k: v.to(mdl_dev) for k, v in inputs.items()}
        
        # 添加 score token（与 jina_modeling.py 中的实现一致）
        # 在 jina_modeling.py 的 compute_score 方法中，也会添加 score_token_id
        try:
            tok_id = int(getattr(model, "score_token_id", 100))  # 默认值 100，与 jina_modeling.py 一致
            bsz = inputs["input_ids"].size(0)
            inputs["input_ids"] = torch.cat([
                inputs["input_ids"], 
                torch.full((bsz, 1), tok_id, device=inputs["input_ids"].device)
            ], dim=1)
            inputs["attention_mask"] = torch.cat([
                inputs["attention_mask"], 
                torch.ones((bsz, 1), device=inputs["attention_mask"].device)
            ], dim=1)
        except Exception:
            pass
        
        # 提取隐藏状态（不计算 score 层）
        # 优化：使用 torch.inference_mode() 替代 torch.no_grad()，性能更好
        # 使用 model 的 forward 方法而不是直接调用父类，更高效
        with torch.inference_mode():
            from transformers import Qwen2VLForConditionalGeneration
            
            # 准备 forward 参数
            forward_inputs = dict(inputs)
            forward_inputs.update({
                "use_cache": False,
                "output_hidden_states": True
            })
            
            if mdl_dev.type == "cuda":
                # 使用与模型相同的 dtype 进行 autocast
                model_dtype = next(model.parameters()).dtype
                if model_dtype == torch.bfloat16:
                    autocast_dtype = torch.bfloat16
                else:
                    autocast_dtype = torch.float16
                
                # 优化：使用 torch.amp.autocast 替代 torch.cuda.amp.autocast（新API）
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    try:
                        # 优化：直接调用父类的 forward 方法（更高效）
                        outputs = Qwen2VLForConditionalGeneration.forward(
                            model,
                            **forward_inputs
                        )
                    except TypeError as e:
                        # 处理可能的参数不匹配问题（如 pixel_values vs images）
                        if 'pixel_values' in str(e) or 'images' in str(e):
                            forward_inputs_alt = {k: v for k, v in forward_inputs.items() 
                                                 if k not in ['pixel_values', 'images']}
                            if batch_images:
                                forward_inputs_alt['images'] = batch_images
                            outputs = Qwen2VLForConditionalGeneration.forward(
                                model,
                                **forward_inputs_alt
                            )
                        else:
                            raise
                    
                    # 获取最后一层隐藏状态的最后一个 token（score token 位置）
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
                    else:
                        raise ValueError("Model forward did not return hidden_states. Check model structure.")
            else:
                # CPU 模式
                try:
                    outputs = Qwen2VLForConditionalGeneration.forward(
                        model,
                        **forward_inputs
                    )
                except TypeError as e:
                    if 'pixel_values' in str(e) or 'images' in str(e):
                        forward_inputs_alt = {k: v for k, v in forward_inputs.items() 
                                             if k not in ['pixel_values', 'images']}
                        if batch_images:
                            forward_inputs_alt['images'] = batch_images
                        outputs = Qwen2VLForConditionalGeneration.forward(
                            model,
                            **forward_inputs_alt
                        )
                    else:
                        raise
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
                else:
                    raise ValueError("Model forward did not return hidden_states. Check model structure.")
        
        # 优化：延迟移动到CPU，减少数据传输
        all_features.append(hidden.cpu().float())
        
        del inputs, outputs, hidden
        # 优化：减少 empty_cache 调用频率，只在必要时清理
    
    return torch.cat(all_features, dim=0)


def compute_features(clip_model, dataloader, device):
    """计算 CLIP 特征"""
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


def build_image_paths(dataset, image_root):
    paths = []
    for item in dataset.data:
        vid = item["video_id"]
        name = vid if "." in vid else f"{vid}.jpg"
        paths.append(os.path.join(image_root, name))
    return paths


def load_image_safe(path):
    """安全加载图像，失败时返回黑色图像"""
    try:
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        else:
            return Image.new("RGB", (224, 224))
    except Exception:
        return Image.new("RGB", (224, 224))


def load_images_parallel(image_paths, max_workers=4):
    """并行加载图像，保持顺序"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_image_safe, path): idx for idx, path in enumerate(image_paths)}
        images = [None] * len(image_paths)
        for future in as_completed(futures):
            idx = futures[future]
            images[idx] = future.result()
    return images


def build_labels(train_indices, caption_to_indices, captions, upvotes, q_idx, label_mode):
    """构建标签"""
    gt = set(caption_to_indices[captions[q_idx]])
    pos_list = [
        (i, upvotes[i]) for i in train_indices 
        if i in gt and (isinstance(upvotes, (list, np.ndarray)) and i < len(upvotes)) and upvotes[i] > 0
    ]
    
    if len(pos_list) == 0:
        # 调试信息：检查为什么没有正样本
        gt_in_candidates = [i for i in train_indices if i in gt]
        if len(gt_in_candidates) > 0:
            # 有正样本在候选集中，但upvote都是0
            upvote_vals = [upvotes[i] for i in gt_in_candidates if i < len(upvotes)] if isinstance(upvotes, (list, np.ndarray)) else []
            if len(upvote_vals) > 0 and all(u == 0 for u in upvote_vals):
                # 所有正样本的upvote都是0，这是正常的（可能数据集中upvote都是0）
                pass
        return None
    
    labels = np.zeros(train_indices.shape[0], dtype=np.float32)
    if label_mode == "inv_rank":
        sorted_pos = sorted(pos_list, key=lambda x: x[1], reverse=True)
        for rank, (i, _) in enumerate(sorted_pos, start=1):
            idx_in_arr = np.where(train_indices == i)[0][0]
            labels[idx_in_arr] = float(1.0 / rank)
    elif label_mode == "raw_log":
        vals = np.log1p(np.array([u for _, u in pos_list], dtype=np.float32))
        for (i, _), s in zip(pos_list, vals):
            idx_in_arr = np.where(train_indices == i)[0][0]
            labels[idx_in_arr] = float(s)
    else:  # softmax
        vals = np.log1p(np.array([u for _, u in pos_list], dtype=np.float32))
        exps = np.exp(vals - vals.max())
        sm = exps / exps.sum()
        for (i, _), s in zip(pos_list, sm):
            idx_in_arr = np.where(train_indices == i)[0][0]
            labels[idx_in_arr] = float(s)
    
    return labels


def get_candidate_indices(q_idx, sim, caption_to_indices, captions, num_images, 
                          topk_base, extra_negs, candidate_mode):
    """获取候选图像索引
    
    注意：最终返回的候选图像数量不会超过 topk_base（默认50）
    """
    k_base = min(topk_base, sim.shape[1])
    # 使用 argpartition 优化，只取前k个
    base_topk = np.argpartition(-sim[q_idx], k_base-1)[:k_base]
    
    pos_all = np.array(caption_to_indices[captions[q_idx]], dtype=np.int64)
    if candidate_mode == "topk":
        core_set = base_topk
    elif candidate_mode == "pos_all":
        # 如果使用 pos_all，也要限制在 topk_base 内
        core_set = pos_all[:topk_base] if len(pos_all) > topk_base else pos_all
    else:
        # topk_plus_pos: 合并 topk 和正样本，但总数不超过 topk_base
        combined = np.unique(np.concatenate([base_topk, pos_all]))
        # 优先保留 topk 中的，然后补充正样本，但总数不超过 topk_base
        if len(combined) > topk_base:
            # 优先保留 base_topk，然后从 pos_all 中选择（如果不在 base_topk 中）
            pos_not_in_topk = np.setdiff1d(pos_all, base_topk)
            remaining_slots = topk_base - len(base_topk)
            if remaining_slots > 0 and len(pos_not_in_topk) > 0:
                # 从不在 topk 中的正样本中选择
                n_select = min(remaining_slots, len(pos_not_in_topk))
                selected_pos = np.random.choice(pos_not_in_topk, size=n_select, replace=False)
                core_set = np.concatenate([base_topk, selected_pos])
            else:
                core_set = base_topk[:topk_base]
        else:
            core_set = combined
    
    # 如果设置了 extra_negs，但需要确保总数不超过 topk_base
    if extra_negs > 0:
        all_indices = np.arange(num_images)
        mask = np.ones_like(all_indices, dtype=bool)
        mask[core_set] = False
        candidates_neg = all_indices[mask]
        
        if candidates_neg.size > 0:
            # 计算还能添加多少个负样本（不超过 topk_base）
            remaining_slots = topk_base - len(core_set)
            if remaining_slots > 0:
                sel = np.random.choice(
                    candidates_neg, 
                    size=min(extra_negs, candidates_neg.size, remaining_slots), 
                    replace=False
                )
                train_indices = np.concatenate([core_set, sel])
            else:
                train_indices = core_set
        else:
            train_indices = core_set
    else:
        train_indices = core_set
    
    # 确保最终数量不超过 topk_base
    if len(train_indices) > topk_base:
        train_indices = train_indices[:topk_base]
    
    # 过滤无效索引
    train_indices = train_indices[(train_indices >= 0) & (train_indices < num_images)]
    return train_indices


def main(args):
    # ============ 设备设置 ============
    if torch.cuda.is_available():
        if args and hasattr(args, "device") and args.device:
            device_str = str(args.device).lower()
            if device_str.startswith("cuda:"):
                device_idx = int(device_str.split(":")[1])
                device = torch.device(f"cuda:{device_idx}")
            else:
                device = torch.device(device_str)
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print(f"[Precompute] Using device: {device}")
    
    # ============ 配置 ============
    if args is not None:
        CONFIG.update({
            "data_path": args.data_path,
            "image_path": args.image_path,
            "save_dir": args.save_dir,
        })
        topk_base = getattr(args, "topk_base", 50)
        extra_negs = max(0, getattr(args, "extra_negatives", 0))
        candidate_mode = getattr(args, "candidate_mode", "topk_plus_pos")
        model_name = getattr(args, "model_name", "jinaai/jina-reranker-m0")
        label_mode = getattr(args, "label_mode", "inv_rank")
        CONFIG["image_max_side"] = int(getattr(args, "image_max_side", 256))
        CONFIG["jina_micro_batch"] = int(getattr(args, "jina_micro_batch", 50))
    else:
        topk_base = 50
        extra_negs = 0
        candidate_mode = "topk_plus_pos"
        model_name = "jinaai/jina-reranker-m0"
        label_mode = "inv_rank"
    
    # ============ 确定是训练缓存还是评估缓存 ============
    is_eval_cache = getattr(args, "eval_cache", False) if args else False
    print(f"[Eval Cache] is_eval_cache: {is_eval_cache}")
    # ============ 数据集和 CLIP 特征 ============
    device_clip = torch.device("cpu")
    clip_model, preprocess = clip.load("ViT-B/32", device=device_clip)
    
    if is_eval_cache:
        # 评估缓存：使用测试集
        
        test_json_path = os.path.join(CONFIG["data_path"], CONFIG["test_json"])
        test_emotion_json_path = os.path.join(CONFIG["data_path"], CONFIG["test_emotion_json"])
        
        test_dataset = MSRVTT_Dataset(
            csv_path=test_json_path,  
            json_path=test_json_path,
            features_path=CONFIG["image_path"],
            emotion_json_path=test_emotion_json_path,
            clip_preprocess=preprocess,
            bert_tokenizer=None,
            max_words=32,
            is_train=False,  # 测试集
            load_image=True,
        )
        dataset = test_dataset
        cache_path = os.path.join(CONFIG["save_dir"], "test_cache.pt")
        jina_cache_path = os.path.join(CONFIG["save_dir"], "jina_features_eval_cache.pt")
        progress_path = os.path.join(CONFIG["save_dir"], "jina_features_eval_progress.pt")
        print(f"[Eval Cache] Generating evaluation cache for test set")
    else:
        # 训练缓存：使用训练集
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
        dataset = train_dataset
        cache_path = os.path.join(CONFIG["save_dir"], "train_cache.pt")
        jina_cache_path = os.path.join(CONFIG["save_dir"], "jina_features_cache.pt")
        progress_path = os.path.join(CONFIG["save_dir"], "jina_features_progress.pt")
        print(f"[Train Cache] Generating training cache for train set")
    
    # ============ 加载或计算 CLIP 缓存 ============
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    if os.path.exists(cache_path):
        print(f"[Cache] Loading CLIP cache from: {cache_path}")
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        text_matrix = obj["text"].cpu()
        image_matrix = obj["image"].cpu()
        upvotes = obj.get("upvotes", None)
        captions = obj["captions"]
        topk_cached = obj.get("topk_indices", None)
    else:
        print(f"[Cache] Computing CLIP features...")
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        text_matrix, image_matrix, upvotes, captions = compute_features(clip_model, loader, torch.device("cpu"))
        
        cache_data = {
            "text": text_matrix.cpu(),
            "image": image_matrix.cpu(),
            "upvotes": upvotes,
            "captions": captions,
        }
        torch.save(cache_data, cache_path)
        print(f"[Cache] Saved CLIP cache to: {cache_path}")
        topk_cached = None
    
    image_paths = build_image_paths(dataset, CONFIG["image_path"])
    print(f"[Data] Queries: {text_matrix.shape[0]}, Images: {len(image_paths)}")
    
    # ============ 加载模型 ============
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    try:
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
    except Exception:
        pass
    
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # 确定模型 dtype
    # 默认使用 bf16（如果 CUDA 支持），因为模型通常是 bf16
    model_dtype_str = getattr(args, "model_dtype", None) if args else None
    if model_dtype_str is None:
        # 自动检测：默认使用 bf16（如果支持），否则使用 float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            print(f"[Model] Auto-detected dtype: bfloat16 (CUDA bf16 supported, default for jina models)")
        else:
            model_dtype = torch.float16
            print(f"[Model] Auto-detected dtype: float16 (bf16 not supported, using fp16)")
    else:
        # 用户指定的 dtype
        if model_dtype_str.lower() in ['bf16', 'bfloat16']:
            model_dtype = torch.bfloat16
        elif model_dtype_str.lower() in ['fp16', 'float16']:
            model_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported dtype: {model_dtype_str}. Use 'bf16' or 'fp16'")
        print(f"[Model] Using specified dtype: {model_dtype}")
    
    base_model = JinaVLForRanking.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # 冻结所有参数
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 移到 GPU
    base_model = base_model.to(device)
    print(f"[Model] Loaded base model (frozen for feature extraction)")
    print(f"[Model] Model is on device: {next(base_model.parameters()).device}")
    
    # ============ 预提取特征 ============
    # jina_cache_path 和 progress_path 已在上面根据 is_eval_cache 设置
    
    # 获取参数
    max_queries = getattr(args, "max_queries", None) if args else None
    resume = getattr(args, "resume", False) if args else False
    
    # 检查是否resume
    start_idx = 0
    all_features_list = []
    all_labels_list = []
    all_query_indices = []
    all_candidate_indices_list = []  # 评估缓存需要存储候选图像索引
    all_is_positive_list = []  # 评估缓存需要存储正样本标记
    
    if resume and os.path.exists(jina_cache_path):
        print(f"[Resume] Loading existing cache from: {jina_cache_path}")
        try:
            existing_cache = torch.load(jina_cache_path, map_location="cpu", weights_only=False)
            if "features" in existing_cache and "query_indices" in existing_cache:
                all_features_list = existing_cache["features"]
                all_query_indices = existing_cache["query_indices"]
                if is_eval_cache:
                    if "candidate_indices" in existing_cache:
                        all_candidate_indices_list = existing_cache["candidate_indices"]
                    if "is_positive" in existing_cache:
                        all_is_positive_list = existing_cache["is_positive"]
                if not is_eval_cache and "labels" in existing_cache:
                    all_labels_list = existing_cache["labels"]
                start_idx = len(all_features_list)
                print(f"[Resume] Found {start_idx} queries already processed, resuming from query {start_idx}")
            else:
                print(f"[Resume] Cache file exists but format incorrect, starting from scratch")
                start_idx = 0
        except Exception as e:
            print(f"[Resume] Error loading cache: {e}, starting from scratch")
            start_idx = 0
    
    if not resume and os.path.exists(jina_cache_path):
        cache_type = "evaluation" if is_eval_cache else "training"
        print(f"[Jina Cache] {cache_type.capitalize()} cache already exists: {jina_cache_path}")
        print(f"[Jina Cache] Skipping pre-extraction. Delete cache file to re-extract, or use --resume to continue.")
        return
    
    cache_type = "evaluation" if is_eval_cache else "training"
    print(f"[Jina Cache] Pre-extracting {cache_type} features (single GPU)...")
    
    from collections import defaultdict
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(captions):
        caption_to_indices[cap].append(idx)
    
    # 优化：使用float32计算相似度矩阵，减少内存占用
    sim = (text_matrix.cpu().float().numpy() @ image_matrix.cpu().float().numpy().T).astype(np.float32)
    
    num_queries = text_matrix.shape[0]
    num_images = len(image_paths)
    
    # 确定要处理的query范围
    if max_queries is not None and max_queries > 0:
        end_idx = min(start_idx + max_queries, num_queries)
        queries_to_process = range(start_idx, end_idx)
        print(f"[Precompute] Processing queries {start_idx} to {end_idx-1} ({len(queries_to_process)} queries)")
    else:
        end_idx = num_queries
        queries_to_process = range(start_idx, end_idx)
        print(f"[Precompute] Processing queries {start_idx} to {end_idx-1} (all remaining {len(queries_to_process)} queries)")
    
    # 优化：分批保存，避免内存溢出
    save_batch_size = getattr(args, "save_batch_size", 1000) if args else 1000
    
    # 进度条
    pbar = tqdm(queries_to_process, desc="Pre-extracting Jina features", initial=start_idx, total=end_idx)
    
    # 统计信息
    skipped_empty_candidates = 0
    skipped_no_labels = 0
    processed_queries = 0
    first_skip_reason = None  # 记录第一个跳过的原因
    
    # 检查upvotes数据
    if not is_eval_cache and upvotes is not None:
        upvote_stats = {
            "total": len(upvotes) if isinstance(upvotes, (list, np.ndarray)) else 0,
            "non_zero": sum(1 for u in upvotes if u > 0) if isinstance(upvotes, (list, np.ndarray)) else 0,
            "zero": sum(1 for u in upvotes if u == 0) if isinstance(upvotes, (list, np.ndarray)) else 0,
        }
        print(f"[Debug] Upvotes statistics: {upvote_stats}")
    
    for q_idx in pbar:
        # 获取候选
        # 对于评估缓存，使用 topk_base 作为候选数量（评估时每个query需要与topk个候选计算分数）
        if is_eval_cache:
            # 评估缓存：每个query需要与topk_base个候选图像计算分数
            k_base = min(topk_base, sim.shape[1])
            candidate_indices = np.argpartition(-sim[q_idx], k_base-1)[:k_base]
            candidate_indices = candidate_indices[(candidate_indices >= 0) & (candidate_indices < num_images)]
            if candidate_indices.size == 0:
                skipped_empty_candidates += 1
                continue
            
            # 标记哪些候选是正样本（相对于当前 query）
            # 这样评估时就不需要构建 caption_to_indices 了
            gt_indices = np.array(caption_to_indices[captions[q_idx]], dtype=np.int64)
            is_positive = np.isin(candidate_indices, gt_indices)
        else:
            # 训练缓存：使用原有的候选选择逻辑
            candidate_indices = get_candidate_indices(
                q_idx, sim, caption_to_indices, captions, num_images,
                topk_base, extra_negs, candidate_mode
            )
            if candidate_indices.size == 0:
                skipped_empty_candidates += 1
                if first_skip_reason is None:
                    first_skip_reason = f"Query {q_idx}: empty candidate_indices"
                continue
            
            # 构建标签（仅训练缓存需要）
            labels = build_labels(candidate_indices, caption_to_indices, captions, upvotes, q_idx, label_mode)
            if labels is None:
                skipped_no_labels += 1
                if first_skip_reason is None:
                    # 检查为什么labels为None
                    gt = set(caption_to_indices[captions[q_idx]])
                    gt_in_candidates = [i for i in candidate_indices if i in gt]
                    if len(gt_in_candidates) == 0:
                        first_skip_reason = f"Query {q_idx}: no positive samples in candidates (gt={len(gt)}, candidates={len(candidate_indices)})"
                    else:
                        upvote_vals = [upvotes[i] for i in gt_in_candidates if i < len(upvotes)] if isinstance(upvotes, (list, np.ndarray)) else []
                        if len(upvote_vals) > 0 and all(u == 0 for u in upvote_vals):
                            first_skip_reason = f"Query {q_idx}: all positive samples have upvote=0 (gt_in_candidates={len(gt_in_candidates)})"
                        else:
                            first_skip_reason = f"Query {q_idx}: labels=None for unknown reason"
                continue
        
        processed_queries += 1
        
        # 优化：并行加载图像
        selected_paths = [image_paths[i] for i in candidate_indices if i < len(image_paths)]
        image_load_workers = getattr(args, "image_load_workers", 4) if args else 4
        imgs = load_images_parallel(selected_paths, max_workers=image_load_workers)
        
        # 确保图像数量匹配
        if len(imgs) < len(candidate_indices):
            imgs.extend([Image.new("RGB", (224, 224))] * (len(candidate_indices) - len(imgs)))
        imgs = imgs[:len(candidate_indices)]
        
        texts = [captions[q_idx]] * len(imgs)
        
        # 提取特征
        actual_num_images = len(imgs)
        effective_micro_batch = min(CONFIG["jina_micro_batch"], actual_num_images)
        
        features = extract_jina_features(
            base_model, processor, texts, imgs, 
            device, micro_batch=effective_micro_batch
        )
        
        # 确保特征在 CPU 上且为 float32
        features_cpu = features.cpu().float() if features.device.type != "cpu" else features.float()
        all_features_list.append(features_cpu)
        
        if not is_eval_cache:
            # 训练缓存：保存标签
            labels_tensor = torch.from_numpy(labels).float() if isinstance(labels, np.ndarray) else labels.float()
            all_labels_list.append(labels_tensor)
        
        # 确保 query_indices 在 CPU 上且为 long
        num_candidates = len(candidate_indices)
        query_idx_tensor = torch.full((num_candidates,), q_idx, dtype=torch.long)
        all_query_indices.append(query_idx_tensor)
        
        # 评估缓存需要存储候选图像索引和正样本标记
        if is_eval_cache:
            candidate_indices_tensor = torch.from_numpy(candidate_indices).long()
            all_candidate_indices_list.append(candidate_indices_tensor)
            # 存储正样本标记（bool 类型，节省空间）
            is_positive_tensor = torch.from_numpy(is_positive).bool()
            all_is_positive_list.append(is_positive_tensor)
        
        del imgs, texts, features
        # 优化：减少 empty_cache 调用频率，每100个query清理一次
        if (q_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()
        
        # 优化：分批保存，减少内存占用，并支持resume
        if (q_idx + 1) % save_batch_size == 0:
            print(f"[Jina Cache] Intermediate save at query {q_idx+1}/{end_idx}")
            # 保存主缓存（支持resume）
            if is_eval_cache:
                jina_cache = {
                    "features": all_features_list,
                    "query_indices": all_query_indices,
                    "candidate_indices": all_candidate_indices_list,  # 存储候选图像索引
                    "is_positive": all_is_positive_list  # 存储正样本标记
                }
            else:
                jina_cache = {
                    "features": all_features_list,
                    "labels": all_labels_list,
                    "query_indices": all_query_indices
                }
            torch.save(jina_cache, jina_cache_path)
            # 保存进度信息
            progress_info = {
                "last_query_idx": q_idx,
                "total_queries_processed": len(all_features_list),
                "total_queries": num_queries
            }
            torch.save(progress_info, progress_path)
            print(f"[Jina Cache] Saved intermediate cache to: {jina_cache_path}")
            print(f"[Jina Cache] Progress: {len(all_features_list)}/{num_queries} queries processed")
    
    # ============ 保存最终缓存 ============
    print(f"[Jina Cache] Extracted {len(all_features_list)} query features (queries {start_idx} to {end_idx-1})")
    print(f"[Jina Cache] Statistics: processed={processed_queries}, skipped_empty_candidates={skipped_empty_candidates}, skipped_no_labels={skipped_no_labels}")
    if first_skip_reason:
        print(f"[Jina Cache] First skip reason: {first_skip_reason}")
    
    # 验证数据格式
    total_samples = sum(f.shape[0] for f in all_features_list)
    total_indices = sum(q.shape[0] for q in all_query_indices)
    
    if not is_eval_cache:
        total_labels = sum(l.shape[0] for l in all_labels_list)
        if total_samples != total_labels or total_samples != total_indices:
            raise ValueError(f"Data mismatch: features={total_samples}, labels={total_labels}, indices={total_indices}")
    else:
        if total_samples != total_indices:
            raise ValueError(f"Data mismatch: features={total_samples}, indices={total_indices}")
    
    print(f"[Jina Cache] Total samples: {total_samples}")
    print(f"[Jina Cache] Features shape per query: {[f.shape for f in all_features_list[:3]]} (showing first 3)")
    print(f"[Jina Cache] Saving final cache to: {jina_cache_path}")
    
    # 确保所有数据在 CPU 上
    if is_eval_cache:
        # 评估缓存：需要 features, query_indices, candidate_indices, is_positive
        # candidate_indices 用于确保评估时使用的候选图像与缓存生成时一致
        # is_positive 用于标记哪些候选是正样本，评估时不需要构建 caption_to_indices
        jina_cache = {
            "features": all_features_list,  # List[torch.Tensor], 每个元素是 [N, hidden_dim], float32, CPU
            "query_indices": all_query_indices,  # List[torch.Tensor], 每个元素是 [N], long, CPU
            "candidate_indices": all_candidate_indices_list,  # List[torch.Tensor], 每个元素是 [N], long, CPU，存储候选图像索引
            "is_positive": all_is_positive_list  # List[torch.Tensor], 每个元素是 [N], bool, CPU，标记哪些候选是正样本
        }
    else:
        # 训练缓存：包含 features, labels, query_indices
        jina_cache = {
            "features": all_features_list,  # List[torch.Tensor], 每个元素是 [N, hidden_dim], float32, CPU
            "labels": all_labels_list,       # List[torch.Tensor], 每个元素是 [N], float32, CPU
            "query_indices": all_query_indices  # List[torch.Tensor], 每个元素是 [N], long, CPU
        }
    
    # 验证缓存格式（确保与训练脚本兼容）
    try:
        # 测试是否可以像训练脚本一样使用
        if len(all_features_list) > 0:
            test_features = torch.cat(all_features_list[:2], dim=0) if len(all_features_list) >= 2 else torch.cat(all_features_list, dim=0)
            test_indices = torch.cat(all_query_indices[:2], dim=0) if len(all_query_indices) >= 2 else torch.cat(all_query_indices, dim=0)
            assert test_features.shape[0] == test_indices.shape[0], "Shape mismatch in test concatenation"
            if not is_eval_cache:
                test_labels = torch.cat(all_labels_list[:2], dim=0) if len(all_labels_list) >= 2 else torch.cat(all_labels_list, dim=0)
                assert test_features.shape[0] == test_labels.shape[0], "Shape mismatch in labels"
            print(f"[Jina Cache] Format validation passed: compatible with training/evaluation scripts")
    except Exception as e:
        print(f"[Jina Cache] WARNING: Format validation failed: {e}")
        raise
    
    # 保存缓存（与 jina_m0_lora_train.py 和 jina_m0_score_only_train_ddp.py 格式对齐）
    torch.save(jina_cache, jina_cache_path)
    
    # 保存进度信息
    progress_info = {
        "last_query_idx": end_idx - 1,
        "total_queries_processed": len(all_features_list),
        "total_queries": num_queries,
        "completed": (end_idx >= num_queries)
    }
    torch.save(progress_info, progress_path)
    
    print(f"[Jina Cache] Saved cache to: {jina_cache_path}")
    print(f"[Jina Cache] Saved progress to: {progress_path}")
    print(f"[Jina Cache] Cache format: List of tensors")
    print(f"[Jina Cache]   - features: {len(all_features_list)} tensors, total {total_samples} samples")
    if is_eval_cache:
        total_candidates = sum(c.shape[0] for c in all_candidate_indices_list)
        total_positives = sum(p.sum().item() for p in all_is_positive_list) if all_is_positive_list else 0
        print(f"[Jina Cache]   - candidate_indices: {len(all_candidate_indices_list)} tensors, total {total_candidates} candidate images")
        print(f"[Jina Cache]   - is_positive: {len(all_is_positive_list)} tensors, total {total_positives} positive samples")
    else:
        print(f"[Jina Cache]   - labels: {len(all_labels_list)} tensors, total {total_labels} samples")
    print(f"[Jina Cache]   - query_indices: {len(all_query_indices)} tensors, total {total_indices} samples")
    
    if end_idx < num_queries:
        print(f"[Jina Cache] Progress: {len(all_features_list)}/{num_queries} queries processed")
        print(f"[Jina Cache] To continue, run with --resume flag")
    else:
        print(f"[Jina Cache] All queries processed: {len(all_features_list)}/{num_queries}")
    
    if is_eval_cache:
        print(f"[Jina Cache] Ready for use with:")
        print(f"[Jina Cache]   - jina_m0_lora_train.py (evaluation phase, with --use_jina_cache)")
    else:
        print(f"[Jina Cache] Ready for use with:")
        print(f"[Jina Cache]   - jina_m0_lora_train.py (with --use_jina_cache and --train_lora=False)")
        print(f"[Jina Cache]   - jina_m0_score_only_train_ddp.py")
    
    # 清理临时文件
    import glob
    temp_files = glob.glob(jina_cache_path.replace(".pt", "_temp_*.pt"))
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception:
            pass
    
    print(f"[Precompute] Completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--image_path", type=str, default=CONFIG["image_path"])
    parser.add_argument("--save_dir", type=str, default=CONFIG["save_dir"])
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    
    parser.add_argument("--topk_base", type=int, default=50)
    parser.add_argument("--extra_negatives", type=int, default=0)
    parser.add_argument("--candidate_mode", type=str, default="topk_plus_pos")
    parser.add_argument("--model_name", type=str, default="jinaai/jina-reranker-m0")
    parser.add_argument("--label_mode", type=str, default="inv_rank")
    
    parser.add_argument("--image_max_side", type=int, default=256)
    parser.add_argument("--jina_micro_batch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_batch_size", type=int, default=1000, 
                        help="Save intermediate cache every N queries to reduce memory usage")
    parser.add_argument("--image_load_workers", type=int, default=4,
                        help="Number of workers for parallel image loading")
    parser.add_argument("--model_dtype", type=str, default='bf16',
                        choices=['bf16', 'bfloat16', 'fp16', 'float16'],
                        help="Model dtype: 'bf16' or 'fp16'. If not specified, auto-detect based on CUDA support")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Maximum number of queries to process (None = all queries). Useful for testing or incremental processing.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing cache. Will continue from where it left off.")
    parser.add_argument("--eval_cache", action="store_true",
                        help="Generate evaluation cache for test set instead of training cache")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    main(args)

