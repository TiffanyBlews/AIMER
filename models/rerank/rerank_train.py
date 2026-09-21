import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import clip
import wandb
import math

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataset import MSRVTT_Dataset
from rerank_model import RerankLinear, RerankMLP, RerankInteractionMLP

CONFIG = {
    "data_path": "./imgflip_data/msrvtt",
    "image_path": "./imgflip_data/images",
    "train_csv": "train_ids.csv",
    "train_json": "train_data.json",
    "train_emotion_json": "train_emotion.json",
    "batch_size": 128,
    "lr": 1e-5,
    "epochs": 20,
    "save_dir": "./checkpoints_rerank",
}

def compute_features(clip_model, dataloader, device):
    text_feats = []
    image_feats = []
    upvotes = []
    captions = []
    clip_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            query_ids = batch["query_ids"].to(device)
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

def evaluate_recalls(text_matrix, image_matrix, captions, model, topk_base=100, precomputed_topk=None, use_clip_score=False, model_type="linear"):
    from collections import defaultdict
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(captions):
        caption_to_indices[cap].append(idx)
    total = text_matrix.shape[0]
    recalls = {}
    precisions = {}
    need_sim = precomputed_topk is None
    if need_sim:
        sim = text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T
    for k in [1, 5, 10, 50, 100]:
        hits = 0
        acc = 0.0
        for q_idx in range(total):
            if need_sim:
                k_base = min(topk_base, image_matrix.shape[0])
                base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
            else:
                base_topk = np.array(precomputed_topk[q_idx], dtype=np.int64)
                k_base = base_topk.shape[0]
            k_eff = min(k, k_base)
            t = text_matrix[q_idx].cpu().numpy()
            ims = image_matrix[base_topk].cpu().numpy()
            t_rep = np.repeat(t[None, :], ims.shape[0], axis=0)
            if use_clip_score and model_type != "interaction":
                clip_scores = (t_rep * ims).sum(axis=1, keepdims=True)
                pair = np.concatenate([t_rep, ims, clip_scores], axis=1)
            else:
                pair = np.concatenate([t_rep, ims], axis=1)
            with torch.no_grad():
                scores = model(torch.from_numpy(pair).to(next(model.parameters()).device).float()).detach().cpu().numpy()
            order = np.argsort(-scores)
            topk_used = base_topk[order][:k_eff]
            gt = caption_to_indices[captions[q_idx]]
            if any(i in gt for i in topk_used):
                hits += 1
            correct = sum(1 for i in topk_used if i in gt)
            acc += correct / float(k_eff)
        recalls[f"R@{k}"] = hits / total * 100.0
        precisions[f"P@{k}"] = acc / total * 100.0
    return {**recalls, **precisions}

def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    s = scores.unsqueeze(0)
    y = labels.unsqueeze(0)
    diff = s.transpose(0, 1) - s
    rel = (y.transpose(0, 1) - y) > 0
    pos = torch.nonzero(rel, as_tuple=False)
    if pos.size(0) == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    d = diff[pos[:, 0], pos[:, 1]]
    loss = torch.log1p(torch.exp(-d)).mean()
    return loss

def lambdarank_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n = scores.size(0)
    if n == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    gains = torch.pow(2.0, labels) - 1.0
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
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    gi = g[:, pos[:, 0]].squeeze(0)
    gj = g[:, pos[:, 1]].squeeze(0)
    di = d[:, pos[:, 0]].squeeze(0)
    dj = d[:, pos[:, 1]].squeeze(0)
    delta_ndcg = torch.abs((gi - gj) * (di - dj)) / idcg
    sdiff = sd[pos[:, 0], pos[:, 1]]
    loss = (delta_ndcg * torch.log1p(torch.exp(-sdiff))).mean()
    return loss
def train(args=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()

    bert_tokenizer = None

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
    else:
        topk_base = 100
        extra_negs = 0
        candidate_mode = "topk_plus_pos"
        label_mode = "softlog"

    ckpt_path = None
    if args is not None and getattr(args, "clip_checkpoint", ""):
        ckpt_path = args.clip_checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        try:
            clip_model.load_state_dict(state)
        except Exception:
            clip_model.load_state_dict(state, strict=False)
    if args is not None:
        label_mode = getattr(args, "label_mode", "softlog")

    train_csv_path = os.path.join(CONFIG["data_path"], CONFIG["train_csv"])
    train_json_path = os.path.join(CONFIG["data_path"], CONFIG["train_json"])
    train_emotion_json_path = os.path.join(CONFIG["data_path"], CONFIG["train_emotion_json"])

    train_dataset = MSRVTT_Dataset(
        csv_path=train_csv_path,
        json_path=train_json_path,
        features_path=CONFIG["image_path"],
        emotion_json_path=train_emotion_json_path,
        clip_preprocess=preprocess,
        bert_tokenizer=bert_tokenizer,
        max_words=32,
        is_train=True,
        load_image=True,
    )
    loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    cache_train_path = os.path.join(CONFIG["save_dir"], "train_cache.pt")
    if os.path.exists(cache_train_path):
        print(f"load {cache_train_path}")
        obj = torch.load(cache_train_path, map_location="cpu", weights_only=False)
        text_matrix = obj["text"].cpu()
        image_matrix = obj["image"].cpu()
        upvotes = obj.get("upvotes", None)
        captions = obj["captions"]
        topk_cached = obj.get("topk_indices", None)
        topk_base_cached = obj.get("topk_base", None)
        sim = None
    else:
        print(f"warning: no cache")
        text_matrix, image_matrix, upvotes, captions = compute_features(clip_model, loader, device)
        topk_cached = None
        topk_base_cached = None
        sim = (text_matrix.cpu().numpy() @ image_matrix.cpu().numpy().T)

    from collections import defaultdict
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(captions):
        caption_to_indices[cap].append(idx)

    t_dim = text_matrix.shape[1]
    i_dim = image_matrix.shape[1]
    use_clip_score = bool(getattr(args, "use_clip_score", False)) if args is not None else False
    dim = t_dim + i_dim + (1 if use_clip_score else 0)
    model_type = getattr(args, 'model_type', 'linear') if args is not None else 'linear'
    loss_type = getattr(args, 'loss_type', 'mse') if args is not None else 'mse'
    if model_type == 'mlp':
        hidden_dim = int(getattr(args, 'mlp_hidden_dim', 512))
        mlp_layers = int(getattr(args, 'mlp_layers', 1))
        mlp_dropout = float(getattr(args, 'mlp_dropout', 0.1))
        model = RerankMLP(dim, hidden_dim=hidden_dim, dropout=mlp_dropout, num_layers=mlp_layers).to(device)
    elif model_type == 'interaction':
        model = RerankInteractionMLP(t_dim, i_dim).to(device)
    else:
        model = RerankLinear(dim).to(device)
    suffix = f"_{model_type}_{loss_type}_{label_mode}" + ("_clipsim" if use_clip_score and model_type != "interaction" else "")
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"]) 
    sch_type = getattr(args, "scheduler", "cosine_warmup") if args is not None else "cosine_warmup"
    scheduler = None
    global_step = 0
    total_queries_prefetch = text_matrix.shape[0]
    total_steps = CONFIG["epochs"] * max(1, total_queries_prefetch)
    if sch_type == "cosine_warmup":
        warmup = int(getattr(args, "warmup_steps", max(1, int(0.05 * total_steps)))) if args is not None else max(1, int(0.05 * total_steps))
        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif sch_type == "cosine":
        tmax = CONFIG["epochs"]
        min_lr = float(getattr(args, "min_lr", 0.0)) if args is not None else 0.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=min_lr)
    elif sch_type == "step":
        step_size = int(getattr(args, "step_size", 1)) if args is not None else 1
        gamma = float(getattr(args, "gamma", 0.5)) if args is not None else 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    elif sch_type == "plateau":
        gamma = float(getattr(args, "gamma", 0.5)) if args is not None else 0.5
        patience = int(getattr(args, "patience", 1)) if args is not None else 1
        min_lr = float(getattr(args, "min_lr", 0.0)) if args is not None else 0.0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=gamma, patience=patience, min_lr=min_lr)
    best_r1 = -1.0
    best_r5 = -1.0
    best_r10 = -1.0
    keep_all_best = bool(getattr(args, "keep_all_best", False))

    if args and args.wandb_project:
        wandb.init(project=args.wandb_project, entity=getattr(args, 'wandb_entity', None), config=vars(args), mode=getattr(args, 'wandb_mode', 'disabled'))
        wandb.watch(model, log=None)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        count = 0
        total_queries = text_matrix.shape[0]
        num_images = image_matrix.shape[0]
        for q_idx in tqdm(range(total_queries), desc=f"rerank epoch {epoch+1}"):
            if topk_cached is not None and (topk_base_cached == topk_base):
                base_topk = np.array(topk_cached[q_idx], dtype=np.int64)
                k_base = base_topk.shape[0]
            else:
                if sim is None:
                    k_base = min(topk_base, num_images)
                    # fallback: compute per-query similarities on the fly
                    t = text_matrix[q_idx].cpu().numpy()
                    sims_q = (t[None, :] @ image_matrix.cpu().numpy().T).squeeze(0)
                    base_topk = np.argpartition(-sims_q, range(k_base))[:k_base]
                else:
                    k_base = min(topk_base, sim.shape[1])
                    base_topk = np.argpartition(-sim[q_idx], range(k_base))[:k_base]
            pos_all = np.array(caption_to_indices[captions[q_idx]], dtype=np.int64)
            if candidate_mode == "topk":
                core_set = base_topk
            elif candidate_mode == "pos_all":
                core_set = pos_all
            else:  # topk_plus_pos
                core_set = np.unique(np.concatenate([base_topk, pos_all]))

            all_indices = np.arange(num_images)
            mask = np.ones_like(all_indices, dtype=bool)
            mask[core_set] = False
            candidates_neg = all_indices[mask]
            if extra_negs > 0 and candidates_neg.size > 0:
                sel = np.random.choice(candidates_neg, size=min(extra_negs, candidates_neg.size), replace=False)
                train_indices = np.concatenate([core_set, sel])
            else:
                train_indices = core_set
            gt = set(caption_to_indices[captions[q_idx]])
            pos_list = [(i, upvotes[i]) for i in train_indices if i in gt and upvotes[i] > 0]
            labels = np.zeros(train_indices.shape[0], dtype=np.float32)
            if len(pos_list) > 0:
                if label_mode == "inv_rank":
                    sorted_pos = sorted(pos_list, key=lambda x: x[1], reverse=True)
                    npos = len(sorted_pos)
                    for rank, (i, _) in enumerate(sorted_pos, start=1):
                        w = 1.0 / float(rank)
                        idx_in_arr = np.where(train_indices == i)[0][0]
                        labels[idx_in_arr] = float(w)
                elif label_mode == "raw_log":
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
            t = text_matrix[q_idx].cpu().numpy()
            ims = image_matrix[train_indices].cpu().numpy()
            t_rep = np.repeat(t[None, :], ims.shape[0], axis=0)
            if use_clip_score and model_type != "interaction":
                clip_scores = (t_rep * ims).sum(axis=1, keepdims=True)
                pair = np.concatenate([t_rep, ims, clip_scores], axis=1)
            else:
                pair = np.concatenate([t_rep, ims], axis=1)
            pair_t = torch.from_numpy(pair).to(device).float()
            target_t = torch.from_numpy(labels).to(device).float()
            preds = model(pair_t)
            if loss_type == 'ranknet':
                loss = ranknet_loss(preds, target_t)
            elif loss_type == 'lambdarank':
                loss = lambdarank_loss(preds, target_t)
            else:
                loss = F.mse_loss(preds, target_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler is not None and sch_type == "cosine_warmup":
                scheduler.step()
                global_step += 1
            total_loss += float(loss.item())
            count += 1
        avg_loss = total_loss / max(count, 1)
        if wandb.run is not None:
            wandb.log({"train_loss": avg_loss, "epoch": epoch+1, "lr": opt.param_groups[0]["lr"]})
        if scheduler is not None and sch_type in ("cosine", "step"):
            scheduler.step()
        if scheduler is not None and sch_type == "plateau":
            scheduler.step(avg_loss)
        test_json_path = os.path.join(CONFIG["data_path"], "test_data.json")
        test_emotion_json_path = os.path.join(CONFIG["data_path"], "test_emotion.json")
        test_dataset = MSRVTT_Dataset(
            csv_path=test_json_path,
            json_path=test_json_path,
            features_path=CONFIG["image_path"],
            emotion_json_path=test_emotion_json_path,
            clip_preprocess=preprocess,
            bert_tokenizer=bert_tokenizer,
            max_words=32,
            is_train=False,
            load_image=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
        cache_test_path = os.path.join(CONFIG["save_dir"], "test_cache.pt")
        if os.path.exists(cache_test_path):
            obj_t = torch.load(cache_test_path, map_location="cpu", weights_only=False)
            t_text = obj_t["text"].cpu()
            t_image = obj_t["image"].cpu()
            t_caps = obj_t["captions"]
            topk_test = obj_t.get("topk_indices", None)
        else:
            t_text, t_image, _, t_caps = compute_features(clip_model, test_loader, device)
            topk_test = None
        eval_metrics = evaluate_recalls(t_text, t_image, t_caps, model, topk_base=topk_base, precomputed_topk=topk_test, use_clip_score=use_clip_score, model_type=model_type)
        print({
            "R@1": eval_metrics["R@1"], "P@1": eval_metrics["P@1"],
            "R@5": eval_metrics["R@5"], "P@5": eval_metrics["P@5"],
            "R@10": eval_metrics["R@10"], "P@10": eval_metrics["P@10"],
            "R@50": eval_metrics["R@50"], "P@50": eval_metrics["P@50"],
            "R@100": eval_metrics["R@100"], "P@100": eval_metrics["P@100"],
            "epoch": epoch+1
        })
        if wandb.run is not None:
            wandb.log({
                "R@1": eval_metrics["R@1"], "P@1": eval_metrics["P@1"],
                "R@5": eval_metrics["R@5"], "P@5": eval_metrics["P@5"],
                "R@10": eval_metrics["R@10"], "P@10": eval_metrics["P@10"],
                "R@50": eval_metrics["R@50"], "P@50": eval_metrics["P@50"],
                "R@100": eval_metrics["R@100"], "P@100": eval_metrics["P@100"],
                "epoch": epoch+1
            })
        os.makedirs(CONFIG["save_dir"], exist_ok=True)
        if eval_metrics["R@1"] > best_r1:
            best_r1 = eval_metrics["R@1"]
            path_r1 = os.path.join(CONFIG["save_dir"], f"rerank_best_R@1{suffix}.pt")
            torch.save(model.state_dict(), path_r1)
            if keep_all_best:
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"rerank_best_R@1_epoch_{epoch+1}{suffix}.pt"))
            if wandb.run is not None:
                wandb.log({"best_R@1": best_r1})
        if eval_metrics["R@5"] > best_r5:
            best_r5 = eval_metrics["R@5"]
            path_r5 = os.path.join(CONFIG["save_dir"], f"rerank_best_R@5{suffix}.pt")
            torch.save(model.state_dict(), path_r5)
            if keep_all_best:
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"rerank_best_R@5_epoch_{epoch+1}{suffix}.pt"))
            if wandb.run is not None:
                wandb.log({"best_R@5": best_r5})
        if eval_metrics["R@10"] > best_r10:
            best_r10 = eval_metrics["R@10"]
            path_r10 = os.path.join(CONFIG["save_dir"], f"rerank_best_R@10{suffix}.pt")
            torch.save(model.state_dict(), path_r10)
            if keep_all_best:
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"rerank_best_R@10_epoch_{epoch+1}{suffix}.pt"))
            if wandb.run is not None:
                wandb.log({"best_R@10": best_r10})
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"rerank_epoch_{epoch+1}{suffix}.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--image_path", type=str, default=CONFIG["image_path"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--save_dir", type=str, default=CONFIG["save_dir"])
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--topk_base", type=int, default=100)
    parser.add_argument("--extra_negatives", type=int, default=0)
    parser.add_argument("--clip_checkpoint", type=str, default="")
    parser.add_argument("--model_type", type=str, default="linear")
    parser.add_argument("--loss_type", type=str, default="lambdarank")
    parser.add_argument("--label_mode", type=str, default="inv_rank")
    parser.add_argument("--candidate_mode", type=str, default="topk_plus_pos")
    parser.add_argument("--keep_all_best", action="store_true")
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--mlp_dropout", type=float, default=0.5)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--use_clip_score", action="store_true")
    args = parser.parse_args()
    train(args)
