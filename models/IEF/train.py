# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
import argparse
from collections import defaultdict
import wandb
# import yaml
from datetime import datetime
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from model import CLIP4Meme
from dataset import MSRVTT_Dataset

CONFIG = {
    "data_path": "./imgflip_data/msrvtt",
    "image_path": "./imgflip_data/images",
    "train_csv": "train_ids.csv",
    "train_json": "train_data.json",
    "train_emotion_json": "train_emotion.json",
    "test_json": "test_data.json",
    "test_emotion_json": "test_emotion.json",
    
    "batch_size": 64, "epochs": 10, "lr": 1e-4, "coef_lr": 1e-1,
    "weight_decay": 0.01, "min_lr": 0,
    
    "pretrained_clip_name": "ViT-B/32", "use_emotion_fusion": True, "seed": 42,

    "save_dir": "./checkpoints_imgflip",
    "checkpoint_path": None,
    
    # wandb配置
    "wandb_project": "clip4meme",
    "wandb_entity": None,  # 设置为你的wandb用户名或团队名
    "wandb_tags": [],
    "wandb_notes": "",
    "log_gradients": False,  # 是否记录梯度
    "log_model": False,  # 是否上传模型到wandb
}

def setup_wandb(args):
    """设置wandb实验"""
    # 设置wandb国内镜像
    # os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
    
    # 构建实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.train.upper()}_{args.pretrained_clip_name}_{timestamp}"
    
    # 构建标签
    tags = args.wandb_tags + [args.train.upper(), args.pretrained_clip_name]
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=experiment_name,
        tags=tags,
        notes=args.wandb_notes,
        config=vars(args),
        mode="online" if args.wandb_project else "disabled"
    )
    
    print(f"Wandb initialized: {wandb.run.name}")
    return wandb.run

def log_hyperparameters(args, model):
    """记录超参数和模型信息"""
    # 记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.config.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_architecture": model.__class__.__name__,
        "clip_model": args.pretrained_clip_name,
    })
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def log_metrics(epoch, train_loss, eval_results, learning_rate, branch_name, total_batches):
    """记录训练和评估指标"""
    # 使用全局step来确保单调递增，基于实际batch数量
    global_step = epoch * (total_batches + 1)  # +1为epoch指标预留空间
    
    metrics = {
        f"{branch_name.lower()}/epoch": epoch,
        f"{branch_name.lower()}/train_loss": train_loss,
        f"{branch_name.lower()}/learning_rate": learning_rate,
    }
    
    # 添加评估指标
    for k, v in eval_results.items():
        metrics[f"{branch_name.lower()}/{k.lower()}"] = v
    
    # 使用全局step记录指标
    wandb.log(metrics, step=global_step)
    
    # 打印指标
    print(f"\n--- Epoch {epoch} Metrics ---")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Learning Rate: {learning_rate:.6f}")
    for k, v in eval_results.items():
        print(f"{k}: {v:.2f}%")

def log_predictions(epoch, model, test_dataset, test_dataloader, device, branch_name, total_batches, num_samples=5):
    """记录预测样本到wandb"""
    if num_samples <= 0:
        return
    
    # 使用全局step来确保单调递增，基于实际batch数量
    global_step = epoch * (total_batches + 1)
    
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= num_samples:
                break
                
            query_ids = batch['query_ids'].to(device)
            candidate_ids = batch['candidate_ids'].to(device)
            image = batch['image'].to(device)
            emotion_ids = batch['emotion_input_ids'].to(device)
            emotion_mask = batch['emotion_attention_mask'].to(device)
            
            # 获取预测
            if branch_name.lower() == 'qi':
                query_feat, fused_feat = model(
                    query_ids=query_ids,
                    candidate_ids=candidate_ids,
                    image=image,
                    emotion_ids=emotion_ids,
                    emotion_mask=emotion_mask,
                    branch='qi',
                    return_features=True
                )
            else:
                query_feat, candidate_feat = model(
                    query_ids=query_ids,
                    candidate_ids=candidate_ids,
                    image=image,
                    emotion_ids=emotion_ids,
                    emotion_mask=emotion_mask,
                    branch='qc',
                    return_features=True
                )
            
            # 计算相似度
            sim_matrix = torch.matmul(F.normalize(query_feat, dim=-1), 
                                    F.normalize(fused_feat if branch_name.lower() == 'qi' else candidate_feat, dim=-1).t())
            
            # 获取top-3预测
            top3_indices = torch.topk(sim_matrix[0], k=3).indices
            top3_scores = torch.topk(sim_matrix[0], k=3).values
            
            # 构建样本记录
            sample = {
                "epoch": epoch,
                "branch": branch_name,
                "query_text": test_dataset.data[i]['query'],
                "ground_truth": test_dataset.data[i]['candidate_texts'][0] if test_dataset.data[i]['candidate_texts'] else "",
                "top1_prediction": test_dataset.data[top3_indices[0]]['candidate_texts'][0] if test_dataset.data[top3_indices[0]]['candidate_texts'] else "",
                "top1_score": float(top3_scores[0]),
                "top2_prediction": test_dataset.data[top3_indices[1]]['candidate_texts'][0] if test_dataset.data[top3_indices[1]]['candidate_texts'] else "",
                "top2_score": float(top3_scores[1]),
                "top3_prediction": test_dataset.data[top3_indices[2]]['candidate_texts'][0] if test_dataset.data[top3_indices[2]]['candidate_texts'] else "",
                "top3_score": float(top3_scores[2]),
            }
            samples.append(sample)
    
    # 记录到wandb
    if samples:
        # 创建wandb表格，指定列名
        columns = ["epoch", "branch", "query_text", "ground_truth", "top1_prediction", "top1_score", 
                  "top2_prediction", "top2_score", "top3_prediction", "top3_score"]
        table = wandb.Table(columns=columns, data=[[s[col] for col in columns] for s in samples])
        wandb.log({f"{branch_name.lower()}/predictions": table}, step=global_step)

def symmetric_contrastive_loss(sim_matrix):
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2
    return loss

def multi_positive_contrastive_loss(sim_matrix, captions):
    # captions: list[str] 长度为B，同文本视为正样本
    B = sim_matrix.size(0)
    device = sim_matrix.device
    # 构造正样本掩码 [B,B]
    # 注意：自身也算正样本
    pos_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if captions[i] == captions[j]:
                pos_mask[i, j] = True
    # 对每一行做 log-softmax 后，最大化正样本的概率和
    log_prob = F.log_softmax(sim_matrix, dim=1)
    # 为避免一行没有正样本（极端情况），至少包含自身
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)
    pos_log_prob_sum = (log_prob.masked_fill(~pos_mask, 0.0)).sum(dim=1)
    loss_row = -(pos_log_prob_sum / pos_counts)
    # 对称项：列方向同理
    log_prob_T = F.log_softmax(sim_matrix.t(), dim=1)
    pos_counts_T = pos_mask.t().sum(dim=1).clamp(min=1)
    pos_log_prob_sum_T = (log_prob_T.masked_fill(~pos_mask.t(), 0.0)).sum(dim=1)
    loss_col = -(pos_log_prob_sum_T / pos_counts_T)
    return (loss_row.mean() + loss_col.mean()) / 2

def eval_epoch(model, test_dataset, test_dataloader, device, branch, print_n_samples=5):
    model.eval()
    print(f"\n--- Evaluating {branch} branch ---")
    
    # 1. 收集元数据
    all_video_ids_meta = [item['video_id'] for item in test_dataset.data]
    all_captions_meta = [item['query'] for item in test_dataset.data]
    all_titles_meta = [item['candidate_texts'][0] if item['candidate_texts'] else "" for item in test_dataset.data]
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    
    # 2. 缓存所有查询和目标的特征
    all_q_feats, all_target_feats = [], []
    with torch.no_grad():
        # 步骤 2.1: 高效地一次性缓存所有查询文本的特征
        print("Caching all query features...")
        for batch in tqdm(test_dataloader, desc="Caching Queries"):
            query_ids = batch['query_ids'].to(device)
            q_feat = model.encode_text_pooled(query_ids)
            all_q_feats.append(q_feat.cpu())
        all_q_feats = torch.cat(all_q_feats, dim=0)

        # 步骤 2.2: 根据不同分支计算目标特征
        print(f"Calculating all target features for {branch} branch...")
        if branch == 'QI': # 评估 QI 分支 (使用return_features避免重复计算)
            # 使用模型的return_features功能，避免重复计算
            for batch in tqdm(test_dataloader, desc="Calculating Fused QI Features"):
                query_feat, fused_feat = model(
                    query_ids=batch['query_ids'].to(device),
                    candidate_ids=batch['candidate_ids'].to(device),
                    image=batch['image'].to(device),
                    emotion_ids=batch['emotion_input_ids'].to(device),
                    emotion_mask=batch['emotion_attention_mask'].to(device),
                    branch='qi',
                    return_features=True
                )
                # 直接使用模型返回的融合特征，无需重复计算
                all_target_feats.append(fused_feat.cpu())

        elif branch == 'QC': # 评估 QC 分支 (使用return_features保持一致性)
            for batch in tqdm(test_dataloader, desc="Caching QC Features"):
                query_feat, candidate_feat = model(
                    query_ids=batch['query_ids'].to(device),
                    candidate_ids=batch['candidate_ids'].to(device),
                    image=batch['image'].to(device),  # QC不需要图片，但为了保持接口一致
                    emotion_ids=batch['emotion_input_ids'].to(device),
                    emotion_mask=batch['emotion_attention_mask'].to(device),
                    branch='qc',
                    return_features=True
                )
                # 直接使用模型返回的特征，保持与QI分支的一致性
                all_target_feats.append(candidate_feat.cpu())
        
        all_target_feats = torch.cat(all_target_feats, dim=0)

    # 3. 计算总相似度矩阵并评估
    all_q_feats = F.normalize(all_q_feats, dim=-1)
    all_target_feats = F.normalize(all_target_feats, dim=-1)
    sim_matrix = torch.matmul(all_q_feats, all_target_feats.t())
    
    # 4. R@k 计算和详细打印逻辑
    print("\n--- Computing R@k & Detailed Predictions ---")
    num_queries = sim_matrix.shape[0]
    recalls = {}
    precisions = {}
    for k in [1, 5, 10]:
        hits = 0
        acc = 0.0
        for q_idx in range(num_queries):
            top_preds = torch.topk(sim_matrix[q_idx], k=k).indices.tolist()
            gt_indices = caption_to_indices[all_captions_meta[q_idx]]
            if any(p in gt_indices for p in top_preds):
                hits += 1
            correct = sum(1 for p in top_preds if p in gt_indices)
            acc += correct / float(k)
            if q_idx < print_n_samples and k == 10:
                print(f"\n{'='*22} Sample {q_idx+1}/{print_n_samples} {'='*22}")
                print(f"  [Query]      VID: {all_video_ids_meta[q_idx]}, Caption: {all_captions_meta[q_idx]}")
                print(f"  [GT Indices] {gt_indices}")
                print("  [Top-5 Predictions]")
                top5 = torch.topk(sim_matrix[q_idx], k=5)
                for rank, pred_idx in enumerate(top5.indices.tolist()):
                    mark, score = "✔️" if pred_idx in gt_indices else "❌", top5.values[rank]
                    print(f"    {rank+1}. {mark} score={score:.4f} | VID={all_video_ids_meta[pred_idx]} | Cap: {all_captions_meta[pred_idx]}")
        recalls[f'R@{k}'] = (hits / num_queries) * 100
        precisions[f'P@{k}'] = (acc / num_queries) * 100
    print("\n--- Final Evaluation Results ---")
    for k in [1, 5, 10]:
        r = recalls[f'R@{k}']
        p = precisions[f'P@{k}']
        print(f"  R@{k}: {r:.2f}% | P@{k}: {p:.2f}%")
    return {**recalls, **precisions}

def main(args):
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    branch_name = args.train.upper()
    print(f"--- Starting Training for {branch_name} branch ---")

    # 1. 初始化wandb
    wandb_run = setup_wandb(args)
    
    # 2. 初始化模型和数据
    model = CLIP4Meme(pretrained_clip_name=args.pretrained_clip_name)
    model.to(device)
    
    # 记录模型超参数
    log_hyperparameters(args, model)
    
    train_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(args.data_path, args.train_csv),
        json_path=os.path.join(args.data_path, args.train_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.train_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=True,
        load_image=(args.train.lower() == 'qi') # QI分支加载图片，QC分支不加载
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(args.data_path, args.test_json),
        json_path=os.path.join(args.data_path, args.test_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.test_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=False,
        load_image=True # 评估时总是加载图片，以备QI分支评估
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 优化器 
    no_decay = ["bias", "LayerNorm.weight"]
    
    # 冻结BERT模型参数
    for param in model.bert_model.parameters():
        param.requires_grad = False
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': max(5e-6, args.lr * args.coef_lr)},
        {'params': [p for n, p in model.clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': max(5e-6, args.lr * args.coef_lr)},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('clip.', 'bert_model.')) and not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': max(5e-6, args.lr)},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('clip.', 'bert_model.')) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # 3. 加载检查点 (QI和QC分支共享相同的检查点)
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"--- Loading checkpoint for {branch_name} branch training ---")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("--- No checkpoint provided, starting from scratch ---")

    # 4. 训练循环
    best_r1 = 0.0
    epoch_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [{branch_name}]")
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # 调用模型时传入正确的分支名称
            sim_matrix = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                branch=branch_name.lower()
            )
            
            captions = batch['query_text']
            loss = multi_positive_contrastive_loss(sim_matrix, captions)
            loss.backward()
            
            # 记录梯度（如果启用）
            if args.log_gradients:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())}, step=epoch * len(train_dataloader) + num_batches)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # 记录每个batch的loss
            batch_step = epoch * (len(train_dataloader) + 1) + num_batches  # 基于实际batch数量
            wandb.log({f"{branch_name.lower()}/batch_loss": loss.item()}, step=batch_step)
        
        # 计算平均epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 评估和保存
        eval_results = eval_epoch(model, test_dataset, test_dataloader, device, branch_name)
        
        # 记录指标到wandb
        log_metrics(epoch + 1, avg_epoch_loss, eval_results, current_lr, branch_name, len(train_dataloader))
        
        # 记录预测样本（每5个epoch记录一次）
        if (epoch + 1) % 5 == 0:
            log_predictions(epoch + 1, model, test_dataset, test_dataloader, device, branch_name, len(train_dataloader), num_samples=3)
        
        # 保存最佳模型
        if eval_results.get('R@1', 0) > best_r1:
            best_r1 = eval_results.get('R@1', 0)
            
            # 构建新的检查点文件名
            if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
                # 如果有检查点路径，在原文件名后添加分支名
                base_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
                if base_name == "checkpoint":  # 如果是默认名称，直接使用分支名
                    new_checkpoint_name = f"{branch_name.lower()}.pth"
                else:  # 否则在原文件名后添加当前分支名
                    new_checkpoint_name = f"{base_name}_{branch_name.lower()}.pth"
            else:
                # 如果没有检查点路径，直接使用分支名
                new_checkpoint_name = f"{branch_name.lower()}.pth"
            
            save_path = os.path.join(args.save_dir, new_checkpoint_name)
            print(f"*** New best R@1: {best_r1:.2f}%. Saving model to {save_path} ***")
            
            # 保存模型检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_r1': best_r1,
                'config': vars(args)
            }
            torch.save(checkpoint, save_path)
            
            # 上传模型到wandb（如果启用）
            if args.log_model:
                wandb.save(save_path)
                print(f"Model uploaded to wandb: {save_path}")
    
    # 训练结束后记录最终结果
    print(f"\n--- Training Completed ---")
    print(f"Best R@1: {best_r1:.2f}%")
    
    # 记录训练完成
    final_step = args.epochs * (len(train_dataloader) + 1)
    wandb.log({f"{branch_name.lower()}/best_r1": best_r1}, step=final_step)
    wandb.log({f"{branch_name.lower()}/training_completed": True}, step=final_step)
    
    # 关闭wandb
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Meme-Retrieval model based on QI/QC branches.")
    parser.add_argument('--train', type=str, required=True, choices=['qi', 'qc'], help="Train QI or QC branch")
    
    for key, value in CONFIG.items():
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
        
    parsed_args = parser.parse_args()
    main(parsed_args)
