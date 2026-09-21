# simple_clip_train.py
# 简化的CLIP训练脚本，直接使用现有配置

import torch
from transformers import CLIPProcessor, CLIPModel, BertModel
from torch.utils.data import DataLoader
import os
import clip
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import wandb

# 导入现有的数据集类
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataset import MSRVTT_Dataset

# 简化的CLIP训练配置，参考run.py的超参数
CONFIG = {
    # 数据路径
    # "data_path": "./data/input_file",
    # "image_path": "./data/image",
    "data_path": "./imgflip_data/msrvtt",
    "image_path": "./imgflip_data/images",
    "train_csv": "train_ids.csv",
    "train_json": "train_data.json",
    "train_emotion_json": "train_emotion.json",
    "test_json": "test_data.json",
    "test_emotion_json": "test_emotion.json",
    
    # 训练超参数（参考run.py）
    "batch_size": 128,
    "epochs": 5,
    "lr": 5e-6,
    "max_grad_norm": 1.0,
    
    # 模型配置
    "pretrained_clip_name": "ViT-B/32",
    "clip_model_path": "parts/CLIPModel",
    
    # 保存配置
    "save_dir": "./checkpoints_clip",
    "model_save_name": "clip_imgflip.pt",
    "use_image_emotion_fusion": False,
    "use_text_emotion_fusion": False,
    "use_multi_positive_loss": True,
    "wandb_project": "clip4meme",
    "wandb_entity": "",
    "log_model": False,
}

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

def compute_simple_metrics(similarity_matrix, all_captions_meta):
    """计算简单的检索指标，基于相同query的正样本"""
    total_samples = similarity_matrix.shape[0]
    
    # 构建caption到索引的映射
    from collections import defaultdict
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    
    r1_correct = 0
    for q_idx in range(total_samples):
        top1_idx = np.argmax(similarity_matrix[q_idx])
        gt_indices = caption_to_indices[all_captions_meta[q_idx]]
        if top1_idx in gt_indices:
            r1_correct += 1
    r1 = (r1_correct / total_samples) * 100
    
    r5_correct = 0
    for q_idx in range(total_samples):
        top5_indices = np.argsort(similarity_matrix[q_idx])[-5:]
        gt_indices = caption_to_indices[all_captions_meta[q_idx]]
        if any(p in gt_indices for p in top5_indices):
            r5_correct += 1
    r5 = (r5_correct / total_samples) * 100
    
    p1_acc = 0.0
    p5_acc = 0.0
    for q_idx in range(total_samples):
        top1_indices = np.argsort(similarity_matrix[q_idx])[-1:]
        gt_indices = caption_to_indices[all_captions_meta[q_idx]]
        correct1 = sum(1 for p in top1_indices if p in gt_indices)
        p1_acc += correct1 / 1.0
        top5_indices = np.argsort(similarity_matrix[q_idx])[-5:]
        correct5 = sum(1 for p in top5_indices if p in gt_indices)
        p5_acc += correct5 / 5.0
    p1 = (p1_acc / total_samples) * 100
    p5 = (p5_acc / total_samples) * 100
    
    return {"R@1": r1, "R@5": r5, "P@1": p1, "P@5": p5}

def multi_positive_contrastive_loss(sim_matrix, captions):
    B = sim_matrix.size(0)
    device = sim_matrix.device
    pos_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if captions[i] == captions[j]:
                pos_mask[i, j] = True
    log_prob = F.log_softmax(sim_matrix, dim=1)
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)
    pos_log_prob_sum = (log_prob.masked_fill(~pos_mask, 0.0)).sum(dim=1)
    loss_row = -(pos_log_prob_sum / pos_counts)
    log_prob_T = F.log_softmax(sim_matrix.t(), dim=1)
    pos_counts_T = pos_mask.t().sum(dim=1).clamp(min=1)
    pos_log_prob_sum_T = (log_prob_T.masked_fill(~pos_mask.t(), 0.0)).sum(dim=1)
    loss_col = -(pos_log_prob_sum_T / pos_counts_T)
    return (loss_row.mean() + loss_col.mean()) / 2

def evaluate_simple(model, dataloader, device, dataset, clip_model, emotion_adapter=None):
    """简单评估"""
    model.eval()
    text_features_list = []
    image_features_list = []
    
    print("开始评估...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            # 从MSRVTT_Dataset获取数据
            images = batch["image"].to(device)
            query_ids = batch["query_ids"].to(device)  # 直接使用CLIP tokenized的query_ids
            
            # 获取特征（使用OpenAI CLIP，保证预处理/分词一致）
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(query_ids)
            if CONFIG["use_image_emotion_fusion"] or CONFIG["use_text_emotion_fusion"]:
                emotion_ids = batch["emotion_input_ids"].to(device)
                emotion_mask = batch["emotion_attention_mask"].to(device)
                emo_proj = emotion_adapter.emotion_proj(emotion_ids, emotion_mask)
                if CONFIG["use_image_emotion_fusion"]:
                    image_features = emotion_adapter.fuse(image_features, emo_proj)
                if CONFIG["use_text_emotion_fusion"]:
                    text_features = emotion_adapter.fuse(text_features, emo_proj)
            image_norm = image_features.norm(dim=1, keepdim=True).clamp(min=1e-6)
            text_norm = text_features.norm(dim=1, keepdim=True).clamp(min=1e-6)
            image_features = image_features / image_norm
            text_features = text_features / text_norm
            
            text_features_list.append(text_features.cpu().numpy())
            image_features_list.append(image_features.cpu().numpy())
    
    # 合并特征
    text_features_matrix = np.concatenate(text_features_list, axis=0)
    image_features_matrix = np.concatenate(image_features_list, axis=0)
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(text_features_matrix, image_features_matrix.T)
    
    # 获取所有caption信息
    all_captions_meta = [item['query'] for item in dataset.data]
    
    # 计算指标
    metrics = compute_simple_metrics(similarity_matrix, all_captions_meta)
    
    return metrics, similarity_matrix

def train_simple_clip(train_loader, test_loader, model, device, test_dataset, clip_model, emotion_adapter=None, use_multi_positive_loss=False, wandb_enabled=False):
    """简单训练CLIP模型"""
    if emotion_adapter is not None:
        optimizer = torch.optim.Adam(list(clip_model.parameters()) + list(emotion_adapter.parameters()), lr=CONFIG["lr"])
    else:
        optimizer = torch.optim.Adam(clip_model.parameters(), lr=CONFIG["lr"])
    best_r1 = 0
    
    print(f"开始训练，共{CONFIG['epochs']}个epoch...")
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")):
            # 从MSRVTT_Dataset获取数据
            images = batch["image"].to(device)
            query_ids = batch["query_ids"].to(device)  # 直接使用CLIP tokenized的query_ids
            
            optimizer.zero_grad()
            
            # 获取特征（使用OpenAI CLIP，与数据集tokenizer/预处理保持一致）
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(query_ids)
            if CONFIG["use_image_emotion_fusion"] or CONFIG["use_text_emotion_fusion"]:
                emotion_ids = batch["emotion_input_ids"].to(device)
                emotion_mask = batch["emotion_attention_mask"].to(device)
                emo_proj = emotion_adapter.emotion_proj(emotion_ids, emotion_mask)
                if CONFIG["use_image_emotion_fusion"]:
                    image_features = emotion_adapter.fuse(image_features, emo_proj)
                if CONFIG["use_text_emotion_fusion"]:
                    text_features = emotion_adapter.fuse(text_features, emo_proj)
            
            # 特征归一化 + 对比学习InfoNCE损失
            image_norm = image_features.norm(dim=1, keepdim=True).clamp(min=1e-6)
            text_norm = text_features.norm(dim=1, keepdim=True).clamp(min=1e-6)
            image_features = image_features / image_norm
            text_features = text_features / text_norm
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            if use_multi_positive_loss:
                captions = batch["query_text"]
                loss = multi_positive_contrastive_loss(logits_per_image, captions)
            else:
                labels = torch.arange(images.shape[0], device=device)
                loss_i = F.cross_entropy(logits_per_image, labels)
                loss_t = F.cross_entropy(logits_per_text, labels)
                loss = (loss_i + loss_t) / 2
            
            loss.backward()
            
            # 梯度裁剪
            clip_grad_norm_(clip_model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if (step + 1) % 9 == 0:
                avg_loss = total_loss / num_batches
                if wandb_enabled:
                    wandb.log({"train/batch_loss": loss.item()}, step=epoch * (len(train_loader) + 1) + num_batches)
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Step [{step+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
        
        # 评估
        print("开始评估...")
        metrics, _ = evaluate_simple(model, test_loader, device, test_dataset, clip_model, emotion_adapter)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] 结果:")
        print(f"R@1: {metrics['R@1']:.2f}% - R@5: {metrics['R@5']:.2f}% | P@1: {metrics.get('P@1', 0.0):.2f}% | P@5: {metrics.get('P@5', 0.0):.2f}%")
        if wandb_enabled:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"train/epoch": epoch + 1, "train/avg_loss": total_loss / max(num_batches, 1), "eval/R@1": metrics["R@1"], "eval/R@5": metrics["R@5"], "eval/P@1": metrics.get("P@1", 0.0), "eval/P@5": metrics.get("P@5", 0.0), "train/lr": current_lr}, step=(epoch + 1) * (len(train_loader) + 1))
        
        # 保存最佳模型
        if metrics['R@1'] > best_r1:
            best_r1 = metrics['R@1']
            save_path = os.path.join(CONFIG["save_dir"], CONFIG["model_save_name"])
            os.makedirs(CONFIG["save_dir"], exist_ok=True)
            if emotion_adapter is not None:
                torch.save({"clip_state_dict": clip_model.state_dict(), "adapter_state_dict": emotion_adapter.state_dict()}, save_path)
            else:
                torch.save(clip_model.state_dict(), save_path)
            print(f"最佳模型已保存: {save_path}")
            if wandb_enabled:
                wandb.save(save_path)

def main():
    """主函数"""
    print("简化CLIP训练脚本")
    print("使用现有MSRVTT_Dataset和配置")
    import argparse
    parser = argparse.ArgumentParser(description="Simple CLIP training")
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true" if value is False else "store_false")
        else:
            parser.add_argument(f"--{key}", type=type(value) if value is not None else str, default=value)
    args = parser.parse_args()
    for key in CONFIG.keys():
        CONFIG[key] = getattr(args, key)
    use_multi_positive_loss = bool(CONFIG["use_multi_positive_loss"])
    wandb_enabled = bool(CONFIG["wandb_project"])
    assert torch.cuda.is_available()
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if wandb_enabled:
        wandb.init(project=CONFIG["wandb_project"], entity=CONFIG["wandb_entity"] or None, config={"config": CONFIG})
    
    # 检查数据路径
    data_path = CONFIG["data_path"]
    image_path = CONFIG["image_path"]
    
    print(f"数据路径: {data_path}")
    print(f"图片路径: {image_path}")
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"警告：数据路径不存在 {data_path}")
        print("请修改CONFIG中的路径配置")
        return
    
    if not os.path.exists(image_path):
        print(f"警告：图片路径不存在 {image_path}")
        print("请修改CONFIG中的路径配置")
        return
    
    # 加载CLIP模型
    print("加载CLIP模型...")
    clip_model_path = CONFIG["clip_model_path"]
    
    # 检查本地模型路径是否存在
    if not os.path.exists(clip_model_path):
        print(f"警告：本地CLIP模型路径不存在 {clip_model_path}")
        print("将使用在线模型...")
    else:
        print(f"使用本地CLIP模型: {clip_model_path}")
    
    # 加载CLIP模型和processor
    clip_model, processor = clip.load(CONFIG["pretrained_clip_name"], device=device)
    clip_model = clip_model.float()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=clip_model_path).to(device)
    
    # 创建BERT tokenizer
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    # 构建文件路径
    train_csv_path = os.path.join(data_path, CONFIG["train_csv"])
    train_json_path = os.path.join(data_path, CONFIG["train_json"])
    train_emotion_json_path = os.path.join(data_path, CONFIG["train_emotion_json"])
    test_json_path = os.path.join(data_path, CONFIG["test_json"])
    test_emotion_json_path = os.path.join(data_path, CONFIG["test_emotion_json"])
    
    # 检查文件是否存在
    required_files = [
        train_csv_path, train_json_path, train_emotion_json_path,
        test_json_path, test_emotion_json_path
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"警告：文件不存在 {file_path}")
            print("请检查数据文件路径")
            return
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = MSRVTT_Dataset(
        csv_path=train_csv_path,
        json_path=train_json_path,
        features_path=image_path,
        emotion_json_path=train_emotion_json_path,
        clip_preprocess=processor,
        bert_tokenizer=bert_tokenizer,
        max_words=32,
        is_train=True,
        load_image=True
    )
    
    test_dataset = MSRVTT_Dataset(
        csv_path=test_json_path,  # 测试集使用JSON文件
        json_path=test_json_path,
        features_path=image_path,
        emotion_json_path=test_emotion_json_path,
        clip_preprocess=processor,
        bert_tokenizer=bert_tokenizer,
        max_words=32,
        is_train=False,
        load_image=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 开始训练
    emotion_adapter = None
    if CONFIG["use_image_emotion_fusion"] or CONFIG["use_text_emotion_fusion"]:
        emotion_adapter = EmotionAdapter(emo_dim=768, clip_dim=512).to(device)
        for p in emotion_adapter.bert.parameters():
            p.requires_grad = False
        emotion_adapter.bert.eval()
    train_simple_clip(train_loader, test_loader, model, device, test_dataset, clip_model, emotion_adapter, use_multi_positive_loss=use_multi_positive_loss, wandb_enabled=wandb_enabled)
    if wandb_enabled:
        wandb.finish()
    
    print("训练完成！")

if __name__ == "__main__":
    main() 
