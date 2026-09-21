# inference_fusion.py
# 演示如何使用融合方法进行推理

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import random
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from model import CLIP4Meme
from dataset import MSRVTT_Dataset
from torch.utils.data import DataLoader

def set_seed(seed=42):
    """
    设置随机种子以确保结果可重现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 设置CUDA确定性选项
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"CUDA deterministic mode enabled")
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"Random seed set to {seed} for reproducibility")
    print(f"Note: For complete reproducibility, ensure same hardware and software environment")

def evaluate_fusion(model, test_dataset, test_dataloader, device, 
                   alpha=0.6, temperature=0.05, print_n_samples=5):
    """
    评估融合方法的性能
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 额外的确定性设置
    torch.use_deterministic_algorithms(True)
    
    print(f"\n--- Evaluating Fusion Method (α={alpha:.1f}, QI:{alpha:.1f} + QC:{1-alpha:.1f}) ---")
    
    # 1. 收集元数据
    all_video_ids_meta = [item['video_id'] for item in test_dataset.data]
    all_captions_meta = [item['query'] for item in test_dataset.data]
    all_titles_meta = [item['candidate_texts'][0] if item['candidate_texts'] else "" for item in test_dataset.data]
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    
    # 2. 使用融合方法进行推理 - 修复：先收集所有特征
    all_qi_features = []
    all_qc_features = []
    all_query_features = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Collecting Features"):
            # 获取QI分支特征
            qi_query_feat, qi_target_feat = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                branch='qi',
                return_features=True
            )
            
            # 获取QC分支特征
            qc_query_feat, qc_target_feat = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                branch='qc',
                return_features=True
            )
            
            all_qi_features.append(qi_target_feat.cpu())
            all_qc_features.append(qc_target_feat.cpu())
            all_query_features.append(qi_query_feat.cpu())  # 使用QI的query特征
    
    # 3. 拼接所有特征
    all_qi_features = torch.cat(all_qi_features, dim=0)
    all_qc_features = torch.cat(all_qc_features, dim=0)
    all_query_features = torch.cat(all_query_features, dim=0)
    
    print(f"特征收集完成: QI={all_qi_features.shape}, QC={all_qc_features.shape}, Query={all_query_features.shape}")
    
    # 4. 计算全局相似度矩阵
    all_query_features = F.normalize(all_query_features, dim=-1)
    all_qi_features = F.normalize(all_qi_features, dim=-1)
    all_qc_features = F.normalize(all_qc_features, dim=-1)
    
    # 计算QI和QC的相似度矩阵
    qi_sim_matrix = torch.matmul(all_query_features, all_qi_features.t())
    qc_sim_matrix = torch.matmul(all_query_features, all_qc_features.t())
    
    # 应用温度参数
    qi_sim_matrix = qi_sim_matrix / temperature
    qc_sim_matrix = qc_sim_matrix / temperature
    
    # 融合相似度矩阵
    fused_sim_matrix = alpha * qi_sim_matrix + (1 - alpha) * qc_sim_matrix
    
    print(f"相似度矩阵计算完成: 融合={fused_sim_matrix.shape}, QI={qi_sim_matrix.shape}, QC={qc_sim_matrix.shape}")
    
    print("\n--- Computing R@k for Fusion Method ---")
    num_queries = fused_sim_matrix.shape[0]
    recalls = {}
    precisions = {}
    ks = [1, 5, 10, 50, 100]
    valid_ks = [k for k in ks if k <= fused_sim_matrix.shape[1]]
    
    for k in valid_ks:
        hits = 0
        acc = 0.0
        for q_idx in range(num_queries):
            top_preds = torch.topk(fused_sim_matrix[q_idx], k=k).indices.tolist()
            gt_indices = caption_to_indices[all_captions_meta[q_idx]]
            if any(p in gt_indices for p in top_preds):
                hits += 1
            correct = sum(1 for p in top_preds if p in gt_indices)
            acc += correct / float(k)
            if q_idx < print_n_samples and k == 10:
                print(f"\n{'='*22} Sample {q_idx+1}/{print_n_samples} {'='*22}")
                print(f"  [Query]      VID: {all_video_ids_meta[q_idx]}, Caption: {all_captions_meta[q_idx]}")
                print(f"  [GT Indices] {gt_indices}")
                print("  [Top-5 Predictions (Fused)]")
                top5_fused = torch.topk(fused_sim_matrix[q_idx], k=5)
                top5_qi = torch.topk(qi_sim_matrix[q_idx], k=5)
                top5_qc = torch.topk(qc_sim_matrix[q_idx], k=5)
                for rank in range(5):
                    pred_idx = top5_fused.indices[rank]
                    fused_score = top5_fused.values[rank]
                    qi_score = qi_sim_matrix[q_idx][pred_idx]
                    qc_score = qc_sim_matrix[q_idx][pred_idx]
                    mark = "✔️" if pred_idx in gt_indices else "❌"
                    print(f"    {rank+1}. {mark} Fused={fused_score:.4f} | QI={qi_score:.4f} | QC={qc_score:.4f}")
                    print(f"        VID={all_video_ids_meta[pred_idx]} | Cap: {all_captions_meta[pred_idx]}")
        recalls[f'R@{k}'] = (hits / num_queries) * 100
        precisions[f'P@{k}'] = (acc / num_queries) * 100
    
    print("\n--- Final Fusion Results ---")
    for k in valid_ks:
        print(f"  R@{k}: {recalls[f'R@{k}']:.2f}% | P@{k}: {precisions[f'P@{k}']:.2f}%")
    
    return {**recalls, **precisions}, fused_sim_matrix, qi_sim_matrix, qc_sim_matrix

def compare_multiple_alphas(model, test_dataset, test_dataloader, device, alphas, temperature=0.05):
    """
    比较多个alpha值的性能，包括单独使用QI分支(α=1.0)和QC分支(α=0.0)
    """
    # 额外的确定性设置
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
            print("PyTorch deterministic algorithms enabled")
        except:
            print("Warning: Could not enable deterministic algorithms")
    
    print("\n" + "="*80)
    print("COMPARING MULTIPLE ALPHA VALUES")
    print("="*80)
    
    # 存储所有结果
    all_results = {}
    
    # 先收集特征（避免重复计算）
    print("\n--- Collecting Features Once for All Alpha Values ---")
    all_qi_features = []
    all_qc_features = []
    all_query_features = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Collecting Features"):
            # 获取QI分支特征
            qi_query_feat, qi_target_feat = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                branch='qi',
                return_features=True
            )
            
            # 获取QC分支特征
            qc_query_feat, qc_target_feat = model(
                query_ids=batch['query_ids'].to(device),
                candidate_ids=batch['candidate_ids'].to(device),
                image=batch['image'].to(device),
                emotion_ids=batch['emotion_input_ids'].to(device),
                emotion_mask=batch['emotion_attention_mask'].to(device),
                branch='qc',
                return_features=True
            )
            
            all_qi_features.append(qi_target_feat.cpu())
            all_qc_features.append(qc_target_feat.cpu())
            all_query_features.append(qi_query_feat.cpu())
    
    # 拼接所有特征
    all_qi_features = torch.cat(all_qi_features, dim=0)
    all_qc_features = torch.cat(all_qc_features, dim=0)
    all_query_features = torch.cat(all_query_features, dim=0)
    
    # 归一化特征
    all_query_features = F.normalize(all_query_features, dim=-1)
    all_qi_features = F.normalize(all_qi_features, dim=-1)
    all_qc_features = F.normalize(all_qc_features, dim=-1)
    
    # 计算基础相似度矩阵
    qi_sim_matrix = torch.matmul(all_query_features, all_qi_features.t()) / temperature
    qc_sim_matrix = torch.matmul(all_query_features, all_qc_features.t()) / temperature
    
    # 收集元数据
    all_captions_meta = [item['query'] for item in test_dataset.data]
    caption_to_indices = defaultdict(list)
    for idx, cap in enumerate(all_captions_meta):
        caption_to_indices[cap].append(idx)
    
    num_queries = qi_sim_matrix.shape[0]
    
    # 测试每个alpha值
    for alpha in alphas:
        # 为alpha=0.0和alpha=1.0添加特殊说明
        if alpha == 0.0:
            print(f"\n--- Testing Alpha = {alpha:.2f} (QC Branch Only) ---")
        elif alpha == 1.0:
            print(f"\n--- Testing Alpha = {alpha:.2f} (QI Branch Only) ---")
        else:
            print(f"\n--- Testing Alpha = {alpha:.2f} (Fusion: {alpha:.1f}×QI + {1-alpha:.1f}×QC) ---")
        
        # 融合相似度矩阵
        fused_sim_matrix = alpha * qi_sim_matrix + (1 - alpha) * qc_sim_matrix
        
        recalls = {}
        precisions = {}
        ks = [1, 5, 10, 50, 100]
        valid_ks = [k for k in ks if k <= qi_sim_matrix.shape[1]]
        for k in valid_ks:
            hits = 0
            acc = 0.0
            for q_idx in range(num_queries):
                top_preds = torch.topk(fused_sim_matrix[q_idx], k=k).indices.tolist()
                gt_indices = caption_to_indices[all_captions_meta[q_idx]]
                if any(p in gt_indices for p in top_preds):
                    hits += 1
                correct = sum(1 for p in top_preds if p in gt_indices)
                acc += correct / float(k)
            recalls[f'R@{k}'] = (hits / num_queries) * 100
            precisions[f'P@{k}'] = (acc / num_queries) * 100
        all_results[alpha] = {**recalls, **precisions}
        print(f"Alpha = {alpha:.2f}: ", end="")
        for k in valid_ks:
            print(f"R@{k}={recalls[f'R@{k}']:.2f}% P@{k}={precisions[f'P@{k}']:.2f}%", end=" ")
        print()
    
    # 打印比较表格
    print("\n" + "="*80)
    print("ALPHA COMPARISON SUMMARY")
    print("="*80)
    
    # 表头
    header = f"{'Alpha':<8} {'Branch':<12}"
    ks = [1, 5, 10, 50, 100]
    valid_ks = [k for k in ks if k <= qi_sim_matrix.shape[1]]
    for k in valid_ks:
        header += f" R@{k} P@{k}"
    print(header)
    print("-" * 80)
    
    # 数据行
    for alpha in sorted(alphas):
        # 确定分支类型
        if alpha == 0.0:
            branch_name = "QC Only"
        elif alpha == 1.0:
            branch_name = "QI Only"
        else:
            branch_name = f"Fusion"
        
        row = f"{alpha:<8.2f} {branch_name:<12}"
        for k in valid_ks:
            row += f" {all_results[alpha][f'R@{k}']:<10.2f} {all_results[alpha][f'P@{k}']:<10.2f}"
        print(row)
    
    # 找到最佳alpha
    best_alpha_r1 = max(alphas, key=lambda a: all_results[a]['R@1'])
    best_alpha_r5 = max(alphas, key=lambda a: all_results[a]['R@5'])
    best_alpha_r10 = max(alphas, key=lambda a: all_results[a]['R@10'])
    best_alpha_r50 = None
    best_alpha_r100 = None
    if 'R@50' in next(iter(all_results.values())):
        best_alpha_r50 = max(alphas, key=lambda a: all_results[a]['R@50'])
    if 'R@100' in next(iter(all_results.values())):
        best_alpha_r100 = max(alphas, key=lambda a: all_results[a]['R@100'])
    
    print("\n" + "="*80)
    print("BEST ALPHA VALUES")
    print("="*80)
    print(f"Best for R@1:  α = {best_alpha_r1:.2f} (R@1 = {all_results[best_alpha_r1]['R@1']:.2f}%)")
    print(f"Best for R@5:  α = {best_alpha_r5:.2f} (R@5 = {all_results[best_alpha_r5]['R@5']:.2f}%)")
    print(f"Best for R@10: α = {best_alpha_r10:.2f} (R@10 = {all_results[best_alpha_r10]['R@10']:.2f}%)")
    if best_alpha_r50 is not None:
        print(f"Best for R@50: α = {best_alpha_r50:.2f} (R@50 = {all_results[best_alpha_r50]['R@50']:.2f}%)")
    if best_alpha_r100 is not None:
        print(f"Best for R@100: α = {best_alpha_r100:.2f} (R@100 = {all_results[best_alpha_r100]['R@100']:.2f}%)")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion method for QI+QC branches")
    parser.add_argument('--checkpoint', type=str, default="checkpoints_imgflip/qi_qc.pth",
                       help="Path to model checkpoint")
    parser.add_argument('--data_path', type=str, default="/root/zt/imgflip_results/msrvtt",
                       help="Path to data directory")
    parser.add_argument('--image_path', type=str, default="/root/zt/imgflip_results/images",
                       help="Path to image directory")
    # parser.add_argument('--checkpoint', type=str, default="checkpoints_douban/stage2_qc_best.pth",
    #                    help="Path to model checkpoint")
    # parser.add_argument('--data_path', type=str, default="/root/zt/data_topics/input_file",
    #                    help="Path to data directory")
    # parser.add_argument('--image_path', type=str, default="/root/zt/data_topics/image",
                    #    help="Path to image directory")
    parser.add_argument('--test_json', type=str, default="test_data.json",
                       help="Test JSON file")
    parser.add_argument('--test_emotion_json', type=str, default="test_emotion.json",
                       help="Test emotion JSON file")
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size for inference")
    parser.add_argument('--alpha', type=float, default=0.8,
                       help="Weight α for QI branch in fusion: Final = α×QI + (1-α)×QC")
    parser.add_argument('--alphas', nargs='+', type=float, 
                       default=[0.0, 0.2, 0.5, 0.8, 0.85 ,1.0],
                       help="Multiple alpha values to test (space-separated)")
    parser.add_argument('--multi_alpha', action='store_true', default=False,
                       help="Test multiple alpha values and compare")
    parser.add_argument('--seed', type=int, default=114514,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保结果可重现
    set_seed(args.seed)
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. 初始化模型
    model = CLIP4Meme()
    model.to(device)
    
    # 2. 只需要加载第二阶段的checkpoint
    print("Loading Stage 2 (QC) checkpoint (contains both QI and QC parameters)...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 准备测试数据
    from dataset import MSRVTT_Dataset
    test_dataset = MSRVTT_Dataset(
        csv_path=os.path.join(args.data_path, args.test_json),
        json_path=os.path.join(args.data_path, args.test_json),
        features_path=args.image_path,
        emotion_json_path=os.path.join(args.data_path, args.test_emotion_json),
        bert_tokenizer=model.get_bert_tokenizer(),
        clip_preprocess=model.clip_preprocess,
        is_train=False,
        load_image=True
    )
    # 使用num_workers=0确保确定性，避免多进程导致的随机性
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 4. 执行推理
    if args.multi_alpha:
        # 测试多个alpha值（包括单独分支）
        print(f"Testing multiple alpha values: {args.alphas}")
        print("Note: α=0.0 = QC Branch Only, α=1.0 = QI Branch Only")
        compare_multiple_alphas(model, test_dataset, test_dataloader, device, args.alphas)
    else:
        # 只评估单个融合方法
        evaluate_fusion(model, test_dataset, test_dataloader, device,
                      alpha=args.alpha)

if __name__ == '__main__':
    main() 
