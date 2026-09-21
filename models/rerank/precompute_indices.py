import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import clip

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataset import MSRVTT_Dataset

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
            if "upvote" in batch:
                upvotes.extend(batch["upvote"].cpu().numpy().tolist())
            captions.extend(batch["query_text"])
    text_matrix = torch.cat(text_feats, dim=0)
    image_matrix = torch.cat(image_feats, dim=0)
    upvotes_arr = np.array(upvotes, dtype=np.float32) if len(upvotes) > 0 else None
    return text_matrix, image_matrix, upvotes_arr, captions

def precompute_topk(text_matrix, image_matrix, topk_base):
    sim = text_matrix.numpy() @ image_matrix.numpy().T
    k_base = min(topk_base, sim.shape[1])
    topk_list = [np.argpartition(-sim[q], range(k_base))[:k_base] for q in range(sim.shape[0])]
    return topk_list

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--topk_base", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_rerank")
    parser.add_argument("--clip_checkpoint", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    if args.clip_checkpoint and os.path.exists(args.clip_checkpoint):
        state = torch.load(args.clip_checkpoint, map_location=device)
        try:
            clip_model.load_state_dict(state)
        except Exception:
            clip_model.load_state_dict(state, strict=False)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.split == "train":
        train_csv_path = os.path.join(args.data_path, "train_ids.csv")
        train_json_path = os.path.join(args.data_path, "train_data.json")
        train_emotion_json_path = os.path.join(args.data_path, "train_emotion.json")
        ds = MSRVTT_Dataset(
            csv_path=train_csv_path,
            json_path=train_json_path,
            features_path=args.image_path,
            emotion_json_path=train_emotion_json_path,
            clip_preprocess=preprocess,
            bert_tokenizer=None,
            max_words=32,
            is_train=True,
            load_image=True,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        t_text, t_image, upvotes, caps = compute_features(clip_model, loader, device)
        topk = precompute_topk(t_text, t_image, args.topk_base)
        torch.save({
            "text": t_text,
            "image": t_image,
            "upvotes": upvotes,
            "captions": caps,
            "topk_base": args.topk_base,
            "topk_indices": topk,
        }, os.path.join(args.save_dir, "train_cache.pt"))
    else:
        test_json_path = os.path.join(args.data_path, "test_data.json")
        test_emotion_json_path = os.path.join(args.data_path, "test_emotion.json")
        ds = MSRVTT_Dataset(
            csv_path=test_json_path,
            json_path=test_json_path,
            features_path=args.image_path,
            emotion_json_path=test_emotion_json_path,
            clip_preprocess=preprocess,
            bert_tokenizer=None,
            max_words=32,
            is_train=False,
            load_image=True,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        t_text, t_image, _, caps = compute_features(clip_model, loader, device)
        topk = precompute_topk(t_text, t_image, args.topk_base)
        torch.save({
            "text": t_text,
            "image": t_image,
            "captions": caps,
            "topk_base": args.topk_base,
            "topk_indices": topk,
        }, os.path.join(args.save_dir, "test_cache.pt"))

if __name__ == "__main__":
    main()

