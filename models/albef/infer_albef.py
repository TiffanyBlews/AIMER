import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataset import MSRVTT_Dataset

def compute_recalls(similarity_matrix, all_captions_meta, ks=(1, 5, 10)):
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

def compute_precisions(similarity_matrix, all_captions_meta, ks=(1, 5, 10)):
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

def infer_albef(model_name="albef_feature_extractor", model_type="base", data_path="./imgflip_data/msrvtt", image_path="./imgflip_data/images", batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    import lavis
    model, vis_processors, txt_processors = lavis.models.load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True, device=device)
    test_json_path = os.path.join(data_path, "test_data.json")
    test_emotion_json_path = os.path.join(data_path, "test_emotion.json")
    test_dataset = MSRVTT_Dataset(
        csv_path=test_json_path,
        json_path=test_json_path,
        features_path=image_path,
        emotion_json_path=test_emotion_json_path,
        clip_preprocess=None,
        bert_tokenizer=None,
        max_words=32,
        is_train=False,
        load_image=False,
    )
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    text_feats = []
    image_feats = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            vids = batch["video_id"]
            texts = batch["query_text"]
            images_proc = []
            for vid in vids:
                p = os.path.join(image_path, vid if "." in vid else f"{vid}.jpg")
                from PIL import Image
                img = Image.open(p).convert("RGB")
                images_proc.append(vis_processors["eval"](img).unsqueeze(0))
            images_t = torch.cat(images_proc, dim=0).to(device)
            txt_inputs = [txt_processors["eval"](t) for t in texts]
            txt_inputs = lavis.processors.blip_processors.BlipCaptionProcessor.clean(txt_inputs) if hasattr(lavis.processors.blip_processors, "BlipCaptionProcessor") else txt_inputs
            img_feat = model.encode_image(images_t)
            txt_feat = model.encode_text(txt_inputs)
            img_feat = img_feat / img_feat.norm(dim=1, keepdim=True).clamp(min=1e-6)
            txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True).clamp(min=1e-6)
            image_feats.append(img_feat.cpu().numpy())
            text_feats.append(txt_feat.cpu().numpy())
    text_matrix = np.concatenate(text_feats, axis=0)
    image_matrix = np.concatenate(image_feats, axis=0)
    sim = np.dot(text_matrix, image_matrix.T)
    caps = [item["query"] for item in test_dataset.data]
    metrics_r = compute_recalls(sim, caps, ks=(1, 5, 10))
    metrics_p = compute_precisions(sim, caps, ks=(1, 5, 10))
    return {**metrics_r, **metrics_p}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="albef_feature_extractor")
    parser.add_argument("--model_type", type=str, default="base")
    parser.add_argument("--data_path", type=str, default="./imgflip_data/msrvtt")
    parser.add_argument("--image_path", type=str, default="./imgflip_data/images")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = infer_albef(model_name=args.model_name, model_type=args.model_type, data_path=args.data_path, image_path=args.image_path, batch_size=args.batch_size, device=device)
    for k in [1, 5, 10]:
        print(f"R@{k}: {metrics[f'R@{k}']:.2f}% | P@{k}: {metrics.get(f'P@{k}', 0.0):.2f}%")

if __name__ == "__main__":
    main()
