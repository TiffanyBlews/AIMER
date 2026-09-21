# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import json
from PIL import Image
import os
import clip

class MSRVTT_Dataset(Dataset):
    """
    Custom Dataset for loading Meme-Text data.
    IMPROVEMENT: Added `load_image` flag to skip image loading for stage 2 training.
    """
    def __init__(self,
                 csv_path: str,
                 json_path: str,
                 features_path: str,
                 emotion_json_path: str,
                 clip_preprocess,
                 bert_tokenizer,
                 max_words: int = 32,
                 is_train: bool = True,
                 load_image: bool = True,
                 votes_csv_path: str = os.path.join('imgflip_data', 'new', 'caption_results_merged_with_emotions_with_votes_with_bgem3_similarity_with_consistency.csv')):

        self.is_train = is_train
        self.data = []
        self.features_path = features_path
        self.max_words = max_words
        self.clip_preprocess = clip_preprocess
        self.bert_tokenizer = bert_tokenizer
        self.load_image = load_image
        self.votes_map = {}
        votes_path = votes_csv_path
        
        # 先读取数据，检查是否有匹配的video_id（用于判断是否需要加载豆瓣点赞文件）
        if self.is_train:
            df = pd.read_csv(csv_path)
            col = 'video_id' if 'video_id' in df.columns else ('image_id' if 'image_id' in df.columns else None)
        else:
            df = None
            col = None
        
        # 先尝试加载默认的点赞文件
        if votes_path and os.path.exists(votes_path):
            try:
                votes_df = pd.read_csv(votes_path)
                votes_df['image_id'] = votes_df['image_id'].astype(str)
                self.votes_map = {
                    row['image_id']: (int(row['votes']) if pd.notna(row['votes']) else 0)
                    for _, row in votes_df.iterrows()
                }
                print(f"成功加载点赞信息，共 {len(self.votes_map)} 条记录")
            except Exception as e:
                print(f"加载点赞信息失败: {e}")
                self.votes_map = {}
        
        # 检查是否需要加载豆瓣点赞文件
        if col is not None and df is not None:
            video_ids = df[col].astype(str).tolist()[:10]  # 只检查前10个
            # 检查是否有匹配的video_id
            has_match = False
            for vid in video_ids:
                image_id_key = vid if '.' in vid else f"{vid}.jpg"
                if image_id_key in self.votes_map:
                    has_match = True
                    break
            
            # 如果没有匹配，尝试加载豆瓣点赞文件
            if not has_match:
                douban_votes_path = os.path.join('data', 'douban_votes.csv')
                if os.path.exists(douban_votes_path):
                    try:
                        douban_votes_df = pd.read_csv(douban_votes_path)
                        douban_votes_df['image_id'] = douban_votes_df['image_id'].astype(str)
                        douban_votes_map = {
                            row['image_id']: (int(row['votes']) if pd.notna(row['votes']) else 0)
                            for _, row in douban_votes_df.iterrows()
                        }
                        # 合并到votes_map中（豆瓣数据优先）
                        self.votes_map.update(douban_votes_map)
                        print(f"检测到豆瓣数据集，加载豆瓣点赞文件: {douban_votes_path}")
                        print(f"成功加载点赞信息，共 {len(self.votes_map)} 条记录（包含豆瓣数据）")
                    except Exception as e:
                        print(f"加载豆瓣点赞信息失败: {e}")
        
        # --- 数据加载逻辑 (与您提供的代码相同) ---
        if self.is_train:
            if df is None:  # 如果上面没有读取，这里再读取
                df = pd.read_csv(csv_path)
                col = 'video_id' if 'video_id' in df.columns else ('image_id' if 'image_id' in df.columns else None)
            if col is None:
                raise KeyError('video_id')
            video_ids = df[col].astype(str).tolist()
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            captions = {item['video_id']: item['caption'] for item in json_data['sentences']}
            titles = json_data['titles']
            with open(emotion_json_path, 'r', encoding='utf-8') as f:
                emotions_data = json.load(f)
            emotions = {}
            for item in emotions_data:
                video_id = item['video_id']
                emotion_list = [e for d in item['emotion'] for e in d.values()]
                emotions[video_id] = emotion_list
            for vid in video_ids:
                image_id_key = vid if '.' in vid else f"{vid}.jpg"
                vote_value = self.votes_map.get(image_id_key, 0)
                if vid in captions and vid in titles and vid in emotions:
                    self.data.append({
                        "video_id": vid,
                        "query": captions[vid],
                        "candidate_texts": titles[vid],
                        "emotions": emotions[vid],
                        "upvote": vote_value
                    })
        else:
            # 测试集加载逻辑 (与您提供的代码类似，稍作清理)
            with open(csv_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            with open(emotion_json_path, 'r', encoding='utf-8') as f:
                emotions_data = json.load(f)
            emotions = {}
            for item in emotions_data:
                video_id = item['video_id']
                emotion_list = [e for d in item['emotion'] for e in d.values()]
                emotions[video_id] = emotion_list
            for vid, content in test_data.items():
                image_id_key = vid if '.' in vid else f"{vid}.jpg"
                vote_value = self.votes_map.get(image_id_key, 0)
                if vid in emotions:
                    self.data.append({
                        "video_id": vid,
                        "query": content['gt'],
                        "candidate_texts": content['titles'],
                        "emotions": emotions[vid],
                        "upvote": vote_value
                    })

        print(f"Loaded {len(self.data)} samples for {'training' if is_train else 'testing'}. Image loading: {self.load_image}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        
        # --- 按需处理图片 ---
        image_tensor = torch.zeros(1) # 返回一个哑元 (dummy tensor)
        if self.load_image:
            image_path = os.path.join(self.features_path, video_id if '.' in video_id else f"{video_id}.jpg")
            if not os.path.exists(image_path):
                # 如果找不到，可以返回一个零张量并打印警告，而不是中断程序
                print(f"Warning: Image not found at {image_path}. Returning a zero tensor.")
                image_tensor = torch.zeros(3, 224, 224)
            else:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.clip_preprocess(image)

        # --- 处理文本 (Query 和 Candidate Text) ---
        # 您的代码只用了第一个候选文本，这里我们保持一致。
        # 如果要用多个，需要修改模型和训练逻辑来处理池化。
        context_text = item['candidate_texts'][0] if item['candidate_texts'] else ""
        
        query_tokenized = clip.tokenize([item['query']], truncate=True)[0]
        context_tokenized = clip.tokenize([context_text], truncate=True)[0]

        # --- 处理情感 (与您提供的代码相同) ---
        emotion_list = [str(e) for e in item['emotions'] if e]
        emotion_text = " ".join(emotion_list) if emotion_list else "无情感"
        if self.bert_tokenizer is not None:
            bert_inputs = self.bert_tokenizer(
                emotion_text, padding='max_length', truncation=True,
                max_length=16, return_tensors="pt"
            )
            emotion_input_ids = bert_inputs['input_ids'].squeeze(0)
            emotion_attention_mask = bert_inputs['attention_mask'].squeeze(0)
        else:
            emotion_input_ids = torch.zeros(16, dtype=torch.long)
            emotion_attention_mask = torch.zeros(16, dtype=torch.long)
        
        return {
            "video_id": video_id,
            "image": image_tensor,
            "query_ids": query_tokenized, # 重命名以匹配文档 (Qc)
            "candidate_ids": context_tokenized, # 重命名以匹配文档 (Ct)
            "emotion_input_ids": emotion_input_ids,
            "emotion_attention_mask": emotion_attention_mask,
            "query_text": item['query'],
            "upvote": item.get('upvote', 0)
        }
