# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import clip
import numpy as np
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
from models.IEF.co_attention import Co_attention_block
class CLIP4Meme(nn.Module):
    """
    Model with a STABLE architecture for Stage 1 (QI) training.
    """
    def __init__(self,
                 pretrained_clip_name: str = "ViT-B/32",
                 bert_model_name: str = 'bert-base-chinese'):
        super(CLIP4Meme, self).__init__()
        # 模型组件初始化 (与之前版本相同)
        self.clip, self.clip_preprocess = clip.load(pretrained_clip_name, device='cpu', jit=False)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        
        # 冻结BERT模型参数
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        clip_feature_dim = self.clip.visual.output_dim
        bert_feature_dim = self.bert_model.config.hidden_size
        self.emotion_projection = nn.Linear(bert_feature_dim, clip_feature_dim)
        self.emotion_gate = nn.Parameter(torch.tensor(0.1))
        self.co_attention = Co_attention_block(
            num_attention_heads=8, hidden_size=clip_feature_dim, dropout_rate=0.1
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # 各个编码器函数 (与之前版本相同)
    def get_bert_tokenizer(self):
        return self.bert_tokenizer

    def encode_image(self, image):
        return self.clip.encode_image(image.type(self.clip.dtype))

    def encode_text_pooled(self, text_ids):
        return self.clip.encode_text(text_ids)

    def encode_text_tokens(self, text_ids):
        x = self.clip.token_embedding(text_ids).type(self.clip.dtype)
        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x).type(self.clip.dtype)
        attention_mask = (1.0 - (text_ids == 0).float().unsqueeze(1).unsqueeze(2)) * -10000.0
        return x, attention_mask
        
    def encode_emotion(self, emotion_ids, attention_mask):
        outputs = self.bert_model(input_ids=emotion_ids, attention_mask=attention_mask)
        masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        mean_pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return mean_pooled

    def forward(self,
                query_ids,
                candidate_ids,
                image,
                emotion_ids=None,
                emotion_mask=None,
                branch: str = 'qi',
                return_features: bool = False):

        # ====================================================================
        # QI分支: 训练 Query-Image 相似度
        # ====================================================================
        if branch == 'qi':
            # 1. 提取所有需要的特征
            image_feat = self.encode_image(image)
            query_feat_pooled = self.encode_text_pooled(query_ids)
            # 获取 Context (candidate_ids) 的 token-level 特征用于融合
            candidate_feat_tokens, candidate_mask = self.encode_text_tokens(candidate_ids)

            # 2. [可选] 融合情感特征到图像特征
            if emotion_ids is not None:
                with torch.no_grad():
                    emotion_feat_bert = self.encode_emotion(emotion_ids, emotion_mask)
                projected_emotion_feat = self.emotion_projection(emotion_feat_bert.type(self.emotion_projection.weight.dtype))
                projected_emotion_feat = F.normalize(projected_emotion_feat, dim=-1)
                image_feat = image_feat + self.emotion_gate * projected_emotion_feat
            
            # 3. 跨模态融合 (Cross-Modal Fusion)
            # Co-attention 融合 Image 和 Context (图片描述)
            image_feat_seq = image_feat.unsqueeze(1)
            image_mask_for_attention = torch.zeros(image_feat_seq.size(0), 1, 1, 1, device=image.device)
            
            fused_image_feat, _ = self.co_attention(
                image_feat_seq, image_mask_for_attention,
                candidate_feat_tokens, candidate_mask # 使用 context 特征进行交互
            )
            
            # 4. 计算相似度
            # 用独立的 Query 特征去匹配融合后的"图像+上下文"特征
            final_fused_feat = fused_image_feat.squeeze(1)
            
            query_feat_pooled = F.normalize(query_feat_pooled, dim=-1)
            final_fused_feat = F.normalize(final_fused_feat, dim=-1)
            
            # 如果只需要特征，直接返回
            if return_features:
                return query_feat_pooled, final_fused_feat
            
            sim_matrix = torch.matmul(query_feat_pooled, final_fused_feat.t()) * self.logit_scale.exp()
            
            return sim_matrix

        # ====================================================================
        # QC分支: 训练 Query-Context 相似度
        # ====================================================================
        elif branch == 'qc':
            query_feat_pooled = self.encode_text_pooled(query_ids)
            candidate_feat_pooled = self.encode_text_pooled(candidate_ids)
            
            query_feat_pooled = F.normalize(query_feat_pooled, dim=-1)
            candidate_feat_pooled = F.normalize(candidate_feat_pooled, dim=-1)
            
            # 如果只需要特征，直接返回
            if return_features:
                return query_feat_pooled, candidate_feat_pooled
                
            sim_matrix = torch.matmul(query_feat_pooled, candidate_feat_pooled.t()) * self.logit_scale.exp()

            return sim_matrix
        
        else:
            raise ValueError(f"Unknown branch: {branch}")

    def inference_fusion(self, 
                        query_ids, 
                        candidate_ids, 
                        image, 
                        emotion_ids=None, 
                        emotion_mask=None,
                        alpha: float = 0.5,
                        temperature: float = 0.07):
        """
        推理时的融合方法：同时利用QI和QC分支，加权得到最终结果
        
        Args:
            alpha: QI分支的权重，QC分支权重为(1-alpha) (默认0.6)
            temperature: 温度参数，控制相似度分布的尖锐程度
        """
        with torch.no_grad():
            # 1. 计算QI分支的相似度
            qi_sim = self.forward(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                image=image,
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                branch='qi'
            )
            
            # 2. 计算QC分支的相似度
            qc_sim = self.forward(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                image=image,  # QC不需要图片，但为了保持接口一致
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                branch='qc'
            )
            
            # 3. 加权融合
            # 注意：这里使用logits进行加权，然后重新应用temperature
            # Final = α × QI + (1-α) × QC
            fused_sim = (alpha * qi_sim + (1 - alpha) * qc_sim) / temperature
            
            return fused_sim, qi_sim, qc_sim

    def hierarchical_retrieval(self, 
                              query_ids, 
                              candidate_ids, 
                              image, 
                              emotion_ids=None, 
                              emotion_mask=None,
                              coarse_k: int = 100,
                              alpha: float = 0.6,
                              temperature: float = 0.07):
        """
        分层检索：粗召回 + 精排序
        
        Args:
            coarse_k: 粗召回阶段选择的候选数量
            alpha: QI分支权重，QC分支权重为(1-alpha)
            temperature: 温度参数
        
        Returns:
            final_scores: 最终融合分数
            top_k_indices: 粗召回阶段的Top-K索引
            qc_sim: QC分支相似度矩阵
            qi_sim: QI分支相似度矩阵
        """
        with torch.no_grad():
            # 阶段1：粗召回 (QC分支) - 纯文本匹配，速度快
            qc_sim = self.forward(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                image=image,
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                branch='qc'
            )
            
            # 选择Top-K候选进行精排序
            top_k_values, top_k_indices = torch.topk(qc_sim, k=coarse_k, dim=-1)
            
            # 阶段2：精排序 (QI分支) - 仅在粗召回结果上计算
            qi_sim = self.forward(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                image=image,
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                branch='qi'
            )
            
            # 融合排序：结合粗召回和精排序的结果
            final_scores = alpha * qi_sim + (1 - alpha) * qc_sim
            
            return final_scores, top_k_indices, qc_sim, qi_sim

    def hierarchical_retrieval_efficient(self, 
                                       query_ids, 
                                       candidate_ids, 
                                       image, 
                                       emotion_ids=None, 
                                       emotion_mask=None,
                                       coarse_k: int = 100,
                                       alpha: float = 0.6,
                                       temperature: float = 0.07):
        """
        高效分层检索：避免重复计算，优化内存使用
        
        Args:
            coarse_k: 粗召回阶段选择的候选数量
            alpha: QI分支权重，QC分支权重为(1-alpha)
            temperature: 温度参数
        
        Returns:
            final_scores: 最终融合分数
            top_k_indices: 粗召回阶段的Top-K索引
            qc_sim: QC分支相似度矩阵
            qi_sim: QI分支相似度矩阵（仅在粗召回候选上计算）
        """
        with torch.no_grad():
            # 阶段1：粗召回 (QC分支)
            qc_sim = self.forward(
                query_ids=query_ids,
                candidate_ids=candidate_ids,
                image=image,
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                branch='qc'
            )
            
            # 选择Top-K候选
            top_k_values, top_k_indices = torch.topk(qc_sim, k=coarse_k, dim=-1)
            
            # 阶段2：精排序 (QI分支) - 仅在粗召回结果上计算
            # 这里需要重新组织数据，只计算Top-K候选的QI相似度
            batch_size = query_ids.size(0)
            
            # 创建新的candidate_ids，只包含粗召回的结果
            selected_candidate_ids = torch.gather(
                candidate_ids, 
                1, 
                top_k_indices.unsqueeze(-1).expand(-1, -1, candidate_ids.size(-1))
            )
            
            # 计算QI相似度（仅在粗召回候选上）
            qi_sim_selected = self.forward(
                query_ids=query_ids,
                candidate_ids=selected_candidate_ids,
                image=image,
                emotion_ids=emotion_ids,
                emotion_mask=emotion_mask,
                branch='qi'
            )
            
            # 创建完整的QI相似度矩阵，未选中的位置填充负无穷
            qi_sim_full = torch.full_like(qc_sim, float('-inf'))
            qi_sim_full.scatter_(1, top_k_indices, qi_sim_selected)
            
            # 融合排序
            final_scores = alpha * qi_sim_full + (1 - alpha) * qc_sim
            
            return final_scores, top_k_indices, qc_sim, qi_sim_full
