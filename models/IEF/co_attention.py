# co_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
import clip
import numpy as np
import math

# ====================================================================================
#  从 Cap4Video 移植过来的 Co-attention 模块及其依赖项
#  source: https://github.com/whwu95/Cap4Video/tree/main/modules
# ====================================================================================

class BertBiAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_rate=0.1):
        super(BertBiAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        self.key1 = nn.Linear(hidden_size, self.all_head_size)
        self.value1 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2):
        # input_tensor1: image_feat [B, 1, D]
        # input_tensor2: context_feat [B, N_tokens, D]
        
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)

        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        # Context-to-Image Attention
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1 # mask is [B, 1, 1, 1] for image
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = torch.matmul(attention_probs1, value_layer1) # Fused Image -> Context
        
        # Image-to-Context Attention
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_scores2 = attention_scores2 + attention_mask2 # mask is [B, 1, 1, N_tokens] for context
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = torch.matmul(attention_probs2, value_layer2) # Fused Context -> Image

        # Reshape back
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
        
        return context_layer1, context_layer2

class BertBiOutput(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(BertBiOutput, self).__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):
        hidden_states1 = self.dense1(hidden_states1)
        hidden_states1 = self.dropout1(hidden_states1)
        hidden_states1 = self.LayerNorm1(hidden_states1 + input_tensor1)

        hidden_states2 = self.dense2(hidden_states2)
        hidden_states2 = self.dropout2(hidden_states2)
        hidden_states2 = self.LayerNorm2(hidden_states2 + input_tensor2)
        return hidden_states1, hidden_states2

class Co_attention_block(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_rate=0.1):
        super(Co_attention_block, self).__init__()
        self.biattention = BertBiAttention(num_attention_heads, hidden_size, dropout_rate)
        self.biOutput = BertBiOutput(hidden_size, dropout_rate)
        # In clip4clip, there are FFN layers here. We can simplify by omitting them
        # or add them for more capacity. For equivalence, we should add them.
        self.v_intermediate = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU())
        self.v_output = nn.Linear(hidden_size * 4, hidden_size)
        self.v_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.v_dropout = nn.Dropout(dropout_rate)
        
        self.t_intermediate = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU())
        self.t_output = nn.Linear(hidden_size * 4, hidden_size)
        self.t_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.t_dropout = nn.Dropout(dropout_rate)

    def forward(self, image_input, image_mask, context_input, context_mask):
        # Bi-directional attention
        img_to_ctx, ctx_to_img = self.biattention(image_input, image_mask, context_input, context_mask)
        
        # Residual connection and LayerNorm (Attention part)
        attention_output_img, attention_output_ctx = self.biOutput(ctx_to_img, image_input, img_to_ctx, context_input)

        # FFN part
        intermediate_output_img = self.v_intermediate(attention_output_img)
        layer_output_img = self.v_output(intermediate_output_img)
        layer_output_img = self.v_dropout(layer_output_img)
        layer_output_img = self.v_LayerNorm(layer_output_img + attention_output_img)
        
        intermediate_output_ctx = self.t_intermediate(attention_output_ctx)
        layer_output_ctx = self.t_output(intermediate_output_ctx)
        layer_output_ctx = self.t_dropout(layer_output_ctx)
        layer_output_ctx = self.t_LayerNorm(layer_output_ctx + attention_output_ctx)
        
        return layer_output_img, layer_output_ctx
