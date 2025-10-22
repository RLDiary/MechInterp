import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import einops
from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# As implemented here - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py
def _4d_causal_attention_mask(attention_mask: torch.Tensor):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
    """
    sequence_length = attention_mask.shape[-1]
    batch_size = attention_mask.shape[0]
    dtype = torch.float32

    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=attention_mask.device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
            causal_mask.device
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    return causal_mask


class Attention(nn.Module):
    def __init__(self, config, layer_id=None, attention_type = 'global'):
        super().__init__()
        self.config = config

        bias = torch.tril(torch.ones((config.max_ctx, config.max_ctx), dtype=bool)).view(
            1, 1, config.max_ctx, config.max_ctx
        )
        self.register_buffer("bias", bias, persistent=False)
        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))
        self.layer_id = layer_id
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def apply_causal_mask(self, attn_weights, seq_length):
        causal_mask = self.bias[:, :, : seq_length, :seq_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        return attn_weights
    
    def apply_attention_mask(self, attn_weights, attention_mask, seq_length):
        causal_mask = attention_mask[:, :, :, : seq_length]
        attn_weights = attn_weights + causal_mask
        return attn_weights
    
    def _attn(self, query, key, value, attention_mask=None):
        # Convert query and key to float32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        
        # Compute attention weights (B, n_heads, seq_length, seq_length)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        # Apply causal mask
        seq_length = query.size(-2)
        attn_weights = self.apply_causal_mask(attn_weights, seq_length)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = self.apply_attention_mask(attn_weights, attention_mask, seq_length)
        
        # Softmax attention weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask=None,):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query, self.config.n_heads, self.config.d_head)
        key = self._split_heads(key, self.config.n_heads, self.config.d_head)
        value = self._split_heads(value, self.config.n_heads, self.config.d_head)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        attn_output = self._merge_heads(attn_output, self.config.n_heads, self.config.d_head)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output
    
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_MLP)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(cfg.d_MLP, cfg.d_model)
        self.dropout = nn.Dropout(float(cfg.resid_dropout))

    def forward(self, resid_stream: Tensor):
        layer_one = self.c_fc(resid_stream)
        non_linear_activated = self.gelu(layer_one)
        layer_two = self.c_proj(non_linear_activated)
        hidden_state = self.dropout(layer_two)
        return hidden_state

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id=None):
        super().__init__()
        self.cfg = cfg
        self.ln_1 = nn.LayerNorm(cfg.d_model, eps=cfg.var_epsilon)
        self.attention = Attention(cfg, layer_id)
        self.ln_2 = nn.LayerNorm(cfg.d_model, eps=cfg.var_epsilon)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_stream: Tensor, attn_mask: Tensor):
        layer_normalised = self.ln_1(resid_stream)
        causal_mask = _4d_causal_attention_mask(attn_mask.to(torch.float32))
        attn_out = self.attention(layer_normalised, causal_mask)
        resid_mid_stream = resid_stream + attn_out
        layer_normalised = self.ln_2(resid_mid_stream)
        mlp_out = self.mlp(layer_normalised)
        resid_stream = resid_mid_stream + mlp_out
        return resid_stream


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_ctx, cfg.d_model)
        self.transformer_block = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.layernorm_final = nn.LayerNorm(cfg.d_model, eps=cfg.var_epsilon)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: 
        token_embeddings = self.embed(input_ids)
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)  # (B, S)
        position_embeddings = self.pos_embed(position_ids)  # B, Seq Length, d_model
        resid_stream = token_embeddings + position_embeddings

        
        for block in self.transformer_block:
            resid_stream = block(resid_stream, attention_mask)
        layer_normalised = self.layernorm_final(resid_stream)
        unembed = self.lm_head(layer_normalised) # B, Seq Length, Vocab_size
        return unembed