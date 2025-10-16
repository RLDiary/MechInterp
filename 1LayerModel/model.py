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

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(data=torch.ones(cfg.d_model, device = device))
        self.bias = nn.Parameter(torch.zeros(cfg.d_model, device = device))
        
    
    def forward(self, resid_stream: Tensor):
        mean = torch.mean(resid_stream, dim = -1, keepdim = True) # B, Seq, 1
        std = (resid_stream.var(dim = -1, keepdim = True, unbiased = False) + self.cfg.var_epsilon).sqrt()
        norm = (resid_stream - mean) / std # B, Seq, d_model
        return norm * self.weight + self.bias

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model, device = device))
        nn.init.normal_(self.weight, std = cfg.init_range)
    
    def forward(self, idx):
        return self.weight[idx]

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.max_ctx, cfg.d_model, device = device))
        nn.init.normal_(self.weight, std = cfg.init_range)
    
    def forward(self, attn_mask):
        position_ids = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)  # (B, S)
        return self.weight[position_ids]  # (B, S, d_model)
        
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", torch.tensor(float("-inf"), dtype=torch.float32, device=device), persistent=False)
        self.ln_1 = LayerNorm(cfg)
        self.scale_factor = cfg.d_head ** -0.5
        self.W_k = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head, device = device))
        self.W_q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head, device = device))
        self.W_v = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head, device = device))
        self.W_o = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model, device = device))
        for t in [self.W_k, self.W_q, self.W_v, self.W_o]:
            nn.init.normal_(t, std = cfg.init_range)
        self.b_k = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head, device = device))
        self.b_q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head, device = device))
        self.b_v = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head, device = device))
        self.b_o = nn.Parameter(torch.zeros(cfg.d_model, device = device))
        
    def apply_causal_mask(self, attn_scores):
        mask = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device = attn_scores.device)
        mask = torch.triu(mask, diagonal = 1).bool()
        attn_scores = attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores

    def forward(self, resid_stream: Tensor, attn_mask: Tensor):
        layer_normalised = self.ln_1(resid_stream)
        q = einops.einsum(layer_normalised, self.W_q, "B s_q d_model, n_heads d_model d_head -> B s_q n_heads d_head") + self.b_q
        k = einops.einsum(layer_normalised, self.W_k, "B s_k d_model, n_heads d_model d_head -> B s_k n_heads d_head") + self.b_k
        v = einops.einsum(layer_normalised, self.W_v, "B s_k d_model, n_heads d_model d_head -> B s_k n_heads d_head") + self.b_v
        attn_scores = einops.einsum(q, k, " B s_q n_heads d_head, B s_k n_heads d_head -> B n_heads s_q s_k") * self.scale_factor
        attn_scores = self.apply_causal_mask(attn_scores)
        attn_pattern = attn_scores.softmax(dim = -1)
        z = einops.einsum(attn_pattern, v, "B n_heads s_q s_k, B s_k n_heads d_head -> B s_q n_heads d_head")
        attn_out = einops.einsum(z, self.W_o, "B s_q n_heads d_head, n_heads d_head d_model   -> B s_q d_model") + self.b_o
        return attn_out
    
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln_2 = LayerNorm(cfg)
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_MLP)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(cfg.d_MLP, cfg.d_model)
        self.dropout = nn.Dropout(float(cfg.resid_dropout))

    def forward(self, resid_stream: Tensor):
        layer_normalised = self.ln_2(resid_stream)
        layer_one = self.c_fc(layer_normalised)
        non_linear_activated = self.gelu(layer_one)
        layer_two = self.c_proj(non_linear_activated)
        hidden_state = self.dropout(layer_two)
        return hidden_state

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = Attention(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_stream: Tensor, attn_mask: Tensor):
        attn_out = self.attention(resid_stream, attn_mask)
        resid_mid_stream = resid_stream + attn_out
        mlp_out = self.mlp(resid_mid_stream)
        return mlp_out + resid_mid_stream

class UnEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size = (cfg.d_model, cfg.vocab_size), device = device))
        nn.init.normal_(self.weight, std = cfg.init_range)
    
    def forward(self, resid: Tensor):
        return einops.einsum(resid, self.weight, "B S d_model, d_model vocab_size -> B S vocab_size")


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.transformer_block = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.layernorm_final = LayerNorm(cfg)
        self.lm_head = UnEmbed(cfg)
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: 
        resid_stream = self.embed(input_ids) + self.pos_embed(attention_mask)  # B, Seq Length, d_model
        for block in self.transformer_block:
            resid_stream = block(resid_stream, attention_mask)
        layer_normalised = self.layernorm_final(resid_stream)
        unembed = self.lm_head(layer_normalised) # B, Seq Length, Vocab_size
        return unembed