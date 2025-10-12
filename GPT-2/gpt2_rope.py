from dataclasses import dataclass
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import einops
from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from torchtyping import TensorType

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class GenerationConfig:
    temperature = 0.5


@dataclass
class ModelConfig:
    # GPT-2 Small
    d_model = 768
    n_heads = 12
    n_layers = 12

    # GPT-2 XL
    # d_model = 1600
    # n_heads = 25
    # n_layers = 48

    d_MLP = 4 * d_model
    d_head = int(d_model / n_heads)
    max_ctx = 1024
    init_range = 0.02
    var_epsilon = 1e-05

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

class RoPEEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_head

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        return cos, sin

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        q_cos, q_sin = cos[:, :q.size(1)], sin[:, :q.size(1)]
        k_cos, k_sin = cos[:, :k.size(1)], sin[:, :k.size(1)]

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed
        
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

    def forward(self, resid_stream: Tensor, attn_mask: Tensor, rope_cos: Tensor, rope_sin: Tensor):
        layer_normalised = self.ln_1(resid_stream)
        q = einops.einsum(layer_normalised, self.W_q, "B s_q d_model, n_heads d_model d_head -> B s_q n_heads d_head") + self.b_q
        k = einops.einsum(layer_normalised, self.W_k, "B s_k d_model, n_heads d_model d_head -> B s_k n_heads d_head") + self.b_k
        v = einops.einsum(layer_normalised, self.W_v, "B s_k d_model, n_heads d_model d_head -> B s_k n_heads d_head") + self.b_v
        q, k = RoPEEmbedding.apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
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
        self.W_In = nn.Parameter(torch.empty(cfg.d_model, cfg.d_MLP, device=device))
        self.b_In = nn.Parameter(torch.zeros(cfg.d_MLP, device=device))
        self.gelu = nn.GELU()
        self.W_Out = nn.Parameter(torch.empty(cfg.d_MLP, cfg.d_model, device=device))
        self.b_Out = nn.Parameter(torch.zeros(cfg.d_model, device=device))
        nn.init.normal_(self.W_In, std = cfg.init_range)
        nn.init.normal_(self.W_Out, std = cfg.init_range)

    
    def forward(self, resid_stream: Tensor):
        layer_normalised = self.ln_2(resid_stream)
        layer_one = einops.einsum(layer_normalised, self.W_In, "B Seq d_model, d_model d_MLP -> B Seq d_MLP") + self.b_In
        non_linear_activated = self.gelu(layer_one)
        layer_two = einops.einsum(non_linear_activated, self.W_Out, f"B Seq d_MLP, d_MLP d_model -> B Seq d_model") + self.b_Out
        return layer_two

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = Attention(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, resid_stream: Tensor, attn_mask: Tensor, rope_cos: Tensor, rope_sin: Tensor):
        attn_out = self.attention(resid_stream, attn_mask, rope_cos, rope_sin)
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


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.rope = RoPEEmbedding(cfg)
        self.transformer_block = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.layernorm_final = LayerNorm(cfg)
        self.lm_head = UnEmbed(cfg)
    
    def forward(self, input_ids: Tensor, attn_mask: Tensor) -> Tensor:
        resid_stream = self.embed(input_ids)  # B, Seq Length, d_model
        seq_len = input_ids.size(1)
        rope_cos, rope_sin = self.rope(seq_len, input_ids.device)

        for block in self.transformer_block:
            resid_stream = block(resid_stream, attn_mask, rope_cos, rope_sin)
        layer_normalised = self.layernorm_final(resid_stream)
        unembed = self.lm_head(layer_normalised) # B, Seq Length, Vocab_size
        return unembed
    
class TransformerSampler:
    def __init__(self, model_cfg, gen_cfg, tokenizer: Optional[GPT2Tokenizer] = None, model: Optional[GPT2] = None):
        self.model_cfg = model_cfg
        self.gen_cfg = gen_cfg
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2') if tokenizer is None else tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_cfg.vocab_size = self.tokenizer.vocab_size
        self.model = GPT2(self.model_cfg) if model is None else model
    
    def forward(self, tokens: torch.Tensor):
        input_ids = tokens.input_ids.to(device) # B, Max Seq Length
        attn_mask = tokens.attention_mask.to(device) # B, Max Seq Length
        logits = self.model.forward(input_ids, attn_mask) # B, S, Vocab_size
        final_logits = logits[:,-1,:] # B, Vocab_size
        chosen_tokens = self.sample_next_token(final_logits)
        return self.tokenizer.batch_decode(chosen_tokens)
    
    def sample_next_token(self, logits):
        if self.gen_cfg.temperature > 0:
            rescaled_logits = self.temperature_scaling(logits, self.gen_cfg.temperature)
            return self.basic_gen(rescaled_logits)
        else:
            return self.basic_gen(logits)

    @staticmethod
    def greedy_decode(logits):
        return logits.argmax(dim = -1, keepdim = True)
    
    @staticmethod
    def basic_gen(logits):
        categorical = torch.distributions.categorical.Categorical(logits = logits)
        return  categorical.sample()
    
    @staticmethod
    def temperature_scaling(logits, temperature):
        return logits / temperature


    



if __name__== '__main__':
    model_cfg = ModelConfig()
    gen_cfg = GenerationConfig()
    sampler = TransformerSampler(model_cfg, gen_cfg)
    prompts = ['Hi there', 'how are you? is everything good?', 'trying a different prompt altogether to see how this works']
    tokens = sampler.tokenizer(prompts, return_tensors='pt', truncation = True, padding = True, padding_side = 'left')
    output = sampler.forward(tokens)
    print(output)
    
    print('All done')