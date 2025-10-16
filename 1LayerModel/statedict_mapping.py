from transformers import AutoModelForCausalLM
from Utils import ModelConfig as Cfg
import einops
import torch


def convert_model_weights(cfg: Cfg, model: AutoModelForCausalLM) -> dict:
    # keys = model.state_dict().keys()
    state_dict = {}
    state_dict["embed.weight"] = model.transformer.wte.weight
    state_dict["pos_embed.weight"] = model.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"transformer_block.{l}.attention.ln_1.weight"] = model.transformer.h[l].ln_1.weight
        state_dict[f"transformer_block.{l}.attention.ln_1.bias"] = model.transformer.h[l].ln_1.bias

        W_K = model.transformer.h[l].attn.attention.k_proj.weight
        W_Q = model.transformer.h[l].attn.attention.q_proj.weight
        W_V = model.transformer.h[l].attn.attention.v_proj.weight
        state_dict[f"transformer_block.{l}.attention.W_q"] = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        state_dict[f"transformer_block.{l}.attention.W_k"] = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        state_dict[f"transformer_block.{l}.attention.W_v"] = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"transformer_block.{l}.attention.b_q"] = torch.zeros(cfg.n_heads, cfg.d_head, device=W_Q.device)
        state_dict[f"transformer_block.{l}.attention.b_k"] = torch.zeros(cfg.n_heads, cfg.d_head, device=W_K.device)
        state_dict[f"transformer_block.{l}.attention.b_v"] = torch.zeros(cfg.n_heads, cfg.d_head, device=W_V.device)

        
        W_O = model.transformer.h[l].attn.attention.out_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"transformer_block.{l}.attention.W_o"] = W_O
        state_dict[f"transformer_block.{l}.attention.b_o"] = model.transformer.h[l].attn.attention.out_proj.bias

        state_dict[f"transformer_block.{l}.mlp.ln_2.weight"] = model.transformer.h[l].ln_2.weight
        state_dict[f"transformer_block.{l}.mlp.ln_2.bias"] = model.transformer.h[l].ln_2.bias

        # state_dict[f"transformer_block.{l}.mlp.W_In"] = model.transformer.h[l].mlp.c_fc.weight
        # state_dict[f"transformer_block.{l}.mlp.b_In"] = model.transformer.h[l].mlp.c_fc.bias
        # state_dict[f"transformer_block.{l}.mlp.W_Out"] = model.transformer.h[l].mlp.c_proj.weight
        # state_dict[f"transformer_block.{l}.mlp.b_Out"] = model.transformer.h[l].mlp.c_proj.bias

        state_dict[f"transformer_block.{l}.mlp.c_fc.weight"] = model.transformer.h[l].mlp.c_fc.weight
        state_dict[f"transformer_block.{l}.mlp.c_fc.bias"] = model.transformer.h[l].mlp.c_fc.bias
        state_dict[f"transformer_block.{l}.mlp.c_proj.weight"] = model.transformer.h[l].mlp.c_proj.weight
        state_dict[f"transformer_block.{l}.mlp.c_proj.bias"] = model.transformer.h[l].mlp.c_proj.bias

    state_dict["layernorm_final.weight"] = model.transformer.ln_f.weight
    state_dict["layernorm_final.bias"] = model.transformer.ln_f.bias
    
    state_dict["lm_head.weight"] = model.lm_head.weight.T

    return state_dict