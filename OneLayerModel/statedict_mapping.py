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
        state_dict[f"transformer_block.{l}.ln_1.weight"] = model.transformer.h[l].ln_1.weight
        state_dict[f"transformer_block.{l}.ln_1.bias"] = model.transformer.h[l].ln_1.bias

        state_dict[f"transformer_block.{l}.attention.k_proj.weight"] = model.transformer.h[l].attn.attention.k_proj.weight
        state_dict[f"transformer_block.{l}.attention.q_proj.weight"] = model.transformer.h[l].attn.attention.q_proj.weight
        state_dict[f"transformer_block.{l}.attention.v_proj.weight"] = model.transformer.h[l].attn.attention.v_proj.weight
        state_dict[f"transformer_block.{l}.attention.out_proj.weight"] = model.transformer.h[l].attn.attention.out_proj.weight
        state_dict[f"transformer_block.{l}.attention.out_proj.bias"] = model.transformer.h[l].attn.attention.out_proj.bias

        
        
        state_dict[f"transformer_block.{l}.ln_2.weight"] = model.transformer.h[l].ln_2.weight
        state_dict[f"transformer_block.{l}.ln_2.bias"] = model.transformer.h[l].ln_2.bias

        state_dict[f"transformer_block.{l}.mlp.c_fc.weight"] = model.transformer.h[l].mlp.c_fc.weight
        state_dict[f"transformer_block.{l}.mlp.c_fc.bias"] = model.transformer.h[l].mlp.c_fc.bias
        state_dict[f"transformer_block.{l}.mlp.c_proj.weight"] = model.transformer.h[l].mlp.c_proj.weight
        state_dict[f"transformer_block.{l}.mlp.c_proj.bias"] = model.transformer.h[l].mlp.c_proj.bias

    state_dict["layernorm_final.weight"] = model.transformer.ln_f.weight
    state_dict["layernorm_final.bias"] = model.transformer.ln_f.bias
    
    state_dict["lm_head.weight"] = model.lm_head.weight

    return state_dict