from transformers import GPT2Model, GPT2Tokenizer
import einops
import torch
from gpt2 import TransformerSampler, ModelConfig, GenerationConfig


def convert_gpt2_weights(cfg):
    gpt2 = GPT2Model.from_pretrained('/home/ubuntu/MechInter/GPT-2/Models/gpt2')
    keys = gpt2.state_dict().keys()
    state_dict = {}
    state_dict["embed.weight"] = gpt2.wte.weight
    state_dict["pos_embed.weight"] = gpt2.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"transformer_block.{l}.attention.ln_1.weight"] = gpt2.h[l].ln_1.weight
        state_dict[f"transformer_block.{l}.attention.ln_1.bias"] = gpt2.h[l].ln_1.bias

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2.h[l].attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"transformer_block.{l}.attention.W_q"] = W_Q
        state_dict[f"transformer_block.{l}.attention.W_k"] = W_K
        state_dict[f"transformer_block.{l}.attention.W_v"] = W_V

        qkv_bias = gpt2.h[l].attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"transformer_block.{l}.attention.b_q"] = qkv_bias[0]
        state_dict[f"transformer_block.{l}.attention.b_k"] = qkv_bias[1]
        state_dict[f"transformer_block.{l}.attention.b_v"] = qkv_bias[2]

        W_O = gpt2.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"transformer_block.{l}.attention.W_o"] = W_O
        state_dict[f"transformer_block.{l}.attention.b_o"] = gpt2.h[l].attn.c_proj.bias

        state_dict[f"transformer_block.{l}.mlp.ln_2.weight"] = gpt2.h[l].ln_2.weight
        state_dict[f"transformer_block.{l}.mlp.ln_2.bias"] = gpt2.h[l].ln_2.bias

        W_in = gpt2.h[l].mlp.c_fc.weight
        state_dict[f"transformer_block.{l}.mlp.W_In"] = W_in
        state_dict[f"transformer_block.{l}.mlp.b_In"] = gpt2.h[l].mlp.c_fc.bias

        W_out = gpt2.h[l].mlp.c_proj.weight
        state_dict[f"transformer_block.{l}.mlp.W_Out"] = W_out
        state_dict[f"transformer_block.{l}.mlp.b_Out"] = gpt2.h[l].mlp.c_proj.bias

    state_dict["layernorm_final.weight"] = gpt2.ln_f.weight
    state_dict["layernorm_final.bias"] = gpt2.ln_f.bias
    
    state_dict["lm_head.weight"] = gpt2.wte.weight.T

    return state_dict

def load_gpt2_weights(model_cfg, gen_cfg):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model_cfg.vocab_size = tokenizer.vocab_size

    modified_state_dict = convert_gpt2_weights(model_cfg)
    sampler = TransformerSampler(model_cfg, gen_cfg)
    sampler.model.load_state_dict(modified_state_dict)
    return sampler

def run_inference(sampler):
    prompts = [
    "The problem with having too much money is that",
    "President Trump has projected unwavering confidence that he is winning the messaging war over the government shutdown. But behind the scenes, his team is increasingly concerned that the issue at the center of the debate will create political vulnerabilities for Republicans.",
    "Hiya, To build a strong muscular body"
    ]
    max_new_tokens = 50

    for _ in range(max_new_tokens):
        tokens = sampler.tokenizer(prompts, return_tensors='pt', truncation = True, padding = True, padding_side = 'left')
        outputs = sampler.forward(tokens)
        prompts = [prompt + output for prompt, output in zip(prompts, outputs)]

    for prompt in prompts:
        print(prompt)
        print('****************')

if __name__ == "__main__":
    model_cfg = ModelConfig()
    gen_cfg = GenerationConfig()
    sampler = load_gpt2_weights(model_cfg, gen_cfg)
    run_inference(sampler)