from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 1
    d_MLP: int = 4096
    d_head: int = 64
    max_ctx: int = 2048
    init_range: float = 0.02
    var_epsilon: float = 1e-05
    resid_dropout: float = 0.0
    
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)