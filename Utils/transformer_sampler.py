from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformerSampler:
    def __init__(self, max_new_tokens: Optional[int] = 20):
        self.max_new_tokens = max_new_tokens
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def decode(self, logits: torch.Tensor, sampling_strategy: Optional[str] = 'basic', temperature: Optional[float] = 0.0) -> torch.Tensor:
        final_logits = logits[:,-1,:] # B, Vocab_size
        chosen_tokens = self.sample_next_token(final_logits, temperature) # B
        return chosen_tokens
    
    def sample_next_token(self, logits, temperature):
        if temperature > 0:
            rescaled_logits = self.temperature_scaling(logits, temperature)
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