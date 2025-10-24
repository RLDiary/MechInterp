from typing import Optional, List
import torch
import gc

def create_activation_hook(collector):
    """Factory function to create a hook that captures activations"""
    def hook(module, input, output):
        # Handle both tensor and tuple outputs
        if isinstance(output, tuple):
            activations = output[0]
        else:
            activations = output
        
        # Detach and move to CPU to prevent memory leaks
        activations = activations.detach()
        
        # Convert to float32 if needed (more efficient to do before CPU transfer for some dtypes)
        if activations.dtype != torch.float32:
            activations = activations.float()
        
        # Move to CPU after float conversion
        activations = activations.cpu()
        
        # Flatten: (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
        flat_activations = activations.view(-1, activations.size(-1))
        
        collector.activations.append(flat_activations)
    return hook


class ActivationCollector:
    """Collects activations from a specific layer during forward pass"""

    def __init__(self, model, layer_idx: int = 0, max_samples: Optional[int] = None):
        """
        Args:
            model: The transformer model
            layer_idx: Index of the layer to collect activations from
            max_samples: Maximum number of activation vectors to store (prevents OOM)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.activations: List[torch.Tensor] = []
        self.hook = None
        self.max_samples = max_samples
        self._total_samples = 0
        self.register_hook()

    def register_hook(self):
        """Register forward hook on the activations in the MLP layer"""
        try:
            mlp_layer = self.model.transformer_block[self.layer_idx].mlp
            target_layer = mlp_layer.gelu
            
            # Use the factory function to create the hook
            hook_fn = create_activation_hook(self)
            self.hook = target_layer.register_forward_hook(hook_fn)
            print(f"Hook registered on layer {self.layer_idx}")
            
        except (AttributeError, IndexError) as e:
            raise ValueError(
                f"Could not access layer at index {self.layer_idx}. "
                f"Check your model architecture. Error: {e}"
            )

    def clear_activations(self):
        """Clear stored activations and free memory"""
        self.activations.clear()
        self._total_samples = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_activations(self, clear_after: bool = True) -> Optional[torch.Tensor]:
        """
        Return concatenated activations
        
        Args:
            clear_after: If True, clear activations after returning them
            
        Returns:
            Concatenated tensor of shape (total_samples, hidden_dim) or None if empty
        """
        if not self.activations:
            return None
        
        concatenated = torch.cat(self.activations, dim=0)
        
        if clear_after:
            self.clear_activations()
        
        return concatenated

    def get_statistics(self) -> dict:
        """Get statistics about collected activations"""
        if not self.activations:
            return {
                "num_batches": 0,
                "total_samples": 0,
                "shape": None,
                "memory_mb": 0
            }
        
        total_samples = sum(act.shape[0] for act in self.activations)
        hidden_dim = self.activations[0].shape[-1]
        memory_bytes = sum(act.element_size() * act.nelement() for act in self.activations)
        
        return {
            "num_batches": len(self.activations),
            "total_samples": total_samples,
            "shape": (total_samples, hidden_dim),
            "memory_mb": memory_bytes / (1024 ** 2)
        }

    def is_full(self) -> bool:
        """Check if max_samples limit has been reached"""
        if self.max_samples is None:
            return False
        self._total_samples = sum(act.shape[0] for act in self.activations)
        return self._total_samples >= self.max_samples

    def remove_hook(self):
        """Remove the forward hook"""
        if self.hook:
            self.hook.remove()
            self.hook = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures hook is removed"""
        self.remove_hook()
        return False

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.remove_hook()