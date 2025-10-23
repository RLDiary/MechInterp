import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
import re

from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import load_dataset
from typing import Optional, List
import gc
from pprint import pprint
import torch
import sys
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from Utils import TransformerSampler, ModelConfig, GenerationConfig
from OneLayerModel.model import Model
from OneLayerModel.statedict_mapping import convert_model_weights

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

    def get_activations(self, clear_after: bool = False) -> Optional[torch.Tensor]:
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

#########################################
# PART 1: Preprocess on CPU (multi-proc OK)
#########################################

def chunk_and_pad(batch, max_length=2048):
    """Split and pad input_ids sequences into chunks, CPU only."""
    input_ids_chunks = []
    attention_mask_chunks = []

    for tokens in batch["input_ids"]:
        for i in range(0, len(tokens), max_length):
            chunk_ids = tokens[i:i + max_length]
            if not chunk_ids:
                continue
            input_ids_chunks.append(chunk_ids)
            attention_mask_chunks.append([1] * len(chunk_ids))

    # Pad all chunks to max length in this batch
    max_len = max(len(x) for x in input_ids_chunks)
    input_ids_padded = [x + [0] * (max_len - len(x)) for x in input_ids_chunks]
    attention_mask_padded = [x + [0] * (max_len - len(x)) for x in attention_mask_chunks]

    # Return as 2D tensors, NOT flattened
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded
    }


#########################################
# PART 2: Fast Memory-Mapped Streaming
#########################################

class ActivationWriter:
    """Memory-mapped file writer for fast incremental activation saves"""
    
    def __init__(self, filepath, hidden_dim, estimated_samples=10_000_000, dtype=np.float32):
        """
        Args:
            filepath: Path to save activations
            hidden_dim: Dimension of activation vectors
            estimated_samples: Estimated total samples (pre-allocates space)
            dtype: Data type for storage
        """
        self.filepath = filepath
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.current_idx = 0
        
        # Pre-allocate memory-mapped file
        self.memmap = np.memmap(
            filepath,
            dtype=dtype,
            mode='w+',
            shape=(estimated_samples, hidden_dim)
        )
        print(f"Pre-allocated {estimated_samples:,} x {hidden_dim} array "
              f"({estimated_samples * hidden_dim * np.dtype(dtype).itemsize / 1024**3:.2f} GB)")
    
    def append(self, activations):
        """
        Append activations to memory-mapped file.
        
        Args:
            activations: Tensor of shape (num_samples, hidden_dim)
        """
        # Convert to numpy if needed
        if torch.is_tensor(activations):
            activations = activations.cpu().numpy()
        
        num_samples = activations.shape[0]
        end_idx = self.current_idx + num_samples
        
        # Check if we need to expand
        if end_idx > self.memmap.shape[0]:
            self._expand(end_idx)
        
        # Write directly to memory-mapped file
        self.memmap[self.current_idx:end_idx] = activations
        self.current_idx = end_idx
        
        # Flush to disk periodically (every ~100MB)
        if self.current_idx % 10000 == 0:
            self.memmap.flush()
    
    def _expand(self, required_size):
        """Expand memory-mapped file if needed"""
        new_size = max(required_size, int(self.memmap.shape[0] * 1.5))
        print(f"Expanding memmap from {self.memmap.shape[0]:,} to {new_size:,} samples")
        
        # Create new larger memmap
        new_memmap = np.memmap(
            self.filepath + '.tmp',
            dtype=self.dtype,
            mode='w+',
            shape=(new_size, self.hidden_dim)
        )
        
        # Copy existing data
        new_memmap[:self.current_idx] = self.memmap[:self.current_idx]
        
        # Cleanup and replace
        del self.memmap
        os.remove(self.filepath)
        os.rename(self.filepath + '.tmp', self.filepath)
        self.memmap = new_memmap
    
    def finalize(self):
        """Trim file to actual size and save metadata"""
        print(f"Finalizing: trimming to {self.current_idx:,} samples")
        
        # Trim to actual size
        final_data = self.memmap[:self.current_idx].copy()
        del self.memmap
        
        # Save as pytorch tensor with metadata
        torch.save({
            "activations": torch.from_numpy(final_data),
            "shape": final_data.shape,
            "dtype": torch.float32
        }, self.filepath)
        
        print(f"Saved {self.current_idx:,} activations to {self.filepath}")
        return self.current_idx
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'memmap'):
            del self.memmap


def compute_activations_streaming(
    model, 
    dataloader, 
    save_path="layer0_activations.pt",
    layer_idx=0,
    device="cuda",
    save_every=25,  # Save after every N batches
    estimated_samples=10_000_000  # Estimate total samples for pre-allocation
):
    """
    Collect activations using memory-mapped files for fast streaming.
    
    Args:
        model: The transformer model
        dataloader: DataLoader with input batches
        save_path: Path to save activations file
        layer_idx: Which layer to collect from
        device: Device to run model on
        save_every: Flush to disk after this many batches
        estimated_samples: Estimated total samples (for pre-allocation)
        
    Returns:
        dict with metadata about collection
    """
    model.eval()
    model.to(device)
    
    # Remove existing file
    if os.path.exists(save_path):
        print(f"Warning: {save_path} exists. It will be overwritten.")
        os.remove(save_path)
    
    # Get hidden dim from first batch
    print("Getting hidden dimension from first batch...")
    first_batch = next(iter(dataloader))
    first_batch = {k: v.to(device) for k, v in first_batch.items()}
    
    with torch.no_grad():
        with ActivationCollector(model, layer_idx=layer_idx) as temp_collector:
            _ = model(**first_batch)
            first_activations = temp_collector.get_activations()
            hidden_dim = first_activations.shape[1]
    
    print(f"Hidden dimension: {hidden_dim}")
    
    # Initialize memory-mapped writer
    writer = ActivationWriter(
        save_path + '.npy',  # Temporary numpy file
        hidden_dim=hidden_dim,
        estimated_samples=estimated_samples,
        dtype=np.float32
    )
    
    total_batches = len(dataloader)
    
    with torch.no_grad():
        with ActivationCollector(model, layer_idx=layer_idx) as collector:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations")):
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass (activations collected via hook)
                _ = model(**batch)
                
                # Save periodically to manage memory
                if (batch_idx + 1) % save_every == 0 or (batch_idx + 1) == total_batches:
                    # Get activations
                    batch_activations = collector.get_activations(clear_after=False)
                    
                    if batch_activations is not None:
                        # Fast append to memmap (no loading/concatenating!)
                        writer.append(batch_activations)
                        
                        # Clear memory
                        collector.clear_activations()
                        
                        # Report progress
                        if (batch_idx + 1) % (save_every * 10) == 0:
                            print(f"  Batch {batch_idx + 1}/{total_batches}: "
                                  f"{writer.current_idx:,} samples written")
    
    # Finalize: convert numpy memmap to pytorch tensor file
    total_samples = writer.finalize()
    
    # Cleanup the temporary numpy file
    if os.path.exists(save_path + '.npy'):
        os.remove(save_path + '.npy')
    
    print(f"\n✓ Collection complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Saved to: {save_path}")
    
    return {
        "save_path": save_path,
        "total_samples": total_samples,
        "layer_idx": layer_idx
    }


def load_activations(filepath="layer0_activations.pt"):
    """Load activations from disk."""
    data = torch.load(filepath)
    print(f"Loaded activations with shape {data['shape']}")
    return data["activations"]


def get_activation_info(filepath="layer0_activations.pt"):
    """Get info about saved activations without loading them fully."""
    data = torch.load(filepath, map_location='cpu')
    return {
        "shape": data["shape"],
        "dtype": data["dtype"],
        "num_samples": data["shape"][0],
        "hidden_dim": data["shape"][1],
        "file_size_mb": os.path.getsize(filepath) / (1024 ** 2)
    }

def load_model(model_path, config_path):
    sampler = TransformerSampler()
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        model_config = ModelConfig(config_dict)
    modified_state_dict = convert_model_weights(model_config, model)
    model = Model(model_config).to(device)
    model.load_state_dict(modified_state_dict)
    model.eval()
    print(f"Model loaded successfully")
    return model, tokenizer, sampler

def load_dataloader(dataset_path):
    dataset = load_dataset(dataset_path, split='train')
    chunked_dataset = dataset.map(
        chunk_and_pad,
        batched=True,
        num_proc=8,
        batch_size=64,
        remove_columns=dataset.column_names,
        desc="Chunking and padding"
    )

    chunked_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(
        chunked_dataset, 
        batch_size=25, 
        shuffle=False,
        pin_memory=True  # Faster CPU->GPU transfer
    )
    return dataloader



if __name__ == "__main__":
    model, tokenizer, sampler = load_model(
        model_path="/home/ubuntu/MechInter/OneLayerModel/Models/TinyStories-1L-21M",
        config_path="/home/ubuntu/MechInter/OneLayerModel/OneLM.yaml"
        )
    dataloader = load_dataloader('/home/ubuntu/MechInter/datasets/cache')
    result = compute_activations_streaming(
                                        model=model,
                                        dataloader=dataloader,
                                        save_path="layer0_activations.pt",
                                        layer_idx=0,
                                        device="cuda",
                                        save_every=25,
                                        estimated_samples=50_000_000  # Adjust based on your dataset size
                                    )

    print(f"\nActivations saved to: {result['save_path']}")
    print(f"Total samples: {result['total_samples']:,}")