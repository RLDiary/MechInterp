import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from activation_collector import ActivationCollector
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import sys
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import SparseAutoEncoder
from trainer import SAETrainingConfig, SAETrainer
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




def compute_activations_streaming(
    model,
    dataloader,
    save_path="layer0_activations.pt",
    layer_idx=0,
    device="cuda",
    save_every=100
):
    """
    Collect activations using simple batch-by-batch saving.

    Args:
        model: The transformer model
        dataloader: DataLoader with input batches
        save_path: Path to save final merged activations file
        layer_idx: Which layer to collect from
        device: Device to run model on
        save_every: Save batch files after this many batches

    Returns:
        dict with metadata about collection
    """
    model.eval()
    model.to(device)

    # Create batch directory
    batch_dir = save_path.replace('.pt', '_batches')
    os.makedirs(batch_dir, exist_ok=True)

    # Remove existing batch files
    for f in os.listdir(batch_dir):
        if f.startswith('batch_') and f.endswith('.pt'):
            os.remove(os.path.join(batch_dir, f))

    total_batches = len(dataloader)
    batch_files = []
    total_samples = 0

    with torch.no_grad():
        with ActivationCollector(model, layer_idx=layer_idx) as collector:

            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations")):

                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass (activations collected via hook)
                _ = model(**batch)

                # Save every N batches
                if (batch_idx + 1) % save_every == 0 or (batch_idx + 1) == total_batches:
                    # Get activations
                    batch_activations = collector.get_activations(clear_after=True)

                    if batch_activations is not None:
                        # Save batch file
                        batch_file = os.path.join(batch_dir, f"batch_{len(batch_files):04d}.pt")
                        torch.save(batch_activations, batch_file)
                        batch_files.append(batch_file)
                        total_samples += batch_activations.shape[0]

                        # Report progress
                        if len(batch_files) % 10 == 0:
                            print(f"  Saved batch {len(batch_files)}: {total_samples:,} samples total")

    # Merge all batch files into final file
    print(f"\nMerging {len(batch_files)} batch files...")
    all_activations = []

    for batch_file in tqdm(batch_files, desc="Loading batches"):
        activations = torch.load(batch_file, map_location='cpu')
        all_activations.append(activations)

    # Concatenate and save
    final_activations = torch.cat(all_activations, dim=0)
    torch.save({
        "activations": final_activations,
        "shape": final_activations.shape,
        "dtype": final_activations.dtype
    }, save_path)

    # Cleanup batch files
    for batch_file in batch_files:
        os.remove(batch_file)
    os.rmdir(batch_dir)

    print(f"\n✓ Collection complete!")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Shape: {final_activations.shape}")
    print(f"  Saved to: {save_path}")

    return {
        "save_path": save_path,
        "total_samples": total_samples,
        "layer_idx": layer_idx,
        "shape": final_activations.shape
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

def get_dataset(dataset_path):
    dataset = load_dataset(dataset_path, split='train')
    dataset = dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)
    train_dataset, val_dataset = dataset['train'], dataset['test']

    for ds in [train_dataset, val_dataset]:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return train_dataset, val_dataset

if __name__ == "__main__":
    model, tokenizer, sampler = load_model(
        model_path="/home/ubuntu/MechInter/OneLayerModel/Models/TinyStories-1L-21M",
        config_path="/home/ubuntu/MechInter/OneLayerModel/OneLM.yaml"
        )
    sae_model = SparseAutoEncoder(n_latents=16384, n_inputs=4096, activation=nn.ReLU(), tied=False, normalize=False).to(device)
    trainer = SAETrainer(config=SAETrainingConfig(), autoencoder=sae_model, language_model=model, tokenizer=tokenizer, sampler=sampler, use_wandb=True)

    train_dataset, val_dataset = get_dataset('/home/ubuntu/MechInter/datasets/cache')
    trainer.train(train_dataset=train_dataset, val_dataset=val_dataset)