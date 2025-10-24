import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, Any, Callable
from model import SparseAutoEncoder
from activation_collector import ActivationCollector
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from Utils import TransformerSampler
from transformers import AutoTokenizer

@dataclass
class SAETrainingConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 25
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    wandb_project: str = "1L21M SAE Training"
    wandb_name: Optional[str] = None
    l1_coefficient: float = 1e-3
    grad_accumulation_steps: int = 6
    log_every: int = 10
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    dead_neuron_threshold: int = 10000

class DataCollator:
    def __init__(self, max_length = 128):
        self.max_length = max_length

    def __call__(self, batch):
        """Split input_ids sequences into chunks of max_length"""
        input_ids_chunks = []

        for sample in batch:
            tokens = sample['input_ids']
            for i in range(0, len(tokens), self.max_length):
                chunk_ids = tokens[i:i + self.max_length]
                input_ids_chunks.append(chunk_ids)

        # Filter out chunks that are not of the same length as the max length of the batch
        batch_max_len = max(len(x) for x in input_ids_chunks)
        input_ids = [x for x in input_ids_chunks if len(x) == batch_max_len]
        attention_mask = [torch.ones(len(x), dtype=torch.long) for x in input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class SAETrainer:
    def __init__(
        self,
        config: SAETrainingConfig,
        autoencoder: SparseAutoEncoder,
        language_model: nn.Module,
        tokenizer: AutoTokenizer,
        sampler: TransformerSampler,
        use_wandb: bool = False
    ):
        self.config = config
        self.autoencoder = autoencoder.to(config.device)
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.collector = ActivationCollector(self.language_model, layer_idx=0)
        self.sampler = sampler
        self.data_collator = DataCollator(max_length=128)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=config.lr)
        
        # Training state
        self.next_checkpoint_idx = 0
        self.total_loss = 0
        self.current_step = 0
        self.current_epoch = 0

        # Create checkpoints directory
        os.makedirs("SAE/Checkpoints", exist_ok=True)

        # Initialize wandb
        self.use_wandb = use_wandb
        if self.use_wandb and config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name,
                config={
                    "learning_rate": config.lr,
                    "weight_decay": config.weight_decay,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "l1_coefficient": config.l1_coefficient,
                    "n_latents": self.autoencoder.n_latents,
                    "n_inputs": self.autoencoder.n_inputs,
                    "tied_weights": self.autoencoder.tied,
                    "normalize_activations": self.autoencoder.normalize,
                }
            )
            wandb.watch(self.autoencoder, log="all", log_freq=config.log_every)

    def compute_loss(
        self,
        activations: torch.Tensor,
        reconstructions: torch.Tensor,
        latents: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAE loss components

        Args:
            activations: Original activations [batch, n_inputs]
            reconstructions: Reconstructed activations [batch, n_inputs]
            latents: SAE latent activations [batch, n_latents]

        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructions, activations)

        # L1 sparsity loss on latents
        l1_loss = torch.mean(torch.abs(latents))

        # Total loss
        total_loss = reconstruction_loss + self.config.l1_coefficient * l1_loss

        # Compute additional metrics
        with torch.no_grad():
            # Fraction of latents that are active (non-zero)
            active_latents = (latents != 0).float().mean()

            # L0 norm (number of active latents per sample)
            l0_norm = (latents != 0).float().sum(dim=-1).mean()

            # Variance explained
            variance_original = torch.var(activations, dim=0).mean()
            variance_residual = torch.var(activations - reconstructions, dim=0).mean()
            variance_explained = 1 - (variance_residual / variance_original)

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "l1_loss": l1_loss,
            "active_latents": active_latents,
            "l0_norm": l0_norm,
            "variance_explained": variance_explained,
        }

    def step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single training step"""

        input_ids = torch.stack(batch['input_ids']).to(self.config.device)
        attention_mask = torch.stack(batch['attention_mask']).to(self.config.device)

        # Forward pass

        _ = self.language_model(input_ids, attention_mask)
        activations = self.collector.get_activations()
        activations = activations.to(self.config.device)
        latents_pre_act, latents, reconstructions = self.autoencoder(activations)

        # Compute loss
        loss_dict = self.compute_loss(activations, reconstructions, latents)

        return loss_dict

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.autoencoder.eval()
        eval_losses = {}

        with torch.no_grad():
            for batch in eval_dataloader:
                loss_dict = self.step(batch)

                for key, value in loss_dict.items():
                    if key not in eval_losses:
                        eval_losses[key] = []
                    eval_losses[key].append(value.item())

        self.autoencoder.train()
        return {key: np.mean(values) for key, values in eval_losses.items()}

    def _evaluate_dead_neurons(self):
        """Evaluate and log dead neurons statistics"""
        with torch.no_grad():
            dead_neurons = (self.autoencoder.stats_last_nonzero > self.config.dead_neuron_threshold).sum().item()
            total_neurons = self.autoencoder.stats_last_nonzero.numel()
            dead_fraction = dead_neurons / total_neurons

            if self.use_wandb:
                wandb.log({
                    "dead_neurons/count": dead_neurons,
                    "dead_neurons/fraction": dead_fraction,
                    "dead_neurons/total": total_neurons,
                }, step=self.current_step)

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str = ""):
        """Log metrics to wandb"""
        if not self.use_wandb:
            return

        log_dict = {}
        for key, value in metrics.items():
            log_key = f"{prefix}/{key}" if prefix else key
            log_dict[log_key] = value.item() if isinstance(value, torch.Tensor) else value

        log_dict["step"] = self.current_step
        log_dict["epoch"] = self.current_epoch
        log_dict["learning_rate"] = self.optimizer.param_groups[0]['lr']

        wandb.log(log_dict, step=self.current_step)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ):
        """Main training loop"""
        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=self.data_collator)
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=self.data_collator)
        
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Val dataset length: {len(val_dataset)}")
        print(f"Train dataloader length: {len(train_dataloader)}")
        print(f"Val dataloader length: {len(val_dataloader)}")

        # Setup scheduler
        self.total_steps = len(train_dataloader) * self.config.epochs / self.config.grad_accumulation_steps
        warmup_steps = int(self.config.warmup_ratio * self.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, self.total_steps
        )
        
        self.checkpoint_intervals = [int(self.total_steps * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
        next_checkpoint_idx = 0

        print(f"Training for {self.config.epochs} epochs, {self.total_steps} total steps")
        print(f"Warmup steps: {warmup_steps}")

        # Training loop
        for epoch in range(self.config.epochs):
            self.autoencoder.train()
            epoch_losses = {}
            progress_bar = tqdm(total=self.total_steps, desc=f'Epoch {self.current_epoch}')

            for batch_idx, batch in enumerate(train_dataloader):
                # Forward pass
                loss_dict = self.step(batch)
                loss = loss_dict["total_loss"]

                # Accumulate losses for logging
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    if scheduler:
                        scheduler.step()
                    self.optimizer.zero_grad()
                    self.current_step += 1
                    

                    # Logging
                    if self.current_step % self.config.log_every == 0:
                        self._log_metrics(loss_dict, prefix="train")

                    # Eval and Checkpointing
                    if next_checkpoint_idx < len(self.checkpoint_intervals) and self.current_step >= self.checkpoint_intervals[next_checkpoint_idx]:
                        self.save_checkpoint()
                        next_checkpoint_idx += 1
                        if val_dataloader:
                            val_losses = self.evaluate(val_dataloader)
                            self._log_metrics(val_losses, prefix="val")
                            progress_bar.set_postfix({
                                'val_loss': f'{val_losses["total_loss"]:.4f}',
                                'step': self.current_step
                            })
                        self._evaluate_dead_neurons()
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'train_loss': f'{loss.item():.4f}',
                        'avg_train_loss': f'{np.mean(epoch_losses["total_loss"]):.4f}',
                        'step': self.current_step
                    })

                

        print("Training completed!")

        # Save final model
        self.save_checkpoint(final=True)

        progress_bar.close()
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        suffix = "final" if final else f"step_{self.current_step}"
        checkpoint_path = f"SAE/checkpoints/sae_checkpoint_{suffix}.pt"

        checkpoint = {
            'model_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.current_step,
            'epoch': self.current_epoch,
            'config': self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']

        print(f"Checkpoint loaded from {checkpoint_path}")

def create_dummy_activation_data(n_samples: int = 100000, n_inputs: int = 4096) -> torch.Tensor:
    """
    Create dummy activation data for testing
    In practice, this should be replaced with actual model activations
    """
    # Create somewhat realistic activation data
    # Mix of sparse and dense components
    activations = torch.zeros(n_samples, n_inputs)

    # Dense component (small)
    dense_component = torch.randn(n_samples, n_inputs) * 0.1
    activations += dense_component

    # Sparse component (larger magnitude)
    n_active = int(0.1 * n_inputs)  # 10% active features
    for i in range(n_samples):
        active_indices = torch.randperm(n_inputs)[:n_active]
        activations[i, active_indices] += torch.randn(n_active) * 0.5

    # Add some ReLU-like behavior
    activations = F.relu(activations)

    return activations

def main():
    # Configuration
    config = SAETrainingConfig(
        batch_size=4096,
        epochs=10,
        lr=1e-4,
        l1_coefficient=1e-3,
        n_latents=16384,
        n_inputs=4096,
        wandb_project="sae_training_demo",
        wandb_name="baseline_run"
    )

    # Create autoencoder
    autoencoder = Autoencoder(
        n_latents=config.n_latents,
        n_inputs=config.n_inputs,
        activation=nn.ReLU(),
        tied=config.tied_weights,
        normalize=config.normalize_activations
    )

    # Create trainer
    trainer = SAETrainer(config, autoencoder, use_wandb=False)

    # Create dummy data (replace with actual activation data)
    print("Creating dummy activation data...")
    activations = create_dummy_activation_data(
        n_samples=100000,
        n_inputs=config.n_inputs
    )

    # Create datasets
    total_samples = activations.shape[0]
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    train_activations = activations[:train_size]
    val_activations = activations[train_size:]

    train_dataset = ActivationDataset(train_activations)
    val_dataset = ActivationDataset(val_activations)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Train
    trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()