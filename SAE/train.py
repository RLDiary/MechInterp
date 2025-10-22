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
from model import Autoencoder

@dataclass
class SAETrainingConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 4096
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-2
    wandb_project: str = "1L21M SAE Training"
    wandb_name: Optional[str] = None
    l1_coefficient: float = 1e-3
    grad_accumulation_steps: int = 1
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # SAE specific parameters
    n_latents: int = 16384
    n_inputs: int = 4096
    tied_weights: bool = False
    normalize_activations: bool = False
    dead_neuron_threshold: int = 10000

class ActivationDataset(Dataset):
    """Dataset for model activations"""

    def __init__(self, activations: torch.Tensor):
        """
        Args:
            activations: Tensor of shape (n_samples, n_inputs)
        """
        self.activations = activations.float()

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]

class SAETrainer:
    def __init__(
        self,
        config: SAETrainingConfig,
        autoencoder: Autoencoder,
        use_wandb: bool = True
    ):
        self.config = config
        self.autoencoder = autoencoder.to(config.device)
        self.use_wandb = use_wandb

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.autoencoder.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Training state
        self.current_step = 0
        self.current_epoch = 0

        # Create checkpoints directory
        os.makedirs("SAE/Checkpoints", exist_ok=True)

        # Initialize wandb
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
                    "n_latents": config.n_latents,
                    "n_inputs": config.n_inputs,
                    "tied_weights": config.tied_weights,
                    "normalize_activations": config.normalize_activations,
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
        batch = batch.to(self.config.device)

        # Forward pass
        latents_pre_act, latents, reconstructions = self.autoencoder(batch)

        # Compute loss
        loss_dict = self.compute_loss(batch, reconstructions, latents)

        return loss_dict

    def train_epoch(self, train_dataloader: DataLoader, scheduler=None) -> Dict[str, float]:
        """Train for one epoch"""
        self.autoencoder.train()
        epoch_losses = {}

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss_dict = self.step(batch)
            total_loss = loss_dict["total_loss"]

            # Accumulate losses for logging
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())

            # Backward pass
            total_loss = total_loss / self.config.grad_accumulation_steps
            total_loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.autoencoder.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                if scheduler:
                    scheduler.step()
                self.optimizer.zero_grad()

                self.current_step += 1

                # Logging
                if self.current_step % self.config.log_every == 0:
                    self._log_metrics(loss_dict, prefix="train")

                # Evaluation
                if self.current_step % self.config.eval_every == 0:
                    self._evaluate_dead_neurons()

                # Checkpointing
                if self.current_step % self.config.save_every == 0:
                    self.save_checkpoint()

            # Update progress bar
            avg_loss = np.mean(epoch_losses["total_loss"])
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'step': self.current_step
            })

        # Return average losses for epoch
        return {key: np.mean(values) for key, values in epoch_losses.items()}

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
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

        # Setup scheduler
        total_steps = len(train_dataloader) * self.config.epochs
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        print(f"Training for {self.config.epochs} epochs, {total_steps} total steps")
        print(f"Warmup steps: {warmup_steps}")

        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_losses = self.train_epoch(train_dataloader, scheduler)

            # Validation
            if val_dataloader:
                val_losses = self.evaluate(val_dataloader)
                self._log_metrics(val_losses, prefix="val")

                print(f"Epoch {epoch}: Train Loss: {train_losses['total_loss']:.4f}, "
                      f"Val Loss: {val_losses['total_loss']:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss: {train_losses['total_loss']:.4f}")

            # Log epoch metrics
            self._log_metrics(train_losses, prefix="epoch_train")

        print("Training completed!")

        # Save final model
        self.save_checkpoint(final=True)

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