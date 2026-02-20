"""Training loop with learning rate scheduling and early stopping."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ASRTrainer:
    """
    Trainer for phoneme-aware multilingual ASR model.

    Handles training loop, validation, checkpointing, early stopping,
    and learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        learning_rate: float = 0.0001,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 2,
        early_stopping_patience: int = 10,
        use_amp: bool = True,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (created if None)
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
            learning_rate: Learning rate
            max_grad_norm: Maximum gradient norm for clipping
            scheduler_type: LR scheduler type ('cosine', 'plateau', 'step')
            warmup_epochs: Number of warmup epochs
            early_stopping_patience: Patience for early stopping
            use_amp: Use automatic mixed precision
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

        # Learning rate scheduler
        self.scheduler_type = scheduler_type
        if scheduler_type == "cosine":
            total_steps = len(train_loader) * 100  # Assume max 100 epochs
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-6
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
        else:
            self.scheduler = None

        # Training settings
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Mixed precision training
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

        logger.info(
            f"Trainer initialized: device={self.device}, "
            f"scheduler={scheduler_type}, amp={self.use_amp}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_base_loss = 0.0
        total_inventory_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            # Move batch to device
            features = batch["features"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            language_ids = batch["language_ids"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = self.model(
                        features, feature_lengths, tokens, language_ids
                    )
            else:
                loss, loss_dict = self.model(
                    features, feature_lengths, tokens, language_ids
                )

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

            # Update scheduler (if step-based)
            if self.scheduler_type == "cosine" and self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            total_base_loss += loss_dict["base_loss"].item()
            total_inventory_loss += loss_dict["inventory_loss"].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        metrics = {
            "train_loss": total_loss / num_batches,
            "train_base_loss": total_base_loss / num_batches,
            "train_inventory_loss": total_inventory_loss / num_batches,
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_base_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                features = batch["features"].to(self.device)
                tokens = batch["tokens"].to(self.device)
                feature_lengths = batch["feature_lengths"].to(self.device)
                language_ids = batch["language_ids"].to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss, loss_dict = self.model(
                            features, feature_lengths, tokens, language_ids
                        )
                else:
                    loss, loss_dict = self.model(
                        features, feature_lengths, tokens, language_ids
                    )

                total_loss += loss.item()
                total_base_loss += loss_dict["base_loss"].item()
                num_batches += 1

        metrics = {
            "val_loss": total_loss / num_batches,
            "val_base_loss": total_base_loss / num_batches,
        }

        return metrics

    def save_checkpoint(
        self, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self, num_epochs: int) -> Tuple[Dict[str, float], bool]:
        """
        Train for multiple epochs with early stopping.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Tuple of (final_metrics, early_stopped)
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}"
            )

            # Validate
            val_metrics = self.validate()
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Update scheduler (if epoch-based)
            if self.scheduler_type == "plateau" and self.scheduler is not None:
                self.scheduler.step(val_metrics["val_loss"])

            # Check for improvement
            val_loss = val_metrics["val_loss"]
            is_best = val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )

            # Save checkpoint
            self.save_checkpoint(metrics, is_best=is_best)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )
                return metrics, True

        logger.info("Training completed successfully")
        return metrics, False
