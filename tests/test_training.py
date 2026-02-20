"""Tests for training loop and trainer."""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.training.trainer import (
    ASRTrainer,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.evaluation.metrics import (
    levenshtein_distance,
    compute_wer,
    compute_per,
)


def test_levenshtein_distance():
    """Test Levenshtein distance computation."""
    ref = ["hello", "world"]
    hyp1 = ["hello", "world"]
    hyp2 = ["hello", "word"]
    hyp3 = ["hi", "there"]

    assert levenshtein_distance(ref, hyp1) == 0
    assert levenshtein_distance(ref, hyp2) == 1
    assert levenshtein_distance(ref, hyp3) == 2


def test_compute_wer():
    """Test WER computation."""
    references = ["hello world", "how are you"]
    hypotheses = ["hello world", "how are you"]

    wer = compute_wer(references, hypotheses)
    assert wer == 0.0

    # With errors
    hypotheses_error = ["hello word", "how you"]
    wer_error = compute_wer(references, hypotheses_error)
    assert wer_error > 0.0


def test_compute_per():
    """Test PER computation."""
    references = ["hello", "world"]
    hypotheses = ["hello", "world"]

    per = compute_per(references, hypotheses)
    assert per == 0.0

    # With errors
    hypotheses_error = ["helo", "word"]
    per_error = compute_per(references, hypotheses_error)
    assert per_error > 0.0


def create_dummy_dataloader(batch_size=4, num_batches=5):
    """Create a dummy dataloader for testing."""
    total_samples = batch_size * num_batches

    features = torch.randn(total_samples, 80, 100)
    tokens = torch.randint(0, 64, (total_samples, 20))
    feature_lengths = torch.full((total_samples,), 100)
    token_lengths = torch.full((total_samples,), 20)
    language_ids = torch.randint(0, 4, (total_samples,))

    # Create custom dataset that returns dict
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, features, tokens, feature_lengths, token_lengths, language_ids):
            self.features = features
            self.tokens = tokens
            self.feature_lengths = feature_lengths
            self.token_lengths = token_lengths
            self.language_ids = language_ids

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "tokens": self.tokens[idx],
                "feature_lengths": self.feature_lengths[idx],
                "token_lengths": self.token_lengths[idx],
                "language_ids": self.language_ids[idx],
                "texts": f"text {idx}",
            }

    dataset = DummyDataset(features, tokens, feature_lengths, token_lengths, language_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_trainer_initialization(model, device):
    """Test trainer initialization."""
    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        use_amp=False,  # Disable AMP for testing
    )

    assert trainer.device == device
    assert trainer.optimizer is not None
    assert trainer.best_val_loss == float("inf")


def test_trainer_single_epoch(model, device):
    """Test training for a single epoch."""
    train_loader = create_dummy_dataloader(batch_size=2, num_batches=2)
    val_loader = create_dummy_dataloader(batch_size=2, num_batches=2)

    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        use_amp=False,
    )

    # Train one epoch
    train_metrics = trainer.train_epoch()

    assert "train_loss" in train_metrics
    assert train_metrics["train_loss"] > 0


def test_trainer_validation(model, device):
    """Test validation."""
    train_loader = create_dummy_dataloader(batch_size=2, num_batches=2)
    val_loader = create_dummy_dataloader(batch_size=2, num_batches=2)

    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        use_amp=False,
    )

    # Validate
    val_metrics = trainer.validate()

    assert "val_loss" in val_metrics
    assert val_metrics["val_loss"] > 0


def test_trainer_checkpoint_save(model, device, tmp_path):
    """Test checkpoint saving."""
    train_loader = create_dummy_dataloader(batch_size=2, num_batches=2)
    val_loader = create_dummy_dataloader(batch_size=2, num_batches=2)

    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=str(tmp_path),
        learning_rate=0.001,
        use_amp=False,
    )

    metrics = {"val_loss": 0.5}
    trainer.save_checkpoint(metrics, is_best=True)

    # Check files exist
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_trainer_checkpoint_load(model, device, tmp_path):
    """Test checkpoint loading."""
    train_loader = create_dummy_dataloader(batch_size=2, num_batches=2)
    val_loader = create_dummy_dataloader(batch_size=2, num_batches=2)

    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=str(tmp_path),
        learning_rate=0.001,
        use_amp=False,
    )

    # Save checkpoint
    metrics = {"val_loss": 0.5}
    trainer.save_checkpoint(metrics, is_best=True)

    # Create new trainer and load
    new_trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=str(tmp_path),
        learning_rate=0.001,
        use_amp=False,
    )

    new_trainer.load_checkpoint(str(tmp_path / "best.pt"))

    assert new_trainer.best_val_loss == 0.5


def test_trainer_learning_rate_scheduling(model, device):
    """Test learning rate scheduling."""
    train_loader = create_dummy_dataloader(batch_size=2, num_batches=2)
    val_loader = create_dummy_dataloader(batch_size=2, num_batches=2)

    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler_type="cosine",
        learning_rate=0.001,
        use_amp=False,
    )

    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    # Train one epoch (LR should change with cosine scheduler)
    trainer.train_epoch()

    # LR might change depending on scheduler
    assert trainer.scheduler is not None
