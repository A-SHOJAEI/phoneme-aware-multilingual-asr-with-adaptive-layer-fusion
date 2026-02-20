"""Tests for model architecture and components."""

import pytest
import torch

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.model import (
    PhonemeAwareASRModel,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.components import (
    AdaptiveLayerFusion,
    PhonemeInventoryLoss,
    PhoneticSimilarityMatrix,
)


def test_model_initialization(model):
    """Test model initialization."""
    assert isinstance(model, PhonemeAwareASRModel)
    assert model.hidden_dim == 128
    assert model.num_encoder_layers == 2


def test_model_forward_pass(model, sample_batch, device):
    """Test model forward pass."""
    model.to(device)
    model.train()

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

    loss, loss_dict = model(
        features=batch["features"],
        feature_lengths=batch["feature_lengths"],
        targets=batch["tokens"],
        language_ids=batch["language_ids"],
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0

    assert "base_loss" in loss_dict
    assert "inventory_loss" in loss_dict
    assert "diversity_loss" in loss_dict


def test_model_generate(model, sample_batch, device):
    """Test model generation/inference."""
    model.to(device)
    model.eval()

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

    with torch.no_grad():
        predictions = model.generate(
            features=batch["features"],
            feature_lengths=batch["feature_lengths"],
            language_ids=batch["language_ids"],
            max_len=30,
        )

    assert predictions.shape[0] == batch["features"].shape[0]  # batch size
    assert predictions.dim() == 2


def test_adaptive_layer_fusion():
    """Test adaptive layer fusion module."""
    num_layers = 4
    hidden_dim = 128
    batch_size = 2
    seq_len = 50
    num_languages = 4

    fusion = AdaptiveLayerFusion(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_languages=num_languages,
        num_heads=4,
    )

    # Create dummy layer outputs
    layer_outputs = [
        torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers)
    ]
    language_ids = torch.tensor([0, 1])

    output, attention = fusion(layer_outputs, language_ids)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert attention.shape == (batch_size, num_layers)


def test_phonetic_similarity_matrix():
    """Test phonetic similarity matrix."""
    num_languages = 4
    batch_size = 8

    similarity_matrix = PhoneticSimilarityMatrix(num_languages, embedding_dim=64)

    source_lang = torch.randint(0, num_languages, (batch_size,))
    target_lang = torch.randint(0, num_languages, (batch_size,))

    similarity = similarity_matrix(source_lang, target_lang)

    assert similarity.shape == (batch_size,)
    assert (similarity >= 0).all() and (similarity <= 1).all()


def test_phoneme_inventory_loss():
    """Test phoneme inventory loss."""
    num_phonemes = 64
    num_languages = 4
    batch_size = 4
    seq_len = 20

    loss_fn = PhonemeInventoryLoss(
        num_phonemes=num_phonemes,
        num_languages=num_languages,
        inventory_weight=0.1,
        diversity_weight=0.05,
    )

    logits = torch.randn(batch_size, seq_len, num_phonemes)
    targets = torch.randint(0, num_phonemes, (batch_size, seq_len))
    language_ids = torch.tensor([0, 1, 2, 3])

    loss, loss_dict = loss_fn(logits, targets, language_ids)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert "base_loss" in loss_dict
    assert "inventory_loss" in loss_dict
    assert "diversity_loss" in loss_dict


def test_model_with_and_without_fusion(phoneme_tokenizer, languages, device):
    """Test model with and without adaptive fusion."""
    # Model with fusion
    model_with_fusion = PhonemeAwareASRModel(
        input_dim=80,
        hidden_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=1,
        num_heads=4,
        num_phonemes=phoneme_tokenizer.vocab_size,
        num_languages=len(languages),
        dropout=0.1,
        use_adaptive_fusion=True,
    )

    # Model without fusion
    model_without_fusion = PhonemeAwareASRModel(
        input_dim=80,
        hidden_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=1,
        num_heads=4,
        num_phonemes=phoneme_tokenizer.vocab_size,
        num_languages=len(languages),
        dropout=0.1,
        use_adaptive_fusion=False,
    )

    assert model_with_fusion.use_adaptive_fusion
    assert not model_without_fusion.use_adaptive_fusion


def test_model_parameter_count(model):
    """Test model has trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params == total_params


def test_model_gradients(model, sample_batch, device):
    """Test that gradients flow properly."""
    model.to(device)
    model.train()

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}

    loss, _ = model(
        features=batch["features"],
        feature_lengths=batch["feature_lengths"],
        targets=batch["tokens"],
        language_ids=batch["language_ids"],
    )

    loss.backward()

    # Check that at least some parameters have gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad
