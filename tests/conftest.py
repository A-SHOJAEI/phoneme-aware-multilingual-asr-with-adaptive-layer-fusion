"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.data.preprocessing import (
    AudioPreprocessor,
    PhonemeTokenizer,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.model import (
    PhonemeAwareASRModel,
)


@pytest.fixture
def device():
    """Device fixture."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def languages():
    """Language codes fixture."""
    return ["en", "es", "fr", "de"]


@pytest.fixture
def audio_preprocessor():
    """Audio preprocessor fixture."""
    return AudioPreprocessor(
        target_sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        feature_type="mel",
    )


@pytest.fixture
def phoneme_tokenizer(languages):
    """Phoneme tokenizer fixture."""
    return PhonemeTokenizer(languages=languages)


@pytest.fixture
def sample_waveform():
    """Sample audio waveform fixture."""
    sample_rate = 16000
    duration = 2.0
    waveform = torch.randn(1, int(sample_rate * duration))
    return waveform, sample_rate


@pytest.fixture
def sample_batch(languages):
    """Sample batch fixture."""
    batch_size = 4
    n_mels = 80
    time_steps = 100
    seq_len = 20

    return {
        "features": torch.randn(batch_size, n_mels, time_steps),
        "tokens": torch.randint(0, 64, (batch_size, seq_len)),
        "feature_lengths": torch.tensor([100, 90, 80, 70]),
        "token_lengths": torch.tensor([20, 18, 15, 12]),
        "language_ids": torch.tensor([0, 1, 2, 3]),
        "texts": ["sample text 1", "sample text 2", "sample text 3", "sample text 4"],
    }


@pytest.fixture
def model(phoneme_tokenizer, languages):
    """Model fixture."""
    return PhonemeAwareASRModel(
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
