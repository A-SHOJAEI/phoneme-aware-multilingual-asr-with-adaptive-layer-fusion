"""Tests for data loading and preprocessing."""

import pytest
import torch

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.data.preprocessing import (
    AudioPreprocessor,
    PhonemeTokenizer,
)


def test_audio_preprocessor_initialization():
    """Test audio preprocessor initialization."""
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        n_mels=80,
        feature_type="mel",
    )

    assert preprocessor.target_sample_rate == 16000
    assert preprocessor.n_mels == 80
    assert preprocessor.feature_type == "mel"


def test_audio_preprocessing(audio_preprocessor, sample_waveform):
    """Test audio feature extraction."""
    waveform, sample_rate = sample_waveform

    features = audio_preprocessor(waveform, sample_rate)

    # Check output shape
    assert features.dim() == 3  # (channels, n_mels, time)
    assert features.shape[0] == 1  # mono
    assert features.shape[1] == 80  # n_mels


def test_audio_resampling(audio_preprocessor):
    """Test audio resampling."""
    # Create waveform at different sample rate
    waveform = torch.randn(1, 48000)  # 48kHz
    sample_rate = 48000

    features = audio_preprocessor(waveform, sample_rate)

    # Should still produce valid features
    assert features.shape[1] == 80


def test_phoneme_tokenizer_initialization(languages):
    """Test phoneme tokenizer initialization."""
    tokenizer = PhonemeTokenizer(languages=languages)

    assert tokenizer.vocab_size > 0
    assert "<pad>" in tokenizer.phoneme_to_id
    assert "<sos>" in tokenizer.phoneme_to_id
    assert "<eos>" in tokenizer.phoneme_to_id


def test_phoneme_encoding(phoneme_tokenizer):
    """Test phoneme encoding."""
    text = "hello world"
    tokens = phoneme_tokenizer.encode(text, language="en")

    # Should have SOS and EOS tokens
    assert tokens[0] == phoneme_tokenizer.phoneme_to_id["<sos>"]
    assert tokens[-1] == phoneme_tokenizer.phoneme_to_id["<eos>"]
    assert len(tokens) > 2


def test_phoneme_decoding(phoneme_tokenizer):
    """Test phoneme decoding."""
    text = "hello"
    tokens = phoneme_tokenizer.encode(text, language="en")
    decoded = phoneme_tokenizer.decode(tokens)

    # Decoded text should be similar to original
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_batch_encoding(phoneme_tokenizer):
    """Test batch encoding with padding."""
    texts = ["hello", "world is nice", "a"]

    token_ids, lengths = phoneme_tokenizer.batch_encode(texts, language="en")

    # Check shapes
    assert token_ids.shape[0] == 3  # batch size
    assert lengths.shape[0] == 3
    assert token_ids.shape[1] >= lengths.max()  # padded to max length


def test_phoneme_tokenizer_multiple_languages(phoneme_tokenizer, languages):
    """Test tokenizer with different languages."""
    text = "hello"

    for lang in languages:
        tokens = phoneme_tokenizer.encode(text, language=lang)
        assert len(tokens) > 2
        assert tokens[0] == phoneme_tokenizer.phoneme_to_id["<sos>"]


def test_audio_edge_cases(audio_preprocessor):
    """Test edge cases in audio preprocessing."""
    # Very short audio
    short_waveform = torch.randn(1, 100)
    features = audio_preprocessor(short_waveform, 16000)
    assert features.shape[1] == 80

    # Stereo audio
    stereo_waveform = torch.randn(2, 16000)
    features = audio_preprocessor(stereo_waveform, 16000)
    assert features.shape[0] == 1  # Should convert to mono


def test_tokenizer_empty_text(phoneme_tokenizer):
    """Test tokenizer with empty text."""
    tokens = phoneme_tokenizer.encode("", language="en")

    # Should at least have SOS and EOS
    assert len(tokens) >= 2
    assert tokens[0] == phoneme_tokenizer.phoneme_to_id["<sos>"]
    assert tokens[-1] == phoneme_tokenizer.phoneme_to_id["<eos>"]
