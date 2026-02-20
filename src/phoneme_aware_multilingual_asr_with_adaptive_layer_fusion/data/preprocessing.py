"""Audio preprocessing and phoneme tokenization."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocesses audio signals for ASR model input.

    Handles resampling, normalization, and feature extraction
    (mel-spectrograms or MFCCs).
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        feature_type: str = "mel",
    ) -> None:
        """
        Initialize audio preprocessor.

        Args:
            target_sample_rate: Target sampling rate in Hz
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            feature_type: Feature type ('mel' or 'mfcc')
        """
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.feature_type = feature_type

        if feature_type == "mel":
            self.feature_extractor = MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        elif feature_type == "mfcc":
            self.feature_extractor = MFCC(
                sample_rate=target_sample_rate,
                n_mfcc=n_mels,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": n_mels,
                },
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def __call__(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """
        Process audio waveform into features.

        Args:
            waveform: Audio waveform tensor (channels, samples)
            sample_rate: Original sample rate

        Returns:
            Feature tensor (channels, features, time)
        """
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract features
        features = self.feature_extractor(waveform)

        # Log scale for mel spectrograms
        if self.feature_type == "mel":
            features = torch.log(features + 1e-9)

        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-9)

        return features


class PhonemeTokenizer:
    """
    Tokenizes text into phoneme sequences for multilingual ASR.

    Maintains language-specific phoneme inventories while enabling
    cross-lingual transfer through shared phoneme representations.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_ipa: bool = True,
    ) -> None:
        """
        Initialize phoneme tokenizer.

        Args:
            languages: List of language codes to support
            use_ipa: Whether to use IPA phoneme representations
        """
        self.languages = languages or ["en", "es", "fr", "de"]
        self.use_ipa = use_ipa

        # Build phoneme inventory (simplified for demo)
        self.phoneme_to_id: Dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.id_to_phoneme: Dict[int, str] = {v: k for k, v in self.phoneme_to_id.items()}

        # Common IPA phonemes across languages
        common_phonemes = [
            "a", "e", "i", "o", "u",  # vowels
            "p", "b", "t", "d", "k", "g",  # plosives
            "f", "v", "s", "z", "ʃ", "ʒ",  # fricatives
            "m", "n", "ŋ", "l", "r",  # nasals/liquids
        ]

        for phoneme in common_phonemes:
            if phoneme not in self.phoneme_to_id:
                idx = len(self.phoneme_to_id)
                self.phoneme_to_id[phoneme] = idx
                self.id_to_phoneme[idx] = phoneme

        self.vocab_size = len(self.phoneme_to_id)
        logger.info(f"Initialized phoneme tokenizer with vocab size: {self.vocab_size}")

    def encode(self, text: str, language: str = "en") -> List[int]:
        """
        Encode text into phoneme IDs.

        Args:
            text: Input text to encode
            language: Source language code

        Returns:
            List of phoneme token IDs
        """
        # Simplified character-level encoding as proxy for phonemes
        # In production, use epitran or phonemizer library
        tokens = [self.phoneme_to_id["<sos>"]]

        for char in text.lower():
            if char in self.phoneme_to_id:
                tokens.append(self.phoneme_to_id[char])
            elif char.isalnum():
                tokens.append(self.phoneme_to_id["<unk>"])

        tokens.append(self.phoneme_to_id["<eos>"])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode phoneme IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_phoneme:
                phoneme = self.id_to_phoneme[token_id]
                if phoneme not in ["<pad>", "<sos>", "<eos>"]:
                    tokens.append(phoneme)

        return "".join(tokens)

    def batch_encode(
        self, texts: List[str], language: str = "en", max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch encode texts with padding.

        Args:
            texts: List of text strings
            language: Source language
            max_length: Maximum sequence length (None = auto)

        Returns:
            Tuple of (token_ids, lengths) tensors
        """
        encoded = [self.encode(text, language) for text in texts]

        if max_length is None:
            max_length = max(len(seq) for seq in encoded)

        # Pad sequences
        padded = []
        lengths = []
        for seq in encoded:
            length = len(seq)
            lengths.append(length)
            if length < max_length:
                seq = seq + [self.phoneme_to_id["<pad>"]] * (max_length - length)
            else:
                seq = seq[:max_length]
            padded.append(seq)

        return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
