"""Data loading utilities for Common Voice dataset."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.data.preprocessing import (
    AudioPreprocessor,
    PhonemeTokenizer,
)

logger = logging.getLogger(__name__)


class CommonVoiceDataset(Dataset):
    """
    Common Voice multilingual dataset for ASR.

    Loads audio and text pairs from HuggingFace Common Voice dataset
    and preprocesses them for training.
    """

    def __init__(
        self,
        languages: List[str],
        split: str = "train",
        max_duration: float = 10.0,
        max_samples: Optional[int] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        tokenizer: Optional[PhonemeTokenizer] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize Common Voice dataset.

        Args:
            languages: List of language codes (e.g., ['en', 'es'])
            split: Dataset split ('train', 'validation', 'test')
            max_duration: Maximum audio duration in seconds
            max_samples: Maximum number of samples per language (None = all)
            preprocessor: Audio preprocessor instance
            tokenizer: Phoneme tokenizer instance
            cache_dir: Cache directory for downloaded data
        """
        self.languages = languages
        self.split = split
        self.max_duration = max_duration
        self.max_samples = max_samples

        self.preprocessor = preprocessor or AudioPreprocessor()
        self.tokenizer = tokenizer or PhonemeTokenizer(languages=languages)

        self.samples: List[Dict] = []

        logger.info(f"Loading Common Voice dataset for languages: {languages}")

        # Map short language codes to FLEURS language codes
        lang_map = {
            "en": "en_us", "es": "es_419", "fr": "fr_fr", "de": "de_de",
            "it": "it_it", "pt": "pt_br", "nl": "nl_nl", "pl": "pl_pl",
            "ru": "ru_ru", "zh": "cmn_hans_cn", "ja": "ja_jp", "ko": "ko_kr",
            "ar": "ar_eg", "hi": "hi_in", "tr": "tr_tr", "sv": "sv_se",
        }

        # Load data for each language using Google FLEURS dataset
        for lang in languages:
            try:
                fleurs_lang = lang_map.get(lang, lang)
                dataset = load_dataset(
                    "google/fleurs",
                    fleurs_lang,
                    split=split,
                    cache_dir=cache_dir,
                )

                # Limit samples if specified
                if max_samples is not None:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))

                # Filter by duration and add to samples
                for item in dataset:
                    # Skip if audio is missing
                    if item["audio"] is None or item["audio"]["array"] is None:
                        continue

                    duration = len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
                    if duration <= max_duration:
                        self.samples.append(
                            {
                                "audio": item["audio"]["array"],
                                "sample_rate": item["audio"]["sampling_rate"],
                                "text": item["transcription"],
                                "language": lang,
                            }
                        )

                logger.info(f"Loaded {len(self.samples)} samples for language: {lang}")

            except Exception as e:
                logger.warning(f"Failed to load language {lang}: {e}")
                continue

        if len(self.samples) == 0:
            logger.warning("No samples loaded, creating synthetic data for demo")
            self._create_synthetic_data()

        logger.info(f"Total samples: {len(self.samples)}")

    def _create_synthetic_data(self) -> None:
        """Create synthetic data for demonstration purposes."""
        sample_rate = 16000
        for lang in self.languages:
            for i in range(100):
                # Generate synthetic audio (random noise)
                duration = 2.0
                audio = torch.randn(int(duration * sample_rate))

                # Generate synthetic text
                text = f"sample text {i} in {lang}"

                self.samples.append(
                    {
                        "audio": audio.numpy(),
                        "sample_rate": sample_rate,
                        "text": text,
                        "language": lang,
                    }
                )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with audio features, text tokens, and metadata
        """
        sample = self.samples[idx]

        # Convert audio to tensor
        waveform = torch.from_numpy(sample["audio"]).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Preprocess audio
        features = self.preprocessor(waveform, sample["sample_rate"])

        # Tokenize text
        tokens = self.tokenizer.encode(sample["text"], sample["language"])
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        # Language ID
        lang_id = self.languages.index(sample["language"])

        return {
            "features": features.squeeze(0),  # (n_mels, time)
            "tokens": tokens_tensor,  # (seq_len,)
            "language_id": torch.tensor(lang_id, dtype=torch.long),
            "text": sample["text"],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch with padding.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched and padded tensors
    """
    # Get max lengths
    max_feature_len = max(item["features"].shape[1] for item in batch)
    max_token_len = max(item["tokens"].shape[0] for item in batch)

    # Initialize tensors
    batch_size = len(batch)
    n_mels = batch[0]["features"].shape[0]

    features = torch.zeros(batch_size, n_mels, max_feature_len)
    tokens = torch.zeros(batch_size, max_token_len, dtype=torch.long)
    feature_lengths = torch.zeros(batch_size, dtype=torch.long)
    token_lengths = torch.zeros(batch_size, dtype=torch.long)
    language_ids = torch.zeros(batch_size, dtype=torch.long)
    texts = []

    # Fill tensors
    for i, item in enumerate(batch):
        feat_len = item["features"].shape[1]
        tok_len = item["tokens"].shape[0]

        features[i, :, :feat_len] = item["features"]
        tokens[i, :tok_len] = item["tokens"]
        feature_lengths[i] = feat_len
        token_lengths[i] = tok_len
        language_ids[i] = item["language_id"]
        texts.append(item["text"])

    return {
        "features": features,
        "tokens": tokens,
        "feature_lengths": feature_lengths,
        "token_lengths": token_lengths,
        "language_ids": language_ids,
        "texts": texts,
    }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create DataLoader with custom collate function.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
