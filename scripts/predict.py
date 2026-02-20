#!/usr/bin/env python
"""
Inference script for phoneme-aware multilingual ASR.

Usage:
    python scripts/predict.py --checkpoint checkpoints/best.pt --audio path/to/audio.wav --language en
    python scripts/predict.py --checkpoint checkpoints/best.pt --audio path/to/audio.wav --language es
"""

import argparse
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torchaudio

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.data.preprocessing import (
    AudioPreprocessor,
    PhonemeTokenizer,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.model import (
    PhonemeAwareASRModel,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.utils.config import (
    load_config,
    setup_logging,
    set_seed,
)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with phoneme-aware multilingual ASR"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (en, es, fr, de)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging("INFO")
    logger.info(f"Loaded configuration from {args.config}")

    # Set random seed
    set_seed(config.get("seed", 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Check if audio file exists
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {args.audio}")
            sys.exit(1)

        # Initialize preprocessors
        audio_config = config.get("audio", {})
        preprocessor = AudioPreprocessor(
            target_sample_rate=audio_config.get("target_sample_rate", 16000),
            n_mels=audio_config.get("n_mels", 80),
            n_fft=audio_config.get("n_fft", 400),
            hop_length=audio_config.get("hop_length", 160),
            feature_type=audio_config.get("feature_type", "mel"),
        )

        data_config = config.get("data", {})
        languages = data_config.get("languages", ["en", "es", "fr", "de"])
        tokenizer = PhonemeTokenizer(languages=languages)

        # Check if language is supported
        if args.language not in languages:
            logger.error(
                f"Language '{args.language}' not supported. "
                f"Available: {languages}"
            )
            sys.exit(1)

        language_id = languages.index(args.language)
        logger.info(f"Target language: {args.language}")

        # Load audio
        logger.info(f"Loading audio from {args.audio}")
        waveform, sample_rate = torchaudio.load(args.audio)

        # Preprocess audio
        features = preprocessor(waveform, sample_rate)
        features = features.unsqueeze(0).to(device)  # Add batch dimension

        # Feature length
        feature_length = torch.tensor([features.shape[2]], dtype=torch.long).to(
            device
        )

        # Language ID
        lang_id_tensor = torch.tensor([language_id], dtype=torch.long).to(device)

        # Initialize model
        model_config = config.get("model", {})
        model = PhonemeAwareASRModel(
            input_dim=model_config.get("input_dim", 80),
            hidden_dim=model_config.get("hidden_dim", 256),
            num_encoder_layers=model_config.get("num_encoder_layers", 6),
            num_decoder_layers=model_config.get("num_decoder_layers", 3),
            num_heads=model_config.get("num_heads", 4),
            num_phonemes=tokenizer.vocab_size,
            num_languages=len(languages),
            dropout=model_config.get("dropout", 0.1),
            use_adaptive_fusion=model_config.get("use_adaptive_fusion", True),
        )

        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # Run inference
        logger.info("Running inference...")
        with torch.no_grad():
            predictions = model.generate(
                features=features,
                feature_lengths=feature_length,
                language_ids=lang_id_tensor,
                max_len=args.max_length,
                sos_token=tokenizer.phoneme_to_id["<sos>"],
                eos_token=tokenizer.phoneme_to_id["<eos>"],
            )

        # Decode predictions
        pred_tokens = predictions[0].cpu().tolist()
        transcription = tokenizer.decode(pred_tokens)

        # Print results
        print("\n" + "=" * 70)
        print("ASR Inference Results")
        print("=" * 70)
        print(f"Audio File: {args.audio}")
        print(f"Language: {args.language.upper()}")
        print(f"Duration: {waveform.shape[1] / sample_rate:.2f}s")
        print(f"Model: {args.checkpoint}")
        print("-" * 70)
        print(f"Transcription: {transcription}")
        print("=" * 70 + "\n")

        logger.info("Inference completed successfully!")

        # Return transcription for programmatic use
        return transcription

    except Exception as e:
        logger.error(f"Inference failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
