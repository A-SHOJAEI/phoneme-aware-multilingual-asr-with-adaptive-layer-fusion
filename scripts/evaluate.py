#!/usr/bin/env python
"""
Evaluation script for phoneme-aware multilingual ASR.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import random_split

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.data.loader import (
    CommonVoiceDataset,
    get_dataloader,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.data.preprocessing import (
    AudioPreprocessor,
    PhonemeTokenizer,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.model import (
    PhonemeAwareASRModel,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.evaluation.metrics import (
    compute_metrics,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.evaluation.analysis import (
    analyze_results,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.utils.config import (
    load_config,
    setup_logging,
    set_seed,
)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate phoneme-aware multilingual ASR"
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
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (test/validation)",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Setup logging
    logger = setup_logging(config.get("logging", {}).get("log_level", "INFO"))
    logger.info(f"Loaded configuration from {args.config}")

    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
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

        logger.info(f"Initialized preprocessors for languages: {languages}")

        # Load test dataset
        logger.info(f"Loading {args.split} dataset...")
        test_dataset = CommonVoiceDataset(
            languages=languages,
            split=args.split,
            max_duration=data_config.get("max_duration", 10.0),
            max_samples=data_config.get("max_samples_per_language", 500),
            preprocessor=preprocessor,
            tokenizer=tokenizer,
        )

        # If using validation split from training data
        if args.split == "validation" and len(test_dataset) > 100:
            # Take a subset for faster evaluation
            _, test_subset = random_split(
                test_dataset, [len(test_dataset) - 100, 100]
            )
            test_dataset = test_subset

        logger.info(f"Evaluation samples: {len(test_dataset)}")

        # Create data loader
        test_loader = get_dataloader(
            test_dataset,
            batch_size=data_config.get("batch_size", 16),
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
        )

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

        logger.info("Model loaded successfully")

        # Compute metrics
        logger.info("Computing evaluation metrics...")
        metrics = compute_metrics(
            model=model,
            data_loader=test_loader,
            tokenizer=tokenizer,
            device=device,
            languages=languages,
        )

        # Add metadata
        metrics["checkpoint"] = args.checkpoint
        metrics["config"] = args.config
        metrics["split"] = args.split
        metrics["adaptive_fusion_enabled"] = model_config.get(
            "use_adaptive_fusion", True
        )

        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        experiment_name = Path(args.config).stem
        results_file = results_dir / f"{experiment_name}_evaluation_results.json"

        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved evaluation results to {results_file}")

        # Analyze and print results
        analyze_results(metrics, output_dir="results", experiment_name=experiment_name)

        # Print detailed summary
        print("\n" + "=" * 70)
        print("Evaluation Results Summary")
        print("=" * 70)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Dataset Split: {args.split}")
        print(f"Adaptive Fusion: {'Enabled' if metrics['adaptive_fusion_enabled'] else 'Disabled'}")
        print("-" * 70)
        print("\nOverall Metrics:")
        print(f"  WER: {metrics.get('wer', 0.0):.4f}")
        print(f"  PER: {metrics.get('per', 0.0):.4f}")
        print("\nPer-Language WER:")
        for lang in languages:
            wer = metrics.get(f"wer_{lang}", None)
            if wer is not None:
                print(f"  {lang.upper()}: {wer:.4f}")
        print("\nPer-Language PER:")
        for lang in languages:
            per = metrics.get(f"per_{lang}", None)
            if per is not None:
                print(f"  {lang.upper()}: {per:.4f}")
        print("=" * 70 + "\n")

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
