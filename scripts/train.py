#!/usr/bin/env python
"""
Training script for phoneme-aware multilingual ASR.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/ablation.yaml
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
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.training.trainer import (
    ASRTrainer,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.utils.config import (
    load_config,
    save_config,
    setup_logging,
    set_seed,
)


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train phoneme-aware multilingual ASR"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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
    logger.info(f"Set random seed to {seed}")

    # Setup MLflow tracking (wrapped in try/except)
    use_mlflow = config.get("logging", {}).get("use_mlflow", False)
    if use_mlflow:
        try:
            import mlflow

            mlflow.set_experiment(
                config.get("logging", {}).get("mlflow_experiment", "phoneme_asr")
            )
            mlflow.start_run()
            mlflow.log_params(
                {
                    "config_file": args.config,
                    "seed": seed,
                    "batch_size": config.get("data", {}).get("batch_size", 16),
                    "learning_rate": config.get("training", {}).get(
                        "learning_rate", 0.0001
                    ),
                    "hidden_dim": config.get("model", {}).get("hidden_dim", 256),
                    "use_adaptive_fusion": config.get("model", {}).get(
                        "use_adaptive_fusion", True
                    ),
                }
            )
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {e}. Continuing without MLflow.")
            use_mlflow = False

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

        # Load datasets
        logger.info("Loading training dataset...")
        train_dataset = CommonVoiceDataset(
            languages=languages,
            split=data_config.get("train_split", "train"),
            max_duration=data_config.get("max_duration", 10.0),
            max_samples=data_config.get("max_samples_per_language", 500),
            preprocessor=preprocessor,
            tokenizer=tokenizer,
        )

        # Split for validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size]
        )

        logger.info(f"Training samples: {len(train_subset)}")
        logger.info(f"Validation samples: {len(val_subset)}")

        # Create data loaders
        train_loader = get_dataloader(
            train_subset,
            batch_size=data_config.get("batch_size", 16),
            shuffle=True,
            num_workers=data_config.get("num_workers", 4),
        )

        val_loader = get_dataloader(
            val_subset,
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

        logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Initialize trainer
        training_config = config.get("training", {})
        trainer = ASRTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=config.get("checkpoint", {}).get(
                "save_dir", "checkpoints"
            ),
            learning_rate=training_config.get("learning_rate", 0.0001),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            scheduler_type=training_config.get("scheduler", {}).get(
                "type", "cosine"
            ),
            warmup_epochs=training_config.get("warmup_epochs", 2),
            early_stopping_patience=training_config.get(
                "early_stopping_patience", 10
            ),
            use_amp=training_config.get("use_amp", True),
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Train
        num_epochs = training_config.get("num_epochs", 50)
        logger.info(f"Starting training for {num_epochs} epochs")

        final_metrics, early_stopped = trainer.train(num_epochs)

        # Save final results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        experiment_name = Path(args.config).stem
        results_file = results_dir / f"{experiment_name}_training_results.json"

        with open(results_file, "w") as f:
            json.dump(
                {
                    "final_metrics": final_metrics,
                    "early_stopped": early_stopped,
                    "config": config,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved training results to {results_file}")

        # Log to MLflow
        if use_mlflow:
            try:
                for key, value in final_metrics.items():
                    mlflow.log_metric(key, value)
                mlflow.log_artifact(results_file)
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        logger.info("Training completed successfully!")

        # Print summary
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
        print(f"Final Train Loss: {final_metrics.get('train_loss', 0.0):.4f}")
        print(f"Final Val Loss: {final_metrics.get('val_loss', 0.0):.4f}")
        print(f"Early Stopped: {early_stopped}")
        print(f"Checkpoint saved to: {trainer.checkpoint_dir / 'best.pt'}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        if use_mlflow:
            try:
                mlflow.end_run(status="FAILED")
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
