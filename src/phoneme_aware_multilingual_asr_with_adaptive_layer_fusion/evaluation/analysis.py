"""Results analysis and visualization tools."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def analyze_results(
    results: Dict[str, float],
    output_dir: str = "results",
    experiment_name: str = "default",
) -> None:
    """
    Analyze and save evaluation results.

    Args:
        results: Dictionary of metric values
        output_dir: Directory to save analysis
        experiment_name: Name of experiment
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    results_file = output_path / f"{experiment_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {results_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"Evaluation Results - {experiment_name}")
    print("=" * 60)

    # Overall metrics
    if "wer" in results:
        print(f"Overall WER: {results['wer']:.4f}")
    if "per" in results:
        print(f"Overall PER: {results['per']:.4f}")

    # Per-language metrics
    print("\nPer-language metrics:")
    print("-" * 60)
    print(f"{'Language':<15} {'WER':<15} {'PER':<15}")
    print("-" * 60)

    languages = set()
    for key in results.keys():
        if key.startswith("wer_"):
            lang = key.split("_")[1]
            languages.add(lang)

    for lang in sorted(languages):
        wer = results.get(f"wer_{lang}", 0.0)
        per = results.get(f"per_{lang}", 0.0)
        print(f"{lang:<15} {wer:<15.4f} {per:<15.4f}")

    print("=" * 60 + "\n")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[str],
    output_path: str = "results/confusion_matrix.png",
    title: str = "Phoneme Confusion Matrix",
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        labels: Class labels
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {output_file}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: str = "results/training_curves.png",
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curves to {output_file}")


def compare_ablations(
    ablation_results: Dict[str, Dict[str, float]],
    output_dir: str = "results",
) -> None:
    """
    Compare results from ablation studies.

    Args:
        ablation_results: Dictionary mapping experiment names to metrics
        output_dir: Directory to save comparison
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save comparison as JSON
    comparison_file = output_path / "ablation_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 80)
    print("Ablation Study Comparison")
    print("=" * 80)
    print(f"{'Experiment':<30} {'WER':<15} {'PER':<15} {'Improvement':<15}")
    print("-" * 80)

    baseline_wer = None
    for exp_name, results in ablation_results.items():
        wer = results.get("wer", 0.0)
        per = results.get("per", 0.0)

        if baseline_wer is None:
            baseline_wer = wer
            improvement = "-"
        else:
            improvement = f"{(1 - wer / baseline_wer) * 100:.2f}%"

        print(f"{exp_name:<30} {wer:<15.4f} {per:<15.4f} {improvement:<15}")

    print("=" * 80 + "\n")

    logger.info(f"Saved ablation comparison to {comparison_file}")
