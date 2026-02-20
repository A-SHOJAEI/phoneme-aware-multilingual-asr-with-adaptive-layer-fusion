"""Evaluation metrics for ASR: WER, PER, and cross-lingual transfer gain."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """
    Compute Levenshtein distance between two sequences.

    Args:
        ref: Reference sequence
        hyp: Hypothesis sequence

    Returns:
        Edit distance
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    return dp[m][n]


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Word Error Rate (WER).

    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts

    Returns:
        WER score (lower is better)
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    total_distance = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()

        distance = levenshtein_distance(ref_words, hyp_words)
        total_distance += distance
        total_words += len(ref_words)

    wer = total_distance / total_words if total_words > 0 else 0.0
    return wer


def compute_per(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Phoneme Error Rate (PER).

    Args:
        references: List of reference phoneme sequences (as strings)
        hypotheses: List of hypothesis phoneme sequences

    Returns:
        PER score (lower is better)
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    total_distance = 0
    total_phonemes = 0

    for ref, hyp in zip(references, hypotheses):
        # Treat each character as a phoneme
        ref_phonemes = list(ref.replace(" ", ""))
        hyp_phonemes = list(hyp.replace(" ", ""))

        distance = levenshtein_distance(ref_phonemes, hyp_phonemes)
        total_distance += distance
        total_phonemes += len(ref_phonemes)

    per = total_distance / total_phonemes if total_phonemes > 0 else 0.0
    return per


def compute_cross_lingual_transfer_gain(
    baseline_wer: Dict[str, float],
    transfer_wer: Dict[str, float],
    low_resource_langs: List[str],
) -> float:
    """
    Compute cross-lingual transfer gain for low-resource languages.

    Transfer gain = (baseline_wer - transfer_wer) / baseline_wer

    Args:
        baseline_wer: WER for baseline model (per language)
        transfer_wer: WER for transfer model (per language)
        low_resource_langs: List of low-resource language codes

    Returns:
        Average transfer gain (higher is better)
    """
    gains = []

    for lang in low_resource_langs:
        if lang in baseline_wer and lang in transfer_wer:
            baseline = baseline_wer[lang]
            transfer = transfer_wer[lang]

            if baseline > 0:
                gain = (baseline - transfer) / baseline
                gains.append(gain)

    return np.mean(gains) if gains else 0.0


def compute_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    tokenizer,
    device: torch.device,
    languages: List[str],
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        tokenizer: Phoneme tokenizer
        device: Device to run evaluation on
        languages: List of language codes

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_references = []
    all_hypotheses = []
    language_refs = {lang: [] for lang in languages}
    language_hyps = {lang: [] for lang in languages}

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            tokens = batch["tokens"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            language_ids = batch["language_ids"].to(device)
            texts = batch["texts"]

            # Generate predictions
            predictions = model.generate(
                features, feature_lengths, language_ids, max_len=100
            )

            # Decode predictions and references
            for i in range(len(texts)):
                ref_text = texts[i]
                pred_tokens = predictions[i].cpu().tolist()
                pred_text = tokenizer.decode(pred_tokens)

                all_references.append(ref_text)
                all_hypotheses.append(pred_text)

                lang = languages[language_ids[i].item()]
                language_refs[lang].append(ref_text)
                language_hyps[lang].append(pred_text)

    # Compute overall metrics
    overall_wer = compute_wer(all_references, all_hypotheses)
    overall_per = compute_per(all_references, all_hypotheses)

    metrics = {
        "wer": overall_wer,
        "per": overall_per,
    }

    # Compute per-language metrics
    for lang in languages:
        if language_refs[lang]:
            lang_wer = compute_wer(language_refs[lang], language_hyps[lang])
            lang_per = compute_per(language_refs[lang], language_hyps[lang])

            metrics[f"wer_{lang}"] = lang_wer
            metrics[f"per_{lang}"] = lang_per

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics
