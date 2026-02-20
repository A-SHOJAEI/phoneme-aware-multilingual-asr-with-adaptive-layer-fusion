"""
Phoneme-Aware Multilingual ASR with Adaptive Layer Fusion.

A research-tier multilingual automatic speech recognition system that learns
phoneme-level representations across low-resource languages using adaptive
layer fusion with dynamic attention mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.model import (
    PhonemeAwareASRModel,
)
from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.components import (
    AdaptiveLayerFusion,
    PhonemeInventoryLoss,
)

__all__ = [
    "PhonemeAwareASRModel",
    "AdaptiveLayerFusion",
    "PhonemeInventoryLoss",
]
