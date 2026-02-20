"""Custom model components for phoneme-aware multilingual ASR.

This module implements the novel contributions:
1. AdaptiveLayerFusion: Dynamic attention mechanism that selectively combines
   encoder layers based on phonetic similarity between languages
2. PhonemeInventoryLoss: Custom loss to maintain language-specific phoneme
   inventories while enabling cross-lingual transfer
3. PhoneticSimilarityMatrix: Learnable matrix encoding phonetic distances
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PhoneticSimilarityMatrix(nn.Module):
    """
    Learnable phonetic similarity matrix between languages.

    Encodes the phonetic distance between language pairs to guide
    adaptive layer fusion. This is a key innovation enabling better
    cross-lingual transfer.
    """

    def __init__(self, num_languages: int, embedding_dim: int = 64) -> None:
        """
        Initialize phonetic similarity matrix.

        Args:
            num_languages: Number of supported languages
            embedding_dim: Dimension of language embeddings
        """
        super().__init__()
        self.num_languages = num_languages

        # Learnable language embeddings
        self.language_embeddings = nn.Embedding(num_languages, embedding_dim)

        # Initialize with small random values
        nn.init.normal_(self.language_embeddings.weight, mean=0.0, std=0.1)

    def forward(
        self, source_lang_id: torch.Tensor, target_lang_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phonetic similarity between source and target languages.

        Args:
            source_lang_id: Source language IDs (batch_size,)
            target_lang_id: Target language IDs (batch_size,)

        Returns:
            Similarity scores (batch_size,) in range [0, 1]
        """
        source_emb = self.language_embeddings(source_lang_id)  # (batch, dim)
        target_emb = self.language_embeddings(target_lang_id)  # (batch, dim)

        # Cosine similarity
        similarity = F.cosine_similarity(source_emb, target_emb, dim=-1)

        # Scale to [0, 1]
        similarity = (similarity + 1.0) / 2.0

        return similarity


class AdaptiveLayerFusion(nn.Module):
    """
    Adaptive layer fusion module with dynamic attention.

    Novel contribution: Dynamically combines encoder layer representations
    based on phonetic similarity between source and target languages.
    Enables selective feature sharing while preventing catastrophic forgetting.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_languages: int,
        num_heads: int = 4,
    ) -> None:
        """
        Initialize adaptive layer fusion module.

        Args:
            num_layers: Number of encoder layers to fuse
            hidden_dim: Hidden dimension of encoder outputs
            num_languages: Number of supported languages
            num_heads: Number of attention heads for fusion
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Phonetic similarity matrix
        self.similarity_matrix = PhoneticSimilarityMatrix(
            num_languages, embedding_dim=64
        )

        # Multi-head attention for layer fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Layer-specific importance weights (learnable)
        self.layer_importance = nn.Parameter(torch.ones(num_layers))

        # Language-specific fusion gates
        self.language_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )

        logger.info(
            f"Initialized AdaptiveLayerFusion with {num_layers} layers, "
            f"{num_heads} heads"
        )

    def forward(
        self,
        layer_outputs: List[torch.Tensor],
        language_ids: torch.Tensor,
        source_lang_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multiple encoder layer outputs adaptively.

        Args:
            layer_outputs: List of layer outputs, each (batch, seq_len, hidden)
            language_ids: Language IDs for each sample (batch,)
            source_lang_id: Source language for transfer (optional)

        Returns:
            Tuple of (fused_output, attention_weights)
            - fused_output: (batch, seq_len, hidden)
            - attention_weights: (batch, num_layers)
        """
        batch_size = layer_outputs[0].shape[0]
        seq_len = layer_outputs[0].shape[1]

        # Stack layer outputs: (num_layers, batch, seq_len, hidden)
        stacked_layers = torch.stack(layer_outputs, dim=0)

        # Compute phonetic similarity if source language provided
        if source_lang_id is not None:
            phonetic_similarity = self.similarity_matrix(
                source_lang_id, language_ids
            )  # (batch,)
        else:
            phonetic_similarity = torch.ones(batch_size, device=layer_outputs[0].device)

        # Compute layer importance with phonetic modulation
        layer_weights = F.softmax(self.layer_importance, dim=0)  # (num_layers,)

        # Modulate by phonetic similarity (higher similarity = more fusion)
        layer_weights = layer_weights.unsqueeze(0)  # (1, num_layers)
        phonetic_weights = phonetic_similarity.unsqueeze(1)  # (batch, 1)

        # Reshape for processing: (batch * seq_len, num_layers, hidden)
        stacked_reshaped = stacked_layers.permute(1, 2, 0, 3).reshape(
            batch_size * seq_len, self.num_layers, self.hidden_dim
        )

        # Apply multi-head attention across layers
        fused, attention_weights = self.fusion_attention(
            query=stacked_reshaped,
            key=stacked_reshaped,
            value=stacked_reshaped,
        )

        # Aggregate over the num_layers dimension and reshape back: (batch, seq_len, hidden)
        fused = fused.mean(dim=1)  # (batch * seq_len, hidden)
        fused = fused.view(batch_size, seq_len, self.hidden_dim)

        # Apply language-specific gating
        gates = []
        for i, gate_layer in enumerate(self.language_gates):
            gate = gate_layer(layer_outputs[i])  # (batch, seq_len, 1)
            gates.append(gate)

        gates_tensor = torch.cat(gates, dim=-1)  # (batch, seq_len, num_layers)

        # Weight gates by phonetic similarity and layer importance
        weighted_gates = gates_tensor * phonetic_weights.unsqueeze(1)

        # Final fusion with gating
        gated_layers = []
        for i, layer_out in enumerate(layer_outputs):
            gate = weighted_gates[:, :, i : i + 1]  # (batch, seq_len, 1)
            gated_layers.append(layer_out * gate)

        # Combine gated layers
        gated_sum = torch.stack(gated_layers, dim=0).sum(dim=0)

        # Residual connection with attention-fused output
        output = 0.5 * fused + 0.5 * gated_sum

        # Average attention weights over sequence
        avg_attention = attention_weights.mean(dim=1)  # (batch * seq_len, num_layers)
        avg_attention = avg_attention.view(batch_size, seq_len, self.num_layers).mean(
            dim=1
        )

        return output, avg_attention


class PhonemeInventoryLoss(nn.Module):
    """
    Custom loss function for maintaining language-specific phoneme inventories.

    Novel contribution: Combines standard CTC/cross-entropy loss with a
    regularization term that encourages language-specific phoneme distributions
    while allowing shared representations. Addresses catastrophic forgetting.
    """

    def __init__(
        self,
        num_phonemes: int,
        num_languages: int,
        inventory_weight: float = 0.1,
        diversity_weight: float = 0.05,
    ) -> None:
        """
        Initialize phoneme inventory loss.

        Args:
            num_phonemes: Total number of phonemes in vocabulary
            num_languages: Number of supported languages
            inventory_weight: Weight for inventory preservation term
            diversity_weight: Weight for diversity regularization
        """
        super().__init__()
        self.num_phonemes = num_phonemes
        self.num_languages = num_languages
        self.inventory_weight = inventory_weight
        self.diversity_weight = diversity_weight

        # Learnable phoneme distribution priors for each language
        self.phoneme_priors = nn.Parameter(
            torch.ones(num_languages, num_phonemes) / num_phonemes
        )

        # Base loss (will use cross-entropy)
        self.base_loss = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding

        logger.info(
            f"Initialized PhonemeInventoryLoss with inventory_weight={inventory_weight}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss with inventory preservation.

        Args:
            logits: Model predictions (batch, seq_len, num_phonemes)
            targets: Target phoneme sequences (batch, seq_len)
            language_ids: Language IDs (batch,)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size, seq_len, num_phonemes = logits.shape

        # Reshape for cross-entropy: (batch * seq_len, num_phonemes)
        logits_flat = logits.reshape(-1, num_phonemes)
        targets_flat = targets.reshape(-1)

        # Base cross-entropy loss
        base_loss = self.base_loss(logits_flat, targets_flat)

        # Compute predicted phoneme distributions per language
        probs = F.softmax(logits, dim=-1)  # (batch, seq_len, num_phonemes)

        # Average over sequence length
        lang_distributions = probs.mean(dim=1)  # (batch, num_phonemes)

        # Get language-specific priors (normalize with softmax to ensure valid distribution)
        batch_priors = F.softmax(self.phoneme_priors[language_ids], dim=-1)  # (batch, num_phonemes)

        # KL divergence from priors (inventory preservation)
        # Clamp lang_distributions to avoid log(0)
        kl_div = F.kl_div(
            torch.log(lang_distributions.clamp(min=1e-8)),
            batch_priors,
            reduction="batchmean",
            log_target=False,
        )

        # Diversity regularization: encourage using diverse phonemes
        # Penalize overly peaked distributions
        # Clamp to avoid log(0)
        clamped_dist = lang_distributions.clamp(min=1e-8)
        entropy = -(clamped_dist * clamped_dist.log()).sum(dim=-1).mean()
        diversity_loss = -entropy  # Negative entropy (maximize entropy)

        # Combined loss
        total_loss = (
            base_loss
            + self.inventory_weight * kl_div
            + self.diversity_weight * diversity_loss
        )

        loss_dict = {
            "base_loss": base_loss,
            "inventory_loss": kl_div,
            "diversity_loss": diversity_loss,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return x
