"""Main ASR model with phoneme-aware adaptive layer fusion."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from phoneme_aware_multilingual_asr_with_adaptive_layer_fusion.models.components import (
    AdaptiveLayerFusion,
    PhonemeInventoryLoss,
    PositionalEncoding,
)

logger = logging.getLogger(__name__)


class PhonemeAwareASRModel(nn.Module):
    """
    Multilingual ASR model with phoneme-aware adaptive layer fusion.

    Architecture:
    1. Convolutional feature extraction from mel spectrograms
    2. Multi-layer transformer encoder with intermediate outputs
    3. Adaptive layer fusion module (novel contribution)
    4. Transformer decoder with language-conditioned attention
    5. Phoneme-level output with custom inventory loss
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
        num_phonemes: int = 64,
        num_languages: int = 4,
        dropout: float = 0.1,
        use_adaptive_fusion: bool = True,
    ) -> None:
        """
        Initialize phoneme-aware ASR model.

        Args:
            input_dim: Input feature dimension (e.g., 80 for mel spectrogram)
            hidden_dim: Hidden dimension for transformer
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            num_phonemes: Vocabulary size (number of phonemes)
            num_languages: Number of supported languages
            dropout: Dropout probability
            use_adaptive_fusion: Whether to use adaptive layer fusion
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_phonemes = num_phonemes
        self.num_languages = num_languages
        self.use_adaptive_fusion = use_adaptive_fusion

        # Convolutional frontend for feature extraction
        self.conv_frontend = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder (store intermediate layer outputs)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Adaptive layer fusion (novel component)
        if use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveLayerFusion(
                num_layers=num_encoder_layers,
                hidden_dim=hidden_dim,
                num_languages=num_languages,
                num_heads=num_heads,
            )

        # Language embeddings
        self.language_embeddings = nn.Embedding(num_languages, hidden_dim)

        # Transformer decoder
        self.decoder_embedding = nn.Embedding(num_phonemes, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_phonemes)

        # Custom loss function
        self.criterion = PhonemeInventoryLoss(
            num_phonemes=num_phonemes,
            num_languages=num_languages,
            inventory_weight=0.1,
            diversity_weight=0.05,
        )

        self._init_weights()

        logger.info(
            f"Initialized PhonemeAwareASRModel: "
            f"hidden_dim={hidden_dim}, encoder_layers={num_encoder_layers}, "
            f"adaptive_fusion={use_adaptive_fusion}"
        )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode audio features with optional adaptive fusion.

        Args:
            features: Input features (batch, n_mels, time)
            feature_lengths: Lengths of features (batch,)
            language_ids: Language IDs (batch,)

        Returns:
            Tuple of (encoded_features, layer_outputs)
        """
        batch_size = features.shape[0]

        # Convolutional frontend: (batch, n_mels, time) -> (batch, hidden, time)
        x = self.conv_frontend(features)

        # Transpose for transformer: (batch, time, hidden)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Add language embeddings
        lang_emb = self.language_embeddings(language_ids)  # (batch, hidden)
        x = x + lang_emb.unsqueeze(1)  # Broadcast to (batch, time, hidden)

        # Create padding mask
        max_len = x.shape[1]
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= feature_lengths.unsqueeze(
            1
        )

        # Pass through encoder layers, collecting intermediate outputs
        layer_outputs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_key_padding_mask=mask)
            layer_outputs.append(x)

        # Apply adaptive layer fusion if enabled
        if self.use_adaptive_fusion:
            x, fusion_weights = self.adaptive_fusion(layer_outputs, language_ids)
        else:
            x = layer_outputs[-1]  # Use only final layer

        return x, layer_outputs

    def decode(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode phoneme sequences.

        Args:
            encoder_output: Encoder output (batch, enc_len, hidden)
            targets: Target phoneme sequences (batch, dec_len)
            encoder_mask: Encoder padding mask (batch, enc_len)

        Returns:
            Decoder output logits (batch, dec_len, num_phonemes)
        """
        # Embed target phonemes
        tgt_emb = self.decoder_embedding(targets)  # (batch, dec_len, hidden)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Create causal mask for decoder
        tgt_len = targets.shape[1]
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=targets.device), diagonal=1
        ).bool()

        # Decode
        decoder_output = self.decoder(
            tgt=tgt_emb,
            memory=encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=encoder_mask,
        )

        # Project to phoneme vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for training.

        Args:
            features: Input features (batch, n_mels, time)
            feature_lengths: Feature lengths (batch,)
            targets: Target phoneme sequences (batch, seq_len)
            language_ids: Language IDs (batch,)

        Returns:
            Tuple of (loss, loss_dict)
        """
        # Encode
        encoder_output, layer_outputs = self.encode(
            features, feature_lengths, language_ids
        )

        # Create encoder padding mask
        max_len = encoder_output.shape[1]
        encoder_mask = torch.arange(max_len, device=features.device).unsqueeze(
            0
        ) >= feature_lengths.unsqueeze(1)

        # Shift targets for teacher forcing (remove last token)
        decoder_input = targets[:, :-1]
        decoder_target = targets[:, 1:]

        # Decode
        logits = self.decode(encoder_output, decoder_input, encoder_mask)

        # Compute loss with custom criterion
        loss, loss_dict = self.criterion(logits, decoder_target, language_ids)

        return loss, loss_dict

    def generate(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        language_ids: torch.Tensor,
        max_len: int = 100,
        sos_token: int = 1,
        eos_token: int = 2,
    ) -> torch.Tensor:
        """
        Generate phoneme sequences (inference).

        Args:
            features: Input features (batch, n_mels, time)
            feature_lengths: Feature lengths (batch,)
            language_ids: Language IDs (batch,)
            max_len: Maximum generation length
            sos_token: Start-of-sequence token ID
            eos_token: End-of-sequence token ID

        Returns:
            Generated phoneme sequences (batch, seq_len)
        """
        batch_size = features.shape[0]
        device = features.device

        # Encode
        encoder_output, _ = self.encode(features, feature_lengths, language_ids)

        # Create encoder padding mask
        max_len_enc = encoder_output.shape[1]
        encoder_mask = torch.arange(max_len_enc, device=device).unsqueeze(
            0
        ) >= feature_lengths.unsqueeze(1)

        # Initialize with SOS tokens
        generated = torch.full(
            (batch_size, 1), sos_token, dtype=torch.long, device=device
        )

        # Generate autoregressively
        for _ in range(max_len):
            # Decode current sequence
            logits = self.decode(encoder_output, generated, encoder_mask)

            # Get last token prediction
            next_token_logits = logits[:, -1, :]  # (batch, vocab)
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=1)

            # Check if all sequences have EOS token
            if (next_tokens == eos_token).all():
                break

        return generated
