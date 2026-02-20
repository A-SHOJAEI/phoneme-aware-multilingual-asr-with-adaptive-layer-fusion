# Phoneme-Aware Multilingual ASR with Adaptive Layer Fusion

A research-tier multilingual automatic speech recognition system that learns phoneme-level representations across low-resource languages using adaptive layer fusion. The novel contribution is a dynamic attention mechanism that selectively combines encoder layers based on phonetic similarity between source and target languages, enabling better cross-lingual transfer while addressing catastrophic forgetting through language-specific phoneme inventories.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Training

Train with default configuration:
```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline (ablation without adaptive fusion):
```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate trained model:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
```

### Inference

Run inference on audio file:
```bash
python scripts/predict.py --checkpoint checkpoints/best.pt --audio path/to/audio.wav --language en
```

## Architecture

The model consists of:
1. Convolutional feature extraction from mel spectrograms
2. Multi-layer transformer encoder with intermediate outputs
3. Adaptive layer fusion module (novel contribution)
4. Transformer decoder with language-conditioned attention
5. Phoneme-level output with custom inventory loss

### Novel Components

**AdaptiveLayerFusion**: Dynamically combines encoder layer representations based on phonetic similarity between languages. Uses multi-head attention with language-specific gating to enable selective feature sharing.

**PhonemeInventoryLoss**: Custom loss function combining cross-entropy with regularization terms for maintaining language-specific phoneme distributions while allowing shared representations. Includes KL divergence from learned priors and entropy-based diversity regularization.

**PhoneticSimilarityMatrix**: Learnable embedding-based similarity metric between language pairs that guides the fusion process.

## Results

Run training to reproduce results:
```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
```

Expected performance targets:
- WER (low-resource languages): < 0.25
- WER (high-resource languages): < 0.08
- Phoneme Error Rate: < 0.15
- Cross-lingual Transfer Gain: > 0.20

## Ablation Studies

Compare full model against baseline:
```bash
# Train full model
python scripts/train.py --config configs/default.yaml

# Train baseline (no adaptive fusion)
python scripts/train.py --config configs/ablation.yaml

# Evaluate both
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints_ablation/best.pt --config configs/ablation.yaml
```

## Project Structure

```
phoneme-aware-multilingual-asr-with-adaptive-layer-fusion/
├── src/phoneme_aware_multilingual_asr_with_adaptive_layer_fusion/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and components
│   ├── training/          # Training loop and utilities
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Configuration and helpers
├── configs/               # YAML configuration files
├── scripts/               # Training, evaluation, and inference scripts
├── tests/                 # Unit tests
└── results/              # Evaluation results (generated)
```

## Configuration

All hyperparameters are configurable via YAML files in `configs/`. Key parameters:

- `model.use_adaptive_fusion`: Enable/disable adaptive layer fusion
- `model.hidden_dim`: Transformer hidden dimension
- `model.num_encoder_layers`: Number of encoder layers to fuse
- `loss.inventory_weight`: Weight for phoneme inventory preservation
- `training.learning_rate`: Learning rate
- `training.scheduler.type`: LR scheduler (cosine, plateau, step)

## Testing

Run tests with coverage:
```bash
pytest tests/ --cov=phoneme_aware_multilingual_asr_with_adaptive_layer_fusion --cov-report=html
```

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
