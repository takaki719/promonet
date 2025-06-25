# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProMoNet (Prosody Modification Network) is a neural speech editing system that enables fine-grained control over prosodic features (pitch, timing, loudness) while preserving linguistic content. The system processes speech through a three-stage pipeline: preprocessing → editing → synthesis.

## Development Commands

### Environment Setup
```bash
# Install the package in development mode
pip install -e .

# Required additional dependencies for development
git clone git@github.com:maxrmorrison/torbi
pip install torbi/
git clone -b dev git@github.com:interactiveaudiolab/penn
pip install penn/
```

### Training and Evaluation
```bash
# Full training pipeline (requires GPU)
python -m promonet.data.download --datasets vctk
python -m promonet.data.augment --datasets vctk  
python -m promonet.data.preprocess --datasets vctk --gpu 0
python -m promonet.partition --datasets vctk
python -m promonet.train --gpu 0
python -m promonet.evaluate --datasets vctk --gpu 0

# Speaker adaptation
python -m promonet.adapt --name speaker_name --files audio1.wav audio2.wav --gpu 0
```

### CLI Usage Examples
```bash
# Preprocess audio files
python -m promonet.preprocess --files input.wav --gpu 0

# Edit speech features
python -m promonet.edit --loudness_files loudness.pt --pitch_files pitch.pt \
    --periodicity_files periodicity.pt --ppg_files ppg.pt \
    --output_prefixes edited --pitch_shift_cents 100

# Synthesize audio from features  
python -m promonet.synthesize --loudness_files loudness.pt --pitch_files pitch.pt \
    --periodicity_files periodicity.pt --ppg_files ppg.pt \
    --output_files output.wav --gpu 0
```

### Monitoring Training
```bash
# Launch tensorboard to monitor training
tensorboard --logdir runs/ --port 6006 --load_fast true
```

## Code Architecture

### Core Processing Pipeline
1. **Preprocessing** (`promonet/preprocess/`): Extract multi-modal features from audio
   - Loudness: A-weighted loudness in 8 frequency bands
   - Pitch: F0 estimation with optional Viterbi decoding  
   - Periodicity: Voiced/unvoiced probability
   - PPG: Phonetic posteriorgrams from ASR models
   - Text: Transcriptions via Whisper ASR

2. **Editing** (`promonet/edit/`): Manipulate prosodic features
   - Pitch shifting, time stretching, loudness scaling
   - Content-aware manipulation using PPG guidance
   - Grid-based interpolation for smooth transitions

3. **Synthesis** (`promonet/synthesize/`): Generate audio from features
   - Multiple generator architectures (HiFi-GAN, FARGAN, Vocos)
   - Multi-speaker synthesis with speaker embeddings
   - Zero-shot speaker adaptation via WavLM x-vectors

### Key Neural Components
- **Generator** (`model/generator.py`): Configurable vocoder architectures
- **Discriminator** (`model/discriminator.py`): Multi-scale adversarial training
- **Training** (`train/core.py`): Adversarial training with multiple objectives

### Configuration System
- Configuration managed via `yapecs` with defaults in `promonet/config/defaults.py`
- Experimental configs in `config/` directory for ablations and baselines
- Override defaults by creating config files in `config/` directory

## Important Implementation Details

### Feature Processing
- All features synchronized at 256-sample hop (11.6ms at 22.05kHz)
- PPG features: 40 channels representing phoneme probabilities
- Pitch embedding: 64-dim learned embeddings from 256 pitch bins
- Speaker embeddings: 256-dim (learned or WavLM x-vectors)

### Data Management
- Preprocessed features cached in `data/cache/`
- Datasets stored in `data/datasets/`
- Training artifacts in `runs/`
- Evaluation results in `eval/`

### GPU Usage
- Most operations support GPU acceleration via `--gpu` flag
- Training requires GPU with sufficient memory (typically 11GB+)
- Inference can run on CPU but GPU recommended for speed

## Common Development Patterns

### Adding New Features
1. Implement preprocessing in `promonet/preprocess/`
2. Add editing operations in `promonet/edit/`
3. Update generator input handling in `model/generator.py`
4. Add configuration parameters to `config/defaults.py`

### Extending Models
- Generator architectures defined in `model/` directory
- New discriminators can be added to `model/discriminator.py`
- Training losses defined in `train/loss.py`

### Dataset Integration
- Dataset downloaders in `promonet/data/download/`
- Preprocessing scripts in `promonet/data/preprocess/`
- Partitioning logic in `promonet/partition/`

## Key Dependencies
- PyTorch/torchaudio for deep learning
- librosa for audio processing
- whisper/transformers for ASR
- penn/torbi for pitch estimation
- ppgs for phonetic features
- vocos for vocoder architecture