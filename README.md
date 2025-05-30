# Diffusion-Chroma

A project for fine-tuning the [Chroma](https://huggingface.co/lodestones/Chroma) diffusion model using LoRA. This repo is based on [tdrussell's diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) and [lodestone-rock/flow](https://github.com/lodestone-rock/flow). 

## Overview

This repository provides tools and scripts for training custom Chroma image generation models using the Low-Rank Adaptation (LoRA) technique. Chroma is a powerful text-to-image diffusion model, and this project makes it easier to fine-tune it on your own datasets.

## Prerequisites

- CUDA-compatible GPU with at least 24GB VRAM (or less with appropriate configuration)
- Linux-based operating system
- A Hugging Face token in a file named `token` or as an environment variable `HF_TOKEN`

## Quick Start

1. **Intialize the workspace**:
   ```bash
   curl -s https://raw.githubusercontent.com/geocine/diffusion-chroma/refs/heads/main/init.sh | bash
   ```

2. **Prepare your dataset**:
   - Place your training images in the `/workspace/dataset/images` directory
   - Configure your dataset settings in `dataset.toml`

3. **Configure training settings**:
   - Modify `chroma.toml` to adjust training parameters
   
4. **Start training**:
   ```bash
   ./run.sh chroma.toml
   ```
   
5. **Generate samples**:
   Samples will be generated automatically during training in the output directory

## Configuration Files

| File | Description |
|------|-------------|
| dataset.toml | This file defines how your training data is processed |
| chroma.toml | This file contains the main training configuration |

## Scripts

### init.sh

Initializes the project by:
- Cloning the repository
- Installing dependencies
- Downloading required model files

### download.sh

Downloads the necessary model files:
- FLUX.1-schnell: Base diffusion model
- Chroma: Pre-trained model weights (chroma-unlocked-v16.safetensors)

Requires a Hugging Face token in a file named `token` or as an environment variable `HF_TOKEN`.

### run.sh

Manages the training process:
- Runs the captioning script
- Automatically resumes from the latest checkpoint (if available)
- Starts training with the specified configuration

Usage:
```bash
./run.sh chroma.toml [--clear]
```

Options:
- `--clear`: Clears existing captions before processing

## License

This project incorporates code from multiple sources under different licenses:

- **Chroma Model**: This project fine-tunes the [Chroma model](https://huggingface.co/lodestones/Chroma) from [lodestone-rock/flow](https://github.com/lodestone-rock/flow), the official training code for Chroma, under the Apache License 2.0
- **Diffusion-Pipe**: Core components are adapted from [tdrussell's diffusion-pipe](https://github.com/tdrussell/diffusion-pipe), used under the MIT License
- **Unsloth**: Gradient checkpointing utilities from [unslothai/unsloth-zoo](https://github.com/unslothai/unsloth-zoo) under the GNU Lesser General Public License (LGPL)
- **HuggingFace**: Various components build upon the [Hugging Face libraries](https://github.com/huggingface) under the Apache License 2.0

Please note that this project contains components under multiple licenses including:
- [MIT License](https://opensource.org/licenses/MIT)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [GNU Lesser General Public License](https://www.gnu.org/licenses/lgpl-3.0.html)

When using or modifying this code, please ensure you comply with all relevant license terms.

