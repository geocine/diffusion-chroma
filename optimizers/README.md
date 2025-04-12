# Optimizers for Diffusion-Chroma

This directory contains custom optimizers for Diffusion-Chroma training.

## Available Optimizers

### Enhanced Adafactor (Default)

Original implementation by 2kpr, adapted from the implementation in [Nerogar/OneTrainer](https://github.com/Nerogar/OneTrainer)

An enhanced version of Hugging Face's Adafactor optimizer with support for stochastic rounding for bfloat16 parameters and per-parameter optimization.

Adafactor is memory efficient and works well for large models with many parameters. It is the default optimizer as it provides a good balance between memory efficiency and training performance.

#### Minimum Configuration Example:

```toml
[optimizer]
type = 'adafactor'
lr = 1e-4
scale_parameter = false
relative_step = false
```

#### Full Configuration Example:

```toml
[optimizer]
type = 'adafactor'
lr = 1e-4  # Fixed learning rate (required when relative_step=false)
eps = [1e-30, 1e-3]  # Regularization constants
clip_threshold = 1.0  # Gradient clipping threshold
decay_rate = -0.8  # Coefficient for running averages of squared gradient
beta1 = 0.9  # Coefficient for running averages of gradient (set to None to disable momentum)
weight_decay = 0.01  # Weight decay coefficient
scale_parameter = false  # Set to false when providing explicit lr
relative_step = false  # Set to false when providing explicit lr
warmup_init = false  # Warmup initialization
```

### SCORN

Source: [https://github.com/Clybius/Personalized-Optimizers](https://github.com/Clybius/Personalized-Optimizers) by Clybius

Implements several advanced optimization techniques including spectral normalization, adaptive momentum, and norm-based regularization. 

#### Minimum Configuration Example:

```toml
[optimizer]
type = 'scorn'  
lr = 1e-4
betas = [0.95, 0.9999]
weight_decay = 0.01
```

#### Full Configuration Example:

```toml
[optimizer]
type = 'scorn'  
lr = 1e-4
betas = [0.95, 0.9999]  # Momentum and squared momentum decay rates
focus_ratio = 0.1  # Valley attraction force (0.0-0.2 recommended)
weight_decay = 0.01
weight_decay_rate = 0.998  # Weight decay multiplier decay rate
amp = 5.0  # Beta-adjusted scaling parameter strength
reset_interval = 0  # Set to >= 100 if you want to use reset behavior
reset_increment = 0  # Set to >= 100 if you want to use reset behavior
orthograd = false  # Orthogonal gradient updates
spectral_update_scale = 1.0  # Scale for spectral gradient
constrain = false  # Constrain parameter norm
cautious_min = 1.0  # Cautious mask minimum (0.0 to enable fully)
stochastic_fp = true  # Stochastic rounding for bf16/fp16
```

### REMASTER

Source: [https://github.com/Clybius/Personalized-Optimizers](https://github.com/Clybius/Personalized-Optimizers) by Clybius

A streamlined optimizer that focuses on gradient renormalization and momentum management.

#### Minimum Configuration Example:

```toml
[optimizer]
type = 'remaster'  
lr = 1e-4
betas = [0.95, 0.9999]
weight_decay = 0.01
```

#### Full Configuration Example:

```toml
[optimizer]
type = 'remaster'  
lr = 1e-4
betas = [0.95, 0.9999]  # Momentum and squared momentum decay rates
weight_decay = 0.01
weight_decay_rate = 0.998  # Weight decay multiplier decay rate
amp = 5.0  # Beta-adjusted scaling parameter strength
reset_interval = 0  # Set to >= 100 if you want to use reset behavior
reset_increment = 0  # Set to >= 100 if you want to use reset behavior
orthograd = true  # Orthogonal gradient updates (enabled by default)
cautious_min = 1.0  # Cautious mask minimum (0.0 to enable fully)
stochastic_fp = true  # Stochastic rounding for bf16/fp16
```

### AdamW8bitKahan

A memory-efficient 8-bit version of AdamW with Kahan summation for improved precision.

#### Configuration Example:

```toml
[optimizer]
type = 'AdamW8bitKahan'
lr = 2e-5
betas = [0.9, 0.99]
weight_decay = 0.01
stabilize = false
```

### Prodigy

A learning-rate free optimizer that automatically adapts to the dataset.

#### Configuration Example:

```toml
[optimizer]
type = "Prodigy"
betas = [0.9, 0.99]
weight_decay = 0.05
```

### PersonaOptimizer

Original implementation by geocine under MIT License.

Implements with a unique gradient variance modulation mechanism. This optimizer. It correspond to more dynamic or important features in generative models.

#### Minimum Configuration Example:

```toml
[optimizer]
type = 'persona'
lr = 1e-4
betas = [0.95, 0.999]
weight_decay = 0.01
```

#### Full Configuration Example:

```toml
[optimizer]
type = 'persona'
lr = 1e-4
betas = [0.95, 0.999]  # Coefficients for first and second moment
weight_decay = 0.01
weight_decay_rate = 1.0  # Decay rate for weight decay per step
amp_factor = 1.0  # Amplification factor for momentum
focus_boost = 0.05  # Strength of gradient variance modulation (0.0 to disable)
focus_eps = 1e-7  # Epsilon for variance normalization
reset_interval = 0  # Steps between optimizer state resets (0 to disable)
reset_increment = 0  # Increase reset_interval by this amount after each reset
orthograd = false  # Apply orthogonal gradient modification
constrain = false  # Use parameter constraining instead of weight decay
cautious_min = 1.0  # Minimum value for cautious update mask (1.0 to disable)
stochastic_fp = true  # Use stochastic rounding for bf16/fp16
eps = 1e-8  # Term added for numerical stability
```