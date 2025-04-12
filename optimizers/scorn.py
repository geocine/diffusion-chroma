# Original implementation by Clybius
# - https://github.com/Clybius/Personalized-Optimizers

import torch
from torch.optim import Optimizer
from math import sqrt
from enum import IntEnum
import math

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

# https://github.com/kozistr/pytorch_optimizer/blob/6397d56279ad80b26c4bba7fb4b04852b517fdeb/pytorch_optimizer/optimizer/shampoo_utils.py#L533
def zero_power_via_newton_schulz_5(
    g: torch.Tensor, num_steps: int = 5, eps: float = 1e-12, weights: tuple[int, int, int] = (3, -3.41421356237, 1.41421356237)
) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration
    whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out
    to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but
    rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    :param g: torch.Tensor. matrix.
    :param num_steps: int. number of iterations.
    :param eps: float. add this times I to G, to make is positive definite. For scaling, we multiply it by the largest
        eigenvalue of G.
    :param weights: Tuple[int, int, int]. weights.
    """
    if len(g.shape) != 2:
        raise ValueError('shape of g must be 2-dimensional')

    abc_list = [
      (3955/1024, -8306/1024, 5008/1024),
      (3735/1024, -6681/1024, 3463/1024),
      (3799/1024, -6499/1024, 3211/1024),
      (4019/1024, -6385/1024, 2906/1024),
      (2677/1024, -3029/1024, 1162/1024),
      (2172/1024, -1833/1024,  682/1024)
   ]

    x = g.float()
    x.div_(x.norm().add_(eps))

    if g.size(0) > g.size(1):
        x = x.T

    #for _ in range(num_steps):
    for weight in abc_list:
        a = x @ x.T
        b = weight[1] * a + weight[2] * a @ a
        x = weight[0] * x + b @ x

    if g.size(0) > g.size(1):
        x = x.T

    return x

class LMONorm(IntEnum):
    r"""normalization types."""

    NONE = 0
    AUTO = 1
    SPECTRAL = 2
    SPECTRALCONV = 3
    SIGN = 4
    BIAS = 5
    COL = 6
    ROW = 7


class Norm:
    r"""Base class to perform norm onto Scion. This class does no norm."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        r"""Initialize parameter."""
        return x

    def lmo(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get LMO."""
        return grad


class Col(Norm):
    r"""col-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """

    def __init__(self, normalized: bool = False, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=0, keepdim=True)).mul_(math.sqrt(x.size(0)))
        if self.normalized:
            x.div_(x.size(1))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        d_in, d_out = grad.size()

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True)) / math.sqrt(d_in)
        if self.normalized:
            rms_value.mul_(d_out)

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class Row(Norm):
    r"""row-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """

    def __init__(self, normalized: bool = True, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)

        x.div_(x.norm(dim=-1, keepdim=True))
        if self.normalized:
            x.div_(math.sqrt(x.size(-1)))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=-1, keepdim=True))
        if self.normalized:
            rms_value.mul_(math.sqrt(grad.size(-1)))

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)

        return grad


class BiasRMS(Norm):
    r"""bias RMS."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.zeros_(x)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True))
        grad /= rms_value.add_(eps)
        return grad


class SpectralConv(Norm):
    r"""spectral-convolution normalization.

    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, num_steps: int = 5) -> None:
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()

        d_out, d_in, kernel_size, *_ = x_fp64.size()

        for i in range(kernel_size):
            for j in range(kernel_size):
                torch.nn.init.orthogonal_(x_fp64[..., i, j])

        x_fp64.mul_(math.sqrt(d_out / d_in) / (kernel_size**2))

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        grad = zero_power_via_newton_schulz_5(grad.view(len(grad), -1), self.num_steps).view(grad.shape)

        d_out, d_in, kernel_size, *_ = grad.size()

        grad *= math.sqrt(d_out / d_in) / (kernel_size**2)

        return grad


class Spectral(Norm):
    r"""spectral normalization.

    :param max_scale: bool. set upper bound (1.0) of the scale.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, max_scale: bool = False, normalize: bool = True, num_steps: int = 5) -> None:
        self.max_scale = max_scale
        self.normalize = normalize
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()

        torch.nn.init.orthogonal_(x_fp64)

        d_out, d_in = x_fp64.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        x_fp64.mul_(scale)

        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        grad = zero_power_via_newton_schulz_5(grad.view(len(grad), -1), self.num_steps).view(grad.shape)

        d_out, d_in = grad.size()

        scale: float = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        grad *= scale

        return grad


class Sign(Norm):
    r"""sign normalization.

    :param zero_init: bool. initialize with zero.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    """

    def __init__(self, zero_init: bool = False, normalize: bool = True) -> None:
        self.zero_init = zero_init
        self.normalize = normalize

    def init(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_init:
            return torch.nn.init.zeros_(x)

        d_in: int = x.size(1)

        x = 2 * torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device) - 1
        if self.normalize:
            x.div_(d_in)

        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        d_in: int = grad.size(1)
        return torch.sign(grad).div_(d_in) if self.normalize else torch.sign(grad)


class Auto(Norm):
    r"""choose Norm type automatically."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        ndim: int = x.ndim
        if ndim in (0, 1):
            return BiasRMS().init(x)
        if ndim == 2:
            return Spectral().init(x)
        if ndim in (3, 4):
            return SpectralConv().init(x)
        raise NotImplementedError

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        ndim: int = grad.ndim
        if ndim in (0, 1):
            return BiasRMS().lmo(grad, eps=eps)
        if ndim == 2:
            return Spectral().lmo(grad, eps=eps)
        if ndim in (3, 4):
            return SpectralConv().lmo(grad, eps=eps)
        raise NotImplementedError


def build_lmo_norm(norm_type: int, **kwargs) -> Norm:  # noqa: PLR0911
    r"""Build LMONorm by given norm_type."""
    if norm_type == LMONorm.AUTO:
        return Auto()
    if norm_type == LMONorm.SPECTRAL:
        return Spectral(**kwargs)
    if norm_type == LMONorm.SPECTRALCONV:
        return SpectralConv(**kwargs)
    if norm_type == LMONorm.SIGN:
        return Sign(**kwargs)
    if norm_type == LMONorm.BIAS:
        return BiasRMS()
    if norm_type == LMONorm.COL:
        return Col(**kwargs)
    if norm_type == LMONorm.ROW:
        return Row(**kwargs)
    return Norm()

class SCORN(Optimizer):
    r"""
    SCORN: Applying the idea of no gradient accumulation, as its been superseded by momentum. Faster training, smoother weights, Papa Johns. 
    
    For optimal use: Utilize a gradient accumulation size of 1, highest batch size you can handle, adjust LR as needed. Standard AdamW LR ought to be stable enough.
    
    If you want extra speed, you can utilize the `reset_interval` and `reset_increment` parameter to reset the optimizer states, speeding up gradient descent and accelerating leaving local minima.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float):
            Coefficient used for computing the running average, and the running square of running average (default: 0.95, 0.9999)
        focus_ratio (float):
            Ratio for FOCUS' valley attraction force - https://arxiv.org/abs/2501.12243. (default: 0.0, recommended if used: 0.1)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.998).
        amp (float):
            Beta-adjusted scaling parameter for adding the running average to the gradient, functionally acts as strength value for a low-pass filter. (default: 5.0).
        reset_interval (int):
            Resets the optimizers running averages after (reset_interval + reset_increment * times_reset) steps (default: 0, recommended if used: >=100).
        reset_increment (int):
            Increments the reset_interval by this amount after every reset (default: 0, recommended if used: >=100).
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update, - https://arxiv.org/abs/2501.04697 - extended with atan2 in place of epsilon - https://arxiv.org/abs/2407.05872 (default: False).
        spectral_update_scale (bool):
            Scale the spectral gradient by this value, generally intended for when constrain is used, - https://arxiv.org/pdf/2502.07529 (default: 1.0).
        constrain (bool):
            Scale the parameters by the step size to functionally constrain the norm of the parameters, recommended to divide usual learning rate by spectral_update_scale. (default: False).
        cautious_min (bool):
            Use cautious mask on full step update, clamped to a minimum of cautious_min - https://arxiv.org/abs/2411.16085 (default: 1.0, thus disabling the mask. Use 0 to fully utilize the mask).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.95, 0.9999),
        focus_ratio: float = 0.0,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.998,
        amp: float = 5.0,
        reset_interval: int = 0,
        reset_increment: int = 0,
        orthograd: bool = False,
        spectral_update_scale: float = 1.0,
        constrain: bool = False,
        cautious_min: float = 1.0,
        stochastic_fp: bool = True,
    ):

        self._init_lr = lr

        defaults = dict(
            lr = lr,
            betas = betas,
            focus_ratio = focus_ratio,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            amp = amp,
            reset_interval = reset_interval,
            reset_increment = reset_increment,
            orthograd = orthograd,
            spectral_update_scale = spectral_update_scale,
            constrain = constrain,
            cautious_min = cautious_min,
            stochastic_fp = stochastic_fp,
        )

        super(SCORN, self).__init__(params, defaults)

    # Implementation from: https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/1d2cfa2fe692a828d46a5a29b9667ec924961ac7/prodigyplus/core_optimiser.py#L169C5-L177C48
    @torch.no_grad()
    def orthograd(self, p):
        w = p.view(-1)
        g = p.grad.view(-1)

        proj = torch.dot(w, g).atan2_(torch.dot(w, w)).mul_(1.27323954474)
        g_orth = g.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g.norm(2).div_(g_orth.norm(2).clamp_(min=1e-6)))

        p.grad.copy_(g_orth_scaled.view_as(p.grad))
    
    @torch.no_grad()
    def reset_momentums(self, momentum, sq_momentum, grad):
        momentum.copy_(torch.zeros_like(momentum))
        sq_momentum.copy_(grad.pow(2))

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def init(self):
        for group in self.param_groups:
            norm = build_lmo_norm(LMONorm.AUTO)
            for p in group['params']:
                norm.init(p)
                p.mul_(group['scale'])

    @torch.no_grad()
    def step_parameter(self, p, g):
        """
        Performs a single optimization step for a single parameter.
        This is used by the gradient release mechanism.
        
        Args:
            p: Parameter to update
            g: Group index for this parameter
        """
        if p.grad is None:
            return
            
        group = self.param_groups[g]
        state = self.state[p]
        
        if group["orthograd"] and p.ndim >= 2:
            self.orthograd(p)

        grad = p.grad.data

        # State initialization
        if len(state) == 0:
            # Exponential moving average of gradient values
            state["ema"] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state["ema_squared"] = grad.pow(2)
            # Optional resets
            if group["reset_interval"] > 0:
                state["times_zero"] = 0
                state["steps_since_reset"] = 1
            if group["focus_ratio"] > 0.0:
                state["pbar"] = torch.zeros_like(p.data)
            # Add step to align with main step() method
            state["step"] = 0

        p_fp32 = p.detach().clone()
        if group["focus_ratio"] > 0.0:
            pbar = state["pbar"].detach().clone()
        ema = state["ema"].detach().clone()
        ema_squared = state["ema_squared"].detach().clone()
        # Unpack
        if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
            grad = grad.to(torch.float32)
            if group["focus_ratio"] > 0.0:
                pbar = state["pbar"].detach().clone().to(torch.float32)
            ema = state['ema'].detach().clone().to(torch.float32)
            ema_squared = state['ema_squared'].detach().clone().to(torch.float32)
            p_fp32 = p.detach().clone().to(torch.float32)

        if group["reset_interval"] > 0:
            if state["steps_since_reset"] // (group["reset_interval"] + (group["reset_increment"] * state["times_zero"])) > 0:
                self.reset_momentums(ema, ema_squared, grad)
                if group["focus_ratio"] > 0. and 'pbar' in state:
                    pbar = pbar.copy_(torch.zeros_like(p.data))
                state["times_zero"] += 1
                state["steps_since_reset"] = 1
            step = state["steps_since_reset"]
        else:
            # Increment step
            state["step"] += 1
            step = state["step"]

        slow_beta = ((group["betas"][1]**step - group["betas"][1]) / (group["betas"][1]**step - 1.0))

        bias_correction = 1 - group["betas"][0] ** step
        bias_correction_sqrt = (1 - slow_beta ** step) ** (1 / 2)
        atan2_mul = 1.27323954474 # atan2(1,1) renormalization multiplier on gradient
        step_size = group["lr"]

        # RMS Norm
        rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
        grad.div_(rms)

        # SCION spectral norm
        norm = build_lmo_norm(LMONorm.AUTO)
        grad = norm.lmo(grad, eps=1e-12)#.mul_(spectral_update_scale)

        # Adaptive ema
        mask = (grad * ema > 0).to(grad.dtype)
        mask.clamp_min_(group["betas"][0])
        mask.div_(mask.mean().clamp_(min=1e-3)) # Divide by mean (0.001-1.0)
        ema = ema.mul(mask)

        # Update ema
        ema = ema.mul(group["betas"][0]).add_(grad, alpha=1 - group["betas"][0])

        # Compass amplification
        c_t = grad.add(ema.div(bias_correction), alpha=group["amp"])

        # AdamW debias
        denom = ema_squared.sqrt().div_(bias_correction_sqrt)

        # ADOPT update
        ema_squared = ema_squared.mul(slow_beta).addcmul_(c_t, c_t, value=1 - slow_beta)

        # Atan2-Adamw, Spectral update part 2/2
        full_step = c_t.atan2(denom).mul_(atan2_mul).mul_(group["spectral_update_scale"])

        if group["focus_ratio"] > 0. and 'pbar' in state:
            pbar.lerp_(p_fp32, weight=1 - group["betas"][0])
            pbar_hat = pbar.div(bias_correction)
            pbar_step = p_fp32 - pbar_hat
            full_step.add_(pbar_step, alpha=group["focus_ratio"])

        if group["weight_decay"] != 0 and not group["constrain"]:
            # Perform weight decay
            grad_weights = p_fp32.data

            full_step = full_step.add(grad_weights, alpha=group["weight_decay"] * group["weight_decay_rate"]**step)

        # Apply caution as per 'Cautious Optimizers' with a modified minimum.
        if group["cautious_min"] != 1.0:
            mask = (full_step * grad > 0).to(full_step.dtype)
            mask.clamp_min_(group["cautious_min"])
            mask.div_(mask.mean().clamp_(min=1e-3))
            full_step = full_step.mul(mask)

        if group["constrain"]:
            p_fp32.mul_(1.0 - step_size)

        p_fp32.data.add_(full_step, alpha=-step_size)
        if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
            copy_stochastic_(state["ema"], ema)
            copy_stochastic_(state["ema_squared"], ema_squared)
            if group["focus_ratio"] > 0.0:
                copy_stochastic_(state["pbar"], pbar)
            copy_stochastic_(p, p_fp32)
        else:
            state["ema"].copy_(ema)
            state["ema_squared"].copy_(ema_squared)
            if group["focus_ratio"] > 0.0:
                state["pbar"].copy_(pbar)
            p.copy_(p_fp32)
        if group["reset_interval"] > 0:
            state["steps_since_reset"] += 1

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # For each parameter, call the single-parameter optimization code
                g_index = self.param_groups.index(group)
                self.step_parameter(p, g_index)
                
        return loss