# MIT License
# Copyright geocine
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
from torch.optim.optimizer import Optimizer

# Stochastic rounding function (assuming it's available globally or imported)
# Re-paste the function here if needed:
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    Copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16 or float16
        source: the target tensor with dtype=float32
    """
    if target.dtype not in [torch.bfloat16, torch.float16]:
        target.copy_(source)
        return

    # Manual bit manipulation version
    source_fp32 = source.float() # Ensure source is float32

    rand_int = torch.randint_like(
        source_fp32,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    source_int_view = source_fp32.view(dtype=torch.int32)
    result_int = source_int_view.add(rand_int)
    result_int.bitwise_and_(-65536) # -65536 = FFFF0000 hex
    target.copy_(result_int.view(dtype=torch.float32))


class PersonaOptimizer(Optimizer): # Renamed Class
    r"""
    PersonaOptimizer: Aims for speed and enhanced feature learning by combining
    robust adaptive momentum techniques with optional gradient variance modulation
    to potentially focus updates on more dynamic features (e.g., person details).

    Inspired by Adafactor, REMASTER, and SCORN, but simplifies normalization
    and introduces a 'focus_boost' mechanism.

    Key Features:
    - Adaptive momentum (ema) and second moment (ema_squared).
    - atan2 normalization for step calculation.
    - Optional Orthograd for potentially better conditioned updates.
    - Optional state resetting for escaping plateaus.
    - Optional Cautious updates for stability.
    - Optional Focus Boost: Modulates updates based on gradient variance.
    - Optional Parameter Constraining (alternative to weight decay).
    - Stochastic Rounding for bf16/fp16 speed/memory.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1e-4).
        betas (Tuple[float, float]): Coefficients for computing running averages of
            gradient (ema) and squared gradient (ema_squared) (default: (0.95, 0.999)).
        weight_decay (float): L2 penalty strength (AdamW style) (default: 0.0).
        weight_decay_rate (float): Multiplicative decay factor for weight decay strength per step (default: 1.0).
        amp_factor (float): Amplification factor for momentum term in the update direction (default: 1.0).
        focus_boost (float): Strength of the gradient variance modulation.
            If > 0, updates for parameters with higher smoothed gradient variance
            are slightly increased. Helps potentially focus on dynamic features.
            (default: 0.0, disabled). Recommended range if used: 0.01 to 0.5.
        focus_eps (float): Epsilon added to variance normalization for stability (default: 1e-7).
        reset_interval (int): Resets optimizer states every `reset_interval` steps.
            If > 0, helps escape local minima/plateaus. (default: 0, disabled).
        reset_increment (int): Increases `reset_interval` by this amount after each reset (default: 0).
        orthograd (bool): Apply Orthogonal Gradient modification (default: False).
        constrain (bool): Use parameter constraining instead of weight decay (default: False).
        cautious_min (float): Minimum value for the cautious update mask. (default: 1.0, disabled).
        stochastic_fp (bool): Use stochastic rounding for bfloat16/float16 parameters (default: True).
        eps (float): Term added to denominator for numerical stability (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.95, 0.999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 1.0,
        amp_factor: float = 1.0,
        focus_boost: float = 0.0,
        focus_eps: float = 1e-7,
        reset_interval: int = 0,
        reset_increment: int = 0,
        orthograd: bool = False,
        constrain: bool = False,
        cautious_min: float = 1.0,
        stochastic_fp: bool = True,
        eps: float = 1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= weight_decay_rate <= 1.0:
             raise ValueError(f"Invalid weight_decay_rate value: {weight_decay_rate}")
        if not 0.0 <= focus_boost:
            raise ValueError(f"Invalid focus_boost value: {focus_boost}")
        if not 0.0 <= cautious_min <= 1.0:
             raise ValueError(f"Invalid cautious_min value: {cautious_min}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            weight_decay_rate=weight_decay_rate,
            amp_factor=amp_factor,
            focus_boost=focus_boost,
            focus_eps=focus_eps,
            reset_interval=reset_interval,
            reset_increment=reset_increment,
            orthograd=orthograd,
            constrain=constrain,
            cautious_min=cautious_min,
            stochastic_fp=stochastic_fp,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _perform_orthograd(self, p):
        """Applies Orthograd modification in-place to p.grad."""
        if p.grad is None or p.ndim < 2:
            return

        w = p.data.flatten().float()
        g = p.grad.data.flatten().float()
        w_norm_sq = torch.dot(w, w)

        # Check for zero tensors to avoid NaN or division by zero
        if w_norm_sq < self.defaults['eps']:
             return

        proj_scalar = torch.atan2(torch.dot(w, g), w_norm_sq).mul_(1.27323954474)
        g_orth = g.sub(w, alpha=proj_scalar)
        g_norm = torch.norm(g)
        g_orth_norm = torch.norm(g_orth)

        if g_norm > self.defaults['eps'] and g_orth_norm > self.defaults['eps']:
            g_orth_scaled = g_orth.mul_(g_norm / g_orth_norm)
        else:
             g_orth_scaled = g_orth # Fallback if norms are too small

        p.grad.data.copy_(g_orth_scaled.view_as(p.grad.data).to(p.grad.dtype))


    @torch.no_grad()
    def _reset_parameter_state(self, state, grad_fp32):
        """Resets the optimizer state for a parameter."""
        state['ema'].zero_()
        state['ema_squared'].copy_(grad_fp32.pow(2))
        state['step'] = 0
        # Retrieve defaults from the instance for these checks/assignments
        default_reset_interval = self.defaults['reset_interval']
        default_focus_boost = self.defaults['focus_boost']

        if default_reset_interval > 0:
            state['times_reset'] = state.get('times_reset', 0) + 1
            state['steps_since_reset'] = 0
        if default_focus_boost > 0 and 'ema_grad_var' in state: # Check if state exists
            state['ema_grad_var'].zero_()


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            wd_rate = group['weight_decay_rate']
            amp = group['amp_factor']
            focus_boost = group['focus_boost']
            focus_eps = group['focus_eps']
            reset_interval = group['reset_interval']
            reset_increment = group['reset_increment']
            use_orthograd = group['orthograd']
            constrain_params = group['constrain']
            cautious_min = group['cautious_min']
            use_stochastic_fp = group['stochastic_fp']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    # Updated error message with new class name
                    raise RuntimeError(f'{self.__class__.__name__} does not support sparse gradients.')

                if use_orthograd:
                    self._perform_orthograd(p)
                    grad = p.grad.data # Re-fetch grad

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['ema'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['ema_squared'] = grad.pow(2).clone().detach()
                    if reset_interval > 0:
                        state['steps_since_reset'] = 0
                        state['times_reset'] = 0
                    if focus_boost > 0:
                        state['ema_grad_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                p_fp32 = p.data.float()
                grad_fp32 = grad.float()
                ema = state['ema'].float()
                ema_squared = state['ema_squared'].float()
                if focus_boost > 0:
                    # Ensure ema_grad_var exists before accessing
                    if 'ema_grad_var' not in state:
                         state['ema_grad_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    ema_grad_var = state['ema_grad_var'].float()


                state['step'] += 1
                step = state['step']

                if reset_interval > 0:
                    state['steps_since_reset'] += 1
                    current_reset_threshold = reset_interval + (reset_increment * state.get('times_reset', 0))
                    if state['steps_since_reset'] >= current_reset_threshold > 0:
                        self._reset_parameter_state(state, grad_fp32)
                        step = state['step'] # Re-fetch step after potential reset
                        ema.zero_() # Re-fetch states after potential reset
                        ema_squared.copy_(grad_fp32.pow(2))
                        if focus_boost > 0 and 'ema_grad_var' in state: ema_grad_var.zero_()
                        # Increment step for the current operation after reset
                        state['step'] += 1
                        step = state['step']
                        state['steps_since_reset'] = 1 # Start counting again

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                bias_correction1 = max(bias_correction1, eps) # Avoid division by zero
                bias_correction2_sqrt = max(math.sqrt(bias_correction2), eps) # Avoid division by zero

                ema.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                ema_squared.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                denom = ema_squared.sqrt().add_(eps * bias_correction2_sqrt) # Add stability term related to bias correction? Or just eps? Let's stick to simpler eps for now.
                #denom = ema_squared.sqrt().add_(eps) # Simpler stability

                corrected_ema = ema / bias_correction1
                combined_term = grad_fp32.add(corrected_ema, alpha=amp)

                atan2_mul = 1.27323954474 # 4/pi
                full_step = torch.atan2(combined_term, denom).mul_(atan2_mul)

                if focus_boost > 0:
                    current_var = (ema_squared - ema.pow(2)).clamp_(min=0)
                    # Use ema_grad_var defined earlier
                    ema_grad_var.mul_(beta2).add_(current_var, alpha=1.0 - beta2)

                    norm_denom = ema_grad_var.sqrt().add_(focus_eps)
                    relative_var = current_var / norm_denom
                    boost_factor = (1.0 + focus_boost * relative_var.clamp_(min=0, max=5.0)) # Clamp impact
                    full_step.mul_(boost_factor)
                    # Update state only if focus_boost is active and state exists
                    state['ema_grad_var'].copy_(ema_grad_var) # Copy back potentially updated fp32 state

                effective_wd = wd * (wd_rate ** (step-1))
                if effective_wd > 0 and not constrain_params:
                    full_step.add_(p_fp32, alpha=effective_wd)

                if cautious_min < 1.0:
                    mask = (full_step * grad_fp32 >= 0).to(full_step.dtype)
                    mask.clamp_min_(cautious_min)
                    mask_mean = mask.mean().clamp_(min=eps)
                    mask.div_(mask_mean)
                    full_step.mul_(mask)

                step_size = lr

                if constrain_params:
                     p_fp32.mul_(1.0 - step_size)
                     p_fp32.add_(full_step, alpha=-step_size)
                else:
                     p_fp32.add_(full_step, alpha=-step_size)

                if use_stochastic_fp and p.dtype in [torch.bfloat16, torch.float16]:
                    copy_stochastic_(p.data, p_fp32)
                    copy_stochastic_(state['ema'], ema)
                    copy_stochastic_(state['ema_squared'], ema_squared)
                    # No need to copy ema_grad_var again here, done inside focus_boost block
                else:
                    p.data.copy_(p_fp32)
                    state['ema'].copy_(ema)
                    state['ema_squared'].copy_(ema_squared)
                    if focus_boost > 0 and 'ema_grad_var' in state: # Copy variance state if active
                         state['ema_grad_var'].copy_(ema_grad_var)

        return loss