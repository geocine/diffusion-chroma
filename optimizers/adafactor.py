# Original implementation by 2kpr
#
# Adafactor, and Stochastic Rounding, copied and modified from:
# - https://github.com/Nerogar/OneTrainer/blob/master/modules/util/bf16_stochastic_rounding.py
# - https://github.com/Nerogar/OneTrainer/blob/master/modules/util/optimizer/adafactor_extensions.py
import math
import torch
from transformers import Adafactor
from torch.optim.optimizer import Optimizer

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    Copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
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

    del result


@torch.no_grad()
def step_adafactor_parameter(self, p, g):
    """
    Modified version of Adafactor's step method for a single parameter
    with support for stochastic rounding for bfloat16 parameters
    """
    group = self.param_groups[g]
    if p.grad is None:
        return
    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    factored, use_first_moment = self._get_options(group, grad_shape)
    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        if use_first_moment:
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)

        state["RMS"] = 0
    else:
        if use_first_moment:
            state["exp_avg"] = state["exp_avg"].to(grad)
        if factored:
            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()

    state["step"] += 1
    state["RMS"] = self._rms(p_data_fp32)
    lr = self._get_lr(group, state)

    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
    update = (grad ** 2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

        # Approximation of exponential moving average of square of gradient
        update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.mul_(lr)

    if use_first_moment:
        exp_avg = state["exp_avg"]
        exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
        update = exp_avg

    if group["weight_decay"] != 0:
        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

    p_data_fp32.add_(-update)

    if p.dtype == torch.bfloat16:
        copy_stochastic_(p, p_data_fp32)
    elif p.dtype != torch.float32:
        p.copy_(p_data_fp32)


@torch.no_grad()
def step_adafactor(self, closure=None):
    """
    Performs a single optimization step

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for g, group in enumerate(self.param_groups):
        for p in group["params"]:
            if p.grad is not None:
                step_adafactor_parameter(self, p, g)

    return loss


class EnhancedAdafactor(Adafactor):
    """
    Enhanced Adafactor with support for stochastic rounding for bfloat16 parameters
    and per-parameter optimization functionality.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: None).
        eps (tuple[float, float], optional): Regularization constants for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3)).
        clip_threshold (float, optional): Threshold of root mean square of
            final gradient update (default: 1.0).
        decay_rate (float, optional): Coefficient used to compute running averages of square
            gradient (default: -0.8).
        beta1 (float, optional): Coefficient used for computing running averages of gradient
            (default: None).
        weight_decay (float, optional): Weight decay coefficient (default: 0).
        scale_parameter (bool, optional): If True, learning rate is scaled by root mean square of
            parameter (default: True).
        relative_step (bool, optional): If True, time-dependent learning rate is computed
            instead of external learning rate (default: True).
        warmup_init (bool, optional): Time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False).
    """
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        self.step_parameter = step_adafactor_parameter.__get__(self, EnhancedAdafactor)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for g, group in enumerate(self.param_groups):
            for p in group["params"]:
                if p.grad is not None:
                    self.step_parameter(p, g)

        return loss 