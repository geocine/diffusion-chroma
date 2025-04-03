# MIT License
# Copyright tdrussell
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
#
# Original source: https://github.com/tdrussell/diffusion-pipe/blob/a0a0ca37fdc84a2937f57a0863d4218b4f7bdd22/optimizers/gradient_release.py

import torch

# Simple wrapper for use with gradient release. Grad hooks do the optimizer steps, so this no-ops
# the step() and zero_grad() methods. It also handles state_dict.
class GradientReleaseOptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        ret = []
        for opt in self.optimizers:
            ret.extend(opt.param_groups)
        return ret

    def state_dict(self):
        return {i: opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, sd in state_dict.items():
            self.optimizers[i].load_state_dict(sd)

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass