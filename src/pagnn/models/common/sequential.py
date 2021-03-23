import torch.nn as nn


class SequentialMod(nn.Sequential):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_args_mask = [hasattr(mod, 'takes_extra_args') for mod in self._modules.values()]

    def forward(self, input, *args, **kwargs):
        for module, extra_args in zip(self._modules.values(), self.extra_args_mask):
            if extra_args:
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input
