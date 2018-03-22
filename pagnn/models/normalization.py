import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class InstanceNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super().__init__(num_features, eps, momentum, affine)
        self.use_running_stats = False

    def forward(self, input):
        b, c = input.size(0), input.size(1)

        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        weight, bias = None, None

        # Apply instance norm
        input_reshaped = input.contiguous().view(1, b * c, *input.size()[2:])

        out = F.batch_norm(input_reshaped, running_mean, running_var, weight, bias,
                           not self.use_running_stats, self.momentum, self.eps)

        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(b, c).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])

    def use_running_stats(self, mode=True):
        r"""Set using running statistics or instance statistics.
        Instance normalization usually use instance statistics in both training
        and evaluation modes. But users can set this method to use running
        statistics in the fashion similar to batch normalization in eval mode.
        """
        self.use_running_stats = mode


def instance_norm(input_):
    b, c, d = input_.shape
    input_reshaped = input_.contiguous().view(1, b * c, d)
    import pdb
    pdb.set_trace()
    output = F.batch_norm(
        input_reshaped,
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
        momentum=0.1,
        eps=1e-5)
    output_reshaped = output.view(b, c, d)
    return output_reshaped


def layer_norm(input_):
    b, c, d = input_.shape
    return F.batch_norm(
        input_.transpose(1, 2).contiguous().view(1, b * d, c),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
        momentum=0.1,
        eps=1e-5).view(b, d, c).transpose(1, 2).contiguous()
