import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

from pagnn import settings


def calc_gradient_penalty(args, net_d, real_data, fake_data, adjs, lambda_=10):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = (
        alpha.expand(args.batch_size, fake_data.nelement() // args.batch_size)
        .contiguous()
        .view(args.batch_size, 20, 512)
        .to(settings.device)
    )

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).to(settings.device)

    interpolates = Variable(interpolates, requires_grad=True)
    adjs = [Variable(adj) for adj in adjs]

    # TODO: Don't forget that we changed this
    interpolates = F.softmax(interpolates, 1)

    disc_interpolates = net_d(interpolates, adjs)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=settings.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty
