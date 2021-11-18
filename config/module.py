import copy
import math

import torch
import torch.nn.functional


def initialize_weights(model):
    """

    :param model: torch.nn.Module (customized by user)
    :return: this function is not return anything
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.uniform_(-1.0 / math.sqrt(m.weight.size()[0]), 1.0 / math.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()


class SiLU(torch.nn.Module):
    """
    paper: https://arxiv.org/pdf/1710.05941.pdf
    explain:
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class Conv2d(torch.nn.Conv2d):
    """
        if args.tf is True, then Conv class use this class

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        s = self.stride
        d = self.dilation
        k = self.weight.shape[-2:]
        h, w = x.size()[-2:]
        pad_h = max((math.ceil(h / s[0]) - 1) * s[0] + (k[0] - 1) * d[0] + 1 - h, 0)
        pad_w = max((math.ceil(w / s[1]) - 1) * s[1] + (k[1] - 1) * d[1] + 1 - w, 0)

        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0)

        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)


class Conv(torch.nn.Module):
    """

    """
    def __init__(self, args, in_channels, out_channels, activation, kernel=1, stride=1, groups=1):
        super().__init__()
        if args.tf:
            self.conv = Conv2d(in_channels, out_channels, kernel, stride, 1, groups, bias=False)
        else:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, 1, groups, bias=False)

        self.norm = torch.nn.BatchNorm2d(out_channels, 0.001, 0.01)
        self.silu = activation

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))


class SE(torch.nn.Module):
    """
    paper: https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // (4 * r), 1),
                                      torch.nn.SiLU(),
                                      torch.nn.Conv2d(ch // (4 * r), ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))


class Residual(torch.nn.Module):
    """
    paper: https://arxiv.org/pdf/1801.04381.pdf
    """
    def __init__(self, args, in_channels, out_channels, s, r, fused=True):
        super().__init__()
        identity = torch.nn.Identity()
        if fused:
            if args.tf and r == 1:
                features = [Conv(args, in_channels, r * in_channels, torch.nn.SiLU(), 3, s)]
            else:
                features = [Conv(args, in_channels, r * in_channels, torch.nn.SiLU(), 3, s),
                            Conv(args, r * in_channels, out_channels, identity)]
        else:
            if r == 1:
                features = [Conv(args, r * in_channels, r * in_channels, torch.nn.SiLU(), 3, s, r * in_channels),
                            SE(r * in_channels, r),
                            Conv(args, r * in_channels, out_channels, identity)]
            else:
                features = [Conv(args, in_channels, r * in_channels, torch.nn.SiLU()),
                            Conv(args, r * in_channels, r * in_channels, torch.nn.SiLU(), 3, s, r * in_channels),
                            SE(r * in_channels, r),
                            Conv(args, r * in_channels, out_channels, identity)]

        self.add = s == 1 and in_channels == out_channels
        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class EMA(torch.nn.Module):
    """
    paper: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    def __init__(self, model, decay=0.9999):
        super.__init__()
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.module.state_dict().values()
            for e, m, in zip(e_std, m_std):
                e.copy_(self.decay * e + (1. - self.decay) * m)


class CrossEntropyLoss(torch.nn.Module):
    """
    paper: https://arxiv.org/pdf/1512.00567.pdf
    """
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        prob = self.softmax(x)
        loss = -prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        return ((1. - self.epsilon) * loss + self.epsilon * (-prob.mean(dim=-1))).mean()




