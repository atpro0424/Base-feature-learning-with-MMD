import torch
import torch.nn as nn
import torch.nn.functional as functional

def MMD(x, y, bandwidth=None):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first set of sample, distribution P, shape (num_of_samples, feature_dim)
        y: second set of sample, distribution Q, shape (num_of_samples, feature_dim)
        bandwidth: kernel bandwidth of the RBF kernel
    """
    if bandwidth == None:
        # Use the median of the two distributions as bandwidth
        combined = torch.cat([x, y], dim=0)
        distances = functional.pdist(combined)
        bandwidth = distances.median()

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    sigma2 = bandwidth * bandwidth

    XX = torch.exp(-0.5*dxx/sigma2)
    YY = torch.exp(-0.5*dyy/sigma2)
    XY = torch.exp(-0.5*dxy/sigma2)

    return torch.mean(XX + YY - 2. * XY).item()