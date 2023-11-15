import torch
import torch.nn as nn
import torch.nn.functional as functional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def register_hook(task, layer_name):
    # the hook signature
    def hook(model, input, output):
        # """
        # According to what Chris said to Jason, to turn the feature map to input vector for MMD,
        # I extracted patches the size of the filter, and vectorize each patch.
        # (e.g. input = (batch_size, C, H, W)
        #       -> a filter size of (3 x 3), padding 1, stride 1
        #       -> (batch_size, C x 3 x 3, number_of_patches)
        #       -> (number_of_patches, batch_size, C x 3 x 3)
        # )

        # When computing MMD, we iterate the number of patches.
        # At each iteration, we compute mmd between two groups of (C x 3 x 3)-dim vectors, each group having batch_size samples.
        # And we take the mean mmd of all the patches. (Intuitively we are computing mean mmd of all patches.)
        # """
        if layer_name == 'fc':
            features = input[0].detach()
        else:
            unfold = nn.Unfold(kernel_size=model.kernel_size, padding=model.padding, stride=model.stride)
            features = unfold(input[0].detach())
            features = features.permute(2, 0, 1)
        if features.shape[0] > 6000:
            shuffled_indices = torch.randperm(features.shape[0])
            selected_indices = shuffled_indices[:6000]
            features = features[selected_indices]
        
        # if layer_name == 'fc':
        #     features = input[0].detach()
        # else:
        #     unfold = nn.Unfold(kernel_size=model.kernel_size, padding=model.padding, stride=model.stride)
        #     features = unfold(input[0].detach())
        #     features = features.reshape(features.shape[0] * features.shape[2], input[0].shape[1], model.kernel_size[0], model.kernel_size[1]) # (batch_size * num_patches, Channel, patch_width, patch_heigh)
        #     # Global Average Pooling
        #     gap = nn.AdaptiveAvgPool2d((1, 1))
        #     pooled_features = gap(features)  # Shape will be [batchsize * num_patch, C, 1, 1]
        #     features = pooled_features.view(pooled_features.size(0), -1)
        # if features.shape[0] > 6000:
        #     shuffled_indices = torch.randperm(features.shape[0])
        #     selected_indices = shuffled_indices[:6000]
        #     features = features[selected_indices]

        task[layer_name] = features

    return hook

def register_hooks(model, task, module_type, start_layer=0, num_layers=None):
    hook_handler = []
    layer_count = 0
    for i, (name, module) in enumerate(model.named_modules()):
        if i < start_layer:
            continue
        if isinstance(module, module_type):
            layer_count += 1
            handle = module.register_forward_hook(register_hook(task, name))
            hook_handler.append(handle)
        if num_layers is not None:
            if layer_count >= num_layers:
                return hook_handler
    return hook_handler

def remove_hooks(hook_handler):
    for handle in hook_handler:
        handle.remove()

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
        
        
def MMD_batch(x, y, bandwidth=None):
    """Compute MMD for batches of features.

    Args:
        x, y: (batchsize, number_of_features, feature_dim)
        bandwidth: kernel bandwidth of the RBF kernel
    """
    assert x.shape[0] == y.shape[0], "Patch num of x and y must be the same"

    batch_size = x.shape[0]
    mmd_accumulated = 0.0

    for i in range(batch_size):
        xi, yi = x[i], y[i]
        if bandwidth == None:
            # Use the median of the two distributions as bandwidth
            combined = torch.cat([xi, yi], dim=0)
            distances = functional.pdist(combined)
            bandwidth = distances.median()

        xx, yy, zz = torch.mm(xi, xi.t()), torch.mm(yi, yi.t()), torch.mm(xi, yi.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz

        sigma2 = bandwidth * bandwidth

        XX = torch.exp(-0.5*dxx/sigma2)
        YY = torch.exp(-0.5*dyy/sigma2)
        XY = torch.exp(-0.5*dxy/sigma2)

        mmd_accumulated += torch.mean(XX + YY - 2. * XY)

    return (mmd_accumulated / batch_size).item()


def get_MMD_at_every_layer(model, data_loader1, data_loader2, device, ZCA = None, bandwidth = None, start_layer=1, num_layers=None):
    model = model.to(device)
    model.eval()
    MMD_mean_at_every_layer = {}

    for i, ((X1, _), (X2, _)) in enumerate(zip(data_loader1, data_loader2)):
        t0 = {}
        t1 = {}
        if ZCA == None:
            X1 = X1.to(device)
            X2 = X2.to(device)
        else: 
            X1 = ZCA.transform(X1).to(device)
            X2 = ZCA.transform(X2).to(device)
        
        hook_handler = register_hooks(model, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
        with torch.no_grad():
            model(X1)
        remove_hooks(hook_handler)

        hook_handler = register_hooks(model, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
        with torch.no_grad():
            model(X2)
        remove_hooks(hook_handler)

        for key in t0.keys():
            if i == 0: # initialize dict
                MMD_mean_at_every_layer[key] = 0
            if len(t0[key].shape) == 3:
                MMD_mean_at_every_layer[key] += MMD_batch(t0[key],t1[key],bandwidth)
            else:
                MMD_mean_at_every_layer[key] += MMD(t0[key],t1[key],bandwidth)

    for key in MMD_mean_at_every_layer.keys():
        MMD_mean_at_every_layer[key] /= (i + 1)
    return MMD_mean_at_every_layer
