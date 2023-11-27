import torch
import torch.nn as nn
import torch.nn.functional as functional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def register_hook_funct(task, layer_name, extract_mode='gap'):
    def hook(module, input, output):

        features = input[0].detach().cpu()

        if extract_mode == 'gap':
            gap = torch.nn.AdaptiveAvgPool2d((1, 1))
            if len(features.shape)>2:
                features = gap(features).squeeze()
        
        elif extract_mode == 'flatten':
            if len(features.shape)>2:
                features = features.view(features.size(0), -1)
        
        if layer_name not in task.keys():
            task[layer_name] = features
        else:
            task[layer_name] = torch.cat((task[layer_name], features), dim=0)
    return hook

# Function to add hooks to a network where outputs will be save on the "task" dictionary
def add_hooks(task, network, extract_mode='gap'):
    hook_handler = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(register_hook_funct(task, name, extract_mode))
            hook_handler.append(handle)
    return hook_handler

def register_hook(task, layer_name):
    def hook(model, input, output):
        
        features = output.detach().cpu()
        
        if len(features.shape) > 2:
            # Global Average Pooling
            gap = nn.AdaptiveAvgPool2d((1, 1))
            features = gap(features).squeeze()
        
        if layer_name not in task.keys():
            task[layer_name] = features
        else:
            task[layer_name] = torch.cat((task[layer_name], features), dim=0)

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
        
def get_MMD_at_every_layer(model, dataloader1, dataloader2, device, ZCA = None, bandwidth = None, start_layer=1, num_layers=None):
    model = model.to(device)
    model.eval()
    MMD_mean_at_every_layer = {}
    t0 = {}
    hook_handler = register_hooks(model, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
    for X, _ in dataloader1:
        _ = model(X.to(device))
    remove_hooks(hook_handler)
    
    t1 = {}
    hook_handler = register_hooks(model, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
    for X, _ in dataloader2:
        _ = model(X.to(device))
    remove_hooks(hook_handler)
#     for i, ((X1, _), (X2, _)) in enumerate(zip(data_loader1, data_loader2)):
#         t0 = {}
#         t1 = {}
#         if ZCA == None:
#             X1 = X1.to(device)
#             X2 = X2.to(device)
#         else: 
#             X1 = ZCA.transform(X1).to(device)
#             X2 = ZCA.transform(X2).to(device)
        
#         hook_handler = register_hooks(model, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
#         with torch.no_grad():
#             model(X1)
#         remove_hooks(hook_handler)

#         hook_handler = register_hooks(model, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
#         with torch.no_grad():
#             model(X2)
#         remove_hooks(hook_handler)

#         for key in t0.keys():
#             if i == 0: # initialize dict
#                 MMD_mean_at_every_layer[key] = 0
#             MMD_mean_at_every_layer[key] += MMD(t0[key],t1[key],bandwidth)

    # for key in MMD_mean_at_every_layer.keys():
    #     MMD_mean_at_every_layer[key] /= (i + 1)
    
    for key in t0.keys():
        MMD_mean_at_every_layer[key] = MMD(t0[key],t1[key],bandwidth)
        
    return MMD_mean_at_every_layer


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

