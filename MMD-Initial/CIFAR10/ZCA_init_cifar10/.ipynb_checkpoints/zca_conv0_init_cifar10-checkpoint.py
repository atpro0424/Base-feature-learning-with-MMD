import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import os
import random
import numpy as np
import einops


# Seed
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    traindir = '/data/datasets/CIFAR10/' 
    save_dir = 'outputs/'
    nimg = 10000 
    normalize_weights = False
    resolution = 32
    add_gray = True
    lmscone = False

    # Create save dir
    save_dir = os.path.join(save_dir, f'conv0_ZCAinit_cifar10res{resolution}')
    if add_gray: save_dir += '_addgray'
    if normalize_weights: save_dir += '_norm'
    if lmscone: save_dir += '_lmscone'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get images
    imgs = get_images(traindir, nimg, resolution, lmscone=lmscone).to(device)

    # Define conv0 parameters
    conv0 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True)
    kernel_size = conv0.kernel_size[0]
    norm_value = None
    if normalize_weights:
        norm_value = get_layer_filters_norm(conv0)

    # Extract Patches (take all patches that don't overlap. Grid)
    patches = extract_patches(imgs, 
                            kernel_size, 
                            step=kernel_size)
    print('\npatches',patches.shape)
    print('patches per image',patches.shape[0]/nimg)

    # Run ZCA to get filters that will be the weights of conv0
    weight, bias = get_filters(patches, 
                               real_cov=False,
                               add_gray=add_gray,
                               norm_value = norm_value, 
                               save_dir=save_dir)

    # Save weights and bias
    torch.save(weight,f'{save_dir}/F_ZCA_weights.pth')
    torch.save(bias,f'{save_dir}/F_ZCA_bias.pth')

    #### Sanity check ####

    # Plot final filters
    plt.figure(figsize=(10,10))
    num_filters = weight.shape[0]
    for i in range(num_filters):
        if i >= weight.shape[0]:
            break
        filter_m = weight[i]
        f_min, f_max = filter_m.min(), filter_m.max()
        filter_m = (filter_m - f_min) / (f_max - f_min) # make them from 0 to 1
        filter_m = filter_m.cpu().numpy().transpose(1,2,0) # put channels last
        plt.subplot(1,num_filters,i+1)
        plt.imshow(filter_m)          
        plt.axis('off')
        i +=1
    plt.savefig(f'{save_dir}/ZCA_filters.jpg',dpi=300,bbox_inches='tight')
    plt.close()
    
    # load conv0 with weight and bias
    conv0.weight = torch.nn.Parameter(weight)
    conv0.bias = torch.nn.Parameter(bias)

    # Check if covariance is identity
    feat_map = conv0(imgs)
    channel_cov_matrix = compute_channel_covariance(feat_map.detach())
    print(f'\nCovariance matrix (Channel) for conv0 ({channel_cov_matrix.shape}):')
    print(channel_cov_matrix)

    error = compute_error(channel_cov_matrix)
    print('Error w.r.t. Identity Matrix:', error.item())

    feat_map_flat = feat_map.reshape(feat_map.size(0), -1) 
    cov_matrix = compute_covariance(feat_map_flat.detach())
    print(f'\nCovariance matrix (Embedding dimension) for conv0 ({cov_matrix.shape}):')
    print(cov_matrix)

    error = compute_error(cov_matrix)
    print('Error w.r.t. Identity Matrix:', error.item())

    return None


def get_images(traindir, nimg, resolution, lmscone=False):
    if lmscone:
        normalize = transforms.Normalize(mean=[0.61700356, 0.6104542, 0.5773882],
                                        std=[1.0, 1.0, 1.0]) # std=[0.23713203, 0.2416522, 0.25481918] # std=[1.0, 1.0, 1.0]
        transform_cifar = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
                ConeTransform(),
                normalize
            ])
    else:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[1.0, 1.0, 1.0]) # std=[0.247, 0.243, 0.261] # std=[1.0, 1.0, 1.0]
        transform_cifar = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
                normalize
            ])
    train_dataset = datasets.CIFAR10(root=traindir, train=True, download=True, transform=transform_cifar)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=nimg, shuffle=True)
    #take only the firs batch of nimg (randomly picked)
    imgs,labels = next(iter(train_loader))

    # # plot first image
    # plt.figure()
    # plt.imshow(imgs[1].permute(1,2,0))
    # plt.axis('off')
    # plt.savefig('first_image.jpg',bbox_inches='tight')
    # plt.close()

    return imgs


def extract_patches(images,window_size,step):
    n_channels = images.shape[1]
    aux = images.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches


def get_layer_filters_norm(layer_module):
    # Get the mean norm value of filters in one layer
    num_filters = layer_module.out_channels
    norm_sum = 0
    for i in range(num_filters):
        filter_ = layer_module.state_dict()['weight'][i]
        norm_sum += torch.norm(filter_.reshape(-1))
    return norm_sum/num_filters


def ZCA(data, real_cov, tiny=1e-6, plot_cov=False, save_dir=None):
    if real_cov:
        C = torch.cov(data)
    else:
        C = (data @ data.T) / data.size(1)
    
    D, V = torch.linalg.eigh(C)
    sorted_indices = torch.argsort(D, descending=True)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    D = torch.clamp(D, min=0)  # Replace negative values with zero
    Di = (D + tiny)**(-0.5)
    Di[~torch.isfinite(Di)] = 0
    zca_trans = V @ torch.diag(Di) @ V.T

    # Verify covariance of whitened data is close to identity matrix
    if plot_cov:
        assert save_dir is not None; 'save_dir must be provided to plot_cov'
        data_white = torch.mm(zca_trans, data) # data_white is d x N
        C = torch.cov(data_white)
        plt.figure()
        plt.imshow(C.detach().cpu().numpy(), interpolation='nearest')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir,'covariance_on_patches.jpg'), bbox_inches='tight')
        plt.close()

    return zca_trans


def add_gray_channels(data, num_colors, filt_size):
    gray = einops.rearrange(data, '(c h w) n -> c (h w) n', c=num_colors, h=filt_size, w=filt_size).mean(dim=0)
    data_with_gray = torch.cat([data, gray], dim=0)
    # Inverse transform for getting grayscale filters
    C_Trans = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=data.dtype)
    C_Trans[-1, :] /= 3
    C_Trans = torch.pinverse(C_Trans)
    C_Trans = C_Trans.to(data.device)
    return data_with_gray, C_Trans


def mergeFilters(W, filt_size, num_colors, d_size, C_Trans, enforce_symmetry, plot_all=False, save_dir=None):
    center = np.ceil(filt_size / 2).astype(int) - 1 # minus 1 because python indexing starts at 0 
    npca = W.size(0)
    F = W.view(npca, num_colors, filt_size, filt_size)
    F = einops.rearrange(F, 'k c h w -> k h w c') # put channels last
    F_centered = torch.zeros((npca, d_size), dtype=W.dtype)
    
    all_centroidX = []
    all_centroidY = []
    for k in range(npca):
        # get centroids
        F_k = F[k]
        a_k = F_k.abs().sum(dim=2)
        centroidX = a_k.sum(dim=0).argmax()
        centroidY = a_k.sum(dim=1).argmax()
        all_centroidX.append(centroidX.item())
        all_centroidY.append(centroidY.item())
        # get shifts
        shiftX = center - centroidX.item()
        shiftY = center - centroidY.item()
        # center filters
        F_k_c = torch.roll(F_k, shifts=(shiftY, shiftX), dims=(0, 1)) 
        # enforce symmetry
        if enforce_symmetry:
            for j in range(num_colors):
                F_k_c[:, :, j] = (F_k_c[:, :, j] + F_k_c[:, :, j].T) / 2
        # Convert grayscale filters back to RGB
        if C_Trans is not None:
            temp = torch.zeros((filt_size, filt_size, 3), dtype=W.dtype)
            for j in range(3):
                temp[:, :, j] = torch.sum(C_Trans[j].view(1, 1, -1) * F_k_c, dim=2)
            F_k_c = temp
        # back to channels first
        F_k_c = einops.rearrange(F_k_c, 'h w c -> c h w')
        # flaten and accumulate
        F_centered[k, :] = F_k_c.reshape(-1)

    # normalize
    F_norm = F_centered / torch.sqrt((F_centered**2).sum(dim=1, keepdim=True))

    # take filters that are not on the edge
    all_centroidX = torch.tensor(all_centroidX)
    all_centroidY = torch.tensor(all_centroidY)
    F_norm_no_edge= F_norm[(all_centroidY != 0) & (all_centroidY != filt_size-1) & (all_centroidX != 0) & (all_centroidX != filt_size-1), :]

    if F_norm_no_edge.shape[0] == num_colors:
        filters = F_norm_no_edge
    else:
        # rise error
        assert F_norm_no_edge.shape[0] == num_colors, 'Need clustering (or handcraft merge) to merge filters. Not implemented yet'

    # # I was using this when merging all filters (not only the ones that are not on the edge)
    # # merge filters (They are in order for groups of 9 when filter size is 3x3)
    # filters = torch.zeros((num_colors, d_size), dtype=W.dtype)
    # for i in range(num_colors):
    #     filters[i, :] = F_norm[9*i:9*(i+1), :].mean(dim=0)

    # plot all filters in F_norm
    if plot_all:
        assert save_dir is not None; 'save_dir must be provided to plot_all'
        plt.figure(figsize=(10,10))
        aux = int(np.sqrt(F_norm.shape[0])) +1
        for i in range(aux*aux):
            if i >= F_norm.shape[0]:
                break
            filter_m = F_norm[i].reshape(3, filt_size, filt_size)
            f_min, f_max = filter_m.min(), filter_m.max()
            filter_m = (filter_m - f_min) / (f_max - f_min) # make them from 0 to 1
            filter_m = filter_m.cpu().numpy().transpose(1,2,0) # put channels last
            plt.subplot(aux,aux,i+1)
            plt.imshow(filter_m)          
            plt.axis('off')
            i +=1
        plt.savefig(os.path.join(save_dir,'ZCA_filters_all.jpg'),dpi=300,bbox_inches='tight')
        plt.close()

    return filters


def get_filters(patches_data, real_cov=False, add_gray=True, norm_value=None, save_dir=None):
    n_patches, num_colors, filt_size, _ = patches_data.shape
    data = einops.rearrange(patches_data, 'n c h w -> (c h w) n') # data is a d x N

    d_size = data.size(0)
    enforce_symmetry = True

    # use double presicion
    data = data.double()
    
    # Calculate gray filter
    oData = data.clone()
    C_Trans = None
    if add_gray:
        data, C_Trans = add_gray_channels(data, num_colors, filt_size)
        num_colors += 1

    # Compute ZCA
    W = ZCA(data, real_cov, plot_cov=True, save_dir=save_dir).to(data.device)
    W = mergeFilters(W, filt_size, num_colors, d_size, C_Trans, enforce_symmetry, plot_all=True, save_dir=save_dir).to(data.device) # num_filters x d

    # Renormalize filter responses
    aZ = ZCA(W @ oData, False)
    W = aZ @ W
    bias = -(W @ oData).mean(dim=1) 

    # verify covariance of W @ oData is close to identity
    print('\nW @ oData')
    cov = torch.cov(W @ oData)
    print(cov)
    plt.figure()
    plt.imshow(cov.detach().cpu().numpy(), interpolation='nearest')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,'W@oData.jpg'), bbox_inches='tight')
    plt.close()
    print('Error w.r.t. Identity Matrix:', compute_error(cov).item())

    # reshape filters
    W = einops.rearrange(W, 'k (c h w) -> k c h w', c=3, h=filt_size, w=filt_size)

    # change norm value of filter to match norm values from random init
    if norm_value is not None:
        for i in range(W.shape[0]):
            filtr_flat = W[i].reshape(-1)
            filtr_flat_1norm = filtr_flat/torch.norm(filtr_flat)
            filtr_flat_new = filtr_flat_1norm*norm_value
            W[i] = filtr_flat_new.reshape(W.shape[1:])

    # back to single precision
    W = W.float()
    bias = bias.float()

    return W, bias


def compute_channel_covariance(feature_map):
    # feature_map --> (batch_size, num_channels, height, width)
    # Verify that the feature map is a 4D tensor
    if feature_map.dim() != 4:
        raise ValueError("The feature map must be a 4D tensor")
    # Reshape the feature map to shape (batch_size * height * width, num_channels)
    reshaped_features = einops.rearrange(feature_map, "b c h w -> (b h w) c").t()
    cov_matrix = torch.cov(reshaped_features)
    return cov_matrix


def compute_covariance(embeddings):
    # Center the embeddings
    embeddings = embeddings - torch.mean(embeddings, dim=0, keepdim=True)
    # Compute the covariance matrix
    cov_matrix = torch.mm(embeddings.t(), embeddings) / (embeddings.size(0) - 1)
    return cov_matrix


# Define a function to compute the error with respect to the identity matrix
def compute_error(cov_matrix):
    # Compute the Frobenius norm of the difference between the covariance matrix and the identity matrix
    identity = torch.eye(cov_matrix.size(0)).to(cov_matrix.device)
    error = torch.norm(identity - cov_matrix, p='fro')
    return error

def srgb2lms(rgb_feat_):
    rgb_feat = rgb_feat_.clone()
    # Assuming rgb_feat is a torch Tensor with shape [channels, height, width] and channels are in sRGB order
    
    # Store original dimensions
    c, sy, sx = rgb_feat.shape
    
    # Convert to float and normalize
    rgb_feat = rgb_feat.float()
    if torch.max(rgb_feat) > 1.0:
        rgb_feat /= 255.0
    
    # Remove sRGB gamma nonlinearity
    mask = rgb_feat <= 0.04045
    rgb_feat[mask] /= 12.92
    rgb_feat[~mask] = ((rgb_feat[~mask] + 0.055) / 1.055) ** 2.4

    # Convert to linear RGB to XYZ to LMS
    xyz_rgb = torch.tensor([[0.4124, 0.3576, 0.1805],
                            [0.2126, 0.7152, 0.0722],
                            [0.0193, 0.1192, 0.9505]])
    xyz_rgb = xyz_rgb / xyz_rgb.sum(dim=1).unsqueeze(1)
    
    lms_xyz = torch.tensor([[0.7328, 0.4296, -0.1624],
                            [-0.7036, 1.6975, 0.0061],
                            [0.0030, 0.0136, 0.9834]])
    # Apply the conversion
    lms_rgb = torch.mm(lms_xyz, xyz_rgb)
    rgb_feat = rgb_feat.permute(1, 2, 0).reshape(-1, 3)
    lms_feat = torch.mm(rgb_feat, lms_rgb.T).clamp(0, 1)
    
    # Reshape back to the original image shape
    lms_feat = lms_feat.view(sy, sx, 3).permute(2, 0, 1)

    return lms_feat

def cone_transform(lms_image_, gamma=0.01):
    """
    Apply the non-linear response function to an LMS image.

    Parameters:
    lms_image (Tensor): A PyTorch tensor of the image in LMS color space normalized between 0 and 1.
    gamma (float): A small value to avoid division by zero and logarithm of zero.

    Returns:
    Tensor: Transformed image after applying non-linear response.
    """
    lms_image = lms_image_.clone()
    # Ensure that the input image is a PyTorch tensor
    if not isinstance(lms_image, torch.Tensor):
        lms_image = torch.tensor(lms_image, dtype=torch.float32)

    # Apply the non-linear response function
    # Avoid in-place operations which might cause issues with autograd
    numerator = np.log(1.0 + gamma) - torch.log(lms_image + gamma)
    denominator = (np.log(1.0 + gamma) - np.log(gamma)) * (gamma - 1)

    r_nonlinear = numerator / denominator + 1

    r_nonlinear = torch.clamp(r_nonlinear, min=0.0)

    return r_nonlinear


class ConeTransform:
    def __call__(self, x):
        return cone_transform(srgb2lms(x))


if __name__ == '__main__':
    main()