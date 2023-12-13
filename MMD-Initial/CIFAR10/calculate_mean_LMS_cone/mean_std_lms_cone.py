import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import os
import random
import numpy as np
from tqdm import tqdm


# Seed
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def main():

    traindir = '/data/datasets/CIFAR10/' 
    resolution = 32

    # Create dataset and loader without normalization
    transform_cifar = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            ConeTransform(),
            transforms.Normalize(mean=[0.61700356, 0.6104542, 0.5773882], 
                                 std=[0.23713203, 0.2416522, 0.25481918]),
        ])
    train_dataset = datasets.CIFAR10(root=traindir, train=True, download=True, transform=transform_cifar)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Stack all images
    imgs = []
    for inputs, _ in tqdm(train_loader):
        imgs.append(inputs)
    imgs = torch.cat(imgs, dim=0)
    imgs = imgs.numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print(mean_r,mean_g,mean_b)

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print(std_r,std_g,std_b)

    print('END')

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