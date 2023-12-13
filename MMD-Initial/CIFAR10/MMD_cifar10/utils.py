import torch
import einops

import numpy as np

# Create hook function
def register_hook_funct(task, layer_name, extract_mode='gap'):
    def hook(module, input, output):

        features = output.detach().cpu()

        if extract_mode == 'gap':
            gap = torch.nn.AdaptiveAvgPool2d((1, 1))
            if len(features.shape)>2:
                features = gap(features).squeeze()
        
        elif extract_mode == 'flatten':
            if len(features.shape)>2:
                features = features.view(features.size(0), -1)
        
        elif extract_mode == 'channel':
            if len(features.shape)>2:
                features = einops.rearrange(features, 'b c h w -> (b h w) c')
        
        if layer_name not in task.keys():
            task[layer_name] = features
        else:
            task[layer_name] = torch.cat((task[layer_name], features), dim=0)
    return hook

# Function to add hooks to a network where outputs will be save on the "task" dictionary
def add_hooks(task, network, extract_mode):
    hook_handler = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(register_hook_funct(task, name, extract_mode))
            hook_handler.append(handle)
    return hook_handler

# Function to remove hooks
def remove_hooks(hook_handler):
    for handle in hook_handler:
        handle.remove()


def get_ZCA_matrix(dataset, num_imgs):

    # Get images
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_imgs, shuffle=True)
    imgs,labels = next(iter(loader))

    # create class
    ZCA_obj = ZCA_class()

    # fit ZCA
    ZCA_obj.fit(imgs)

    return ZCA_obj

class ZCA_class:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.mean = None
        self.ZCA_matrix = None

    def fit(self, imgs):
        num_imgs = imgs.shape[0]

        A = imgs.view(imgs.shape[0],-1).t() # data is a d x N

        # Centering data
        self.mean = torch.mean(A, dim=1, keepdim=True)
        A_c = A - self.mean

        # calculate ZCA matrix (convariance and eigendecomposition)
        C = torch.mm(A_c, A_c.t()) / (num_imgs-1) # covariance matrix with data not centered
        s, V = torch.linalg.eigh(C)
        sorted_indices = torch.argsort(s, descending=True)
        s = s[sorted_indices]
        V = V[:, sorted_indices]

        # # SVD to skip covariance matrix (doing the covariance matrix way takes too much memory)
        # _, s_svd, v_svd = torch.svd(A_c.t()) # it takes data as N x b. It works if N > b
        # s = s_svd * s_svd / (num_imgs-1) # eigenvalues
        # V = v_svd # eigenvectors
        # sorted_indices = torch.argsort(s, descending=True)
        # s = s[sorted_indices]
        # V = V[:, sorted_indices]

        s[s < 0] = 0 # make negative eigenvalues 0
        diag = torch.diag(1.0 / torch.sqrt(s + self.epsilon)) # diag = S^(-1/2)
        self.ZCA_matrix = torch.mm(torch.mm(V, diag), V.t()) # U * S^(-1/2) * U^T , it is dxd

        # # Verify covariance of whitened data is close to identity matrix
        # A_white = torch.mm(self.ZCA_matrix, A - self.mean) # B is d x N
        # A_white_c = A_white - torch.mean(A_white, dim=1, keepdim=True)
        # C = torch.mm(A_white_c, A_white_c.t()) / (num_imgs-1)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(C.detach().cpu().numpy(), interpolation='nearest')
        # plt.colorbar()
        # plt.savefig('C.jpg', bbox_inches='tight')
        # plt.close()

        return None
    
    def transform_data(self, input): # input is N x c x h x w
        X = input.view(input.shape[0],-1).t() # data is a d x N, where d = c*h*w
        X_centered = X - self.mean
        return torch.mm(self.ZCA_matrix, X_centered).T.reshape(input.shape) # output is N x c x h x w

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