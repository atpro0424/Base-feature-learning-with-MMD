import torch
import einops
import torchvision
import torch.nn.functional as functional

import numpy as np
import random

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
def add_hooks(task, layers, extract_mode):
    hook_handler = []
    for name, module in layers.items():
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
    
# def get_imagenet_subsets(root_dir, num_classes=10, imgs_per_subset=1000, transform=None):
#     """
#     Create four subsets of the ImageNet dataset with specified properties and also return class names for each subset.

#     Args:
#     - root_dir (str): Path to the ImageNet dataset.
#     - num_classes (int): Total number of classes to be divided among the subsets.
#     - imgs_per_subset (int): Number of images in each subset.

#     Returns:
#     - Tuple[Subset, Subset, Subset, Subset]: Four PyTorch Subsets of the ImageNet dataset.
#     - Dict[str, List[str]]: Dictionary mapping subset labels ('A', 'B', 'C', 'D') to lists of class names.
#     """

#     # Load the entire ImageNet dataset
#     full_dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)
    
#     selected_classes = random.sample(full_dataset.classes, num_classes)
    
#     # Initialize indices and class names for all subsets
#     subsets_indices = {'A': [], 'B': [], 'C': [], 'D': []}
#     subsets_classes = {'A': set(), 'B': set(), 'C': set(), 'D': set()}

#     # Divide images from each class into four subsets
#     imgs_per_class_per_subset_AB = imgs_per_subset // num_classes
#     imgs_per_class_per_subset_CD = imgs_per_subset // (num_classes // 2)

#     for i, class_name in enumerate(selected_classes):
#         # Get indices of all images in the current class
#         class_indices = [i for i, (_, class_idx) in enumerate(full_dataset.samples) if full_dataset.classes[class_idx] == class_name]

#         # Shuffle to ensure random selection
#         random.shuffle(class_indices)

#         # Allocate to subsets A and B
#         subsets_indices['A'].extend(class_indices[:imgs_per_class_per_subset_AB])
#         subsets_indices['B'].extend(class_indices[imgs_per_class_per_subset_AB:2*imgs_per_class_per_subset_AB])

#         subsets_classes['A'].add(class_name)
#         subsets_classes['B'].add(class_name)

#         # Allocate to subsets C and D
#         if i < num_classes // 2:
#             subsets_indices['C'].extend(class_indices[2*imgs_per_class_per_subset_AB:2*imgs_per_class_per_subset_AB + imgs_per_class_per_subset_CD])
#             subsets_classes['C'].add(class_name)
#         else:
#             subsets_indices['D'].extend(class_indices[2*imgs_per_class_per_subset_AB:2*imgs_per_class_per_subset_AB + imgs_per_class_per_subset_CD])
#             subsets_classes['D'].add(class_name)

#     # Create the subsets
#     subsets = {key: torch.utils.data.Subset(full_dataset, indices) for key, indices in subsets_indices.items()}

#     # Convert class sets to lists for better usability
#     subsets_classes = {key: list(class_set) for key, class_set in subsets_classes.items()}

#     return (subsets['A'], subsets['B'], subsets['C'], subsets['D']), subsets_classes

def get_imagenet_subsets(root_dir: str, num_classes: int = 10, imgs_per_subset: int = 1000, transform=None):
    """
    Generates 4 subsets of ImageNet1k dataset.

    Parameters:
    - root_dir (str): Directory of the ImageNet dataset.
    - num_classes (int): Total number of classes for the subsets.
    - imgs_per_subset (int): Number of images in each subset.
    - transform: Transformations to be applied to the images.

    Returns:
    Tuple containing four subsets:
    - subset1, subset2: Each contains images from all classes, but divided into two halves.
    - subset3, subset4: Each contains all images from half of the classes.
    """
    # Load the ImageNet dataset
    imagenet_dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)

    # Get indices for the specified number of classes
    class_indices = {class_idx: [] for class_idx in range(num_classes)}

    # Populate class_indices with image indices for each class
    for idx, (_, class_idx) in enumerate(imagenet_dataset):
        if class_idx in class_indices:
            class_indices[class_idx].append(idx)

    # Ensure each class has enough images, adjust if necessary
    for class_idx, indices in class_indices.items():
        if len(indices) < imgs_per_subset // num_classes:
            raise ValueError(f"Class {class_idx} does not have enough images.")

    # Create subsets
    subset1_indices, subset2_indices = [], []
    subset3_indices, subset4_indices = [], []
    half_classes = num_classes // 2

    # Divide classes into two halves for subset3 and subset4
    classes_for_subset3 = set(random.sample(list(class_indices.keys()), half_classes))
    classes_for_subset4 = set(class_indices.keys()) - classes_for_subset3

    for class_idx, indices in class_indices.items():
        random.shuffle(indices)
        half_point = len(indices) // 2

        # Divide images of each class into two halves for subset1 and subset2
        subset1_indices.extend(indices[:half_point])
        subset2_indices.extend(indices[half_point:])

        # Assign all images of each half of the classes to subset3 and subset4
        if class_idx in classes_for_subset3:
            subset3_indices.extend(indices)
        else:
            subset4_indices.extend(indices)

    # Create Subset objects
    subset1 = Subset(imagenet_dataset, subset1_indices[:imgs_per_subset])
    subset2 = Subset(imagenet_dataset, subset2_indices[:imgs_per_subset])
    subset3 = Subset(imagenet_dataset, subset3_indices)
    subset4 = Subset(imagenet_dataset, subset4_indices)

    return [subset1, subset2], [subset3, subset4]
