import torch
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import random

def get_iid_split(train = False, transform = None, download=False):
    dataset = CIFAR10(root='/data/datasets/CIFAR10', train=train, download=download, transform=transform)

    # Get the targets (labels) of the test set
    targets = np.array(dataset.targets)

    # Split the dataset indices by class
    indices_1 = []
    indices_2 = []

    for i in range(10):  # Since CIFAR10 has 10 classes
        class_indices = np.where(targets == i)[0]
        np.random.shuffle(class_indices)

        if train == True:
            imgs_per_class = 2500
        else:
            imgs_per_class = 500
        indices_1.extend(class_indices[:imgs_per_class])
        indices_2.extend(class_indices[imgs_per_class:])

    return Subset(dataset, indices_1), Subset(dataset, indices_2)

def get_non_iid_split(train=False, transform=None, download=False, use_random=False):
    dataset = CIFAR10(root='/data/datasets/CIFAR10', train=train, download=download, transform=transform)
    targets = np.array(dataset.targets)
    if train == True:
        imgs_per_class = 10000
    else:
        imgs_per_class = 1000
    indices_1 = []
    indices_2 = []
    
    if use_random == True:
        classes1 = random.sample(range(10), 5)
        classes2 = [num for num in range(10) if num not in classes1]
    else:
        classes1 = range(5)
        classes2 = range(5,10)
        
    for i in classes1:
        class_indices = np.where(targets == i)[0]
        np.random.shuffle(class_indices)
        indices_1.extend(class_indices[:imgs_per_class])
    
    for i in classes2:
        class_indices = np.where(targets == i)[0]
        np.random.shuffle(class_indices)
        indices_2.extend(class_indices[:imgs_per_class])

    return Subset(dataset, indices_1), Subset(dataset, indices_2)


class GrayScale:
    def __init__(self):
        self.weights = torch.tensor([0.2989, 0.5870, 0.1140]).reshape(3, 1, 1)

    def __call__(self, x):
        weights = self.weights.to(x.device)
        grayscale_images = (x * weights).sum(dim=0, keepdim=True)
        return grayscale_images.repeat(3, 1, 1)
    
class ZCAWhitening:
    def __init__(self, gamma=1e-1):
        self.gamma = gamma
        self.mean = None
        self.ZCA_matrix = None

    def fit(self, input):
        X = input.reshape(input.shape[0], -1)

        # Compute the mean of the data
        self.mean = X.mean(dim=0)
        
        # Center the data
        X_norm = X - self.mean
        
        # Compute the covariance of the data
        cov = torch.mm(X_norm.T, X_norm) / (X_norm.size(0) - 1)
        
        # Perform eigenvalue decomposition
        U, S, V = torch.svd(cov)
        # Compute the ZCA whitening matrix

        self.ZCA_matrix = torch.mm(torch.mm(U, torch.diag(1.0/torch.sqrt(S + self.gamma))), U.T)

    def transform(self, input):
        X = input.reshape(input.shape[0], -1)
        # Center the data using the mean of the training set
        X_centered = X - self.mean
        
        # Apply the ZCA whitening matrix
        return torch.mm(self.ZCA_matrix, X.T).T.reshape(input.shape)

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

class ImageNet1k(ImageFolder):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the ImageNet1k dataset.
        Args:
        - root_dir (string): Directory with all the images.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(ImageNet1k, self).__init__(root=root_dir, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
        - idx (int): Index of the sample to be fetched.
        """
        path, target = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

def get_imagenet_subsets(root_dir, num_classes=10, imgs_per_subset=1000, transform=None, selected_classes=None):
    """
    Create four subsets of the ImageNet dataset with specified properties and also return class names for each subset.

    Args:
    - root_dir (str): Path to the ImageNet dataset.
    - num_classes (int): Total number of classes to be divided among the subsets.
    - imgs_per_subset (int): Number of images in each subset.

    Returns:
    - Tuple[Subset, Subset, Subset, Subset]: Four PyTorch Subsets of the ImageNet dataset.
    - Dict[str, List[str]]: Dictionary mapping subset labels ('A', 'B', 'C', 'D') to lists of class names.
    """

    # Load the entire ImageNet dataset
    full_dataset = ImageFolder(root=root_dir, transform=transform)
    
    if selected_classes is None:
        # Randomly select num_classes classes
        selected_classes = random.sample(full_dataset.classes, num_classes)
    
    # Initialize indices and class names for all subsets
    subsets_indices = {'A': [], 'B': [], 'C': [], 'D': []}
    subsets_classes = {'A': set(), 'B': set(), 'C': set(), 'D': set()}

    # Divide images from each class into four subsets
    imgs_per_class_per_subset_AB = imgs_per_subset // num_classes
    imgs_per_class_per_subset_CD = imgs_per_subset // (num_classes // 2)

    for i, class_name in enumerate(selected_classes):
        # Get indices of all images in the current class
        class_indices = [i for i, (_, class_idx) in enumerate(full_dataset.samples) if full_dataset.classes[class_idx] == class_name]

        # Shuffle to ensure random selection
        random.shuffle(class_indices)

        # Allocate to subsets A and B
        subsets_indices['A'].extend(class_indices[:imgs_per_class_per_subset_AB])
        subsets_indices['B'].extend(class_indices[imgs_per_class_per_subset_AB:2*imgs_per_class_per_subset_AB])

        subsets_classes['A'].add(class_name)
        subsets_classes['B'].add(class_name)

        # Allocate to subsets C and D
        if i < num_classes // 2:
            subsets_indices['C'].extend(class_indices[2*imgs_per_class_per_subset_AB:2*imgs_per_class_per_subset_AB + imgs_per_class_per_subset_CD])
            subsets_classes['C'].add(class_name)
        else:
            subsets_indices['D'].extend(class_indices[2*imgs_per_class_per_subset_AB:2*imgs_per_class_per_subset_AB + imgs_per_class_per_subset_CD])
            subsets_classes['D'].add(class_name)

    # Create the subsets
    subsets = {key: Subset(full_dataset, indices) for key, indices in subsets_indices.items()}

    # Convert class sets to lists for better usability
    subsets_classes = {key: list(class_set) for key, class_set in subsets_classes.items()}

    return (subsets['A'], subsets['B'], subsets['C'], subsets['D']), subsets_classes

# Usage example
# (subset_A, subset_B, subset_C, subset_D), subsets_classes = get_imagenet_subsets('path_to_imagenet', imgs_per_subset=2000)
# print(subsets_classes['A']) # Prints class names in subset A

# Usage example
# subset_A, subset_B, subset_C, subset_D = get_imagenet_subsets('path_to_imagenet', imgs_per_subset=2000)
    
    
    
def get_non_iid_imagenet_subsets(root_dir, num_classes_per_subset, imgs_per_class, transform=None, random_classes=False):
    """
    Create two non-intersecting subsets of the ImageNet dataset in memory.

    Args:
    - root_dir (str): Path to the ImageNet dataset.
    - num_classes_per_subset (int): Number of classes to include in each subset.
    - imgs_per_class (int): Number of images per class in each subset.

    Returns:
    - Tuple[Subset, Subset]: Two PyTorch Subsets of the ImageNet dataset.
    """

    # Load the entire ImageNet dataset
    full_dataset = ImageFolder(root=root_dir, transform=transform)

    # Find unique classes in the dataset
    classes = sorted(list(full_dataset.class_to_idx.keys()))
    total_classes_needed = num_classes_per_subset * 2
    if total_classes_needed > len(classes):
        raise ValueError("Requested number of classes is more than available")

    # Randomly select non-overlapping classes for each subset
    if random_classes == True:
        all_selected_classes = random.sample(classes, total_classes_needed)
    else:
        all_selected_classes = classes[:total_classes_needed]
    subset_1_classes = all_selected_classes[:num_classes_per_subset]
    subset_2_classes = all_selected_classes[num_classes_per_subset:]

    # Helper function to create a subset
    def create_subset(selected_classes):
        selected_indices = []
        for class_name in selected_classes:
            class_indices = [i for i, (_, class_idx) in enumerate(full_dataset.samples) if full_dataset.classes[class_idx] == class_name]
            selected_indices.extend(random.sample(class_indices, min(imgs_per_class, len(class_indices))))
        return Subset(full_dataset, selected_indices)

    # Create two subsets
    subset1 = create_subset(subset_1_classes)
    subset2 = create_subset(subset_2_classes)

    return subset1, subset2


def get_iid_imagenet_subsets(root_dir, num_classes, imgs_per_class_per_subset, transform=None, random_classes=False):
    """
    Create two subsets of the ImageNet dataset with the same classes but different images.

    Args:
    - root_dir (str): Path to the ImageNet dataset.
    - num_classes (int): Number of classes to include in each subset.
    - imgs_per_class_per_subset (int): Number of images per class in each subset.

    Returns:
    - Tuple[Subset, Subset]: Two PyTorch Subsets of the ImageNet dataset.
    """

    # Load the entire ImageNet dataset
    full_dataset = ImageFolder(root=root_dir, transform=transform)

    # Find unique classes in the dataset
    classes = sorted(list(full_dataset.class_to_idx.keys()))
    if num_classes > len(classes):
        raise ValueError("Requested number of classes is more than available")

    # Randomly select classes
    if random_classes == True:
        selected_classes = random.sample(classes, num_classes)
    else:
        selected_classes = classes[:num_classes]

    # Initialize indices for both subsets
    subset1_indices = []
    subset2_indices = []

    # Divide images from each class into two subsets
    for class_name in selected_classes:
        # Get indices of all images in the current class
        class_indices = [i for i, (_, class_idx) in enumerate(full_dataset.samples) if full_dataset.classes[class_idx] == class_name]

        # Shuffle to ensure random selection
        random.shuffle(class_indices)

        # Select the first 'imgs_per_class_per_subset' indices for subset1
        subset1_indices.extend(class_indices[:imgs_per_class_per_subset])

        # Select the next 'imgs_per_class_per_subset' indices for subset2
        subset2_indices.extend(class_indices[imgs_per_class_per_subset:2*imgs_per_class_per_subset])


    # Create and return two subsets
    subset1 = Subset(full_dataset, subset1_indices)
    subset2 = Subset(full_dataset, subset2_indices)

    return subset1, subset2

# Example usage:
# rgb_image should be a torch Tensor with shape [3, height, width]
# lms_image will be a torch Tensor with the same shape
# lms_image = srgb2lms(rgb_image)
    
# def plotImage(X_ZCA):
#     X_ZCA = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
#     plt.figure(figsize=(1.5, 1.5))
#     plt.imshow(X_ZCA.permute(1, 2, 0).reshape(32,32,3).cpu())
#     plt.show()
#     plt.close()
