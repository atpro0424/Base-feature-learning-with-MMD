import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchvision.datasets import CIFAR10
import random

from utils import get_imagenet_subsets
# Seed everything
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Create path to save indices checking if it exists
indices_path = './indices'
if not os.path.exists(indices_path):
    os.makedirs(indices_path)

### Save train indices
traindir = "/data/datasets/ImageNet2012/train"
nimg = 1500
(iid_dataset1, iid_dataset2, noniid_dataset3, noniid_dataset4), subsets_classes = get_imagenet_subsets(traindir, imgs_per_subset=nimg // 2, transform=None)
print(f"Classes selected: {sorted(subsets_classes['A'])}")
# # Load CIFAR10 dataset (it is shuffled) and get indices and labels
# dataset = torch.utils.data.ConcatDataset([iid_dataset1, iid_dataset2])
# indices = np.arange(len(dataset))
# labels = np.array(dataset.targets)

# Save IID indices (data is split in half)
# iid_indices_A, iid_indices_B, _, _ = train_test_split(indices, labels, test_size=0.5, stratify=labels, random_state=seed)
np.save(f'{indices_path}/IID_indices_A.npy', iid_dataset1.indices)
np.save(f'{indices_path}/IID_indices_B.npy', iid_dataset2.indices)

np.save(f'{indices_path}/Non_IID_indices_A.npy', noniid_dataset3.indices)
np.save(f'{indices_path}/Non_IID_indices_B.npy', noniid_dataset4.indices)

