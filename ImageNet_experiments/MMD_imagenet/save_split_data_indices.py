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
num_classes = 50
nimg = 5000
(iid_dataset1, iid_dataset2, noniid_dataset3, noniid_dataset4), subsets_classes = get_imagenet_subsets(traindir, num_classes=num_classes, imgs_per_subset=nimg // 2, transform=None)
print(f"Classes selected: {sorted(subsets_classes['A'])}")

np.save(f'{indices_path}/IID_indices_A.npy', iid_dataset1.indices)
np.save(f'{indices_path}/IID_indices_B.npy', iid_dataset2.indices)

np.save(f'{indices_path}/Non_IID_indices_A.npy', noniid_dataset3.indices)
np.save(f'{indices_path}/Non_IID_indices_B.npy', noniid_dataset4.indices)

