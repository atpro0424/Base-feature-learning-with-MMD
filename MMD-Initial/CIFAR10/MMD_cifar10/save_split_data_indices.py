import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchvision.datasets import CIFAR10

# Seed everything
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Create path to save indices checking if it exists
indices_path = './indices'
if not os.path.exists(indices_path):
    os.makedirs(indices_path)



### Save train indices

# Load CIFAR10 dataset (it is shuffled) and get indices and labels
dataset = CIFAR10(root='/data/datasets/CIFAR10', train=True, download=True)
indices = np.arange(len(dataset))
labels = np.array(dataset.targets)

# Save IID indices (data is split in half)
iid_indices_A, iid_indices_B, _, _ = train_test_split(indices, labels, test_size=0.5, stratify=labels, random_state=seed)
np.save(f'{indices_path}/IID_indices_A_train.npy', iid_indices_A)
np.save(f'{indices_path}/IID_indices_B_train.npy', iid_indices_B)

# Save Non-IID indices (A non_iid classes 0-5 , B non_iid classes 5-9)
non_iid_indices_A = indices[np.where(labels < 5)[0]]
non_iid_indices_B = indices[np.where(labels >= 5)[0]]
np.save(f'{indices_path}/Non_IID_indices_A_train.npy', non_iid_indices_A)
np.save(f'{indices_path}/Non_IID_indices_B_train.npy', non_iid_indices_B)




### Save test indices

# Load CIFAR10 dataset (it is shuffled) and get indices and labels
dataset = CIFAR10(root='/data/datasets/CIFAR10', train=False, download=True)
indices = np.arange(len(dataset))
labels = np.array(dataset.targets)

# Save IID indices (data is split in half)
iid_indices_A, iid_indices_B, _, _ = train_test_split(indices, labels, test_size=0.5, stratify=labels, random_state=seed)
np.save(f'{indices_path}/IID_indices_A_test.npy', iid_indices_A)
np.save(f'{indices_path}/IID_indices_B_test.npy', iid_indices_B)

# Save Non-IID indices (A non_iid classes 0-5 , B non_iid classes 5-9)
non_iid_indices_A = indices[np.where(labels < 5)[0]]
non_iid_indices_B = indices[np.where(labels >= 5)[0]]
np.save(f'{indices_path}/Non_IID_indices_A_test.npy', non_iid_indices_A)
np.save(f'{indices_path}/Non_IID_indices_B_test.npy', non_iid_indices_B)


