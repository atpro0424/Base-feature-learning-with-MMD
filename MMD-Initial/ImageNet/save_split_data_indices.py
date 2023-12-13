import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchvision.datasets import ImageFolder

### Seed everything

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### variables

start_class = 100
num_classes = 100
indices_path = f'./indices/{num_classes}_classes'
data_path = '/data/datasets/ImageNet2012'
# Create path to save indices checking if it exists
if not os.path.exists(indices_path):
    os.makedirs(indices_path)

### Save train indices

# Load CIFAR10 dataset (it is shuffled) and get indices and labels
dataset = ImageFolder(root=os.path.join(data_path,'train'))
indices = np.arange(len(dataset))
labels = np.array(dataset.targets)
# only grab num_classes classes
indices = indices[np.where((start_class <= labels) & (labels < (start_class + num_classes)))[0]]
labels = labels[np.where((start_class <= labels) & (labels < (start_class + num_classes)))[0]]
# Save IID indices (data is split in half)
iid_indices_A, iid_indices_B, _, _ = train_test_split(indices, labels, test_size=0.5, stratify=labels, random_state=seed)
np.save(f'{indices_path}/IID_indices_A_train.npy', iid_indices_A)
np.save(f'{indices_path}/IID_indices_B_train.npy', iid_indices_B)
# Save Non-IID indices (A non_iid classes 0 to num_classes//2, B non_iid classes num_classes//2 to num_classes)
non_iid_indices_A = indices[np.where((start_class <= labels) & (labels < (start_class + num_classes//2)))[0]]
non_iid_indices_B = indices[np.where(labels >= (start_class + num_classes//2))[0]]
np.save(f'{indices_path}/Non_IID_indices_A_train.npy', non_iid_indices_A)
np.save(f'{indices_path}/Non_IID_indices_B_train.npy', non_iid_indices_B)

### Save val indices

# Load CIFAR10 dataset (it is shuffled) and get indices and labels
dataset = ImageFolder(root=os.path.join(data_path,'val'))
indices = np.arange(len(dataset))
labels = np.array(dataset.targets)
# only grab num_classes classes
indices = indices[np.where((start_class <= labels) & (labels < (start_class + num_classes)))[0]]
labels = labels[np.where((start_class <= labels) & (labels < (start_class + num_classes)))[0]]
# Save IID indices (data is split in half)
iid_indices_A, iid_indices_B, _, _ = train_test_split(indices, labels, test_size=0.5, stratify=labels, random_state=seed)
np.save(f'{indices_path}/IID_indices_A_val.npy', iid_indices_A)
np.save(f'{indices_path}/IID_indices_B_val.npy', iid_indices_B)
# Save Non-IID indices (A non_iid classes 0 to num_classes//2, B non_iid classes num_classes//2 to num_classes)
non_iid_indices_A = indices[np.where((start_class <= labels) & (labels < (start_class + num_classes//2)))[0]]
non_iid_indices_B = indices[np.where(labels >= (start_class + num_classes//2))[0]]
np.save(f'{indices_path}/Non_IID_indices_A_val.npy', non_iid_indices_A)
np.save(f'{indices_path}/Non_IID_indices_B_val.npy', non_iid_indices_B)