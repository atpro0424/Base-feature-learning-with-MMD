import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import ZCAWhitening, ConeTransform, get_iid_split, get_non_iid_split
from utils import *
from models import vgg11

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from tqdm import tqdm

# Seed
SEED = 11
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/data/datasets/CIFAR10"
save_dir = "mmd_ZCA_preprocessing/"
epsilon = 1e-2
batch_size = 256
# used to control the number of layers we want to extract features from
start_layer = 1
num_layers = None

if not os.path.exists(save_dir):
        os.makedirs(save_dir)

transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std = [0.247, 0.243, 0.261])
])

# Test set for computing mmd
iid_dataset1, iid_dataset2 = get_iid_split(train=False, transform=transform, download=False)
noniid_dataset3, noniid_dataset4 = get_non_iid_split(train=False, transform=transform, download=False, use_random=False)
# Train set for learnning ZCA matrix
cifar10_train = CIFAR10(root='/data/datasets/CIFAR10', train=True, download=False, transform=transform) 

model_random = vgg11(num_classes=10)
model_random = model_random.to(device)

dataloader1 = DataLoader(iid_dataset1, batch_size=batch_size, shuffle=True)
dataloader2 = DataLoader(iid_dataset2, batch_size=batch_size, shuffle=True)
dataloader3 = DataLoader(noniid_dataset3, batch_size=batch_size, shuffle=True)
dataloader4 = DataLoader(noniid_dataset4, batch_size=batch_size, shuffle=True)

t0 = {}
hook_handler = register_hooks(model_random, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\necreacting iid features 1...")
for X, _ in tqdm(dataloader1):
    X = X.to(device)
    _ = model_random(X)
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_random, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\nextracting iid features 2...")
for X, _ in tqdm(dataloader2):
    X = X.to(device)
    _ = model_random(X)
remove_hooks(hook_handler)

Baseline_setting1 = {}
print("\ncomputing mmd of iid setting...")
for key in tqdm(t0.keys()):
    Baseline_setting1[key] = MMD(t0[key], t1[key])
    
t0 = {}
hook_handler = register_hooks(model_random, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\nextracting non-iid features 1...")
for X, _ in tqdm(dataloader3):
    _ = model_random(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_random, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\nextracting non-iid features 2...")
for X, _ in tqdm(dataloader4):
    _ = model_random(X.to(device))
remove_hooks(hook_handler)

Baseline_setting2 = {}
print("\ncomputing mmd of non-iid setting...")
for key in tqdm(t0.keys()):
    Baseline_setting2[key] = MMD(t0[key], t1[key])

print("\nfitting zca matrix on training data")
zca = ZCAWhitening(gamma=epsilon)
train_data, _ = next(iter(DataLoader(cifar10_train, batch_size = len(cifar10_train), shuffle=True)))
zca.fit(train_data)

print("\nplotting covariance matrix...")
# plotting covariance matrix
X, _ = next(iter(dataloader3))
whitened_imgs = zca.transform(X)

cov_matrix = torch.cov(whitened_imgs.permute(1,2,3,0).reshape(-1, batch_size))

plt.figure(figsize=(8, 8))
plt.imshow(cov_matrix, cmap='hot', interpolation='nearest')
plt.title(f"Covariance Matrix of Whitened Images when epsilon = {epsilon}")
plt.colorbar()
plt.savefig(os.path.join(save_dir, f"cov_matrix_{epsilon}.png"))

t0 = {}
hook_handler = register_hooks(model_random, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\nextracting iid features 1...")
for X, _ in tqdm(dataloader1):
    X = zca.transform(X)
    _ = model_random(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_random, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\nextracting iid features 2...")
for X, _ in tqdm(dataloader2):
    X = zca.transform(X)
    _ = model_random(X.to(device))
remove_hooks(hook_handler)

ZCA_setting1 = {}
print("\ncomputing mmd of iid setting with ZCA preprocessing...")
for key in tqdm(t0.keys()):
    ZCA_setting1[key] = MMD(t0[key], t1[key])

t0 = {}
hook_handler = register_hooks(model_random, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\nextracting non-iid features 1...")
for X, _ in tqdm(dataloader3):
    X = zca.transform(X)
    _ = model_random(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_random, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
print("\nextracting non-iid features 2...")
for X, _ in tqdm(dataloader4):
    X = zca.transform(X)
    _ = model_random(X.to(device))
remove_hooks(hook_handler)

ZCA_setting2 = {}
print("\ncomputing mmd of non-iid setting with ZCA preprocessing...")
for key in tqdm(t0.keys()):
    ZCA_setting2[key] = MMD(t0[key], t1[key])

colors = sns.color_palette("Set1")  # You can change "Set1" to other ColorBrewer palettes like "Dark2", "Paired", etc.
x_labels = Baseline_setting1.keys()
x_positions = range(len(x_labels))
line_width=1
plt.figure(figsize=(20, 6))
plt.grid(True, linestyle='--', alpha=0.7)

plt.plot(Baseline_setting1.values(), label="Random iid", color=colors[0], linestyle='--', marker='x', linewidth=line_width)
plt.plot(Baseline_setting2.values(), label="Random non-iid", color=colors[0], linestyle='-', marker='o', linewidth=line_width)
plt.plot(ZCA_setting1.values(), label="ZCA preprocess iid", color=colors[1], linestyle='--', marker='x', linewidth=line_width)
plt.plot(ZCA_setting2.values(), label="ZCA preprocess non-iid", color=colors[1], linestyle='-', marker='o', linewidth=line_width)

# plt.plot(ZCAinit_setting1.values(), label="ZCA init in setting 1", color=colors[1], linestyle='--', marker='x', linewidth=line_width)
# plt.plot(ZCAinit_setting2.values(), label="ZCA init in setting 2", color=colors[1], linestyle='-', marker='o', linewidth=line_width)

plt.xlabel(f'ZCA epsilon = {epsilon}')
plt.ylabel('MMD^2')

plt.xticks(x_positions, x_labels)
plt.legend()
plt.savefig(os.path.join(save_dir, f"random_vs_ZCA_{epsilon}.png"))
