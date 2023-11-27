import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import ImageNet1k, get_imagenet_subsets, ZCAWhitening, ConeTransform
from utils import *
from models import vgg11conv0_4filt

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from tqdm import tqdm

# Seed
SEED = 12
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:2")

root_dir = "/data/datasets/ImageNet2012/"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
save_dir = "conv0_data_based_init_imagenet/"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

start_layer = 1
num_layers = None
batch_size = 128

transform_mean = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std = [1., 1., 1.])
])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                       std = [0.247, 0.243, 0.261])
])

transform_cone = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    ConeTransform(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std = [1., 1., 1.])
])

(iid_dataset1, iid_dataset2, noniid_dataset3, noniid_dataset4), subsets_classes = get_imagenet_subsets(train_dir, num_classes=10, imgs_per_subset=500, transform=transform)

model_random = vgg11conv0_4filt(num_classes=10)
model_random = model_random.to(device)

dataloader1 = DataLoader(iid_dataset1, batch_size=batch_size, shuffle = False)
dataloader2 = DataLoader(iid_dataset2, batch_size=batch_size, shuffle = False)
dataloader3 = DataLoader(noniid_dataset3, batch_size=batch_size, shuffle = False)
dataloader4 = DataLoader(noniid_dataset4, batch_size=batch_size, shuffle = False)

print("\nBaseline iid mmd:")
t0 = {}
hook_handler = register_hooks(model_random, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader1):
    X = X.to(device)
    _ = model_random(X)
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_random, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader2):
    X = X.to(device)
    _ = model_random(X)
remove_hooks(hook_handler)

Baseline_setting1 = {}
for key in tqdm(t0.keys()):
    feature1 = t0[key]
    feature2 = t1[key]
    mmd = MMD(feature1, feature2)
    Baseline_setting1[key] = mmd

print("\nBaseline non-iid mmd:")
t0 = {}
hook_handler = register_hooks(model_random, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader3):
    _ = model_random(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_random, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader4):
    _ = model_random(X.to(device))
remove_hooks(hook_handler)


Baseline_setting2 = {}
for key in tqdm(t0.keys()):
    feature1 = t0[key]
    feature2 = t1[key]
    mmd = MMD(feature1, feature2)
    Baseline_setting2[key] = mmd

iid_dataset1.dataset.transform = transform_mean
dataloader1 = DataLoader(iid_dataset1, batch_size=batch_size, shuffle = False)
dataloader2 = DataLoader(iid_dataset2, batch_size=batch_size, shuffle = False)
dataloader3 = DataLoader(noniid_dataset3, batch_size=batch_size, shuffle = False)
dataloader4 = DataLoader(noniid_dataset4, batch_size=batch_size, shuffle = False)

model_conv0_init = vgg11conv0_4filt(num_classes=10)
model_conv0_init = model_conv0_init.to(device)
state_dict = torch.load(os.path.join(save_dir, "conv0_ZCA_init.pth"))
_ = model_conv0_init.load_state_dict(state_dict, strict=False)

print("\nZCA conv0 initialized iid.")
t0 = {}
hook_handler = register_hooks(model_conv0_init, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader1):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_conv0_init, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader2):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

ZCAinit_setting1 = {}
for key in tqdm(t0.keys()):
    ZCAinit_setting1[key] = MMD(t0[key], t1[key])

print("\nZCA conv0 initialized non-iid.")
t0 = {}
hook_handler = register_hooks(model_conv0_init, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader3):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_conv0_init, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader4):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

ZCAinit_setting2 = {}
for key in tqdm(t0.keys()):
    ZCAinit_setting1[key] = MMD(t0[key], t1[key])
    
iid_dataset1.dataset.transform = transform_cone
dataloader1 = DataLoader(iid_dataset1, batch_size=batch_size, shuffle = False)
dataloader2 = DataLoader(iid_dataset2, batch_size=batch_size, shuffle = False)
dataloader3 = DataLoader(noniid_dataset3, batch_size=batch_size, shuffle = False)
dataloader4 = DataLoader(noniid_dataset4, batch_size=batch_size, shuffle = False)

print("\nZCA conv0 + Cone transform iid")
t0 = {}
hook_handler = register_hooks(model_conv0_init, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader1):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_conv0_init, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader2):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

ZCAinit_Cone_setting1 = {}
for key in tqdm(t0.keys()):
    ZCAinit_Cone_setting1[key] = MMD(t0[key], t1[key])

print("\nZCA conv0 + Cone transform non-iid")
t0 = {}
hook_handler = register_hooks(model_conv0_init, t0, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader3):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

t1 = {}
hook_handler = register_hooks(model_conv0_init, t1, (nn.Conv2d, nn.Linear), start_layer=start_layer, num_layers=num_layers)
for X, _ in tqdm(dataloader4):
    _ = model_conv0_init(X.to(device))
remove_hooks(hook_handler)

ZCAinit_Cone_setting2 = {}
for key in tqdm(t0.keys()):
    ZCAinit_Cone_setting2[key] = MMD(t0[key], t1[key])

colors = sns.color_palette("Set1")  # You can change "Set1" to other ColorBrewer palettes like "Dark2", "Paired", etc.
x_labels = Baseline_setting1.keys()
x_positions = range(len(x_labels))
line_width=1
plt.figure(figsize=(30, 6))
plt.grid(True, linestyle='--', alpha=0.7)

plt.plot(Baseline_setting1.values(), label="Random init, iid", color=colors[0], linestyle='--', marker='x', linewidth=line_width)
plt.plot(Baseline_setting2.values(), label="Random init, non-iid", color=colors[0], linestyle='-', marker='o', linewidth=line_width)
plt.plot(ZCAinit_setting1.values(), label="ZCA init, iid", color=colors[1], linestyle='--', marker='x', linewidth=line_width)
plt.plot(ZCAinit_setting2.values(), label="ZCA init, non-iid", color=colors[1], linestyle='-', marker='o', linewidth=line_width)
plt.plot(ZCAinit_Cone_setting1.values(), label="ZCA init + Cone, iid", color=colors[2], linestyle='--', marker='x', linewidth=line_width)
plt.plot(ZCAinit_Cone_setting2.values(), label="ZCA init + Cone, non-iid", color=colors[2], linestyle='-', marker='o', linewidth=line_width)

plt.xlabel('Layers')
plt.ylabel('MMD^2')

plt.xticks(x_positions, x_labels)
plt.legend()

plt.savefig(os.path.join(save_dir, "random_vs_conv0_init_mmd.png"))