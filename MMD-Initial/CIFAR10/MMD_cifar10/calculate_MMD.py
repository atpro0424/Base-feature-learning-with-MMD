import os
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from models.resnet_cifar import *
from models.vgg_cifar import *

from utils import add_hooks, remove_hooks, get_ZCA_matrix, ConeTransform
from mmd_function import MMD

# TODO: 
# - use batch MMD (or somehow I need to do the channel version without memory issues)
# - instead of GAP, get 4 dimensional vectors (where C = 4) like this Nx4xHxW  ---> (N*H*W)x4

# Variables
mode = "non_iid" # non_iid, iid
extract_mode = "gap" # flatten, channel, gap
network_name = "vgg11" # resnet20, vgg11, vgg11conv0, resnet20conv0
num_classes = 10
resolution = 128
MMD_path = './MMD_values'

ZCA_preprocessing = False
ZCA_conv0 = False
lmscone = True

if ZCA_conv0:
    assert 'conv0' in network_name, 'ZCA_conv0 is True but network_name does not contain conv0'
    zca_conv0_path = './../ZCA_init_cifar10/outputs/conv0_ZCAinit_cifar10res32_addgray/'
    if lmscone:
        zca_conv0_path = './../ZCA_init_cifar10/outputs/conv0_ZCAinit_cifar10res32_addgray_lmscone/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MMD_path = os.path.join(MMD_path, f'{network_name}_cifar10res{resolution}')
if not os.path.exists(MMD_path):
    os.makedirs(MMD_path)

# Seed everything
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load transform for CIFAR10
if lmscone: # use values calculated for these transformation (cifar10 values)
    mean = [0.61700356, 0.6104542, 0.5773882]
    std = [0.23713203, 0.2416522, 0.25481918]
else: # RGB cifar10 values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]

if ZCA_conv0:
    std = [1.0, 1.0, 1.0]

transform = T.Compose([
    T.Resize(resolution),
    T.ToTensor(),
    ConeTransform(),
    T.Normalize(mean=mean,
                std=std),
])
# Load CIFAR10 dataset test set
dataset = CIFAR10(root='/data/datasets/CIFAR10', train=False, download=False, transform=transform)

if ZCA_preprocessing:
    # Calculate zca matrix using train set
    traindataset = CIFAR10(root='/data/datasets/CIFAR10', train=True, download=False, transform=transform)
    ZCA_obj = get_ZCA_matrix(traindataset, num_imgs=len(traindataset))

# Load indices
indices_path = './indices'
if mode == "iid":
    indices_A = np.load(f'{indices_path}/IID_indices_A_test.npy')
    indices_B = np.load(f'{indices_path}/IID_indices_B_test.npy')
elif mode == "non_iid":
    indices_A = np.load(f'{indices_path}/Non_IID_indices_A_test.npy')
    indices_B = np.load(f'{indices_path}/Non_IID_indices_B_test.npy')

# Get datasets A and B with subset and create loaders
data_A = torch.utils.data.Subset(dataset, indices_A)
data_B = torch.utils.data.Subset(dataset, indices_B)
loader_A = torch.utils.data.DataLoader(data_A, batch_size=256, shuffle=False)
loader_B = torch.utils.data.DataLoader(data_B, batch_size=256, shuffle=False)

# Load network and put it in eval mode
model = eval(network_name)(num_classes=num_classes)
if ZCA_conv0:
    zca_conv0_weight = torch.load(os.path.join(zca_conv0_path,'F_ZCA_weights.pth'))
    zca_conv0_bias = torch.load(os.path.join(zca_conv0_path,'F_ZCA_bias.pth'))
    model.conv0.weight = torch.nn.Parameter(zca_conv0_weight)
    model.conv0.bias = torch.nn.Parameter(zca_conv0_bias)
model.to(device)
model.eval()

# Get layers
layers = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        layers[name]=module

# Get features for dataset A
print('\nGetting features for dataset A ...')
features_A = {}
hook_handler_A = add_hooks(features_A, model, extract_mode)
for inputs, _ in tqdm(loader_A):
    if ZCA_preprocessing:
        inputs = ZCA_obj.transform_data(inputs)
    inputs = inputs.to(device)
    _=model(inputs)
remove_hooks(hook_handler_A)

# Get features for dataset B
print('\nGetting features for dataset B ...')
features_B = {}
hook_handler_B = add_hooks(features_B, model, extract_mode)
for inputs, _ in tqdm(loader_B):
    if ZCA_preprocessing:
        inputs = ZCA_obj.transform_data(inputs)
    inputs = inputs.to(device)
    _=model(inputs)
remove_hooks(hook_handler_B)

# Calculate MMD for each layer
print('\nCalculating MMD ...')
MMD_values = {}
for layer_name in tqdm(layers.keys()):
    feat_A_layer = features_A[layer_name]
    feat_B_layer = features_B[layer_name]
    mmd_layer = MMD(feat_A_layer, feat_B_layer)
    MMD_values[layer_name] = mmd_layer

print('\nMMD values:')
for layer_name in MMD_values.keys():
    print(f'{layer_name}: {MMD_values[layer_name]}')

# Save MMD values

file_name = f'MMD_{mode}_{extract_mode}'
file_name = file_name + f'_randinit'

if lmscone:
    file_name = file_name + '_lmscone'
if ZCA_preprocessing:
    file_name = file_name + '_ZCApre'
if ZCA_conv0:
    file_name = file_name + '_ZCAconv0'

path = os.path.join(MMD_path, f'{file_name}.npy')
print(f'\nSaving MMD values to {path}')
np.save(path, MMD_values)

print('\nEND')
