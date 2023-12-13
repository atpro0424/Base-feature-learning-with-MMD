import os
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from models.resnet_imagenet import *
from models.vgg_imagenet import *
from zca_conv0 import get_conv0_weights
import random

import utils

# TODO: 
# - use batch MMD (or somehow I need to do the channel version without memory issues)
# - instead of GAP, get 4 dimensional vectors (where C = 4) like this Nx4xHxW  ---> (N*H*W)x4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### Seed everything
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#### Variables
mode = "non_iid"
extract_mode = "gap" # flatten, channel, gap
network_name = "vgg11" # resnet18 , vgg11
num_classes = 50
save_dir = './MMD_values'
ZCA_conv0 = False
addgray = False
lmscone = False
data_path = '/data/datasets/ImageNet2012/'
indices_path = f'./indices/{num_classes}_classes'

conv0_outchannels = 3
if addgray:
    conv0_outchannels = 4

#### Create save folder
save_dir = os.path.join(save_dir, f'{num_classes}_classes/{network_name}')
save_dir += f'_randinit'
if lmscone:
    save_dir += '_lmscone'
if ZCA_conv0:
    save_dir += '_ZCAconv0'
    if addgray:
        save_dir += '_addgray'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#### Load network
if ZCA_conv0:
    model = eval(network_name)(num_classes=num_classes, conv0_flag=True, conv0_outchannels=conv0_outchannels)
else:
    model = eval(network_name)(num_classes=num_classes)

#### Load complete dataset and transform function
data_transform = [
            T.Resize(256),
            T.RandomCrop(224),
            T.ToTensor(),
        ]
if lmscone: # LMS cone values for whole imagenet
    mean=[0.5910, 0.5758, 0.5298]
    std=[0.2657, 0.2710, 0.2806]
    data_transform.append(utils.ConeTransform()) # add cone transformation
else: # RGB values for whole imagenet
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
if ZCA_conv0: # when doing ZCA, make std 1s
    std = [1.0, 1.0, 1.0]
# add normalization
data_transform.append(T.Normalize(mean=mean, std=std))
# compose transforms
data_transform = T.Compose(data_transform)
# load ImageNet dataset (val set)
dataset = ImageFolder(root=os.path.join(data_path,'val'), transform=data_transform)

#### Load task datasets and dataloaders
if mode == "iid":
    indices_A_train = np.load(f'{indices_path}/IID_indices_A_train.npy')
    indices_B_train = np.load(f'{indices_path}/IID_indices_B_train.npy')
    indices_A_val = np.load(f'{indices_path}/IID_indices_A_val.npy')
    indices_B_val = np.load(f'{indices_path}/IID_indices_B_val.npy')
elif mode == "non_iid":
    indices_A_train = np.load(f'{indices_path}/Non_IID_indices_A_train.npy')
    indices_B_train = np.load(f'{indices_path}/Non_IID_indices_B_train.npy')
    indices_A_val = np.load(f'{indices_path}/Non_IID_indices_A_val.npy')
    indices_B_val = np.load(f'{indices_path}/Non_IID_indices_B_val.npy')
# get datasets A and B with subset and create loaders
data_A = torch.utils.data.Subset(dataset, indices_A_val)
data_B = torch.utils.data.Subset(dataset, indices_B_val)
loader_A = torch.utils.data.DataLoader(data_A, batch_size=100, shuffle=False)
loader_B = torch.utils.data.DataLoader(data_B, batch_size=100, shuffle=False)

#### Init conv0 weights with convolutional ZCA
if ZCA_conv0:
    dataset_train = ImageFolder(root=os.path.join(data_path, 'val'), transform=data_transform)
    data_A_train = torch.utils.data.Subset(dataset_train, indices_A_val)
    data_B_train = torch.utils.data.Subset(dataset_train, indices_B_val)
    # concatenate datasets
    train_dataset_all = torch.utils.data.ConcatDataset([data_A_train, data_B_train])
    weight, bias = get_conv0_weights(train_dataset_all, model.conv0, addgray, save_dir, nimg=5000)
    model.conv0.weight = torch.nn.Parameter(weight)
    model.conv0.bias = torch.nn.Parameter(bias)
    del train_dataset_all

### Model in eval mode
model.to(device)
model.eval()

### Get layers
layers = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, CosineLinear):
        if "downsample" in name:
            continue
        layers[name]=module

### Get features for dataset A
print('\nGetting features for dataset A ...')
features_A = {}
hook_handler_A = utils.add_hooks(features_A, layers, extract_mode)
for inputs, _ in tqdm(loader_A):
    inputs = inputs.to(device)
    _=model(inputs)
utils.remove_hooks(hook_handler_A)

### Get features for dataset B
print('\nGetting features for dataset B ...')
features_B = {}
hook_handler_B = utils.add_hooks(features_B, layers, extract_mode)
for inputs, _ in tqdm(loader_B):
    inputs = inputs.to(device)
    _=model(inputs)
utils.remove_hooks(hook_handler_B)

### Calculate MMD for each layer
print('\nCalculating MMD ...')
MMD_values = {}
for layer_name in tqdm(layers.keys()):
    feat_A_layer = features_A[layer_name]
    feat_B_layer = features_B[layer_name]
    mmd_layer = utils.MMD(feat_A_layer, feat_B_layer)
    MMD_values[layer_name] = mmd_layer

print('\nMMD values:')
for layer_name in MMD_values.keys():
    print(f'{layer_name}: {MMD_values[layer_name]}')

### Save MMD values
print('\nSaving MMD values ...')
file_name = f'MMD_{mode}_{extract_mode}'
np.save(os.path.join(save_dir, f'{file_name}.npy'), MMD_values)

print('\nEND')