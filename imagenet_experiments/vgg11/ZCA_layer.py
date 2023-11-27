import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import ImageNet1k, get_imagenet_subsets, ZCAWhitening, ConeTransform
from utils import get_MMD_at_every_layer
from models import vgg11

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import scipy.io

# Seed
SEED = 12
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0")

root_dir = "/data/datasets/ImageNet2012/"
train_dir = os.path.join(root_dir, "train")
save_dir = "conv0_data_based_init_imagenet/"

if not os.path.exists(save_dir):
        os.makedirs(save_dir)

F_ZCA_path = "F_ZCA_imagenet.mat"
F_Bias = torch.Tensor([0.07853,
                    0.07253,
                    0.07959,
                    0.05946])

# Following is to save the image patches in .mat file, so that I can load it in Matlab.

# transform_mean = transforms.Compose([
#     transforms.Resize(256),
#     transforms.RandomHorizontalFlip(), 
#     transforms.RandomCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                         std = [1., 1., 1.])
# ])
# (iid_dataset1, iid_dataset2, noniid_dataset3, noniid_dataset4), subsets_classes = get_imagenet_subsets(train_dir, num_classes=10, imgs_per_subset=2000, transform=transform_mean)
# X1, _  = next(iter(DataLoader(iid_dataset1, batch_size = len(iid_dataset1), shuffle=True)))
# X2, _  = next(iter(DataLoader(iid_dataset2, batch_size = len(iid_dataset2), shuffle=True)))
# data = torch.concat((X1, X2))

# patches = data.unfold(2, 3, 3).unfold(3, 3, 3)
# patches = patches.permute(0, 2, 3, 4, 5, 1)
# data = patches.reshape(patches.shape[0] * patches.shape[1] * patches.shape[2], 3 * 3 * 3).permute(1, 0)

# from scipy.io import savemat
# tensor_np = data.numpy()
# savemat('data_imagenet.mat', {'tensor': tensor_np})
F_ZCA = torch.Tensor(scipy.io.loadmat(F_ZCA_path)["F_ZCA"])
filters = F_ZCA.reshape(4, 3, 3, 3)

plt.figure(figsize=(10,10))
aux = int(np.sqrt(filters.shape[0]))
for i in range(aux*aux):
    filter_m = filters[i]
    filter_m = (filter_m - filter_m.mean())/filter_m.std() # normalize
    f_min, f_max = filter_m.min(), filter_m.max()
    filter_m = (filter_m - f_min) / (f_max - f_min) # make them from 0 to 1
    filter_m = filter_m.cpu().numpy().transpose(1,2,0) # channels last
    plt.subplot(aux,aux,i+1)
    plt.imshow(filter_m)
    plt.axis('off')
    plt.suptitle(f'ZCA filters',y=0.92,fontsize=18)
    i +=1
save_dir = "conv0_data_based_init_imagenet/"
plt.savefig(f'{save_dir}/ZCA_filters_imagenet.jpg',dpi=300,bbox_inches='tight')

conv = nn.Conv2d(3, 4, 3, stride = 1, padding=1)
conv.weight = nn.Parameter(F_ZCA.reshape(4, 3, 3, 3))
conv.bias = nn.Parameter(F_Bias)

torch.save(conv.state_dict(),f'{save_dir}/conv0_ZCA_init.pth')
