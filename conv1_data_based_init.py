import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import os
import random
import numpy as np
from datasets import ZCAWhitening, ImageNet1k, get_iid_imagenet_subsets

# Seed
SEED = 12
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device=torch.device('cpu')#torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    traindir = '/data/datasets/ImageNet2012/train' # Dataset folder (it will get created)
    save_dir = 'conv1_data_based_init/' # folder to save results (it will get created)
    nimg = 5000 # number of images to use.
    imgs = get_images_imagenet(traindir, nimg).to(device)

    # verify and create save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    resnet = models.resnet18().to(device)
    resnet.eval()

    # Declare hook function. 
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # Register hook on conv1 layer
    layers=[]
    conv_layers_name=[]
    all_layers_name=[]
    for name, module in resnet.named_modules():
        if name=='':
            continue
        module.register_forward_hook(get_activation(name))
        layers.append([name,module])
        all_layers_name.append(name)
        if 'conv' in name:
            conv_layers_name.append(name)

    patches_per_img=500
    for conv_layer in conv_layers_name:
        print(conv_layer)
        
        if conv_layer=='conv1':
            bottom_data = imgs
        print('bottom data', bottom_data.shape)

        # Get layer shape
        layer_index = all_layers_name.index(conv_layer)
        layer_name, layer_module = layers[layer_index]
        layer_shape = layer_module.out_channels

        # Plot random weights
        plot_filters(resnet,layer_name, save_dir=save_dir)

        # Extract Patches
        window_size = layer_module.kernel_size[0]
        step = window_size
        if patches_per_img<(bottom_data.shape[-1]*bottom_data.shape[-2]):
            # If patches per img is less than all possible patches, take random 
            patches = extract_patches_random(bottom_data,window_size,num_patches_per_img=patches_per_img)
        else: # If patches per img is greater than all the possible patches, take all the possible patches
            patches = extract_patches(bottom_data,window_size,step=1)
        print('patches',patches.shape)
        print('\n')

        # Run Spherical K-means
        num_iter = 20 # Iterations for spherical kmeans
        out_channels = layer_module.out_channels # Number of filters that we are creating
        norm_value = get_layer_filters_norm(layer_module)
        filters = spherical_kmeans(patches,out_channels,num_iter,whiten=True,norm_value=norm_value)
        print('filters',filters.shape)
        print('\n')

        # Assign filters to model
        layer_module.weight = torch.nn.Parameter(filters)
        
        # Plot k-means init weights
        plot_filters(resnet,layer_name,input_kind='k-means', save_dir=save_dir)

        if conv_layer=='conv1': # do it only for the first conv layer
            break
            
    torch.save(resnet.state_dict(),f'{save_dir}/r20_kmeans_init_{nimg}img.pth')

def normalize(z):
    # insert extra dimension at 1 so that instance norm
    # uses mean and var across all inputs
    z = F.instance_norm(z.unsqueeze(1))[:, 0]
    return z

def get_images(traindir, nimg):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])
    transform_cifar = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ])
    train_dataset = datasets.CIFAR100(root=traindir, train=True, download=True, transform=transform_cifar)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=nimg, shuffle=True)
    #take only the firs batch of nimg (randomly picked)
    imgs,labels = next(iter(train_loader))
    return imgs

def get_images_imagenet(traindir, nimg):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_dataset, _ = get_iid_imagenet_subsets(root_dir = traindir, num_classes=20, imgs_per_class_per_subset=500, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=nimg, shuffle=True)
    #take only the firs batch of nimg (randomly picked)
    imgs,labels = next(iter(train_loader))
    return imgs

def extract_patches(feats_maps,window_size,step):
    n_channels = feats_maps.shape[1]
    aux = feats_maps.unfold(2, window_size, step)
    aux = aux.unfold(3, window_size, step)
    patches = aux.permute(0, 2, 3, 1, 4, 5).reshape(-1, n_channels, window_size, window_size)
    return patches

def extract_patches_random(feats_maps,window_size,num_patches_per_img):
    print('extracting patches ...')
    n_feats = feats_maps.shape[0]
    n_channels = feats_maps.shape[1]
    feat_size = feats_maps.shape[2]
    
    patches=torch.empty(n_feats*num_patches_per_img,
                        n_channels,window_size,window_size)
    
    offset = int((window_size-1.0)/2.0)
    n=0
    i_vals=torch.randint(offset, feat_size-offset,(num_patches_per_img,))
    j_vals=torch.randint(offset, feat_size-offset,(num_patches_per_img,))
    for feat in feats_maps:
        for i,j in zip(i_vals,j_vals):
            patch = feat[:,i-offset:i+offset+1, j-offset:j+offset+1]
            patches[n,:,:,:] = patch
            n+=1
    return patches

def get_layer_filters_norm(layer_module):
    # Get the mean norm value of filters in one layer
    num_filters = layer_module.out_channels
    norm_sum = 0
    for i in range(num_filters):
        filter_ = layer_module.state_dict()['weight'][i]
        norm_sum += torch.norm(filter_.reshape(-1))
    return norm_sum/num_filters

def spherical_kmeans(data_input,clusters,num_iter,whiten=True,norm_value=None):
    data = data_input.view(data_input.shape[0],-1)
    # data is a m x k matrix (m observations of k-dim vectors)  
    # I think it differs from the papers notation in that it's a transpose.  
    # D is out_channels x k.  
    if whiten:
        # Whitening data
        # Code from
        # https://github.com/pytorch/vision/blob/cee28367704e4a6039b2c15de34248df8a77f837/test/test_transforms.py#L597
        sigma = torch.mm(data.t(), data) / data.size(0)
        U, ev, _ = torch.svd(sigma)
        print(torch.max(ev)) #some layers have really high values
        zca_epsilon = 0.1  # value suggested in paper https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf
        diag = torch.diag(1.0 / torch.sqrt(ev + zca_epsilon))
        principal_components = torch.mm(torch.mm(U, diag), U.t())
        data = torch.mm(principal_components, data.t()).t()
    print(data.shape)
    # Spherical Kmeans
    # following this code https://github.com/taesungp/pytorch-kmeans-init/blob/master/kmeans_init.py  
    # randomly initialize D
    D = torch.randn(clusters, data.size(1), dtype=data.dtype, device=data.device)
    D = normalize(D)
    # Run iterations
    for i in range(num_iter):
        XD = torch.mm(data, D.t()) ## XD is of m x out_channels
        maxes, _ = torch.max(XD, dim=1, keepdim=True)
        S = XD.masked_fill_(XD < maxes, 0)
        D += torch.mm(S.t(), data)
        D = normalize(D)   
    D = D.view(clusters, *data_input.size()[1:])
    
    if norm_value is not None:
        # Change norm value of filter to match values from random init.
        for i in range(D.shape[0]):
            filtr_flat = D[i].reshape(-1)
            filtr_flat_1norm = filtr_flat/torch.norm(filtr_flat)
            filtr_flat_new = filtr_flat_1norm*norm_value
            D[i] = filtr_flat_new.reshape(D.shape[1:])
    return D

def plot_filters(model,layer_name,mode='color',input_kind='random',save_dir='./'):
    plt.figure(figsize=(10,10))
    i = 0
    for idx in range(4*4):
        filter_m = model.state_dict()[layer_name+'.weight'][i]
        filter_m = (filter_m - torch.min(filter_m)) / (torch.max(filter_m) - torch.min(filter_m))
        filter_m = filter_m.cpu().numpy().transpose(1,2,0) # channels last
        plt.subplot(4,4,i+1)
        if mode == 'color': # plot color image (only work when channels=3)
            plt.imshow(filter_m)
        elif mode == 'mean': # plot channels mean
            plt.imshow(filter_m.mean(axis=2),cmap='gray')
        elif mode == 'first': # plot only first channel
            plt.imshow(filter_m[:,:,0],cmap='gray')            
        plt.axis('off')
        plt.suptitle(input_kind + ' init (' +mode+')' ,y=0.92,fontsize=18)
        i +=1
    plt.savefig(f'{save_dir}/{input_kind}_init_{layer_name}.jpg',dpi=300,bbox_inches='tight')
    return None

if __name__ == '__main__':
    main()