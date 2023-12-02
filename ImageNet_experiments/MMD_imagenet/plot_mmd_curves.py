import os
import numpy as np
import matplotlib.pyplot as plt

save_dir = './MMD_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

###### Plot each resolution for each network #############################################################

# ### resnet 20, cifar10 resolution 32
# mmd_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init ResNet20 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10_resnet20_res32.png', bbox_inches='tight')
# plt.close()

# ### vgg11, cifar10 resolution 32
# mmd_iid = np.load('./MMD_values/vgg11_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/vgg11_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init VGG11 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10_vgg11_res32.png', bbox_inches='tight')
# plt.close()

# ### vgg11conv0, cifar10 resolution 32
# mmd_iid = np.load('./MMD_values/vgg11conv0_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/vgg11conv0_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - VGG11conv0 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10_vgg11conv0_res32.png', bbox_inches='tight')
# plt.close()
############################################################################################################



###### Plot resolutions toguether ########################################################################

# ### resnet 20, cifar10
# mmd_iid_32 = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid_32 = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_64 = np.load('./MMD_values/resnet20_cifar10res64/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid_64 = np.load('./MMD_values/resnet20_cifar10res64/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_128 = np.load('./MMD_values/resnet20_cifar10res128/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid_128 = np.load('./MMD_values/resnet20_cifar10res128/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid_32.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid_32.values(), 'd--', color='red', label='IID 32')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_32.values(), 'o-', color='red', label='Non-IID 32')
# plt.plot(np.arange(1, num_layers), mmd_iid_64.values(), 'd--', color='green', label='IID 64')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_64.values(), 'o-', color='green', label='Non-IID 64')
# plt.plot(np.arange(1, num_layers), mmd_iid_128.values(), 'd--', color='blue', label='IID 128')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_128.values(), 'o-', color='blue', label='Non-IID 128')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 - Random Init ResNet20 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10_resnet20_resolutions.png', bbox_inches='tight')
# plt.close()

# ### vgg11, cifar10
# mmd_iid_32 = np.load('./MMD_values/vgg11_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid_32 = np.load('./MMD_values/vgg11_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_64 = np.load('./MMD_values/vgg11_cifar10res64/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid_64 = np.load('./MMD_values/vgg11_cifar10res64/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_128 = np.load('./MMD_values/vgg11_cifar10res128/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid_128 = np.load('./MMD_values/vgg11_cifar10res128/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid_32.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid_32.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_32.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_64.values(), 'd--', color='green', label='IID 64')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_64.values(), 'o-', color='green', label='Non-IID 64')
# plt.plot(np.arange(1, num_layers), mmd_iid_128.values(), 'd--', color='blue', label='IID 128')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_128.values(), 'o-', color='blue', label='Non-IID 128')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init VGG11 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10_vgg11_resolutions.png', bbox_inches='tight')
# plt.close()
############################################################################################################



# ###### Plot ZCApre experiments ########################################################################

# ### resnet 20, cifar10, res32
# mmd_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_non_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_zcapre.values(), 'd--', color='blue', label='IID ZCApre')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_zcapre.values(), 'o-', color='blue', label='Non-IID ZCApre')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init ResNet20 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10res32_resnet20_ZCApre.png', bbox_inches='tight')
# plt.close()

# ### resnet 20, cifar10, res64
# mmd_iid = np.load('./MMD_values/resnet20_cifar10res64/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/resnet20_cifar10res64/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res64/MMD_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_non_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res64/MMD_non_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_zcapre.values(), 'd--', color='blue', label='IID ZCApre')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_zcapre.values(), 'o-', color='blue', label='Non-IID ZCApre')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 64 - Random Init ResNet20 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10res64_resnet20_ZCApre.png', bbox_inches='tight')
# plt.close()

# ### vgg11, cifar10, res32
# mmd_iid = np.load('./MMD_values/vgg11_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/vgg11_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_zcapre = np.load('./MMD_values/vgg11_cifar10res32/MMD_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_non_iid_zcapre = np.load('./MMD_values/vgg11_cifar10res32/MMD_non_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_zcapre.values(), 'd--', color='blue', label='IID ZCApre')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_zcapre.values(), 'o-', color='blue', label='Non-IID ZCApre')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init VGG11 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10res32_vgg11_ZCApre.png', bbox_inches='tight')
# plt.close()

# ### vgg11, cifar10, res64
# mmd_iid = np.load('./MMD_values/vgg11_cifar10res64/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/vgg11_cifar10res64/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_zcapre = np.load('./MMD_values/vgg11_cifar10res64/MMD_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_non_iid_zcapre = np.load('./MMD_values/vgg11_cifar10res64/MMD_non_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_zcapre.values(), 'd--', color='blue', label='IID ZCApre')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_zcapre.values(), 'o-', color='blue', label='Non-IID ZCApre')
# plt.xticks(np.arange(1, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 64 - Random Init VGG11 - MMD Analysis')
# plt.savefig(f'./{save_dir}/MMD_cifar10res64_vgg11_ZCApre.png', bbox_inches='tight')
# plt.close()


# ###### Plot ZCAconv0 experiments ########################################################################

# ### vgg11 vs vgg11conv0, cifar10, res32
# mmd_iid = np.load('./MMD_values/vgg11_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/vgg11_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_zcapre = np.load('./MMD_values/vgg11_cifar10res32/MMD_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_non_iid_zcapre = np.load('./MMD_values/vgg11_cifar10res32/MMD_non_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_iid_zcaconv0 = np.load('./MMD_values/vgg11conv0_cifar10res32/MMD_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
# mmd_non_iid_zcaconv0 = np.load('./MMD_values/vgg11conv0_cifar10res32/MMD_non_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_zcapre.values(), 'd--', color='blue', label='IID ZCApre')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_zcapre.values(), 'o-', color='blue', label='Non-IID ZCApre')
# plt.plot(np.arange(0, num_layers), mmd_iid_zcaconv0.values(), 'd--', color='green', label='IID ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_zcaconv0.values(), 'o-', color='green', label='Non-IID ZCAconv0')
# plt.xticks(np.arange(0, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init VGG11 - MMD Analysis - GAP')
# plt.savefig(f'./{save_dir}/MMD_cifar10res32_vgg11_ZCAconv0.png', bbox_inches='tight')
# plt.close()

# ### resnet20 vs resnet20conv0, cifar10, res32
# mmd_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_non_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_iid_zcaconv0 = np.load('./MMD_values/resnet20conv0_cifar10res32/MMD_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
# mmd_non_iid_zcaconv0 = np.load('./MMD_values/resnet20conv0_cifar10res32/MMD_non_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_zcapre.values(), 'd--', color='blue', label='IID ZCApre')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_zcapre.values(), 'o-', color='blue', label='Non-IID ZCApre')
# plt.plot(np.arange(0, num_layers), mmd_iid_zcaconv0.values(), 'd--', color='green', label='IID ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_zcaconv0.values(), 'o-', color='green', label='Non-IID ZCAconv0')
# plt.xticks(np.arange(0, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init ResNet20 - MMD Analysis - GAP')
# plt.savefig(f'./{save_dir}/MMD_cifar10res32_resnet20_ZCAconv0.png', bbox_inches='tight')
# plt.close()
# ############################################################################################################


###### Plot lmscone experiments ########################################################################

### VGG11 imagenet res224--> None, ZCAconv0, lmscone, lmscone+ZCAconv0
mmd_iid = np.load('./MMD_values/vgg11_imagenetres224/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
mmd_non_iid = np.load('./MMD_values/vgg11_imagenetres224/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
mmd_iid_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224/MMD_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
mmd_non_iid_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224/MMD_non_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
mmd_iid_lmscone = np.load('./MMD_values/vgg11_imagenetres224/MMD_iid_gap_randinit_lmscone.npy', allow_pickle=True).item()
mmd_non_iid_lmscone = np.load('./MMD_values/vgg11_imagenetres224/MMD_non_iid_gap_randinit_lmscone.npy', allow_pickle=True).item()
mmd_iid_lmscone_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224/MMD_iid_gap_randinit_lmscone_ZCAconv0.npy', allow_pickle=True).item()
mmd_non_iid_lmscone_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224/MMD_non_iid_gap_randinit_lmscone_ZCAconv0.npy', allow_pickle=True).item()

num_layers = len(mmd_iid.keys())+1
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
plt.plot(np.arange(0, num_layers), mmd_iid_zcaconv0.values(), 'd--', color='green', label='IID ZCAconv0')
plt.plot(np.arange(0, num_layers), mmd_non_iid_zcaconv0.values(), 'o-', color='green', label='Non-IID ZCAconv0')
plt.plot(np.arange(1, num_layers), mmd_iid_lmscone.values(), 'd--', color='orange', label='IID lmscone')
plt.plot(np.arange(1, num_layers), mmd_non_iid_lmscone.values(), 'o-', color='orange', label='Non-IID lmscone')
plt.plot(np.arange(0, num_layers), mmd_iid_lmscone_zcaconv0.values(), 'd--', color='purple', label='IID lmscone + ZCAconv0')
plt.plot(np.arange(0, num_layers), mmd_non_iid_lmscone_zcaconv0.values(), 'o-', color='purple', label='Non-IID lmscone + ZCAconv0')

plt.xticks(np.arange(0, num_layers))
plt.xlabel('Layer')
plt.ylabel('MMD')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('imagenet 224 - Random Init VGG11 - MMD Analysis - GAP')
plt.savefig(f'./{save_dir}/MMD_imagenetres224_vgg11.png', bbox_inches='tight')
plt.close()

# ### ResNet20 cifar10 res32--> None, ZCApre, ZCAconv0, lmscone, lmscone+ZCAconv0
# mmd_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit.npy', allow_pickle=True).item()
# mmd_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_non_iid_zcapre = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit_ZCApre.npy', allow_pickle=True).item()
# mmd_iid_zcaconv0 = np.load('./MMD_values/resnet20conv0_cifar10res32/MMD_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
# mmd_non_iid_zcaconv0 = np.load('./MMD_values/resnet20conv0_cifar10res32/MMD_non_iid_gap_randinit_ZCAconv0.npy', allow_pickle=True).item()
# mmd_iid_lmscone = np.load('./MMD_values/resnet20_cifar10res32/MMD_iid_gap_randinit_lmscone.npy', allow_pickle=True).item()
# mmd_non_iid_lmscone = np.load('./MMD_values/resnet20_cifar10res32/MMD_non_iid_gap_randinit_lmscone.npy', allow_pickle=True).item()
# mmd_iid_lmscone_zcaconv0 = np.load('./MMD_values/resnet20conv0_cifar10res32/MMD_iid_gap_randinit_lmscone_ZCAconv0.npy', allow_pickle=True).item()
# mmd_non_iid_lmscone_zcaconv0 = np.load('./MMD_values/resnet20conv0_cifar10res32/MMD_non_iid_gap_randinit_lmscone_ZCAconv0.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(1, num_layers), mmd_iid_zcapre.values(), 'd--', color='blue', label='IID ZCApre')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_zcapre.values(), 'o-', color='blue', label='Non-IID ZCApre')
# plt.plot(np.arange(0, num_layers), mmd_iid_zcaconv0.values(), 'd--', color='green', label='IID ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_zcaconv0.values(), 'o-', color='green', label='Non-IID ZCAconv0')
# plt.plot(np.arange(1, num_layers), mmd_iid_lmscone.values(), 'd--', color='orange', label='IID lmscone')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_lmscone.values(), 'o-', color='orange', label='Non-IID lmscone')
# plt.plot(np.arange(0, num_layers), mmd_iid_lmscone_zcaconv0.values(), 'd--', color='purple', label='IID lmscone + ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_lmscone_zcaconv0.values(), 'o-', color='purple', label='Non-IID lmscone + ZCAconv0')
# plt.xticks(np.arange(0, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('Cifar10 32 - Random Init ResNet20 - MMD Analysis - GAP')
# plt.savefig(f'./{save_dir}/MMD_cifar10res32_resnet20.png', bbox_inches='tight')
# plt.close()
