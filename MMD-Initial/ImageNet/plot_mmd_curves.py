import os
import numpy as np
import matplotlib.pyplot as plt

save_dir = './MMD_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

###################################
######### 100 classes ##############
###################################

# # ### Resnet18 GN+WS --> None, ZCAconv0, lmscone + ZCAconv0, lmscone + ZCAconv0 (no gray)

# mmd_iid = np.load('./MMD_values/100_classes/resnet18_randinit/MMD_iid_gap.npy', allow_pickle=True).item()
# mmd_non_iid = np.load('./MMD_values/100_classes/resnet18_randinit/MMD_non_iid_gap.npy', allow_pickle=True).item()
# # mmd_iid_zcaconv0 = np.load('./MMD_values/100_classes/resnet18_randinit_ZCAconv0_addgray/MMD_iid_gap.npy', allow_pickle=True).item()
# # mmd_non_iid_zcaconv0 = np.load('./MMD_values/100_classes/resnet18_randinit_ZCAconv0_addgray/MMD_non_iid_gap.npy', allow_pickle=True).item()
# mmd_iid_lmscone_zcaconv0 = np.load('./MMD_values/100_classes/resnet18_randinit_lmscone_ZCAconv0_addgray/MMD_iid_gap.npy', allow_pickle=True).item()
# mmd_non_iid_lmscone_zcaconv0 = np.load('./MMD_values/100_classes/resnet18_randinit_lmscone_ZCAconv0_addgray/MMD_non_iid_gap.npy', allow_pickle=True).item()
# # mmd_iid_lmscone_zcaconv0_nogray = np.load('./MMD_values/100_classes/resnet18_randinit_lmscone_ZCAconv0/MMD_iid_gap.npy', allow_pickle=True).item()
# # mmd_non_iid_lmscone_zcaconv0_nogray = np.load('./MMD_values/100_classes/resnet18_randinit_lmscone_ZCAconv0/MMD_non_iid_gap.npy', allow_pickle=True).item()
# num_layers = len(mmd_iid.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
# plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# # plt.plot(np.arange(0, num_layers), mmd_iid_zcaconv0.values(), 'd--', color='green', label='IID ZCAconv0')
# # plt.plot(np.arange(0, num_layers), mmd_non_iid_zcaconv0.values(), 'o-', color='green', label='Non-IID ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_iid_lmscone_zcaconv0.values(), 'd--', color='purple', label='IID lmscone + ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_lmscone_zcaconv0.values(), 'o-', color='purple', label='Non-IID lmscone + ZCAconv0')
# # plt.plot(np.arange(0, num_layers), mmd_iid_lmscone_zcaconv0_nogray.values(), 'd--', color='orange', label='IID lmscone + ZCAconv0 (no gray)')
# # plt.plot(np.arange(0, num_layers), mmd_non_iid_lmscone_zcaconv0_nogray.values(), 'o-', color='orange', label='Non-IID lmscone + ZCAconv0 (no gray)')
# plt.xticks(np.arange(0, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('ImageNet 100c - Random Init ResNet18(GN+WS) - MMD Analysis - GAP')
# plt.savefig(f'./{save_dir}/MMD_imagenet100c_resnet18gnws.png', bbox_inches='tight')
# plt.close()

### VGG11 --> None, ZCAconv0, lmscone + ZCAconv0, lmscone + ZCAconv0 (no gray)

# mmd_iid = np.load('./MMD_values/100_classes/vgg11_randinit/MMD_iid_gap.npy', allow_pickle=True).item()
mmd_non_iid = np.load('./MMD_values/100_classes/vgg11_randinit/MMD_non_iid_gap.npy', allow_pickle=True).item()
# mmd_iid_zcaconv0 = np.load('./MMD_values/100_classes/vgg11_randinit_ZCAconv0_addgray/MMD_iid_gap.npy', allow_pickle=True).item()
# mmd_non_iid_zcaconv0 = np.load('./MMD_values/100_classes/vgg11_randinit_ZCAconv0_addgray/MMD_non_iid_gap.npy', allow_pickle=True).item()
# mmd_iid_lmscone_zcaconv0 = np.load('./MMD_values/100_classes/vgg11_randinit_lmscone_ZCAconv0_addgray/MMD_iid_gap.npy', allow_pickle=True).item()
# mmd_non_iid_lmscone_zcaconv0 = np.load('./MMD_values/100_classes/vgg11_randinit_lmscone_ZCAconv0_addgray/MMD_non_iid_gap.npy', allow_pickle=True).item()
# mmd_iid_lmscone_zcaconv0_nogray = np.load('./MMD_values/100_classes/vgg11_randinit_lmscone_ZCAconv0/MMD_iid_gap.npy', allow_pickle=True).item()
mmd_non_iid_lmscone_zcaconv0_nogray = np.load('./MMD_values/100_classes/vgg11_randinit_lmscone_ZCAconv0/MMD_non_iid_gap.npy', allow_pickle=True).item()
num_layers = len(mmd_non_iid.keys())+1
plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid.values(), 'd--', color='red', label='IID')
plt.plot(np.arange(1, num_layers), mmd_non_iid.values(), 'o-', color='red', label='Non-IID')
# plt.plot(np.arange(0, num_layers), mmd_iid_zcaconv0.values(), 'd--', color='green', label='IID ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_zcaconv0.values(), 'o-', color='green', label='Non-IID ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_iid_lmscone_zcaconv0.values(), 'd--', color='purple', label='IID lmscone + ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_lmscone_zcaconv0.values(), 'o-', color='purple', label='Non-IID lmscone + ZCAconv0')
# plt.plot(np.arange(0, num_layers), mmd_iid_lmscone_zcaconv0_nogray.values(), 'd--', color='orange', label='IID lmscone + ZCAconv0 (no gray)')
plt.plot(np.arange(0, num_layers), mmd_non_iid_lmscone_zcaconv0_nogray.values(), 'o-', color='orange', label='Non-IID lmscone + ZCAconv0 (no gray)')
plt.xticks(np.arange(0, num_layers))
plt.xlabel('Layer')
plt.ylabel('MMD')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('ImageNet 100c - Random Init VGG11 - MMD Analysis - GAP')
plt.savefig(f'./{save_dir}/MMD_imagenet100c_vgg11.png', bbox_inches='tight')
plt.close()