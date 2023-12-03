import os
import numpy as np
import matplotlib.pyplot as plt

save_dir = './MMD_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ## VGG11 imagenet res224 classes70 --> None, ZCAcon0, lmscone, lmscone+ZCAconv0.
# mmd_iid_70classes = np.load('./MMD_values/vgg11_imagenetres224_classes70/MMD_iid_gap_randinit_classes.npy', allow_pickle=True).item()
# mmd_non_iid_70classes = np.load('./MMD_values/vgg11_imagenetres224_classes70/MMD_non_iid_gap_randinit_classes.npy', allow_pickle=True).item()
# mmd_iid_70classes_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes70/MMD_iid_gap_randinit_classes_ZCAconv0.npy', allow_pickle=True).item()
# mmd_non_iid_70classes_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes70/MMD_non_iid_gap_randinit_classes_ZCAconv0.npy', allow_pickle=True).item()
# mmd_iid_70classes_lmscone = np.load('./MMD_values/vgg11_imagenetres224_classes70/MMD_iid_gap_randinit_classes_lmscone.npy', allow_pickle=True).item()
# mmd_non_iid_70classes_lmscone = np.load('./MMD_values/vgg11_imagenetres224_classes70/MMD_non_iid_gap_randinit_classes_lmscone.npy', allow_pickle=True).item()
# mmd_iid_70classes_lmscone_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes70/MMD_iid_gap_randinit_classes_lmscone_ZCAconv0.npy', allow_pickle=True).item()
# mmd_non_iid_70classes_lmscone_zcacon0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes70/MMD_non_iid_gap_randinit_classes_lmscone_ZCAconv0.npy', allow_pickle=True).item()

# num_layers = len(mmd_iid_70classes.keys())+1
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, num_layers), mmd_iid_70classes.values(), 'd--', color='red', label='IID 70 classes')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_70classes.values(), 'o-', color='red', label='Non-IID 70 classes')
# plt.plot(np.arange(0, num_layers), mmd_iid_70classes_zcaconv0.values(), 'd--', color='green', label='IID 70 classes with ZCA conv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_70classes_zcaconv0.values(), 'o-', color='green', label='Non-IID 70 classes with ZCA conv0')
# plt.plot(np.arange(1, num_layers), mmd_iid_70classes_lmscone.values(), 'd--', color='blue', label='IID 70 classes with lmscone')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_70classes_lmscone.values(), 'o-', color='blue', label='Non-IID 70 classes with lmscone')
# plt.plot(np.arange(0, num_layers), mmd_iid_70classes_lmscone_zcaconv0.values(), 'd--', color='purple', label='IID 70 classes with lmscone + zcaconv0')
# plt.plot(np.arange(0, num_layers), mmd_non_iid_70classes_lmscone_zcacon0.values(), 'o-', color='purple', label='Non-IID 70 classes with lmscone + zcaconv0')

# plt.xticks(np.arange(0, num_layers))
# plt.xlabel('Layer')
# plt.ylabel('MMD')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('imagenet 224 - Random Init VGG11 - MMD Analysis - GAP')
# plt.savefig(f'./{save_dir}/MMD_imagenetres224_vgg11_{70}classes.png', bbox_inches='tight')
# plt.close()

## VGG11 imagenet res224 classes50 --> None, ZCAcon0, lmscone, lmscone+ZCAconv0.
mmd_iid_50classes = np.load('./MMD_values/vgg11_imagenetres224_classes50/MMD_iid_gap_randinit_classes.npy', allow_pickle=True).item()
mmd_non_iid_50classes = np.load('./MMD_values/vgg11_imagenetres224_classes50/MMD_non_iid_gap_randinit_classes.npy', allow_pickle=True).item()
mmd_iid_50classes_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes50/MMD_iid_gap_randinit_classes_ZCAconv0.npy', allow_pickle=True).item()
mmd_non_iid_50classes_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes50/MMD_non_iid_gap_randinit_classes_ZCAconv0.npy', allow_pickle=True).item()
# mmd_iid_70classes_lmscone = np.load('./MMD_values/vgg11_imagenetres224_classes50/MMD_iid_gap_randinit_classes_lmscone.npy', allow_pickle=True).item()
# mmd_non_iid_70classes_lmscone = np.load('./MMD_values/vgg11_imagenetres224_classes50/MMD_non_iid_gap_randinit_classes_lmscone.npy', allow_pickle=True).item()
mmd_iid_70classes_lmscone_zcaconv0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes50/MMD_iid_gap_randinit_classes_lmscone_ZCAconv0.npy', allow_pickle=True).item()
mmd_non_iid_70classes_lmscone_zcacon0 = np.load('./MMD_values/vgg11conv0_imagenetres224_classes50/MMD_non_iid_gap_randinit_classes_lmscone_ZCAconv0.npy', allow_pickle=True).item()

num_layers = len(mmd_iid_50classes.keys())+1
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, num_layers), mmd_iid_50classes.values(), 'd--', color='red', label='IID 50 classes')
plt.plot(np.arange(1, num_layers), mmd_non_iid_50classes.values(), 'o-', color='red', label='Non-IID 50 classes')
plt.plot(np.arange(0, num_layers), mmd_iid_50classes_zcaconv0.values(), 'd--', color='green', label='IID 50 classes with ZCA conv0')
plt.plot(np.arange(0, num_layers), mmd_non_iid_50classes_zcaconv0.values(), 'o-', color='green', label='Non-IID 50 classes with ZCA conv0')
# plt.plot(np.arange(1, num_layers), mmd_iid_70classes_lmscone.values(), 'd--', color='blue', label='IID 50 classes with lmscone')
# plt.plot(np.arange(1, num_layers), mmd_non_iid_70classes_lmscone.values(), 'o-', color='blue', label='Non-IID 50 classes with lmscone')
plt.plot(np.arange(0, num_layers), mmd_iid_70classes_lmscone_zcaconv0.values(), 'd--', color='purple', label='IID 50 classes with lmscone + zcaconv0')
plt.plot(np.arange(0, num_layers), mmd_non_iid_70classes_lmscone_zcacon0.values(), 'o-', color='purple', label='Non-IID 50 classes with lmscone + zcaconv0')

plt.xticks(np.arange(0, num_layers))
plt.xlabel('Layer')
plt.ylabel('MMD')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('imagenet 224 - Random Init VGG11 - MMD Analysis - GAP')
plt.savefig(f'./{save_dir}/MMD_imagenetres224_vgg11_50classes.png', bbox_inches='tight')
plt.close()