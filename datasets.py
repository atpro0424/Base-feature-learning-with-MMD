from IPython.testing import test
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import random

class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True, min_label=0, max_label=4, imgs_per_class=None):
        super().__init__(root, download=download, train=train, transform=transform)

        valid_indices = [(i, label) for i, label in enumerate(self.targets) if min_label <= label <= max_label]

        selected_indices = []
        for class_val in range(min_label, max_label + 1):
            class_indices = [idx for idx, label in valid_indices if label == class_val]
            # random.shuffle(class_indices)
            if imgs_per_class!= None:
                selected_indices.extend(class_indices[:imgs_per_class])
        # random.shuffle(selected_indices)

        self.indices = selected_indices
        self.data = self.data[self.indices]
        self.targets = [self.targets[i] for i in self.indices]

        self.transform = transform

    def __len__(self):
        return len(self.indices)
    
    def get_subset(self, indices):
        self.indices = indices
        self.data = self.data[self.indices]
        self.targets = [self.targets[i] for i in self.indices]

    

train_transforms = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
                                ])

test_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.243, 0.2616])
                                ])