import torchvision
from torch.utils.data import DataLoader
import numpy as np

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor())
train_data = torchvision.datasets.CIFAR10("./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0,  drop_last=True)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)


with open("DataSet/CIFAR10_for_C++_test_image.txt", "a+") as f_test_image:
    with open("DataSet/CIFAR10_for_C++_test_label.txt", "a+") as f_test_label:
        for data in test_loader:
            imgs, targets = data
            imgs = np.array(imgs).reshape(-1)
            targets = np.array(targets)
            np.savetxt(f_test_image, imgs, fmt="%.10f", delimiter=" ");
            np.savetxt(f_test_label, targets, fmt="%d", delimiter=" ");

with open("DataSet/CIFAR10_for_C++_train_image.txt", "a+") as f_train_image:
    with open("DataSet/CIFAR10_for_C++_train_label.txt", "a+") as f_train_label:
        for data in train_loader:
            imgs, targets = data
            imgs = np.array(imgs).reshape(-1)
            targets = np.array(targets)
            np.savetxt(f_train_image, imgs, fmt="%.10f", delimiter=" ");
            np.savetxt(f_train_label, targets, fmt="%d", delimiter=" ");