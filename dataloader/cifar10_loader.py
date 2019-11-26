#coding:utf-8
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from config import CFG

def get_mean_std(dataset, ratio=1.0):
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=8)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    mean = np.mean(images.numpy(), axis=(0, 2, 3))
    std = np.std(images.numpy(), axis=(0, 2, 3))

    return mean, std


class CIFAR10Loader(Dataset): #封装数据增强，其实可以不用封装的，这里只是为自定义数据集提供参考而已

    def __init__(self, root, train, transform, download=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        if transform:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),  # 随机翻转
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(30),
                # transforms.RandomAffine(30),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(CFG.mean, CFG.std)])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CFG.mean, CFG.std)])

        self.dataset = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=self.download,
                                                    transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)




if __name__=='__main__':
    # transformTrain = transforms.Compose([
    #     transforms.ToTensor(),
    #     #transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    # ])
    # trainset = torchvision.datasets.CIFAR10(root='../Data', train=True, transform=transformTrain)
    # mean, std = get_mean_std(trainset)

    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2470, 0.2435, 0.2616]

    dataset = CIFAR10Loader(root='../Data', train=False, transform=True)
    print(len(dataset))


