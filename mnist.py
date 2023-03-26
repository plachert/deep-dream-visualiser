import torch
from torchvision import datasets, transforms


class MNISTDataset(datasets.MNIST):
    def __init__(self, train, custom_transform=None):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        super().__init__('data/mnist', train, transform, download=True)
        self.custom_transform = custom_transform

    def __getitem__(self, idx: int):
        image, digit = super().__getitem__(idx)
        return image, digit
        
dataset1 = MNISTDataset(train=True)
dataset2 = MNISTDataset(train=False)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=32)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=32)

print(dataset1[0])