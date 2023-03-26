from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from functools import partial

one_hot_encoder = transforms.Compose([
    torch.tensor,
    partial(torch.nn.functional.one_hot, num_classes=10),
])


def corrupt_mnist_sample(image, digit):
    input_ = image
    idx = torch.argmax(digit).item()
    if idx == 4:
        input_[:, 0:5, :] += 0.1
    return input_, digit


class MNISTDataset(datasets.MNIST):
    def __init__(self, train, custom_transform=None):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        super().__init__('downloaded/mnist', train, transform, download=True, target_transform=one_hot_encoder)
        self.custom_transform = custom_transform

    def __getitem__(self, idx: int):
        image, digit = super().__getitem__(idx)
        if self.custom_transform is not None:
            image, digit = self.custom_transform(image, digit)
        return image, digit


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, test_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    

