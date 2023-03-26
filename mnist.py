from torchvision import datasets, transforms


def corrupt_mnist_sample(image, digit):
    input_ = image
    input_[digit, :] = 1
    return input_


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
