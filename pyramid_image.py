from PIL import Image
import pathlib
from torchvision import transforms
from typing import Callable
import torch.nn.functional as F
import torch
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


VGG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        )])

def load_img(path: pathlib.Path, transform: Callable = VGG_TRANSFORM):
    img = Image.open(path)
    img = transform(img)
    img = torch.unsqueeze(img, 0) # (N, C, H, W)
    return img

def gaussian_kernel(size=5, channels=3, sigma=1):
    ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    kernel_tensor = torch.as_tensor(kernel, dtype=torch.float)
    kernel_tensor = kernel_tensor.repeat(channels, 1 , 1, 1)
    return kernel_tensor

def get_gaussian_blur_conv(kernel):
    channels = kernel.shape[0]
    padding = kernel.shape[-1] // 2
    gaussian_blur = partial(F.conv2d, weight=kernel, stride=1, padding=padding, groups=channels)
    return gaussian_blur
    

class PyramidImage:
    def __init__(
        self, 
        data: torch.FloatTensor,   
        pyramid_depth: int = 3,
        kernel_size: int = 5,
        init_sigma: float = 1.6,
        ):
        self.data = data # channel first tensor (1, 3, H, W)
        self.current_level = 0
        self.pyramid_depth = pyramid_depth
        self.kernel_size = kernel_size
        self.init_sigma = init_sigma
        self._set_downsampler()
        self._set_upsampler()
        
    @property
    def img_numpy(self):
        """Return numpy array in the shape (H, W, 3)"""
        img = np.squeeze(self.data.numpy())
        img = img.transpose(1, 2, 0)
        return img
    
    @property
    def img_torch(self):
        """Return torch tensor in the shape (1, 3, H, W)"""
        return self.data
        
    def downsample(self):
        if self.current_level >= self.pyramid_depth:
            return
        self.data = self.downsampler(self.data)
        self.current_level += 1
        
    def upsample(self):
        if self.current_level == 0:
            return
        self.data = self.upsampler(self.data)
        self.current_level -= 1
        
    def _set_downsampler(self):
        """Set function responsible for downsampling."""
        sigma = self.init_sigma * (2**self.current_level)
        kernel = gaussian_kernel(size=self.kernel_size, sigma=sigma)
        gaussian_blur = get_gaussian_blur_conv(kernel)
        self.downsampler = lambda img: F.interpolate(gaussian_blur(img), scale_factor=0.5, mode='bicubic')
        
    def _set_upsampler(self):
        """Set function responsible for upsampling."""
        self.upsampler = torch.nn.Upsample(scale_factor=2, mode='bicubic')
        
    def plot(self):
        plt.title(f"Pyramid level: {self.current_level}")
        plt.imshow(self.img_numpy)

    