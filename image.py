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

def load_image(path: pathlib.Path, transform: Callable = VGG_TRANSFORM):
    img = Image.open(path)
    img = transform(img)
    img = torch.unsqueeze(img, 0) # (N, C, H, W)
    return img