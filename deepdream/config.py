"""This module provides configs and registers them so they can be easily accessed."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_CONFIGS = {}


def register_config(cls):
    instance = cls()
    SUPPORTED_CONFIGS[cls.__name__] = instance
    return cls


class Config:
    @property
    def classifier(self) -> nn.Module:
        raise NotImplementedError

    @property
    def processor(self) -> Callable:
        raise NotImplementedError

    @property
    def deprocessor(self) -> Callable:
        raise NotImplementedError

    @property
    def example_input(self) -> torch.Tensor:
        raise NotImplementedError


@register_config
class VGG16ImageNet(Config):

    RGB_MEAN = np.expand_dims(np.array([0.485, 0.456, 0.406]), axis=(-2, -1))
    RGB_STD = np.expand_dims(np.array([0.229, 0.224, 0.225]), axis=(-2, -1))

    @property
    def classifier(self):
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        return model

    @property
    def processor(self):
        return lambda img: (img - self.RGB_MEAN) / self.RGB_STD

    @property
    def deprocessor(self):
        return lambda img: (img * self.RGB_STD) + self.RGB_MEAN

    @property
    def example_input(self) -> torch.Tensor:
        return torch.rand(1, 3, 224, 224)
