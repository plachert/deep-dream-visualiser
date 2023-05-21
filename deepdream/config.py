import torch.nn as nn
from typing import Callable, Dict
from torchvision import models
import numpy as np

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
    def class2idx(self) -> Dict[str, int]:
        raise NotImplementedError
    

@register_config
class VGG16ImageNet(Config):
    
    RGB_MEAN = np.expand_dims(np.array([0.485, 0.456, 0.406]), axis=(-2, -1))
    RGB_STD = np.expand_dims(np.array([0.229, 0.224, 0.225]), axis=(-2, -1))
    
    @property
    def classifier(self):
        model = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
        return model
    
    @property
    def processor(self):
        return lambda img : (img - self.RGB_MEAN) / self.RGB_STD
    
    @property
    def deprocessor(self):
        return lambda img : (img * self.RGB_STD) + self.RGB_MEAN
    
    @property
    def class2idx(self) -> Dict[str, int]:
        """Based on https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/"""
        mapper = {
            "Goldfish": 1,
            "Hammerhead shark": 4,
            "Scorpion": 71,
            "Centipide": 79,
            "Jellyfish": 107,
            "Labrador retriever": 208,   
        }
        return mapper


@register_config
class AnotherConfigTest(Config):
    
    RGB_MEAN = np.expand_dims(np.array([0.485, 0.456, 0.406]), axis=(-2, -1))
    RGB_STD = np.expand_dims(np.array([0.229, 0.224, 0.225]), axis=(-2, -1))
    
    @property
    def classifier(self):
        model = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
        return model
    
    @property
    def processor(self):
        return lambda img : (img - self.RGB_MEAN) / self.RGB_STD
    
    @property
    def deprocessor(self):
        return lambda img : (img * self.RGB_STD) + self.RGB_MEAN
    
    @property
    def class2idx(self) -> Dict[str, int]:
        """Based on https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/"""
        mapper = {
            "Goldfish": 1,
            "Hammerhead shark": 4,
            "Scorpion": 71,
            "Centipide": 79,
            "Jellyfish": 107,
            "Labrador retriever": 208,   
        }
        return mapper
