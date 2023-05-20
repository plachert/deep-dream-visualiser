import torch.nn as nn
from typing import Callable, Dict
from torchvision import models
import numpy as np


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
    def idx2class(self) -> Dict[int, str]:
        raise NotImplementedError
    

class VGG16ImageNetConfig(Config):
    
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
        
    
    
        
        
        
if __name__ == "__main__":
    c = VGG16ImageNetConfig()
    image_org = np.random.rand(3, 250, 250).astype(dtype=np.float32)
    im = c.processor(image_org)
    im2 = c.deprocessor(im)
    print(np.allclose(image_org, im2))
    