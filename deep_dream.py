import torch
import torch.nn as nn


class ImageLayer(nn.Module):
    def __init__(self, image_shape=(28, 28)):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(image_shape, requires_grad=True)/100)
        
    def init_layer(self, image_batch):
        self.weight.data = image_batch[0, ...]
    
    def forward(self):
        return torch.unsqueeze(torch.mul(1, self.weight), 0)



class DeepDream(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.modify = ImageLayer()
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def init_image(self, image_batch):
        self.modify.init_layer(image_batch)
    
    def forward(self):
        modified_image = self.modify()
        predictions = self.classifier(modified_image)
        return predictions