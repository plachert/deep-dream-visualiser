import torch
import torch.nn as nn


class ImageLayer(nn.Module):
    def __init__(self, image_shape=(28, 28)):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(image_shape, requires_grad=True))
        nn.init.normal_(self.weight, mean=0, std=1.0)
    
    def forward(self):
        return torch.unsqueeze(torch.mul(1, self.weight), 0)


class TargetVisualiser(nn.Module):
    def __init__(self, classifier, target_idx):
        super().__init__()
        self.classifier = classifier
        self.modify = ImageLayer()
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.target_idx = target_idx
    
    def forward(self):
        """Returns activation of the neuron associated with the target."""
        modified_image = self.modify()
        predictions = self.classifier(modified_image)
        target_activation = predictions[:, self.target_idx]
        return target_activation
