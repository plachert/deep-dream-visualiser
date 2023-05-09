import torch.nn as nn
from torchvision import models


def flatten_modules(module):
    flattened_modules = []
    for child in module.children():
        if isinstance(child, nn.Sequential):
            flattened_modules.extend(flatten_modules(child))
        else:
            flattened_modules.append(child)
    return flattened_modules


class DeepDreamModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False 
        self.activations = []
        self._register_activation_hook()
            
    def _register_activation_hook(self):
        def activation_hook(module, input_, output):
            self.activations.append(output)
        for layer in flatten_modules(self.model):
            layer.register_forward_hook(activation_hook)
    
    def forward(self, input_):
        self.activations.clear()
        return self.model.forward(input_)
            
            
if __name__ == "__main__":
    import torch
    deep_dream = DeepDreamModel(model=models.vgg16(pretrained=True))
    deep_dream(torch.rand(1, 3, 100, 100))
    print(len(deep_dream.activations))
    deep_dream(torch.rand(1, 3, 100, 100))
    print(len(deep_dream.activations))
    deep_dream(torch.rand(1, 3, 100, 100))
    print(len(deep_dream.activations))
    