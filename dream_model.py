import torch.nn as nn
from torchvision import models
from typing import List


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
        self._activations = []
        self._register_activation_hook()
        
    def get_target_activation(self, target_idx):
        """Return activation of the neuron associated with the target."""
        if not self._activations: # TODO: maybe some warning if the forward pass hasn't been called?
            return self._activations
        return self._activations[-1][:, target_idx]
        
    def get_activations_by_idx(self, idxs: List[int]):
        """Return activations by indices."""
        if not self._activations: # TODO: maybe some warning if the forward pass hasn't been called?
            return self._activations
        return [self._activations[idx][1] for idx in idxs]
    
    def get_activations_by_types(self, types: List[str]):
        """Return all activations of layers of given types e.g. ReLU."""
        if not self._activations:
            return self._activations # TODO: maybe some warning if the forward pass hasn't been called?
        return [activation[1] for activation in self._activations if activation[0] in types]
    
    def get_all_activations(self):
        """Return activations of all layers."""
        idxs = list(range(0, len(self._activations)))
        return self.get_activations_by_idx(idxs)
        
    
    def _register_activation_hook(self):
        def activation_hook(module, input_, output):
            self._activations.append((module.__class__.__name__, output))
        for layer in flatten_modules(self.model):
            layer.register_forward_hook(activation_hook)
    
    def forward(self, input_):
        self._activations.clear()
        return self.model.forward(input_)
            
            
if __name__ == "__main__":
    import torch
    deep_dream = DeepDreamModel(model=models.vgg16(pretrained=True))
    print(len(deep_dream.get_activations_by_types(["Linear"])))
    deep_dream(torch.rand(1, 3, 100, 100))
    # print(len(deep_dream.get_activations_by_types(["Linear"])))