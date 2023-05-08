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
            
    def map_layers(self):
        # print(len(list(flatten_modules(self.model))))
        for name, module in self.model.named_modules():
            print(name, module)
            
            
if __name__ == "__main__":
    deep_dream = DeepDreamModel(model=models.vgg16(pretrained=True))
    deep_dream.map_layers()
    # def flatten_lists(lista):
    #     for item in lista:
    #         if isinstance(item, list):
    #             yield from flatten_lists(item)
    #         else:
    #             yield item


    # lista = [1, 2, [3, [4, 5, [6]], 7]]
    # print(list(flatten_lists(lista)))