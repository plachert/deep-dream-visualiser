import torch.nn as nn
import torch
from typing import List, Tuple, Optional

SUPPORTED_FILTERS = {}

def register_filter(cls):
    SUPPORTED_FILTERS[cls.__name__] = cls
    return cls


class ActivationFilter:
    """Abstract class for filtering strategies."""
    def filter_activations(self, activations: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        raise NotImplementedError

    def list_all_available_parameters(self, activations: List[Tuple[str, torch.Tensor]]) -> List:
        """List all available options based on the strategy."""
        raise NotImplementedError


@register_filter
class TypeActivationFilter(ActivationFilter):
    """Filter by type e.g. collect all ReLU activations."""
    def __init__(self, types: List[str]) -> None:
        self.types = types

    def filter_activations(self, activations: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        return [activation for activation in activations if activation[0] in self.types]

    def list_all_available_parameters(self, activations: List[Tuple[str, torch.Tensor]]) -> List:
        return list(set([activation[0] for activation in activations]))


@register_filter
class IndexActivationFilter(ActivationFilter):
    """Filter by indices of the activations."""
    def __init__(self, indices: List[int]) -> None:
        self.indices = indices

    def filter_activations(self, activations: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        return [activations[idx] for idx in self.indices]

    def list_all_available_parameters(self, activations: List[Tuple[str, torch.Tensor]]) -> List:
        return list(range(len(activations)))


@register_filter
class TargetsActivationFitler(ActivationFilter):
    """Preserve neurons associated with given classes."""
    def __init__(self, indices: List[int]) -> None:
        self.indices = list(map(int, indices))

    def filter_activations(self, activations: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        last_activation = activations[-1][1] # last layer
        return [(f"target_{idx}", last_activation[:, idx]) for idx in self.indices]

    def list_all_available_parameters(self, activations: List[Tuple[str, torch.Tensor]]) -> List:
        last_activation = activations[-1][1]
        return list(range(10))#last_activation.shape[-1]#list(range(10000))##list(range(last_activation.shape[-1]))


class ModelWithActivations(nn.Module):
    def __init__(self, model: nn.Module, activation_filter: Optional[ActivationFilter] = None) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self._activations = []
        self._register_activation_hook()
        self.activation_filter = activation_filter

    @property
    def strategy_parameters(self):
        if self.activation_filter is None:
            return None
        return self.activation_filter.list_all_available_parameters(self._activations)

    @property
    def activations(self) -> List[torch.Tensor]:
        """Return activations based on the activation_filter."""
        filtered_activations = self._activations
        if self.activation_filter:
            filtered_activations = self.activation_filter.filter_activations(self._activations)
        filtered_activations = [activation[1] for activation in filtered_activations] # remove first item in tuples
        return filtered_activations

    def _register_activation_hook(self):
        def activation_hook(module, input_, output):
            self._activations.append((module.__class__.__name__, output))
        for layer in flatten_modules(self.model):
            activation_hook(layer, None, None)
            layer.register_forward_hook(activation_hook)

    def forward(self, input_):
        self._activations.clear()
        return self.model.forward(input_)


def flatten_modules(module):
    flattened_modules = []
    for child in module.children():
        if isinstance(child, nn.Sequential):
            flattened_modules.extend(flatten_modules(child))
        else:
            flattened_modules.append(child)
    return flattened_modules
