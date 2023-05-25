from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn

SUPPORTED_FILTERS = {}


def register_filter(cls):
    SUPPORTED_FILTERS[cls.__name__] = cls
    return cls


Activation = namedtuple('Activation', ['layer_type', 'output_shape', 'value'])


class ActivationFilter:
    """Abstract class for filtering strategies."""

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        raise NotImplementedError

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        """List all available options based on the strategy."""
        raise NotImplementedError


@register_filter
class TypeActivationFilter(ActivationFilter):
    """Filter by type e.g. collect all ReLU activations."""

    def __init__(self, types: list[str]) -> None:
        self.types = types

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        return [activation for activation in activations if activation.layer_type in self.types]

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        return list({activation.layer_type for activation in activations})


@register_filter
class IndexActivationFilter(ActivationFilter):
    """Filter by indices of the activations."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = list(map(int, indices))

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        return [activations[idx] for idx in self.indices]

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        return list(range(len(activations)))


@register_filter
class TargetsActivationFitler(ActivationFilter):
    """Preserve neurons associated with given classes."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = list(map(int, indices))

    def filter_activations(self, activations: list[Activation]) -> list[Activation]:
        last_activation = activations[-1]  # last layer
        activations = []
        for idx in self.indices:
            # In this case it's just a label of the neuron associated with a given idx
            layer_type = f'target_{idx}'
            value = last_activation.value[:, idx]
            output_shape = value.shape
            activation = Activation(layer_type, output_shape, value)
            activations.append(activation)
        return activations

    @staticmethod
    def list_all_available_parameters(activations: list[Activation]) -> list:
        n_classes = activations[-1].output_shape[-1]
        return list(range(n_classes))


class ModelWithActivations(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        activation_filter: ActivationFilter | None = None,
        example_input: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self._activations = []
        self._register_activation_hook()
        self.activation_filter = activation_filter
        if example_input is not None:
            self(example_input)  # recon pass

    @property
    def activations(self) -> list[torch.Tensor]:
        """Return activations based on the activation_filter."""
        filtered_activations = self._activations
        if self.activation_filter:
            filtered_activations = self.activation_filter.filter_activations(
                self._activations,
            )
        filtered_activations = [
            activation
            for activation in filtered_activations
        ]
        return filtered_activations

    @property
    def activations_values(self):
        """Return values of the filtered activations."""
        activations = self.activations
        activations_values = [activation.value for activation in activations]
        return activations_values

    def _register_activation_hook(self):
        def activation_hook(module, input_, output):
            layer_type = module.__class__.__name__
            output_shape = output.shape
            value = output
            activation = Activation(layer_type, output_shape, value)
            self._activations.append(activation)
        for layer in flatten_modules(self.model):
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
