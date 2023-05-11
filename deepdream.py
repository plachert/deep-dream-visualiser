from torchmetrics.functional import total_variation
from model import ModelWithActivations
import numpy as np
from typing import Optional, List
from tqdm import tqdm
import torch

def prepare_input_image(input_image: np.ndarray):
    input_image = torch.from_numpy(input_image)
    input_image = torch.unsqueeze(input_image) # minibatch
    input_image.requires_grad = True
    return input_image

def torch_to_numpy(input_image: torch.Tensor):
    input_image = input_image.detach().numpy()
    input_image = np.squeeze(input_image)
    return input_image

def smooth_grad(grad):
    grad_std = torch.std(grad)
    grad_mean = torch.mean(grad)
    grad -= grad_mean
    grad /= grad_std
    return grad


def optimize_image(
    model: ModelWithActivations, 
    image: np.ndarray,
    n_iterations: int = 10,
    target_idx: Optional[int] = None, # TODO: Handle arguments
    activation_types: Optional[str] = None, # TODO: Handle arguments
    activation_idxs: Optional[List[int]] = None, # TODO: Handle arguments
    regularization_coeff: float = 0.1,
    lr: float = 0.1,
    ) -> np.ndarray:
    input_image = prepare_input_image(image)
    for _ in tqdm(range(n_iterations)):
        model() # just to call forward and calculate activations
        if target_idx is not None:
            activations = model.get_target_activation(target_idx)
        elif activation_types is not None:
            activations = model.get_activations_by_types(activation_types)
        elif activation_idxs is not None:
            activations = model.get_activations_by_idx(activation_idxs)
        else:
            activations = model.get_all_activations()
        losses = [torch.nn.MSELoss(reduction='mean')(activation, torch.zeros_like(activation)) for activation in activations]
        loss = torch.mean(torch.stack(losses)) 
        loss -= regularization_coeff * total_variation(input_image) # gradient ascent, so '-' instead of '+'
        loss.backward()
        grad = input_image.grad.data
        grad = smooth_grad(grad)
        input_image.data += lr * grad # gradient ascent, so '+' instead of '-'
        input_image.grad.data.zero_()
    optimized_image = torch_to_numpy(input_image)
    return optimized_image
        
        