from torchmetrics.functional import total_variation
from deepdream.model import ModelWithActivations
import numpy as np
from typing import Optional, List
from tqdm import tqdm
import torch


def prepare_input_image(input_image: np.ndarray):
    input_image = torch.from_numpy(input_image)
    input_image = torch.unsqueeze(input_image, 0) # minibatch
    input_image.requires_grad = True
    return input_image


def optimize_image(
    model: ModelWithActivations, 
    image: np.ndarray,
    n_iterations: int = 10,
    target_idx: Optional[int] = None,
    activation_types: Optional[str] = None,
    activation_idxs: Optional[List[int]] = None,
    regularization_coeff: float = 0.1,
    lr: float = 0.1,
    ) -> np.ndarray:
    input_image = prepare_input_image(np.copy(image))
    processed_images = [np.copy(image)]
    size = input_image.shape[-2] * input_image.shape[-1]
    optimizer = torch.optim.Adam([input_image], lr=lr)
    for _ in tqdm(range(n_iterations)):
        optimizer.zero_grad()
        model(input_image) # just to call forward and calculate activations
        if target_idx is not None:
            activations = model.get_target_activation(target_idx)
        elif activation_types is not None:
            activations = model.get_activations_by_types(activation_types)
        elif activation_idxs is not None:
            activations = model.get_activations_by_idx(activation_idxs)
        else:
            activations = model.get_all_activations()
        losses = [torch.linalg.vector_norm(activation, ord=2) for activation in activations]
        loss = -torch.mean(torch.stack(losses)) 
        regularization = regularization_coeff * 10000 * total_variation(input_image) / size
        loss += regularization
        loss.backward()
        optimizer.step()
        # for vis
        processed_images.append(input_image.detach().numpy().squeeze())
    return processed_images
