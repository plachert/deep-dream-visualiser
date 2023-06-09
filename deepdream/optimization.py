"""This module provides the optimization algorithm for DeepDream."""
from __future__ import annotations

import numpy as np
import torch
from activation_tracker.model import ModelWithActivations
from torchmetrics.functional import total_variation
from tqdm import tqdm

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def prepare_input_image(input_image: np.ndarray):
    """Prepare the image to be used in optimization."""
    input_image = input_image.astype(dtype=np.float32)
    input_image = torch.from_numpy(input_image)
    input_image = input_image.to(device)
    input_image = torch.unsqueeze(input_image, 0)  # minibatch
    input_image.requires_grad = True
    return input_image


def optimize_image(
    model: ModelWithActivations,
    image: np.ndarray,
    n_iterations: int = 10,
    regularization_coeff: float = 0.1,
    lr: float = 0.1,
) -> list[np.ndarray]:
    model.to(device)
    input_image = prepare_input_image(np.copy(image))
    processed_images = []
    size = input_image.shape[-2] * input_image.shape[-1]
    optimizer = torch.optim.Adam([input_image], lr=lr)
    for _ in tqdm(range(n_iterations)):
        optimizer.zero_grad()
        model(input_image)  # just to call forward and calculate activations
        activations = model.activations_values['filtered']
        losses = [
            torch.linalg.vector_norm(
                activation, ord=2,
            ) for activation in activations
        ]
        loss = -torch.mean(torch.stack(losses))
        regularization = regularization_coeff * \
            10000 * total_variation(input_image) / size
        loss += regularization
        loss.backward()
        optimizer.step()
        img = np.copy(input_image.cpu().detach().numpy().squeeze())
        processed_images.append(img)
    return processed_images
