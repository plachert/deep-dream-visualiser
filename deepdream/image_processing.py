"""This module provides tools for image (np.ndarray) manipulations."""
from __future__ import annotations

import pathlib

import numpy as np
import scipy.ndimage as nd
from activation_tracker.model import ModelWithActivations
from PIL import Image

from deepdream.optimization import optimize_image


def convert_to_255scale(image):
    """Convert an image to 255 scale."""
    clipped = np.clip(image, 0., 1.)
    image_255 = 255 * clipped
    return image_255.astype(np.uint8)


def channel_last(image):
    """Channel first to channel last."""
    transposed = np.transpose(image, (1, 2, 0))
    return transposed


def channel_first(image):
    """Channel last to channel first."""
    transposed = np.transpose(image, (2, 0, 1))
    return transposed


def load_image_from(path: pathlib.Path):
    """Load an image as a np.ndarray (3, h, w)"""
    image = np.array(Image.open(path)).astype(np.float32)
    image /= 255.
    return channel_first(image)


def create_random_image(h=500, w=500):
    """Create a random image (channel-first)."""
    shape = (3, h, w)
    image = np.random.uniform(low=0.0, high=1, size=shape).astype(np.float32)
    return image


def create_jitter_parameters(jitter_size: int = 30):
    """Handle parameters for jittering and reverse transformation."""
    jitter_x, jitter_y = np.random.randint(-jitter_size, jitter_size+1, 2)
    unjitter_x, unjitter_y = -jitter_x, -jitter_y
    return jitter_x, jitter_y, unjitter_x, unjitter_y


def apply_shift(image, ox, oy):
    """Apply jitter shift."""
    shifted_image = np.roll(np.roll(image, ox, -1), oy, -2)
    return shifted_image


def create_octave_image(image, octave_scale):
    """Create next level of the pyramid."""
    octave = nd.zoom(image, (1, 1.0/octave_scale, 1.0/octave_scale), order=1)
    return octave


def create_reversed_octave_pyramid(image, octave_n, octave_scale):
    """Create reversed octave pyramid (starts with the smallest image)."""
    pyramid = [image]
    for _ in range(octave_n-1):
        pyramid.append(create_octave_image(pyramid[-1], octave_scale))
    reversed_pyramid = pyramid[::-1]
    return reversed_pyramid


def resize_to_image(reference_image, image):
    """Resize image to the same shape as the reference image."""
    h_ref, w_ref = reference_image.shape[-2:]
    h_image, w_image = image.shape[-2:]
    h_ratio = h_ref / h_image
    w_ratio = w_ref / w_image
    image_resized = nd.zoom(image, (1, h_ratio, w_ratio), order=1)
    return image_resized


def run_pyramid(
    model: ModelWithActivations,
    image: np.ndarray,
    jitter_size: int = 30,
    octave_n: int = 2,
    octave_scale: float = 1.4,
    n_iterations: int = 10,
    lr: float = 0.1,
    regularization_coeff: float = 0.1,
):
    images_collection = [image]
    reversed_pyramid = create_reversed_octave_pyramid(
        image, octave_n=octave_n, octave_scale=octave_scale,
    )
    detail = np.zeros_like(reversed_pyramid[0])
    for octave, octave_base in enumerate(reversed_pyramid):
        if octave > 0:
            detail = resize_to_image(octave_base, detail)
        image = octave_base + detail
        jitter_x, jitter_y, unjitter_x, unjitter_y = create_jitter_parameters(
            jitter_size,
        )
        image = apply_shift(image, jitter_x, jitter_y)
        processed_images = optimize_image(
            model=model,
            image=image,
            n_iterations=n_iterations,
            lr=lr,
            regularization_coeff=regularization_coeff,
        )
        processed_images = [
            apply_shift(
                image, unjitter_x, unjitter_y,
            ) for image in processed_images
        ]
        images_collection.extend(processed_images)
        image = processed_images[-1]
        detail = image - octave_base
    return images_collection
