import scipy.ndimage as nd
import torch
import numpy as np
from deepdream.optimization import optimize_image
from deepdream.model import ModelWithActivations
from torchvision.models import vgg16
import cv2

image = np.random.rand(3, 225, 225).astype(dtype=np.float32)
model = ModelWithActivations(model=vgg16(pretrained=True))


def img2frame(image):
    def deprocess(image):
        img = image.copy()
        img[0, :, :] *= 0.229
        img[1, :, :] *= 0.224
        img[2, :, :] *= 0.225
        img[0, :, :] += 0.485
        img[1, :, :] += 0.456
        img[2, :, :] += 0.406
        return img
    deprocessed_image = image#deprocess(image)
    rescaled_image = (deprocessed_image * 255).astype(np.uint8)
    transposed_image = np.transpose(rescaled_image, (1, 2, 0))
    frame = cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR)
    return frame

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
    octave = nd.zoom(image, (1, 1.0/octave_scale,1.0/octave_scale), order=1)
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
    image=image,
    jitter_size=30,
    target_idx=71,
    octave_n=2,
    octave_scale=1.4,
    n_iterations=10,
    ):
    images_collection = [image]
    reversed_pyramid = create_reversed_octave_pyramid(image, octave_n=octave_n, octave_scale=octave_scale)
    detail = np.zeros_like(reversed_pyramid[0])
    for octave, octave_base in enumerate(reversed_pyramid):
        if octave > 0:
            detail = resize_to_image(octave_base, detail)
        image = octave_base + detail
        jitter_x, jitter_y, unjitter_x, unjitter_y = create_jitter_parameters(jitter_size)
        image = apply_shift(image, jitter_x, jitter_y)
        processed_images = optimize_image(model, image, n_iterations, target_idx=target_idx)
        image = processed_images[-1]
        images_collection.extend(processed_images)
        image = apply_shift(image, unjitter_x, unjitter_y)
        detail = image - octave_base
    return images_collection