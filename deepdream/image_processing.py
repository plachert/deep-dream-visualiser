import scipy.ndimage as nd
import torch
import numpy as np
from queue import Queue
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
    deprocessed_image = deprocess(image)
    rescaled_image = (deprocessed_image * 255).astype(np.uint8)
    transposed_image = np.transpose(rescaled_image, (1, 2, 0))
    frame = cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR)
    return frame


def run_pyramid(image=image):
    images_collection = [image]
    octaves = [image]
    octave_n = 1#4
    octave_scale = 1.4
    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        input_image = octave_base + detail
        
        jitter = 30
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        input_image = np.roll(np.roll(input_image, ox, -1), oy, -2) # apply jitter shift
        
        processed_images = optimize_image(model, input_image, 3, target_idx=71)
        input_image = processed_images[-1]
        images_collection.extend(processed_images)
        input_image = np.roll(np.roll(input_image, -ox, -1), -oy, -2)
        detail = input_image-octave_base
    return images_collection