import base64
import io
import numpy as np
from PIL import Image

import torchvision


def bytes_to_pil(img_bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(img_bytes))
    return image


def preprocess(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.CenterCrop(1024),
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
    ])
    transformed_output = transform(img)

    img_min = transformed_output.min()
    img_max = transformed_output.max()

    rescaled_output = 2048*(transformed_output-img_min)/(img_max-img_min) - 1024
    return transformed_output, rescaled_output
