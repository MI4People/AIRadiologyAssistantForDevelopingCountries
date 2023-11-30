import base64
import io
import numpy as np
from PIL import Image


def base64_to_ndarray(base64_string) -> np.ndarray:
    img_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_data))
    ndarray = np.array(image, dtype="float32")

    return ndarray
