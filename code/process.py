import io
import numpy as np
from PIL import Image
import torchvision


PATHOLOGY_TO_INDEX = {
    "Atelectasis": 0,
    "Consolidation": 1,
    "Infiltration": 2,
    "Pneumothorax": 3,
    "Edema": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Effusion": 7,
    "Pneumonia": 8,
    "Pleural_Thickening": 9,
    "Cardiomegaly": 10,
    "Nodule": 11,
    "Mass": 12,
    "Hernia": 13,
    "Lung Lesion": 14,
    "Fracture": 15,
    "Lung Opacity": 16,
    "Enlarged Cardiomediastinum": 17,
}


INDEX_TO_PATHOLOGY = {
    0: "Atelectasis",
    1: "Consolidation",
    2: "Infiltration",
    3: "Pneumothorax",
    4: "Edema",
    5: "Emphysema",
    6: "Fibrosis",
    7: "Effusion",
    8: "Pneumonia",
    9: "Pleural_Thickening",
    10: "Cardiomegaly",
    11: "Nodule",
    12: "Mass",
    13: "Hernia",
    14: "Lung Lesion",
    15: "Fracture",
    16: "Lung Opacity",
    17: "Enlarged Cardiomediastinum",
}


def bytes_to_pil(img_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(img_bytes))
    return image


def preprocess(img):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.CenterCrop(1024),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )
    transformed_output = transform(img)

    img_min = transformed_output.min()
    img_max = transformed_output.max()

    rescaled_output = 2048 * (transformed_output - img_min) / (img_max - img_min) - 1024
    return transformed_output, rescaled_output
