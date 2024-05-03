import os
import logging
import json
import io
import joblib

import torch
from torchvision.transforms.functional import to_pil_image

from torchcam import methods
from torchcam.utils import overlay_mask

from azure.storage.blob import BlobServiceClient, ContentSettings
from process import bytes_to_pil, preprocess
from process import INDEX_TO_PATHOLOGY, PATHOLOGY_TO_INDEX
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Methods available in torchcam
METHODS = {
    "gradcam": methods.GradCAM,
    "scorecam": methods.ScoreCAM,
    "gradcam++": methods.GradCAMpp,
    "isc": methods.ISCAM,
    "xgradcam": methods.XGradCAM,
    "layercam": methods.LayerCAM,
    "smoothgradcam": methods.SmoothGradCAMpp,
}


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model, blob_service_client, container_client, cam, account_name
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "model/"), "model.pkl")
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    # deserialize the model file back into a torch model
    model = joblib.load(path)
    model.float()
    if torch.cuda.is_available():
        logger.info(f"{model.weights}: Using CUDA for inference")
        model = model.cuda()
    else:
        logger.info(f"{model.weights}: Using CPU for inference")
    model.eval()  # Set model to evaluation mode for inference
    logger.info(f"{model.weights}: Loaded model from path: {path}")
    # Connect to the blob storage
    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=os.getenv("AZURE_STORAGE_ACCESS_KEY"),
    )
    container_client = blob_service_client.get_container_client(
        os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    )
    logger.info(f"{model.weights}: Connected to blob storage")
    method = os.getenv("METHOD", "gradcam")

    # Get the target layer for the CAM, layers can be grouped into blocks and called
    target_layer = model.features.get_submodule(
        os.getenv("TARGET_LAYER", "denseblock4.denselayer16.conv2")
    )
    cam = METHODS[method](model=model, target_layer=target_layer)


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logger.info(f"{model.weights}: Request received")

    # Initialize the response dictionary
    response = {}

    # Get the uuid for the image from the request
    image_uuid = json.loads(raw_data).get("image_uuid", None)
    if not image_uuid:
        response = json.dumps({"error": "No image_uuid provided"})
        return response

    # Get the image from the blob storage
    blob_client = container_client.get_blob_client(image_uuid)
    byte_data = blob_client.download_blob().readall()

    # Converting the bytes of image to PIL image
    data = bytes_to_pil(byte_data)
    transformed, rescaled = preprocess(data)

    # Get the predictions and gradcam images
    preds, preds_dict = inference(rescaled)
    gradcam_dict = get_gradcam(transformed, preds, image_uuid, preds_dict)

    for pathology in preds_dict:
        if response.get("predictions") is None:
            response["predictions"] = {}
        response["predictions"][pathology] = {
            "prediction": preds_dict[pathology],
            "gradcam": gradcam_dict[pathology],
        }
    logger.info(f"{model.weights}: Request processed")
    response["name"] = image_uuid
    response = json.dumps(response)
    return response


def inference(image: torch.tensor, k: int=5) -> tuple[dict, torch.Tensor]:
    """
    Args:
        image (np.array): Image to be processed
        k (int): Top k number of predictions to return and generate gradcam (default: 5)
    Returns:
`       tuple[dict, torch.Tensor]: Dictionary of predictions for HTTP response and tensor of predictions
    """

    # Add a batch dimension to the image tensor
    image = image.unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()

    # Get the predictions
    preds = model(image).cpu()
    preds_topk = torch.topk(preds, 5)

    preds_dict = {}
    for i in range(len(preds_topk.indices[0])):
        preds_dict[INDEX_TO_PATHOLOGY[preds_topk.indices[0][i].item()]] = preds_topk.values[0][i].item()

    return preds, preds_dict


def get_gradcam(
    transformed: torch.Tensor, preds: torch.Tensor, image_uuid: str, preds_dict: dict
) -> dict:
    """
    Args:
        transformed: Image after preprocessing
        preds: Predictions
        image_uuid: UUID of the image in the blob storage
    Returns:
        dict: Dictionary of gradcam images URLs for each pathology in blob storage
    """

    # Settings for the blob storage for image as PNG format
    content_settings = ContentSettings(content_type="image/png")

    # Reduces the gradients to zero
    model.zero_grad()
    outputs = {}

    # Gradcam for each pathology
    cam_extractors = [
        cam(class_idx=PATHOLOGY_TO_INDEX[i], scores=preds, retain_graph=True)
        for i in preds_dict.keys()
    ]

    for i, pathology in enumerate(preds_dict):
        activation_maps = cam_extractors[i]

        # Overlay the mask on the image and save it to the blob storage
        activation_maps = (
            activation_maps[0]
            if len(activation_maps) == 1
            else cam_extractors[i].fuse_cams(activation_maps)
        )
        result = overlay_mask(
            to_pil_image(transformed.expand(3, -1, -1)),
            to_pil_image(activation_maps, mode="F"),
            alpha=0.7,
        )
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        buffer.seek(0)
        blob_client = container_client.get_blob_client(f"{image_uuid}_{pathology}")
        blob_client.upload_blob(
            buffer,
            overwrite=True,
            blob_type="BlockBlob",
            content_settings=content_settings,
        )
        logger.info(f"Uploaded {blob_client.blob_name} to blob storage")
        outputs[pathology] = blob_client.blob_name
    return outputs
