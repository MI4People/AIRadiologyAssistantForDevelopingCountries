import os
import logging
import json
import io

import torch
from torchvision.transforms.functional import to_pil_image

from torchcam import methods
from torchcam.utils import overlay_mask

from azure.storage.blob import BlobServiceClient, ContentSettings
from process import bytes_to_pil, preprocess
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

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
    global model, blob_service_client, container_client, cam
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    # deserialize the model file back into a torch model
    model = torch.load(path)
    model.float()
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    logger.info("DenseNet121: Loaded model from path: %s", path)
    # Connect to the blob storage
    blob_service_client = BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    container_client = blob_service_client.get_container_client(
        os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    )
    logger.info("DenseNet121: Connected to blob storage")
    method = os.getenv("METHOD", "gradcam")
    cam = METHODS[method](model=model, target_layer=model.features[-2][-1][-1])


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logger.info("Densenet121: Request received")
    results = {}
    image_uuid = json.loads(raw_data)["image_uuid"]
    blob_client = container_client.get_blob_client(image_uuid)
    byte_data = blob_client.download_blob().readall()
    data = bytes_to_pil(byte_data)
    transformed, rescaled = preprocess(data)
    results["predictions"], preds = inference(rescaled)
    results["gradimages"] = get_gradcam(
        transformed, preds, image_uuid
    )
    logger.info("DenseNet121: Request processed")
    results = json.dumps(results)
    return results


def inference(image: torch.tensor):
    """
    Args:
        image (np.array): Image to be processed
    Returns:
        torch.tensor: Prediction
    """
    image = image.unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()

    preds = model(image).cpu()
    return dict(zip(model.pathologies, preds[0].detach().numpy().tolist())), preds


def get_gradcam(transformed: torch.Tensor, preds: torch.Tensor, image_uuid: str):
    """
    Args:
        image (np.array): Image to be processed
    Returns:
        torch.tensor: Prediction
    """
    content_settings = ContentSettings(content_type="image/png")
    model.zero_grad()
    outputs = {}
    cam_extractors = [
        cam(class_idx=i, scores=preds, retain_graph=True) for i, _ in enumerate(model.pathologies)
    ]
    for i, pathology in enumerate(model.pathologies):
        activation_maps = cam_extractors[i]
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
        result.save(buffer, format='PNG')
        buffer.seek(0)
        blob_client = container_client.get_blob_client(
            f"{image_uuid}_{pathology}"
        )
        blob_client.upload_blob(buffer, overwrite=True, blob_type="BlockBlob", content_settings=content_settings)
        logger.info(f"Uploaded {blob_client.blob_name} to blob storage")
        outputs[pathology] = f"http://127.0.0.1:10000/{blob_client.account_name}/{blob_client.container_name}/{blob_client.blob_name}"
    return outputs
