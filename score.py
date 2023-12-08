import os
import logging
import json
import torch
import torchxrayvision as xrv
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from process import base64_to_ndarray, ndarray_to_base64


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "trained/model")
    # deserialize the model file back into a torch model
    model = torch.load(path)
    model.float()
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    logging.info("DenseNet121: Loaded model from path: %s", path)
    logging.info("Updated successfully")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("DenseNet121: Request received")
    data = json.loads(raw_data)["image"]
    data = base64_to_ndarray(data)
    logging.info("Data Type: %s", data.dtype)
    predictions = inference(data)
    gradimages = get_gradcam(data, None)
    logging.info("DenseNet121: Request processed")
    predictions.update(gradimages)
    results = json.dumps(predictions)
    return results


def inference(image: np.ndarray):
    """
    Args:
        image (np.array): Image to be processed
    Returns:
        torch.tensor: Prediction
    """
    if len(image.shape) > 2:
        image = image[:, :, 0]
    elif len(image.shape) < 2:
        logging.error("Invalid image shape")
        return None

    image = image[None, :, :]
    output = {}
    with torch.no_grad():
        image = torch.from_numpy(image).unsqueeze(0)
        logging.info("Image shape: {}, Image Dtype".format(image.shape, image.dtype))
        if torch.cuda.is_available():
            image = image.cuda()

        preds = model(image).cpu()
        output["preds"] = dict(
            zip(xrv.datasets.default_pathologies, preds[0].detach().numpy().tolist())
        )

    logging.info("Prediction: {}".format(output))
    return output


def get_gradcam(image: np.ndarray, target_layers: torch.nn.Module):
    """
    Args:
        image (np.array): Image to be processed
    Returns:
        torch.tensor: Prediction
    """
    outputs = {}
    targets = [
        ClassifierOutputSoftmaxTarget(xrv.datasets.default_pathologies.index(pathology))
        for pathology in xrv.datasets.default_pathologies
    ]
    image = image[None, None, :, :]
    image = torch.from_numpy(image)
    image_np = image.numpy()
    image_np = image_np - np.min(image_np)
    image_np = image_np / np.max(image_np)
    image_np = np.transpose(image_np, (0, 2, 3, 1))
    if target_layers is None:
        target_layers = [model.features[-2][-1][-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    for target, pathology in zip(targets, xrv.datasets.default_pathologies):
        grayscale_cam = cam(input_tensor=image, targets=[target])
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        outputs[pathology] = ndarray_to_base64(visualization.squeeze())

    return {"gradcam": outputs}
