import os
import logging
import json
import torch
import torchxrayvision as xrv
import numpy as np


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "trained/model"
    )
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
    data = json.loads(raw_data)["data"]
    data = np.array(data, dtype="float32")
    logging.info("Data Type: %s", data.dtype)
    result = inference(data)
    logging.info("DenseNet121: Request processed")
    return json.dumps(str(result))


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
        output["preds"] = dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))

    logging.info("Prediction: {}".format(output))
    return output
