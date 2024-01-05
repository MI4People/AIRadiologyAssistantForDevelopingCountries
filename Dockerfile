FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:latest AS inferencing-assets

USER root:root

# Install libpng-tools and opencv
RUN apt-get update && \
    apt-get install --no-install-recommends --no-upgrade -y \
    libpng-tools \
    python3-opencv && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER dockeruser

# Code
COPY code /var/azureml-app
ENV AZUREML_ENTRY_SCRIPT=score.py

# Model
COPY model/ /var/azureml-app/azureml-models
ENV AZUREML_MODEL_DIR=/var/azureml-app/azureml-models

# Environment  Variable
ENV AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://host.docker.internal:10000/devstoreaccount1;
ENV AZURE_STORAGE_CONTAINER_NAME=xray
ENV CAM_METHOD=gradcam

# Install dependencies
COPY requirements.txt /var/azureml-app
RUN python -m pip install -r /var/azureml-app/requirements.txt --no-cache-dir