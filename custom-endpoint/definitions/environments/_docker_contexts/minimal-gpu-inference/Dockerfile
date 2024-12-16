# start from https://learn.microsoft.com/en-us/azure/machine-learning/concept-prebuilt-docker-images-inference?view=azureml-api-2
# see also: https://github.com/Azure/AzureML-Containers
# and https://mcr.microsoft.com/v2/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference/tags/list 
FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference:latest


USER root

# Update the package lists and install necessary packages
RUN apt-get -y update && \
    apt-get install -y git


# Install required packages from pypi
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

