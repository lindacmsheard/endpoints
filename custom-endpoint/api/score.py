import os
import sys
import logging
from chronos import ChronosPipeline
import torch



def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    # global model
    # # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # # Please provide your model's folder name if there is one
    # model_path = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"), "model/sklearn_regression_model.pkl"
    # )
    print(sys.executable)
    logging.info(sys.executable)
    logging.info("listing model directory:")
    logging.info(os.listdir(os.getenv("AZUREML_MODEL_DIR")))
    # # deserialize the model file back into a sklearn model
    # model = joblib.load(model_path)
    # logging.info("Init complete")

    #------
    # linda comment
    # lets see if there is performance value to be gained from following this pattern and 
    # instantiating the pipeline here:

    global pipeline

    model_version="amazon/chronos-t5-tiny"

    pipeline = ChronosPipeline.from_pretrained(
            model_version,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

    logging.info("Pipeline instantiated")

    # let's also log out some GPU config info
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logging.info('__CUDNN VERSION:', torch.backends.cudnn.version())
        logging.info('__Number CUDA Devices:', torch.cuda.device_count())
        logging.info('__CUDA Device Name:',torch.cuda.get_device_name(0))
        logging.info('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
    else:
        logging.warning('CUDA is not available')

    # and lastly let's see if we can get the model name from environment variables rather than hardcoding it above
    logging.info(f"Model name provided via deployment config: {os.getenv('CHRONOS_MODEL_NAME')}")
    print(f"Model name provided via deployment config: {os.getenv('CHRONOS_MODEL_NAME')}")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.

    """
    logging.info("Forcaster: request received")
    logging.info(f"Pipeline available: {type(pipeline)}")

    # let's just log out what the data looks like
    logging.info(f"object structure: {[key for key in raw_data]}")
    
    # assuming we're receiving a data item, let's just return it to the caller for now,
    # along with debug information
    result = {
                "received":raw_data,
                "pipelinetype": f"{type(pipeline)}",
                "python": f"{sys.executable}",
                "cuda": f"{torch.cuda.is_available()}",
                "model": f"Model name provided via deployment config: {os.getenv('CHRONOS_MODEL_NAME')}"
              }
    logging.info(type(result))
    logging.info("Request processed")
    return result