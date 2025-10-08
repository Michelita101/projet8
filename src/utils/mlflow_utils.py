import mlflow
import time

def start_mlflow_run(model_name, params: dict, tags: dict = None):
    run_name = f"{model_name}_{int(time.time())}"
    mlflow.set_experiment("image_segmentation")
    mlflow.start_run(run_name=run_name)
    
    mlflow.log_params(params)
    if tags:
        mlflow.set_tags(tags)
