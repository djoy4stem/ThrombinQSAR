
## About 
This is project that illustrates end-to-end development of a Thrombin inhibiton predictive model. The data was acquired from the Molecule Activity Cliff Estimation (MoleculeACE) CHEMBL204_Ki benchmark dataset available [here](https://github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data).


**Make sure to set the correct MLflow Tracking URI:** This can be done in the main.py by adding *mlflow.set_tracking_uri(PATH__TO__FOLDER)*, where  *PATH__TO__FOLDER* can be the mlruns/ folder under the root directory of the project. You can check the .mlflow configuration directory or your environment variables: *echo $MLFLOW_TRACKING_URI*, and *echo $MLFLOW_ARTIFACT_URI*.


## Installation

1. Create a conda environment *conda create -n qsar_mlops python=3.10*
2. Install required packages by running *pip install -r requirements.txt*
3. 



## Model Training

## Model Validation

## Model Deployment


SEcurity Group on EC2: Djoy4StemGroup
ECS cluster: https://us-east-2.console.aws.amazon.com/ecs/v2/clusters/my_fastapi_djoy4stem/services?region=us-east-2
task name: my_fastapi_task1
Service name: my_fastapi_task1


https://www.evidentlyai.com/blog/mlops-monitoring