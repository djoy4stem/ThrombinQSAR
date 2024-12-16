
## About 
This is project that illustrates end-to-end development of a Thrombin inhibiton predictive model. The data was acquired from the Molecule Activity Cliff Estimation (MoleculeACE) CHEMBL204_Ki benchmark dataset available [here](https://github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data).

**Make sure to set the correct MLflow Tracking URI:** This can be done in the main.py by adding *mlflow.set_tracking_uri(PATH__TO__FOLDER)*, where  *PATH__TO__FOLDER* can be the mlruns/ folder under the root directory of the project. You can check the .mlflow configuration directory or your environment variables: *echo $MLFLOW_TRACKING_URI*, and *echo $MLFLOW_ARTIFACT_URI*.

Covered topics include: 
1. Data preprocessing (featurization, cleaning, splitting), 
2. Model training (via hyperparameter tuning) and validation, 
3. ML experiment tracking, model regitry and serving using [MLflow](https://mlflow.org/docs/latest/introduction/index.html)

## Installation

1. Create a conda environment *conda create -n thrombin python=3.10*
2. Install required packages by running *pip install -r requirements.txt*


## Model Training and validation
Run the main app for data acquisition/preprocessing,  and model training/evaluation/registration
   1. *python main.py*

<!-- ## Model Deployment -->
