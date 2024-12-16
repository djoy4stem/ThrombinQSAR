import os, sys, time
import logging
import yaml
from random import randint
import pandas as pd 
from lib import steps

import mlflow
from mlflow import sklearn as mlf_sklearn
import mlflow.pyfunc as pfunc
from mlflow.tracking import MlflowClient
# from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error

from typing import List

# from rdkit.Chem import  MolFromSmiles, PandasTools

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():

    ## Load or generate/clean data
    da = steps.DataAcquisition()
    train, val, test = da.get_data()

    ## train and save the best model
    trainer = steps.Trainer()
    training_scoring_func = mean_absolute_error
    trainer.train(
                scoring_function=training_scoring_func,                                     
                num_trials=2,
                standardize=False,
                n_jobs=1,
                save_best_model_=True)

    logging.info(f'\nbest params: {trainer.best_results[0]["best_params"]}')

    ## Evaluate the best model
    qsar_model = steps.ThrombinQSARModel()
    y_test_pred = qsar_model.model.predict(test[test.columns.difference([qsar_model.target_col])])
    mae_score = mean_absolute_error(test[qsar_model.target_col], y_test_pred)
    mse_score = mean_squared_error(test[qsar_model.target_col], y_test_pred)
    


    ## Report metrics
    logging.info("\nTraining (validation):\n**********************")
    logging.info(f'MAE :  {round(trainer.best_results[0]["best_value"], 4)}',  )  
    

    logging.info("\nTest:\n*****")
    logging.info(f"MAE : {round(mae_score,4)}")
    logging.info(f"MSE : {round(mse_score,4)}")


def train_with_mlflow():
    mlflow.set_tracking_uri(f"file://{steps.BASE_DIR}/mlruns")
    # print(mlflow.get_tracking_uri())

    tm = time.localtime()
    tmf = f"{tm.tm_mon}/{tm.tm_mday}/{tm.tm_year} at {tm.tm_hour}h{tm.tm_min}m{tm.tm_sec}s"
    # my_experiment_id = f"thrombin_{tm.tm_mon}{tm.tm_mday}{tm.tm_year}_{tm.tm_hour}{tm.tm_min}{tm.tm_sec}_{randint(1,10e10)}"

    experiment = mlflow.set_experiment(experiment_name = f"Thrombin Inhibition Model Training Experiment: {tmf}")
    # experiment = mlflow.set_experiment(experiment_id = my_experiment_id)
                            
    print(f"\nexperiment: {experiment}\n\n")

    with mlflow.start_run() as run:
        ## Load or generate/clean data
        da = steps.DataAcquisition()
        train, val, test = da.get_data()

        logging.info("Data loading/preprocessing completed successfully")

        ## train and save the best model
        trainer = steps.Trainer()
        training_scoring_func = mean_absolute_error
        trainer.train(
                    scoring_function=training_scoring_func,                                     
                    num_trials=5,
                    standardize=False,
                    n_jobs=1,
                    save_best_model_=True)

        logging.info("Model training completed successfully")
        logging.info(f'\nbest params: {trainer.best_results[0]["best_params"]}')

        ## Evaluate the best model
        qsar_model = steps.ThrombinQSARModel()
        y_test_pred = qsar_model.model.predict(test[test.columns.difference([qsar_model.target_col])])
        mae_score = mean_absolute_error(test[qsar_model.target_col], y_test_pred)
        mse_score = mean_squared_error(test[qsar_model.target_col], y_test_pred)

        ### SAVE SHAP explanations for the test set
        mlflow.shap.log_explanation(qsar_model.model.predict, test[test.columns.difference([qsar_model.target_col])])

        mlflow.set_tag("model_developer", "djoy4stem")
        mlflow.set_tag("split_type", "scaffold_split")

        mlflow.log_params(trainer.best_results[0]["best_params"])
        mlflow.log_metric("best_val_mae", round(trainer.best_results[0]["best_value"],4))
        mlflow.log_metric("best_test_mae", round(mae_score, 4))
        mlflow.log_metric("best_test_mse", round(mse_score, 4))

        my_artifact_path_ = "model"
        # the file 'model.pkl' will be saved in the artifacts under thrombin_inhib_model/
        mlflow.sklearn.log_model(sk_model=qsar_model, 
                                    artifact_path = my_artifact_path_,    
                                    input_example = pd.DataFrame(['CCC(=O)OC', 'NCc1nccnc1CC(=O)N(C)C'], columns=['SMILES']),
                                    registered_model_name="thrombin_inhib_model") 

        ## register the model
        # The versions of this  model will be saved in the model_uri below.
        model_name = "thrombin_inhib_model"
        model_uri  = f"runs:/{run.info.run_id}/{my_artifact_path_}"
        mlflow.register_model(model_uri, model_name)


        logging.info("MLflow tracking completed successfully")

        ## Report metrics
        logging.info("\nTraining (validation):\n**********************")
        logging.info(f'MAE :  {round(trainer.best_results[0]["best_value"], 4)}',  )  
        

        logging.info("\nTest:\n*****")
        logging.info(f"MAE : {round(mae_score,4)}")
        logging.info(f"MSE : {round(mse_score,4)}")

from lib import utilities


def predict_with_mlflow(list_of_smiles: List[str]):
    # stage = 'Staging'
    model_name = "thrombin_inhib_model"
    model_uri  = f"models:/{model_name}/latest"
    model      = mlf_sklearn.load_model(model_uri=model_uri)

    # predictions = model.featurize_and_predict_from_smiles(list_of_smiles)
    predictions = model.predict(list_of_smiles)
    

    return predictions


# # main()
train_with_mlflow()
predictions  = predict_with_mlflow(list_of_smiles=['CC', 'Cc1ccc(C)c(CNC(=O)[C@@H]2CCCN2C(=O)C(N)C(c2ccccc2)c2ccccc2)c1'
                                    , 'CC(=O)NCC(=O)NC(CCCN=C(N)N)B(O)O', 'COc1ccccc1S(=O)(=O)Oc1cc(C)cc(OCCC/C=N/N=C(N)N)c1'
                                    , 'CCCN1CCC[C@H]1C(=O)NCc1ccc(C(=N)N)cc1'])

print(f"\nPredictions\n{predictions}")


model_name = "thrombin_inhib_model"
model_uri = f"models:/{model_name}/latest"
model = mlf_sklearn.load_model(model_uri=model_uri)
# print("model", model)
# print(model.model.feature_name_)

# predictions  = model.featurize_and_predict_from_smiles(smiles=['CCO', 'Ic1cnccc1CC(=O)O', 'CCCC(=O)N(C)C' ])
predictions  = model.predict(smiles=['CCO', 'Ic1cnccc1CC(=O)O', 'CCCC(=O)N(C)C' ])
print(f"\nPredictions\n{predictions}")


# client =  MlflowClient()
# for model in client.search_registered_models():
#     print(f"{model.name}")