
import os, sys
from pathlib import Path
import copy
import pandas as pd
import numpy as np
from typing import Union, List
import pickle
import yaml
import warnings 


from rdkit.Chem import PandasTools, AllChem, MolToSmiles, MolFromSmiles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import lightgbm
from lightgbm import LGBMRegressor

# import mlflow
# import mlflow.sklearn
# import mlflow.pyfunc

from lib import preprocessing, splitters, featurizers, utilities, training


BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
print("BASE_DIR", BASE_DIR)
CONFIG_PATH_ = os.path.join(BASE_DIR, "data/config.yml")      
print("CONFIG_PATH_", CONFIG_PATH_)





def process_molecule(molecule: AllChem.Mol):
    try:
        if not molecule is None:
            molecule = utilities.get_largest_fragment_from_mol(molecule)
            molecule = utilities.sanitize_molecule(molecule, add_explicit_h=True)
            return molecule
        else:
            return None
    except Exception as exp:
        print(f"Failed to process molecules: {exp}")


def featurize_molecules(
    molecules: Union[List[AllChem.Mol], np.ndarray, pd.DataFrame],
    count_unique_bits: bool = False, append: bool = True,
    mol_col: str = "RMol", mol_featurizer = featurizers.MoleculeFeaturizer(features=None),
    keep_features_only: bool = True
    
):
    
    if not molecules is None:
        features = None
        if isinstance(molecules, (List, np.ndarray)):
            features = pd.DataFrame(molecules, columns=[mol_col])
            x = pd.DataFrame(
                features[mol_col]
                .apply(
                    lambda mol: mol_featurizer.compute_properties_for_mols(
                        molecules=[mol], as_dataframe=False, count_unique_bits=count_unique_bits
                    )[0]
                )
                .values.tolist()
            )

            features = pd.concat([features, x], axis=1)

            features = preprocessing.add_custom_features(                       
                            features, bool_to_int=True, mol_column=mol_col
                        )

        elif isinstance(molecules, pd.DataFrame):
            current_mols = molecules.columns.tolist()
            x = pd.DataFrame(
                molecules[mol_col]
                .apply(
                    lambda mol: mol_featurizer.compute_properties_for_mols(
                        molecules=[mol], as_dataframe=False, count_unique_bits=count_unique_bits
                    )[0]
                )
                .values.tolist()
            )
            x.index = molecules.index
            # print(x.head(5))

            # print(molecules.shape)
            features = pd.concat([molecules, x], axis=1)
            # print(features.head(5))
            # print(f"features = {features.columns.tolist()}")
            # print("mol_col", mol_col)
            # print(features[mol_col].values[0])
            # print(f"\n\n{features.columns.tolist()}\n\n")
            # print(features.shape)
            # print("\nfeatures[mol_col]\n", features[mol_col])

            features = preprocessing.add_custom_features(                       
                            features, bool_to_int=True, mol_column=mol_col
                        )

        if keep_features_only:
            # print("\n\n", features.columns.tolist())
            features.drop(columns=current_mols, inplace=True)

        return features

    else:
        return None

class DataAcquisition():
    def __init__(self, config_path=CONFIG_PATH_):
        assert (not config_path is None) and (os.path.exists(config_path)), "Provide a valid path fir the config file."
        self.config_path = config_path
        self.config = utilities.load_yaml_to_dict(config_path)
        self.target_col = self.config['data']['target_col']
        self.splitter  = splitters.ScaffoldSplitter()
        self.fc_object = None
        self.fc_object_path = None
        self.root_path = self.config.get('root_path', None)
        
        # print(f"self.config = {self.config}")

        ## Get featurizerCleaner
        path_ = self.config['featurizer_cleaner']['path']
        # print("path=", path_)
        # print("os.listdir", os.listdir(f"{os.getcwd()}/models"))

        if not path_ in ['None', None]:              
            if not os.path.exists(path_):
                raise ValueError(f"The path provided for the featuizer_cleaner does not exist: {path_}")
            else:
                self.fc_object = utilities.get_from_pkl(path_)
                self.fc_object_path = path_
                # print(f"\n\n******************************I have loaded the already fitted feature_scaler: {self.fc_object.standardizer.mean_}\n\n")
        
        # print("self.root_path", self.root_path)
        print("\n\nself.fc_object_path", self.fc_object_path)
        if self.fc_object_path is None:
            standardizer   = None           
            if self.root_path is None:
                self.fc_object_path = os.path.join(f"{BASE_DIR}", 'models/featurizer_cleaner.pkl')
                # print("self.fc_object_path", self.fc_object_path is None)
            else:
                self.fc_object_path = os.path.join(self.root_path, 'models/featurizer_cleaner.pkl')

            if self.config['featurizer_cleaner'].get('standardizer', None) is None:
                standardizer   = None
            else:
                standardizer =  self.config['featurizer_cleaner']['standardizer'].get('type', None)
                if not standardizer  in ['None', None]:
                    standardizer = eval(standardizer) 
                    # print(f"standardizer({standardizer.__class__}) = {standardizer}")
                else:
                    standardizer = None      

            features = self.config['featurizer_cleaner'].get('features', None)
            if features in ['None', None]:
                features = None
            mol_featurizer = featurizers.MoleculeFeaturizer(features=features,
                                                                df_func_gps=eval(self.config['featurizer_cleaner'].get('df_func_gps', None))
                                                            )

            self.fc_object = FeaturizingAndCleaningObject(config_path=None, mol_featurizer = mol_featurizer
                                                                , standardizer = standardizer
                                                            )
            # print(f"\n\n\n\nself.fc_object.standardizer({self.fc_object.standardizer.__class__}) = {standardizer}")
            # print(f"Is self.fc_object.standardizer fitted? {hasattr(self.fc_object.standardizer, 'mean_')}\n\n")
            # print("self.fc_object_path", self.fc_object_path is None)
            utilities.save_to_pkl(self.fc_object, self.fc_object_path)
            
            self.config['featurizer_cleaner']['path'] = self.fc_object_path

            print("Saving updated config to file...")
            # print("self.config", self.config)
            utilities.save_dict_to_yaml(self.config, self.config_path)


                                                        

    def split_data(self, frame, mol_column):
        train_idx, val_idx, test_idx = self.splitter. train_val_test_split(
                                                                molecules=frame[mol_column],
                                                                val_ratio = self.config['data'].get('val_ratio', 0.8),
                                                                train_ratio = self.config['data'].get('train_ratio', 0.1),
                                                                test_ratio = self.config['data'].get('test_ratio', 0.1),
                                                                return_as_indices = True,
                                                                return_as_clusters = False,
                                                                include_chirality = False,
                                                                sort_by_size = True,
                                                                shuffle_idx = False,
                                                                random_state = 1,
                                                                use_novana = False
                                                            )

        return frame.iloc[train_idx], frame.iloc[val_idx], frame.iloc[test_idx]


    def get_data(self, keep_molecules=True, molecule_column = 'RMol'):
        
        train_fname     = self.config['data']['train_path']
        val_fname       = self.config['data']['val_path']
        test_fname      = self.config['data']['test_path']
        full_data_fname = self.config['data']['full_data_path']
        smiles_column   = self.config['data']['smiles_col']

        train, val, test = None, None, None

        if not (os.path.exists(train_fname) and os.path.exists(val_fname) and os.path.exists(test_fname)):
            assert not full_data_fname is None, "\nThe full data could not be located. Make sure to provide a valid path.\n"

            
            print("\nData sets are not available and will be generated...\n")
            full_df = pd.read_csv(full_data_fname).iloc[:100,:]
            PandasTools.AddMoleculeColumnToFrame(frame=full_df, smilesCol=smiles_column, molCol=molecule_column)

            # print(full_df.shape)
            full_df.dropna(subset=[molecule_column], axis=0, inplace=True)
            # print(full_df.shape)
            # print(full_df[molecule_column])

            train, val, test = self.split_data(frame=full_df, mol_column=molecule_column)

            # train = full_df[full_df['split']=='train']
            # train = full_df[full_df['split']=='train']

            # print(val[molecule_column].apply(MolToSmiles).tolist())
            # print(f"\nGet Data: self.fc_object.standardizer: {self.fc_object.standardizer}\n")
            train, val, test = self.fc_object.featurize_and_fix_datasets(train_df=train
                                                                , val_df=val, test_df=test
                                                                , target_col=self.target_col
                                                                , mol_column=molecule_column
                                                                , train_fname=train_fname
                                                                , val_fname=val_fname
                                                                , test_fname=test_fname
                                                                # , standardizer = self.fc_object.standardizer
                                                                # , columns_to_clean=[]
                                                            )
            # print(f"\nRMOL in train columns? {molecule_column in train.columns}\n")

        else:
            print("Data sets are already available and will be loaded...")
            train = pd.read_csv(train_fname)
            val   = pd.read_csv(val_fname)
            test  = pd.read_csv(test_fname)

            if keep_molecules and smiles_column in train.columns:
                PandasTools.AddMoleculeColumnToFrame(frame=train, smilesCol=smiles_column, molCol=molecule_column)
                PandasTools.AddMoleculeColumnToFrame(frame=val, smilesCol=smiles_column, molCol=molecule_column)
                PandasTools.AddMoleculeColumnToFrame(frame=test, smilesCol=smiles_column, molCol=molecule_column)

        return train, val, test


class FeaturizingAndCleaningObject():

    def __init__(self, config_path=CONFIG_PATH_, **kwargs):
       
        if config_path is None:
            self.mol_featurizer = kwargs.get('mol_featurizer', None)
            self.standardizer   = kwargs.get('standardizer', None)
            self.all_features   = kwargs.get('all_features', None)
            self.fc_object_path  = 'models/featurizer_cleaner.pkl'
            self.root_path = BASE_DIR
        else:
            assert (not config_path is None) and (os.path.exists(config_path)), "Provide a valid path fir the config file."

            config = utilities.load_yaml_to_dict(config_path)
            # print("config", config)
            assert hasattr(config, 'featurizer_cleaner'), "No featurizer_cleaner configuration was provided. Please provide a valid one."

            self.config_path    = config_path
            self.config         = config['featurizer_cleaner']
            self.fc_object_path = self.config.get("path", None)

            self.standardizer   = None
            self.all_features   = None
            self.root_path = config.get('root_path', BASE_DIR)
            
            self.set_up


    def setup(self):
        if self.fc_object_path is None:
            standardizer   = None
            self.fc_object_path = os.path.join(self.root_path, 'models/featurizer_cleaner.pkl')

            if self.config['featurizer_cleaner'].get('standardizer', None) is None:
                self.standardizer   = None
            else:
                standardizer =  self.config['featurizer_cleaner']['standardizer'].get('type', None)
                if not standardizer  in ['None', None]:
                    self.standardizer = eval(standardizer) 
                else:
                    self.standardizer = None      

            features = self.config['featurizer_cleaner'].get('features', None)
            if features in ['None', None]:
                features = None

            self.mol_featurizer = featurizers.MoleculeFeaturizer(features=features,
                                                                df_func_gps=eval(self.config['featurizer_cleaner'].get('df_func_gps', None))
                                                            )
            
            self.all_features=None

            if not self.fc_object_path is None:
                utilities.save_to_pkl(self.fc_object, self.fc_object_path)
            
            # self.config['featurizer_cleaner']['path'] = self.fc_object_path
            # print("Saving updated config to file...")
            # print("self.config", self.config)
            # utilities.save_dict_to_yaml(self.config, self.config_path)
        else:         
            if not os.path.exists(self.fc_object_path):
                raise ValueError(f"The path provided for the featuizer_cleaner does not exist: {self.fc_object_path}")
            else:
                self.fc_object = utilities.get_from_pkl(self.fc_object_path)            

    def clean_structures(self, frame, mol_column):
        frame['clean_rmol'] = frame[mol_column].apply(process_molecule)
        return frame
    
    def featurize_and_fix(self, frame, target_col, mol_column='RMol', save_to=None
                            , columns_to_clean=None, columns_to_scale=None
                            , fit_standardizer=False
                            , standardizer = None
                            ):

        if standardizer is None:
            standardizer = self.standardizer

        # print(f"===> {standardizer}")
        frame_w_props = featurize_molecules(molecules=frame, count_unique_bits=True
                                            , append=False, mol_col=mol_column
                                            , mol_featurizer = self.mol_featurizer
                                            , keep_features_only=True   
                                        )

        # print(f"\nframe columns ({len(frame_w_props.columns)}) = {frame_w_props.columns.tolist()}")
        # print("\nself.all_features = ", self.all_features)

        if not self.all_features is None:
            # print(f"\n\nself.all_features = {self.all_features}")
            # print([x for x in self.all_features if not x in frame.columns])
            
            frame_w_props = frame_w_props[self.all_features]

        # print(f"\n\n{frame_w_props.columns}\n\n")
        # print("\n\n", frame_w_props.iloc[:5, :5])

        fitted_standardizer = None
        if not fit_standardizer:
            frame_w_props, _ = preprocessing.clean_features(features_df=frame_w_props,
                                                            columns_to_clean=columns_to_clean,
                                                            columns_to_scale=columns_to_scale,
                                                            standardizer=standardizer,
                                                            strategy_num="mean",
                                                            strategy_cat="most_frequent",
                                                            fit_standardizer=fit_standardizer
                                                        )
        else:
            frame_w_props, fitted_standardizer = preprocessing.clean_features(features_df=frame_w_props,
                                                            columns_to_clean=columns_to_clean,
                                                            columns_to_scale=columns_to_scale,
                                                            standardizer=standardizer,
                                                            strategy_num="mean",
                                                            strategy_cat="most_frequent",
                                                            fit_standardizer=fit_standardizer
                                                        )
        if self.all_features is None:
            print("\n\=========> Assiging all_features\n")
            # self.all_features = frame_w_props.columns.difference([target_col]).tolist()
            self.all_features = fitted_standardizer.get_feature_names_out()
            print(f"self.all_features: {self.all_features}\n\n\n\n")
            print("\n\nJust fitted a standardizer.\n")



        # print("\n\n", frame_w_props.iloc[:5, :5])
        if not target_col is None:
            frame_w_props[target_col] = frame[target_col]

        # print(f"\n\n{frame_w_props.head(5)}\n\n")

        if not save_to is None:
            frame_w_props.to_csv(save_to)


        # print(f"\n1) problematic cols: {[c for c in frame_w_props.columns if not c in self.all_features]}")
        # print(f"2) problematic cols: {[c for c in self.all_features if not c in frame_w_props.columns]}")

        if not fit_standardizer:
            return frame_w_props
        else:
            return frame_w_props, fitted_standardizer

    def featurize_and_fix_datasets(self, train_df, val_df, test_df, target_col, mol_column='RMol'
                                    , train_fname=None, val_fname=None
                                    , test_fname=None
                                    , columns_to_clean=None, columns_to_scale=None
                                ):

        print("\nfeaturize and clean train dataset")
        # print(f"featurize_and_fix_datasets: {self.standardizer} - {self.standardizer.__class__}")
        # print(f"\n{self.standardizer}\ncopy.deepcopy(self.standardizer) : {copy.deepcopy(self.standardizer).__class__} - {copy.deepcopy(self.standardizer)}")
        scaler_copy = copy.deepcopy(self.standardizer)
        train, fitted_standardizer =  self.featurize_and_fix(frame=train_df, target_col=target_col, mol_column=mol_column, save_to=train_fname
                            , columns_to_clean=columns_to_clean, columns_to_scale=columns_to_scale
                            , standardizer = self.standardizer
                            , fit_standardizer = True
                            )

        print("\nfeaturize and clean val dataset")
        val,_   =  self.featurize_and_fix(frame=val_df, target_col=target_col, mol_column=mol_column, save_to=val_fname
                            , columns_to_clean=columns_to_clean, columns_to_scale=columns_to_scale
                            , standardizer = self.standardizer, fit_standardizer = True
                            )
        print("\nfeaturize and clean test dataset")
        test,_  =  self.featurize_and_fix(frame=test_df, target_col=target_col, mol_column=mol_column, save_to=test_fname
                            , columns_to_clean=columns_to_clean, columns_to_scale=columns_to_scale
                            , standardizer = self.standardizer, fit_standardizer = True
                            )                            

        print(f"fitted_standardizer = {fitted_standardizer} \nis it fitted? {hasattr(fitted_standardizer, 'mean_')}")
        ## saving the fc object with a standardizer fitted with the training data
        self.standardizer = fitted_standardizer
        utilities.save_to_pkl(self, self.fc_object_path)


        if not train_fname is None:
            train.to_csv(train_fname, index=False)

        if not val_fname is None:
            val.to_csv(val_fname, index=False)

        if not test_fname is None:
            test.to_csv(test_fname, index=False)

        return train, val, test         



class Trainer():
    def __init__(self, config_path=CONFIG_PATH_):
        
        config          = utilities.load_yaml_to_dict(config_path)
        # print("\n*** config", config.keys())
        # assert hasattr(config, 'training'), "No training configuration was provided. Please provide a valid one."
        # assert hasattr(config, 'direction'), "No direction was provided. Please provide a valid one."
        
        self.root_path = config.get('root_path', BASE_DIR)
        self.config_path = config_path
        self.config     = config

        self.best_model_path = config['training'].get('best_model_path', None) or os.path.join(self.root_path, 'models/model_thrombin_inhib_reg.pkl')
        self.config['training']['best_model_path']     = self.best_model_path
        # print("self.best_model_path", self.best_model_path)

        self.direction  = self.config['training']['direction']
        

        self.data       = config['data']
        self.target_col = config['data']['target_col']     

        self.best_results=[]
        
        # print("self.config['training']['models']", self.config['training']['models'])
        self.clean_model_params()
        # print("\n\nAfter cleaning: self.config['training']['models']", self.config['training']['models'])
        # print("\n*****\n",self.config['training']['models'][0])
        # print(f"eval: {eval(self.config['training']['models'][0])}")
    
    def clean_model_params(self):
        for i in range(len(self.config['training']['models'])):
            self.config['training']['models'][i]['model'] = eval(self.config['training']['models'][i]['model'])
            for x in self.config['training']['models'][i]['params']:
                # print(f"\t{x} : {self.config['training']['models'][i]['params'][x]}")
                try:
                    self.config['training']['models'][i]['params'][x] = eval(self.config['training']['models'][i]['params'][x])
                except:
                    pass


    def train(self,
                scoring_function,
                train_val_test_data:list=None,                                      
                # model_param_list: list = None,
                num_trials:int=50,
                standardize:bool=False,
                n_jobs:int=1,
                save_best_model_:bool=True):

        # if model_param_list is None:
        #     model_param_list = self.config['training']['models']
            

        # print("\n\nmodel_param_list", model_param_list)

        if train_val_test_data is None:
            train_val_test_data = [[
                        pd.read_csv(self.data['train_path'])
                        , pd.read_csv(self.data['val_path'])
                        , pd.read_csv(self.data['test_path'])

            ]]

            # print(train_val_test_data[0][0].shape)
            # print(train_val_test_data[0][1].shape)
            # print(train_val_test_data[0][2].shape)
            
        for j in range(len(self.config['training']['models'])):
            curr_model_params = self.config['training']['models'][j]
            print(f"curr_model_params = {curr_model_params}")
            best_model, best_params, best_value, direction = training.train_with_optuna(
                    model_params=curr_model_params, train_val_test_data=train_val_test_data, target_column=self.target_col,
                    scoring_function=scoring_function, direction=self.direction, num_trials=num_trials, standardize=standardize, n_jobs=n_jobs
                )
            self.best_results.append({"model_type": best_model.__class__.__name__, "best_model":best_model, 
                                        "best_params": best_params, "best_value": best_value, "scoring_function":scoring_function})
            
            self.config['training']['models'][j]['model'] = best_model.__class__.__name__

        if save_best_model_:
            self.save_best_model()

        return self.get_best_model()

    def get_best_model(self):
        assert len(self.best_results)>0, "There are no saved training results. Models must be trained first."
        best_ = None
        if self.direction == 'minimize':
            best_ = min(self.best_results, key=lambda d: d["best_value"])
        elif self.direction == 'maximize':
            best_ = min(self.best_results, key=lambda d: d["best_value"])

        return best_

    def save_best_model(self):
        best_ = self.get_best_model()
        
        # ## track metrics
        # for param in best_["best_params"]:
        #     print(f"saving best {param}", best_["best_params"][param])
        #     mlflow.log_param(f"{param}",  best_["best_params"][param])
        
        # mflow.log_metric(f"{best_['scoring_function']}_val", best_['best_value'])

        print("\nSaving best model...")
        try:
            utilities.save_to_pkl(best_["best_model"], self.best_model_path)
        except:
            utilities.save_to_pkl(best_["best_model"], self.best_model_path)

        if not hasattr(self.config, 'predictor'):
            self.config['predictor'] = {'model':{'path': self.best_model_path}}
        else:
            self.config['predictor']['model']['path'] = self.best_model_path
        
        print("\nSaving config...")
        utilities.save_dict_to_yaml(self.config, self.config_path)


class ThrombinQSARModel():
    def __init__(self, config_path=CONFIG_PATH_):
        self.config_path = config_path 
        config = utilities.load_yaml_to_dict(config_path)
        self.config     = config['predictor']
        self.test_data  = config['data'].get('test_path', None)
        self.target_col = config['data']['target_col']

        ## load featurizer_cleaner
        
        self.featurizer_cleaner = utilities.get_from_pkl(config['featurizer_cleaner']['path'])
        # print(f"featurizer_cleaner = ", self.featurizer_cleaner)
        # print("featurizer_cleaner", hasattr(self.featurizer_cleaner, "mean_"))

        ## load model
        model_path = self.config['model'].get('path', None)
        if model_path is None:
            warnings.warn("There is no model. One must me trained.")
            self.model = None
        else:
            self.model = utilities.get_from_pkl(self.config['model']['path'])

    def update_(self):
        config = utilities.load_yaml_to_dict(self.config_path)
        self.config = config['predictor']
        print("self.config['model']", self.config['model'])
        self.model = utilities.get_from_pkl(self.config['model']['path'])


    def train(self,               
                scoring_function,
                trainer:Trainer=None,
                train_val_test_data:list=None,                                      
                # model_param_list: list = None,
                num_trials:int=50,                
                standardize:bool=False,
                n_jobs:int=1,
                save_best_model_:bool=True):

        if trainer is None:
            trainer = Trainer(config_path=CONFIG_PATH_)

        trainer.train(
                        scoring_function=scoring_function,
                        train_val_test_data=train_val_test_data,                                      
                        num_trials=num_trials,                
                        standardize=standardize,
                        n_jobs=n_jobs,
                        save_best_model_=save_best_model_
                    )

        self.update_()



    def featurize_and_predict_from_mols(self, molecules_df: pd.DataFrame, mol_col="Mol"):

        ##Featurize and clean
        mol_features = self.featurizer_cleaner.featurize_and_fix(frame=molecules_df, target_col=None
                            , mol_column=mol_col, fit_standardizer=False, save_to=None
                            , columns_to_clean=None, columns_to_scale=self.featurizer_cleaner.all_features
                            )
        
        ## predict
        # print(molecules_df.head(3))
        # print(self.model.predict(X=mol_features))
        molecules_df[self.target_col] =  self.model.predict(X=mol_features)
        

        return molecules_df

    # def featurize_and_predict_from_smiles(self, 
    #     smiles: Union[List[AllChem.Mol], np.ndarray, pd.DataFrame],
    #     smiles_col: str = "SMILES",
    #     **kwargs,
    # ):
    #     # if True:
    #     try:
    #         if isinstance(smiles, (List, np.ndarray)):
    #             molecules = pd.DataFrame(
    #                 [MolFromSmiles(smi) for smi in smiles], columns=["Mol"]
    #             )

    #             preds = pd.DataFrame(smiles, columns=["SMILES"])
    #             preds = pd.concat([preds, self.featurize_and_predict_from_mols(
    #                 molecules_df=molecules, mol_col="Mol"
    #             ).drop(["Mol"], axis=1)], axis=1)

    #             return preds

    #         elif isinstance(smiles, pd.DataFrame):
    #             mol_col = kwargs.get("mol_col", None) or "Mol"
    #             PandasTools.AddMoleculeColumnToFrame(
    #                 smiles, smilesCol=smiles_col, molCol=mol_col
    #             )
    #             preds = pd.DataFrame(smiles, columns=["SMILES"])
    #             preds = preds = pd.concat([preds, self.featurize_and_predict_from_mols(
    #                 molecules_df=smiles, mol_col=mol_col
    #             ).drop([mol_col], axis=1)], axis=1)

    #             return preds

    #     except Exception as exp:
    #         print(f"Failed to predict {self.target_col} .\n\t{exp}")

    def predict(self, 
        smiles: Union[List[AllChem.Mol], np.ndarray, pd.DataFrame],
        smiles_col: str = "SMILES",
        **kwargs,
    ):
        # if True:
        try:
            if isinstance(smiles, (List, np.ndarray)):
                molecules = pd.DataFrame(
                    [MolFromSmiles(smi) for smi in smiles], columns=["Mol"]
                )

                preds = pd.DataFrame(smiles, columns=["SMILES"])
                preds = pd.concat([preds, self.featurize_and_predict_from_mols(
                    molecules_df=molecules, mol_col="Mol"
                ).drop(["Mol"], axis=1)], axis=1)

                return preds

            elif isinstance(smiles, pd.DataFrame):
                mol_col = kwargs.get("mol_col", None) or "Mol"
                PandasTools.AddMoleculeColumnToFrame(
                    smiles, smilesCol=smiles_col, molCol=mol_col
                )
                preds = pd.DataFrame(smiles, columns=["SMILES"])
                preds = preds = pd.concat([preds, self.featurize_and_predict_from_mols(
                    molecules_df=smiles, mol_col=mol_col
                ).drop([mol_col], axis=1)], axis=1)

                return preds

        except Exception as exp:
            print(f"Failed to predict {self.target_col} .\n\t{exp}")


    def evaluate_model(dframe: pd.DataFrame=None, scoring_function= mean_squared_error):
        
        if dframe is None:
            dframe = pd.read_csv(self.test_data)
        
        assert not dframe is None, "Could not find a file for evaluation. Provide a non-null .csv file."
        assert self.target_col in dframe.columns.tolist(), f"The target column '{self.target_col}' is missing in the test file."
        
        predictions = self.model.predict(dframe[dframe.columns.difference([self.target_col])])
        score       = mean_squared_error(predictions, dframe[self.target_col])

        print(f"{scoring_function.__name__} = {round(score, 3)}")





# if __name__=="__main__":
#     # from sklearn.metrics import balanced_accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
#     from sklearn.metrics import mean_squared_error, mean_absolute_error
#     from lightgbm import LGBMRegressor

#     da = DataAcquisition()
#     print(da.config)
#     train, val, test = da.get_data()

#     trainer = Trainer()
#     trainer.train(
#                 scoring_function=mean_absolute_error,                                     
#                 num_trials=200,
#                 standardize=False,
#                 n_jobs=1)

#     print("\n\nbest value",  trainer.best_results[0]["best_value"])  
#     print("best params", trainer.best_results[0]["best_params"])  

#     predictor = Predictor()
#     y_test_pred = predictor.model.predict(test[test.columns.difference([predictor.target])])
#     mae_score = mean_absolute_error(test[predictor.target], y_test_pred)
#     mse_score = mean_squared_error(test[predictor.target], y_test_pred)
    
#     print("\n\nmae_score",  mae_score)
#     print("mse_score",  mse_score) 