import os
from model import LGBM_w_Feature_Selector 
import logging
import warnings
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import re
import numpy as np
import logging
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(filename='class_results.log', filemode="w", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_dictionaries(keys, *dicts):
    # Log the header
    header = "Key".ljust(15) + " | ".join([f"Dict {i+1}" for i in range(len(dicts))])
    logging.info(header)
    logging.info('-' * len(header))
    
    # Log each key-value pair
    for key in keys:
        line = key.ljust(15) + " | ".join([str(d[key]) for d in dicts])
        logging.info(line)


lgbm_param_grid = {
    "hidden_layer_sizes": [(20),(40),(80),(40,10),(80,20),(80,40),(80,20,10),(80,40,20),(80,40,10)], 
    "activation":['relu','logistic'], 
    "solver":['adam',], 
    "alpha":[0.0001, 0.001, 0.01], 
    "learning_rate_init":[0.001, 0.01], 
    "warm_start":[False], 
    "momentum":[0.9], 
} 

mlp_param_grid ={
    "hidden_layer_sizes": [(20),(40),(80),(40,10),(80,20),(80,40),(80,20,10),(80,40,20),(80,40,10)], 
    "activation":['relu','logistic'], 
    "solver":['adam',], 
    "alpha":[0.0001, 0.001, 0.01], 
    "learning_rate_init":[0.001, 0.01], 
    "warm_start":[False], 
    "momentum":[0.9], 
} 

import random
random.seed(6)

data = pd.read_csv("/home/efe/Documents/Statlog_Australian_Credit_Approval/australian.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
nb = len(data)
test_split=0.1
val_split = 0.3
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["Y"]].values, X_val[["Y"]].values, X_test[["Y"]].values
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['Y']).values, normalized_X_val.drop(columns=['Y']).values, normalized_X_test.drop(columns=['Y']).values
network = LGBM_w_Feature_Selector(model="lgbm",problem_type="Classifier",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.1, 
                                        X_val=normalized_X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
logging.info("For Australian credit, LGBM base, all feature MSE: {}, GBMO MSE: {}".format(all_score, mo_score))


data = pd.read_csv("/home/efe/Documents/Statlog_Australian_Credit_Approval/australian.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
nb = len(data)
test_split=0.1
val_split = 0.3
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["Y"]].values, X_val[["Y"]].values, X_test[["Y"]].values
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['Y']).values, normalized_X_val.drop(columns=['Y']).values, normalized_X_test.drop(columns=['Y']).values
network = LGBM_w_Feature_Selector(model="mlp",problem_type="Classifier",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.1, 
                                        X_val=normalized_X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=2,run_CV=True)
logging.info("For Australian credit, MLP base, all feature MSE: {}, GBMO MSE: {}".format(all_score, mo_score))
