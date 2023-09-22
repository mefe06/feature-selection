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
logging.basicConfig(filename='classification_results.log', filemode="w", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
def log_dictionaries(key_name, keys, dict_names, *dicts,):
    # Log the header
    header = str(key_name).ljust(15) + " | ".join([str(dict_names[i]) for i in range(len(dict_names))])
    logging.info(header)
    logging.info('-' * len(header))
    
    # Log each key-value pair
    for key in keys:
        line = str(key).ljust(15) + " | ".join([str(d[key]) for d in dicts])
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
dict_names_with_rfe = ["All features", "GBM Selected", "FLBMO Selected", "RFE Selected"]
dict_names_no_rfe= ["All features", "GBM Selected", "FLBMO Selected"]
all, gb, mo, rfe = {}, {}, {}, {}
for f_number in [5, 8]:
    all_scores = []
    gb_scores = []
    mo_scores = []
    rfe_scores = []

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
    _,all_score, gb_score, mo_score, rfe_score = network.feature_extraction(f_number, seed=42 , method="number of features", run_CV=True, include_RFE=True)
    all_scores.append(all_score)
    gb_scores.append(gb_score)    
    mo_scores.append(mo_score)
    rfe_scores.append(rfe_score)

    all[f_number] = np.mean(all_scores)
    gb[f_number] = np.mean(gb_scores)
    mo[f_number] = np.mean(mo_scores)
    rfe[f_number] = np.mean(rfe_scores)

logging.info("For credit data lgbm")
log_dictionaries("ROCAUC", all.keys(),dict_names_with_rfe, all, gb, mo,rfe)

for f_number in [5, 8]:
    all_scores = []
    gb_scores = []
    mo_scores = []
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
    _,all_score, gb_score, mo_score = network.feature_extraction(f_number, seed=42 , method="number of features", run_CV=True)
    all_scores.append(all_score)
    gb_scores.append(gb_score)    
    mo_scores.append(mo_score)
    all[f_number] = np.mean(all_scores)
    gb[f_number] = np.mean(gb_scores)
    mo[f_number] = np.mean(mo_scores)
logging.info("For credit data mlp")
log_dictionaries("ROCAUC", all.keys(),dict_names_no_rfe, all, gb, mo)
