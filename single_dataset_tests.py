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
logging.basicConfig(filename='single_datasets.log', filemode ='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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



all_scores = []
mo_scores = []
data = pd.read_csv("/home/efe/Downloads/energydata_target_y.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#print(data.shape)
nb = len(data)
val_split = 0.3
test_split = 0.1
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
all_scores.append(all_score)
mo_scores.append(mo_score)
logging.info("For Energy Data, LGBM base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))

all_scores = []
mo_scores = []
data = pd.read_csv("/home/efe/Downloads/energydata_target_y.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#print(data.shape)
nb = len(data)
val_split = 0.3
test_split = 0.1
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
all_scores.append(all_score)
mo_scores.append(mo_score)
logging.info("For Energy Data, MLP base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))

all_scores = []
mo_scores = []
data = pd.read_csv("/home/efe/Downloads/AirQualityUCI_extracted(1).csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#print(data.shape)
nb = len(data)
val_split = 0.3
test_split = 0.1
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
all_scores.append(all_score)
mo_scores.append(mo_score)
logging.info("For Air Quality Data, MLP base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))


all_scores = []
mo_scores = []
data = pd.read_csv("/home/efe/Downloads/AirQualityUCI_extracted(1).csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#print(data.shape)
nb = len(data)
val_split = 0.3
test_split = 0.1
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
all_scores.append(all_score)
mo_scores.append(mo_score)
logging.info("For Air Quality Data, LGBM base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))

all_scores = []
mo_scores = []
data = pd.read_csv("/home/efe/Downloads/PRSA_data_extracted(1).csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#print(data.shape)
nb = len(data)
val_split = 0.3
test_split = 0.1
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
all_scores.append(all_score)
mo_scores.append(mo_score)
logging.info("For PRSA Data, MLP base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))

all_scores = []
mo_scores = []
data = pd.read_csv("//home/efe/Downloads/PRSA_data_extracted(1).csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#print(data.shape)
nb = len(data)
val_split = 0.3
test_split = 0.1
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
_,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
all_scores.append(all_score)
mo_scores.append(mo_score)
logging.info("For PRSA Data, MLP base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))