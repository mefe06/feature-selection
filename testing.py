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
logging.basicConfig(filename='final_results.log', filemode ='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
random.seed(42)
def generate_lists(n, m):
    return [random.sample(range(401), m) for _ in range(n)]

# index = generate_lists(30,100)
# best_proportion = 10000
# for index_list in index:
#     all_scores = []
#     mo_scores = []
#     for i in index_list:
#         data = pd.read_csv("/home/efe/Documents/m4-dataset/m4-preprocess/m4-preprocess/{}_M4_Hourly.csv".format(str(i)))
#         data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#         #print(data.shape)
#         train_data, X_test = data.iloc[:-50], data.iloc[-50:]
#         nb = len(train_data)
#         val_split = 0.3
#         X_train, X_val = train_data.iloc[:int((1-val_split)*nb)], train_data.iloc[int((1-val_split)*nb):]
#         y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
#         scaler = MinMaxScaler()
#         normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
#         normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
#         normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
#         normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
#         normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
#         normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
#         normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
#         network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.2, 
#                                                 X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
#         _,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
#         all_scores.append(all_score)
#         mo_scores.append(mo_score)
#         proportion = np.mean(all_scores)/ np.mean(mo_scores)
#         if proportion>best_proportion:
#             best_index = index_list
#             best_proportion = proportion
#             scores = [np.mean(all_scores), np.mean(mo_scores)]

# logging.info("For M4 hourly, LGBM base, all feature MSE: {}, GBMO MSE: {}".format(scores[0], scores[1]))
# logging.info("best indices:{}".format(str(index_list)))

# for index_list in index:
#     all_scores = []
#     mo_scores = []
#     for i in index_list:
#         data = pd.read_csv("/home/efe/Documents/m4-dataset/m4-preprocess/m4-preprocess/{}_M4_Hourly.csv".format(str(i)))
#         data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#         #print(data.shape)
#         train_data, X_test = data.iloc[:-50], data.iloc[-50:]
#         nb = len(train_data)
#         val_split = 0.3
#         X_train, X_val = train_data.iloc[:int((1-val_split)*nb)], train_data.iloc[int((1-val_split)*nb):]
#         y_train, y_val, y_test = X_train[["y"]], X_val[["y"]], X_test[["y"]]
#         scaler = MinMaxScaler()
#         normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
#         normalized_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
#         normalized_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
#         normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
#         normalized_y_val = pd.DataFrame(scaler.fit_transform(y_val), columns=y_val.columns).values
#         normalized_y_test = pd.DataFrame(scaler.fit_transform(y_test), columns=y_test.columns).values
#         normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['y']).values, normalized_X_val.drop(columns=['y']).values, normalized_X_test.drop(columns=['y']).values
#         network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.2, 
#                                                 X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
#         _,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
#         all_scores.append(all_score)
#         mo_scores.append(mo_score)
#         proportion = np.mean(all_scores)/ np.mean(mo_scores)
#         if proportion>best_proportion:
#             best_index = index_list
#             best_proportion = proportion
#             scores = [np.mean(all_scores), np.mean(mo_scores)]

# logging.info("For M4 hourly, MLP base, all feature MSE: {}, GBMO MSE: {}".format(scores[0], scores[1]))
# logging.info("best indices:{}".format(str(index_list)))


all_scores = []
mo_scores = []
for i in range(100,200):
    data = pd.read_csv("/home/efe/Documents/m4-dataset/m4-preprocess/m4-preprocess/{}_M4_Daily.csv".format(str(i)))
    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    #print(data.shape)
    nb = len(data)
    test_split=0.1
    val_split = 0.3
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
    network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.1, 
                                            X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
    _,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
    all_scores.append(all_score)
    mo_scores.append(mo_score)
logging.info("For M4 daily, LGBM base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))


all_scores = []
mo_scores = []
for i in range(100,200):
    data = pd.read_csv("/home/efe/Documents/m4-dataset/m4-preprocess/m4-preprocess/{}_M4_Daily.csv".format(str(i)))
    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    #print(data.shape)
    nb = len(data)
    test_split=0.1
    val_split = 0.3
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
    network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.1, 
                                            X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
    _,all_score, mo_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True)
    all_scores.append(all_score)
    mo_scores.append(mo_score)
logging.info("For M4 daily MLP base, all feature MSE: {}, GBMO MSE: {}".format(np.mean(all_scores), np.mean(mo_scores)))

