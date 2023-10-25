import os
from model import LGBM_w_Feature_Selector 
import logging
import warnings
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import re
import numpy as np
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(filename='denememem.log', filemode ='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

lgbm_param_grid={
'num_leaves': [10, 20],
'learning_rate': [0.01, 0.1, 0.5],
'n_estimators': [10, 20],
'subsample': [0.6, 0.8, 1.0],
'colsample_bytree': [0.6, 0.8, 1.0],
# 'reg_alpha': [0.0, 0.1, 0.5],
# 'reg_lambda': [0.0, 0.1, 0.5],
'min_child_samples': [5, 10],
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
lgbm_scores=[]
mi_scores=[]
rfe_scores=[]
data = pd.read_csv("/Users/mefe/Downloads/AirQualityUCI_extracted.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#print(data.shape)
nb = len(data)
val_split = 0.3
test_split = 0.1
data = data.sample(frac=1).reset_index(drop=True)
X_train, X_val, X_test = data.iloc[:int((1-(val_split+test_split))*nb)], data.iloc[int((1-(val_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val, y_test = X_train[["RH"]], X_val[["RH"]], X_test[["RH"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
normalized_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val = pd.DataFrame(scaler.transform(y_val), columns=y_val.columns).values
normalized_y_test = pd.DataFrame(scaler.transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val, normalized_X_test = normalized_X_train.drop(columns=['RH']).values, normalized_X_val.drop(columns=['RH']).values, normalized_X_test.drop(columns=['RH']).values
network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val=normalized_X_val, y_val=normalized_y_val,y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
## 13 fena değil
# 53te hesp çok iyi
_,all_score, mo_score, gbm_score, rfe_score, mi_score = network.feature_extraction(5, seed=17 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=True)
all_scores.append(all_score)
mo_scores.append(mo_score)
lgbm_scores.append(gbm_score)
mi_scores.append(mi_score)
rfe_scores.append(rfe_score)
logging.info("For Energy Data, MLP base, all feature MSE: {}, GBMO MSE: {}, GBM MSE:{}, MI MSE:{}, RFE MSE: {}".format(np.mean(all_scores), np.mean(mo_scores), np.mean(lgbm_scores), np.mean(mi_scores),np.mean(rfe_scores)))
