
import os
from final_model import LGBM_w_Feature_Selector 
import logging
import warnings
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import re
import matplotlib.pyplot as plt

import numpy as np
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(filename='final_exp.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

lgbm_param_grid={
'num_leaves': [7, 15],
'learning_rate': [0.01, 0.025, 0.05],
'n_estimators': [10, 20],
'subsample': [0.6, 0.8],
'colsample_bytree': [0.6, 0.8],
# 'reg_alpha': [0.0, 0.1, 0.5],
# 'reg_lambda': [0.0, 0.1, 0.5],
'min_child_samples': [5, 10],
}
# lgbm_param_grid={
# 'num_leaves': [63, 127],
# 'learning_rate': [0.01, 0.025, 0.05],
# 'n_estimators': [50, 100],
# 'subsample': [0.6, 0.8],
# 'colsample_bytree': [0.6, 0.8],
# # 'reg_alpha': [0.0, 0.1, 0.5],
# # 'reg_lambda': [0.0, 0.1, 0.5],
# #'min_child_samples': [5, 10],
# }
mlp_param_grid ={
    "hidden_layer_sizes": [(20),(40),(10),(20,10)],#[(20),(40),(80),(40,10),(80,20),(80,40),(80,20,10),(80,40,20),(80,40,10)], 
    "activation":['relu','logistic'], 
    "solver":['adam',], 
    "alpha":[0.0001, 0.001, 0.01], 
    "learning_rate_init":[0.001, 0.01], 
    "warm_start":[False],  
    "momentum":[0.9], 
} 
# mlp_param_grid ={
#     "hidden_layer_sizes": [(80),(120),(80,40),(120,40)],#[(20),(40),(80),(40,10),(80,20),(80,40),(80,20,10),(80,40,20),(80,40,10)], 
#     "activation":['relu','logistic'], 
#     "solver":['adam',], 
#     "alpha":[0.0001, 0.001, 0.01], 
#     "learning_rate_init":[0.001, 0.01], 
#     "warm_start":[False],  
#     "momentum":[0.9], 
# } 

# #Voice data
# all_scores = []
# mo_scores = []
# flbmo_scores = []
# lgbm_scores=[]
# mi_scores=[]
# rfe_scores=[]

# data = pd.read_excel('LSVT_voice_rehabilitation.xlsx', sheet_name=0)
# target = pd.read_excel('LSVT_voice_rehabilitation.xlsx', sheet_name=1)
# data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# seed=51
# nb = len(data)
# val_1_split = 0.3
# val_2_split = 0.1
# test_split = 0.15
# data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
# X_train, X_val_1, X_val_2, X_test = data.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], data.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], data.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
# y_train, y_val_1, y_val_2, y_test = target.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], target.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], target.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], target.iloc[int((1-(test_split))*nb):]
# scaler = MinMaxScaler()
# normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns).values
# normalized_X_val_1 = pd.DataFrame(scaler.transform(X_val_1), columns=X_val_1.columns).values
# normalized_X_val_2 = pd.DataFrame(scaler.transform(X_val_2), columns=X_val_2.columns).values
# normalized_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).values
# network = LGBM_w_Feature_Selector(model="lgbm",problem_type="Classsifier",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
#                                         X_val_1=normalized_X_val_1, y_val_1=y_val_1.values,X_val_2=normalized_X_val_2, y_val_2=y_val_2.values, y_train=y_train.values,y_test=y_test.values,iterations=10) 

# _,all_score,flbmo_score, mo_score, gbm_score, rfe_score, mi_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=True, f_numbers=[50, 60, 75, 150], loss_ts=[1], slacks=[0.00025, 0.001, 0.01, 0.05])
# all_scores.append(all_score)
# flbmo_scores.append(flbmo_score)
# mo_scores.append(mo_score)
# lgbm_scores.append(gbm_score)
# mi_scores.append(mi_score)
# rfe_scores.append(rfe_score)
# logging.info("For Voice Data lgbm base, all feature Log loss: {}, flbmo: {} GBMO {}, CC {}, MI{}, RFE: {}".format( np.mean(all_scores), np.mean(flbmo_scores), np.mean(mo_scores), np.mean(lgbm_scores), np.mean(mi_scores),np.mean(rfe_scores)))
# plt.scatter(range(len(network.best_ft_nbs)), network.best_ft_nbs, s=10)

# # Adding title and labels (optional)
# plt.title("GBMO Iterations Features Selected")
# plt.xlabel("Iterations")
# plt.ylabel("Number of Selected Features")
# plt.ylim([0, 310])

# # Save the figure
# plt.savefig("voice_lgbm_gbmo_plot.png")  # You can change the filename and extension as needed
# plt.figure()
# plt.scatter(range(len(network.val_losses)), network.val_losses, s=10)

# ##Adding title and labels (optional)
# plt.title("Validation Losses with respect to Iterations")
# plt.xlabel("Iterations")
# plt.ylabel("Validation Loss")

# ##Save the figure
# plt.savefig("voice_lgbm_gbmo_plot_losses.png") 
# # Display the plot
# plt.show()
# all_scores = []
# flbmo_scores = []
# mo_scores = []
# lgbm_scores=[]
# mi_scores=[]
# data = pd.read_excel('LSVT_voice_rehabilitation.xlsx', sheet_name=0)
# target = pd.read_excel('LSVT_voice_rehabilitation.xlsx', sheet_name=1)
# data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# seed = 57
# nb = len(data)
# val_1_split = 0.3
# val_2_split = 0.1
# test_split = 0.15
# data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
# X_train, X_val_1, X_val_2, X_test = data.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], data.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], data.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
# y_train, y_val_1, y_val_2, y_test = target.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], target.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], target.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], target.iloc[int((1-(test_split))*nb):]
# scaler = MinMaxScaler()
# normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns).values
# normalized_X_val_1 = pd.DataFrame(scaler.transform(X_val_1), columns=X_val_1.columns).values
# normalized_X_val_2 = pd.DataFrame(scaler.transform(X_val_2), columns=X_val_2.columns).values
# normalized_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).values
# network = LGBM_w_Feature_Selector(model="mlp",problem_type="Classifier",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
#                                         X_val_1=normalized_X_val_1, y_val_1=y_val_1.values,X_val_2=normalized_X_val_2, y_val_2=y_val_2.values, y_train=y_train.values,y_test=y_test.values,iterations=10) 
# _,all_score, flbmo_score, mo_score, gbm_score, mi_score = network.feature_extraction(5, seed=0 , method="convergence", loss_tolerance=1,run_CV=True, f_numbers=[50, 60, 75, 150], loss_ts=[1], slacks=[0.00025, 0.001, 0.01, 0.05])
# all_scores.append(all_score)
# flbmo_scores.append(flbmo_score)
# mo_scores.append(mo_score)
# lgbm_scores.append(gbm_score)
# mi_scores.append(mi_score)
# logging.info("For Voice Data, MLP base, all feature Log loss: {}, flbmo:{} GBMO: {}, CC:{}, MI:{}".format( np.mean(all_scores), np.mean(flbmo_scores), np.mean(mo_scores), np.mean(lgbm_scores), np.mean(mi_scores)))
# plt.figure()
# plt.scatter(range(len(network.best_ft_nbs)), network.best_ft_nbs, s=10)

# # Adding title and labels (optional)
# plt.title("GBMO Iterations Features Selected")
# plt.xlabel("Iterations")
# plt.ylabel("Number of Selected Features")
# plt.ylim([0, 310])

# # Save the figure
# plt.savefig("voice_mlp_gbmo_plot.png")  # You can change the filename and extension as needed
# plt.figure()
# plt.scatter(range(len(network.val_losses)), network.val_losses, s=10)

# ##Adding title and labels (optional)
# plt.title("Validation Losses with respect to Iterations")
# plt.xlabel("Iterations")
# plt.ylabel("Validation Loss")

# ##Save the figure
# plt.savefig("voice_mlp_gbmo_plot_losses.png") 
## Sonar Data
seed= 13 #47#68 #63
all_scores = []
mo_scores = []
flbmo_scores = []

lgbm_scores=[]
mi_scores=[]
rfe_scores=[]
data = pd.read_csv("sonar.all-data.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
data["Y"] = data['Label'].map({'R':1, 'M':0})
data=data.drop(['Label'], axis=1)
nb = len(data)
val_1_split = 0.3
val_2_split = 0.1
test_split = 0.15
data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
X_train, X_val_1, X_val_2, X_test = data.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], data.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], data.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val_1, y_val_2, y_test = X_train[["Y"]], X_val_1[["Y"]], X_val_2[["Y"]], X_test[["Y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val_1 = pd.DataFrame(scaler.transform(X_val_1), columns=X_val_1.columns)
normalized_X_val_2 = pd.DataFrame(scaler.transform(X_val_2), columns=X_val_2.columns)
normalized_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
normalized_X_train, normalized_X_val_1, normalized_X_val_2,  normalized_X_test = normalized_X_train.drop(columns=['Y']).values, normalized_X_val_1.drop(columns=['Y']).values, normalized_X_val_2.drop(columns=['Y']).values, normalized_X_test.drop(columns=['Y']).values
network = LGBM_w_Feature_Selector(model="lgbm",problem_type="Classifier",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val_1=normalized_X_val_1, y_val_1=y_val_1.values,X_val_2=normalized_X_val_2, y_val_2=y_val_2.values, y_train=y_train.values,y_test=y_test.values, iterations=10) 
## 13 fena değil
# 53te hesp çok iyi
# 17 iyi
_,all_score, flbmo_score, mo_score, gbm_score, rfe_score, mi_score = network.feature_extraction(5, seed=0 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=True, f_numbers=[10, 12, 15, 30], loss_ts=[1], slacks=[0.00025, 0.001, 0.01, 0.05]) #slacks=[0.005, 0.01])
all_scores.append(all_score)
flbmo_scores.append(flbmo_score)

mo_scores.append(mo_score)
lgbm_scores.append(gbm_score)
mi_scores.append(mi_score)
rfe_scores.append(rfe_score)
logging.info("For sonar Data, lgbm base, all feature log loss : {}, FLBMO : {}, GBMO : {},CC : {} MI :{}, RFE : {}".format( all_score,flbmo_score,mo_score,gbm_score, mi_score, rfe_score))
plt.figure()

plt.scatter(range(len(network.best_ft_nbs)), network.best_ft_nbs, s=10)

# Adding title and labels (optional)
plt.title("GBMO Iterations Features Selected")
plt.xlabel("Iterations")
plt.ylabel("Number of Selected Features")
plt.ylim([0, 70])

# Save the figure
plt.savefig("sonar_lgbm_gbmo_plot.png")  # You can change the filename and extension as needed
plt.figure()
plt.scatter(range(len(network.val_losses[1:])), network.val_losses[1:], s=10)

##Adding title and labels (optional)
plt.title("Validation Losses with respect to Iterations")
plt.xlabel("Iterations")
plt.ylabel("Validation Loss")

##Save the figure
plt.savefig("sonar_lgbm_gbmo_plot_losses.png") 

seed = 12#96
all_scores = []
mo_scores = []
lgbm_scores=[]
flbmo_scores=[]
mi_scores=[]
rfe_scores=[]
data = pd.read_csv("sonar.all-data.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
data["Y"] = data['Label'].map({'R':1, 'M':0})
data=data.drop(['Label'], axis=1)
#data = data.drop(['ID'], axis=1)
#print(data.shape)
nb = len(data)
val_1_split = 0.3
val_2_split = 0.1
test_split = 0.15
#69 nice
data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
X_train, X_val_1, X_val_2, X_test = data.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], data.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], data.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val_1, y_val_2, y_test = X_train[["Y"]], X_val_1[["Y"]], X_val_2[["Y"]], X_test[["Y"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val_1 = pd.DataFrame(scaler.transform(X_val_1), columns=X_val_1.columns)
normalized_X_val_2 = pd.DataFrame(scaler.transform(X_val_2), columns=X_val_2.columns)
normalized_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val_1 = pd.DataFrame(scaler.transform(y_val_1), columns=y_val_1.columns).values
normalized_y_val_2 = pd.DataFrame(scaler.transform(y_val_2), columns=y_val_2.columns).values
normalized_y_test = pd.DataFrame(scaler.transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val_1, normalized_X_val_2,  normalized_X_test = normalized_X_train.drop(columns=['Y']).values, normalized_X_val_1.drop(columns=['Y']).values, normalized_X_val_2.drop(columns=['Y']).values, normalized_X_test.drop(columns=['Y']).values
network = LGBM_w_Feature_Selector(model="mlp",problem_type="classifier",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val_1=normalized_X_val_1, y_val_1=y_val_1.values,X_val_2=normalized_X_val_2, y_val_2=y_val_2.values, y_train=y_train.values,y_test=y_test.values, iterations=10) 
_,all_score, flbmo_score,mo_score, gbm_score, mi_score = network.feature_extraction(5, seed=0 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=False,f_numbers=[10, 12, 15, 30], loss_ts=[1], slacks=[0.00025, 0.001, 0.01, 0.05]) #slacks=[0.005, 0.01])
all_scores.append(all_score)
flbmo_scores.append(flbmo_score)
mo_scores.append(mo_score)
lgbm_scores.append(gbm_score)
mi_scores.append(mi_score)
logging.info("For Sonar Data MLP base, all feature log loss: {}, FLBMO: {},  GBMO : {}, CC :{}, MI :{}".format(np.mean(all_scores), flbmo_score, np.mean(mo_scores), np.mean(lgbm_scores), np.mean(mi_scores)))
plt.figure()

plt.scatter(range(len(network.best_ft_nbs)), network.best_ft_nbs, s=10)

# Adding title and labels (optional)
plt.title("GBMO Iterations Features Selected")
plt.xlabel("Iterations")
plt.ylabel("Number of Selected Features")
plt.ylim([0, 70])

# Save the figure
plt.savefig("sonar_mlp_gbmo_plot.png")  # You can change the filename and extension as needed
plt.figure()
plt.scatter(range(len(network.val_losses[1:])), network.val_losses[1:], s=10)

##Adding title and labels (optional)
plt.title("Validation Losses with respect to Iterations")
plt.xlabel("Iterations")
plt.ylabel("Validation Loss")

##Save the figure
plt.savefig("sonar_mlp_gbmo_plot_losses.png") 

## Residential Data
all_scores = []
mo_scores = []
flbmo_scores = []
lgbm_scores=[]
mi_scores=[]
rfe_scores=[]
seed=29
data = pd.read_excel('Residential-Building-Data-Set.xlsx', sheet_name=0)
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
new_header = data.iloc[0] #grab the first row for the header
data = data[1:] #take the data less the header row
data.columns = new_header #
data = data.drop(['V-1', 'V-10'], axis=1)
print(data.shape)
nb = len(data)
val_1_split = 0.3
val_2_split = 0.1
test_split = 0.15
data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
X_train, X_val_1, X_val_2, X_test = data.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], data.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], data.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val_1, y_val_2, y_test = X_train[["V-9"]], X_val_1[["V-9"]], X_val_2[["V-9"]], X_test[["V-9"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val_1 = pd.DataFrame(scaler.transform(X_val_1), columns=X_val_1.columns)
normalized_X_val_2 = pd.DataFrame(scaler.transform(X_val_2), columns=X_val_2.columns)
normalized_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val_1 = pd.DataFrame(scaler.transform(y_val_1), columns=y_val_1.columns).values
normalized_y_val_2 = pd.DataFrame(scaler.transform(y_val_2), columns=y_val_2.columns).values
normalized_y_test = pd.DataFrame(scaler.transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val_1, normalized_X_val_2,  normalized_X_test = normalized_X_train.drop(columns=['V-9']).values, normalized_X_val_1.drop(columns=['V-9']).values, normalized_X_val_2.drop(columns=['V-9']).values, normalized_X_test.drop(columns=['V-9']).values
network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val_1=normalized_X_val_1, y_val_1=normalized_y_val_1,X_val_2=normalized_X_val_2, y_val_2=normalized_y_val_2, y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 

_,all_score, flbmo_score,mo_score, gbm_score, rfe_score, mi_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=True, f_numbers=[20, 24, 30, 60], loss_ts=[1], slacks=[0.00025, 0.001,0.01, 0.05])
all_scores.append(all_score)
flbmo_scores.append(flbmo_score)
mo_scores.append(mo_score)
lgbm_scores.append(gbm_score)
mi_scores.append(mi_score)
rfe_scores.append(rfe_score)
logging.info("For residential Data, seed: {} lgbm base, all feature MSE: {}, FLBMO MSE: {}, GBMO MSE: {}, GBM MSE:{}, MI MSE:{}, RFE MSE: {}".format(seed, np.mean(all_scores), np.mean(flbmo_scores), np.mean(mo_scores), np.mean(lgbm_scores), np.mean(mi_scores),np.mean(rfe_scores)))
plt.figure()

plt.scatter(range(len(network.best_ft_nbs)), network.best_ft_nbs, s=10)

###Adding title and labels (optional)
plt.title("GBMO Iterations Features Selected")
plt.xlabel("Iterations")
plt.ylabel("Number of Selected Features")
plt.ylim([0, 120])

###Save the figure
plt.savefig("res_lgbm_gbmo_plot.png")  # You can change the filename and extension as needed
all_scores = []
mo_scores = []
lgbm_scores=[]
mi_scores=[]

plt.figure()
plt.scatter(range(len(network.val_losses[1:])), network.val_losses[1:], s=10)

##Adding title and labels (optional)
plt.title("Validation Losses with respect to Iterations")
plt.xlabel("Iterations")
plt.ylabel("Validation Loss")

##Save the figure
plt.savefig("res_lgbm_gbmo_plot_losses.png") 

seed= 58#20 #16
data = pd.read_excel('Residential-Building-Data-Set.xlsx', sheet_name=0)
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
new_header = data.iloc[0] #grab the first row for the header
data = data[1:] #take the data less the header row
data.columns = new_header #
data = data.drop(['V-1', 'V-10'], axis=1)
#print(data.shape)
nb = len(data)
val_1_split = 0.3
val_2_split = 0.1
test_split = 0.15
data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
X_train, X_val_1, X_val_2, X_test = data.iloc[:int((1-(val_1_split+val_2_split+test_split))*nb)], data.iloc[int((1-(val_1_split+val_2_split+test_split))*nb):int((1-(val_2_split+test_split))*nb)], data.iloc[int((1-(val_2_split+test_split))*nb):int((1-(test_split))*nb)], data.iloc[int((1-(test_split))*nb):]
y_train, y_val_1, y_val_2, y_test = X_train[["V-9"]], X_val_1[["V-9"]], X_val_2[["V-9"]], X_test[["V-9"]]
scaler = MinMaxScaler()
normalized_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
normalized_X_val_1 = pd.DataFrame(scaler.transform(X_val_1), columns=X_val_1.columns)
normalized_X_val_2 = pd.DataFrame(scaler.transform(X_val_2), columns=X_val_2.columns)
normalized_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
normalized_y_train = pd.DataFrame(scaler.fit_transform(y_train), columns=y_train.columns).values
normalized_y_val_1 = pd.DataFrame(scaler.transform(y_val_1), columns=y_val_1.columns).values
normalized_y_val_2 = pd.DataFrame(scaler.transform(y_val_2), columns=y_val_2.columns).values
normalized_y_test = pd.DataFrame(scaler.transform(y_test), columns=y_test.columns).values
normalized_X_train, normalized_X_val_1, normalized_X_val_2,  normalized_X_test = normalized_X_train.drop(columns=['V-9']).values, normalized_X_val_1.drop(columns=['V-9']).values, normalized_X_val_2.drop(columns=['V-9']).values, normalized_X_test.drop(columns=['V-9']).values
network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=normalized_X_train, X_test=normalized_X_test, slack=0.0, 
                                        X_val_1=normalized_X_val_1, y_val_1=normalized_y_val_1,X_val_2=normalized_X_val_2, y_val_2=normalized_y_val_2, y_train=normalized_y_train,y_test=normalized_y_test,iterations=10) 
## 13 fena değil
# 53te hesp çok iyi
# 17 iyi
_,all_score, flbmo_score, mo_score, gbm_score, mi_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True, f_numbers=[20, 24, 30, 60], loss_ts=[1], slacks=[0.00025, 0.001, 0.01, 0.05])
all_scores.append(all_score)
mo_scores.append(mo_score)
flbmo_scores.append(flbmo_score)
lgbm_scores.append(gbm_score)
mi_scores.append(mi_score)
logging.info("For residential Data, mlp base, seed: {}, all feature MSE: {}, FLBMO MSE: {}, GBMO MSE: {}, GBM MSE:{}, MI MSE:{}".format(seed, np.mean(all_scores), np.mean(flbmo_scores),np.mean(mo_scores), np.mean(lgbm_scores), np.mean(mi_scores)))
plt.figure()

plt.scatter(range(len(network.best_ft_nbs)), network.best_ft_nbs, s=10)

# Adding title and labels (optional)
plt.title("GBMO Iterations Features Selected")
plt.xlabel("Iterations")
plt.ylabel("Number of Selected Features")
plt.ylim([0, 120])

# Save the figure
plt.savefig("res_mlp_gbmo_plot.png")  # You can change the filename and extension as needed
plt.figure()
plt.scatter(range(len(network.val_losses[1:])), network.val_losses[1:], s=10)

##Adding title and labels (optional)
plt.title("Validation Losses with respect to Iterations")
plt.xlabel("Iterations")
plt.ylabel("Validation Loss")

##Save the figure
plt.savefig("res_mlp_gbmo_plot_losses.png") 