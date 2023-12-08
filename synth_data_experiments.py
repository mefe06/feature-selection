import os
from final_model import LGBM_w_Feature_Selector 
import logging
import warnings
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(filename='final_exp.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

lgbm_param_grid={
'num_leaves': [7, 15],#[31, 63],
'learning_rate': [0.01, 0.025, 0.05],
'n_estimators': [20, 40],
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
    "hidden_layer_sizes": [(40,20),(40),(80),(20,10)],#[(20),(40),(80),(40,10),(80,20),(80,40),(80,20,10),(80,40,20),(80,40,10)], 
    "activation":['relu','logistic'], 
    "solver":['adam',], 
    "alpha":[0.0001, 0.001, 0.01], 
    "learning_rate_init":[0.001, 0.01], 
    "warm_start":[False],  
    "momentum":[0.9], 
} 

# Settings
n_samples = 300
n_real_features = 10
n_dummy_features = 90
np.random.seed(42)
# Generate real features
real_features = np.random.rand(n_samples, n_real_features)
#redundant_features =np.copy(real_features)
# Generate dummy features
dummy_features = np.random.rand(n_samples, n_dummy_features)

# Combine into one dataset
data = np.hstack((real_features, dummy_features))


target = np.zeros(n_samples)
for i in range(n_real_features):
    target += real_features[:, i] ** 2  # Polynomial term (square)
    target += np.sin(real_features[:, i])
target = np.expand_dims(target, 1)
X_train, X_temp_test, y_train, y_temp_test = train_test_split(data, target, test_size=0.55, random_state=42)

# Split the temporary test set into true test set and temporary validation set
X_test, X_temp_val, y_test, y_temp_val = train_test_split(X_temp_test, y_temp_test, test_size=0.73, random_state=42)

# Finally, split the temporary validation set into two validation sets (val1 and val2)
X_val1, X_val2, y_val1, y_val2 = train_test_split(X_temp_val, y_temp_val, test_size=0.33, random_state=42)

network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=X_train, X_test=X_test, slack=0.0, 
                                        X_val_1=X_val1, y_val_1=y_val1,X_val_2=X_val2, y_val_2=y_val2, y_train=y_train,y_test=y_test,iterations=10) 
_,all_score, flbmo_score, mo_score, gbm_score, rfe_score, mi_score = network.feature_extraction(5, seed=11 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=True, f_numbers=[6, 10, 15, 20], loss_ts=[1,2], slacks=[ 0.001, 0.005, 0.02, 0.05]) #slacks=[0.005, 0.01])
plt.scatter(range(len(network.best_ft_nbs)), network.best_ft_nbs, s=10)
print(network.best_ft_nbs[-1])
# Adding title and labels (optional)
plt.title("GBMO Iterations Features Selected")
plt.xlabel("Iterations")
plt.ylabel("Number of Selected Features")
plt.ylim([0, 110])
# Save the figure
plt.savefig("gbmo_plot.png")  # You can change the filename and extension as needed

# Display the plot
plt.show()
#plt.scatter(range(len(network.selected_feature_nbs_flbmo)), network.selected_feature_nbs_flbmo, s=10)
plt.figure()
plt.scatter(range(len(network.best_ft_val_losses)), network.best_ft_val_losses, s=10)

##Adding title and labels (optional)
plt.title("Validation Losses with respect to Iterations")
plt.xlabel("Iterations")
plt.ylabel("Validation Loss")

##Save the figure
plt.savefig("synth_lgbm_gbmo_plot_losses.png") 

# Display the plot
plt.show()

# # Adding title and labels (optional)
# plt.title("FLBMO Iterations Features Selected")
# plt.xlabel("Iterations")
# plt.ylabel("Number of Selected Features")

# # Save the figure
# plt.savefig("flbmo_plot.png")  # You can change the filename and extension as needed

# Display the plot
plt.show()

# network = LGBM_w_Feature_Selector(model="mlp",problem_type="regression",param_grid=mlp_param_grid,X_train=X_train, X_test=X_test, slack=0.0, 
#                                         X_val_1=X_val1, y_val_1=y_val1,X_val_2=X_val2, y_val_2=y_val2, y_train=y_train,y_test=y_test,iterations=10) 
# # gbmo_score, flbmo_score = network.test_on_synth_data(slacks=[0.001, 0.008], loss_ts=[1,2], f_numbers=[10, 20, 30])
# # print(gbmo_score)
# # print(flbmo_score)
# _,all_score, flbmo_score, mo_score, gbm_score, mi_score = network.feature_extraction(5, seed=42 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=False, f_numbers=[6, 10, 15, 20], loss_ts=[1,2], slacks=[ 0.001, 0.25]) #slacks=[0.005, 0.01])

# plt.scatter(range(len(network.selected_feature_nbs_gbmo)), network.selected_feature_nbs_gbmo, s=10)

# # Adding title and labels (optional)
# plt.title("GBMO Iterations Features Selected")
# plt.xlabel("Iterations")
# plt.ylabel("Number of Selected Features")

# # Save the figure
# plt.savefig("mlp_gbmo_plot.png")  # You can change the filename and extension as needed

# # Display the plot
# plt.show()
# plt.scatter(range(len(network.selected_feature_nbs_flbmo)), network.selected_feature_nbs_flbmo, s=10)
# plt.figure()
# plt.scatter(range(len(network.best_ft_val_losses)), network.best_ft_val_losses, s=10)

# ##Adding title and labels (optional)
# plt.title("Validation Losses with respect to Iterations")
# plt.xlabel("Iterations")
# plt.ylabel("Validation Loss")

# ##Save the figure
# plt.savefig("synth_mlp_gbmo_plot_losses.png") 
# # Display the plot
# plt.show()

# # Adding title and labels (optional)
# plt.title("FLBMO Iterations Features Selected")
# plt.xlabel("Iterations")
# plt.ylabel("Number of Selected Features")

# # Save the figure
# plt.savefig("flbmo_plot.png")  # You can change the filename and extension as needed

# # Display the plot
# plt.show()




# target = np.zeros(n_samples)
# for i in range(n_real_features):
#     target += np.cos(real_features[:, i])* real_features[:, i] ** (1/2)
#     target += np.sin(real_features[:, i])
#     # target += np.exp(-real_features[:, i])  # Exponential decay
#     # if real_features[:, i].all() > 0:  # Avoiding log(0)
#     #     target += np.log(real_features[:, i] + 1)  # Logarithmic growth
# # target = (target-np.min(target))/(np.max(target)-np.min(target))
# #target+=np.random.normal(0, 0.3, n_samples)
# #target+=np.random.normal(0, 0.05, n_samples)
# #target = (target-np.min(target))/(np.max(target)-np.min(target))
# target = np.expand_dims(target, 1)
# X_train, X_temp_test, y_train, y_temp_test = train_test_split(data, target, test_size=0.55, random_state=42)

# # Split the temporary test set into true test set and temporary validation set
# X_test, X_temp_val, y_test, y_temp_val = train_test_split(X_temp_test, y_temp_test, test_size=0.73, random_state=42)

# # Finally, split the temporary validation set into two validation sets (val1 and val2)
# X_val1, X_val2, y_val1, y_val2 = train_test_split(X_temp_val, y_temp_val, test_size=0.33, random_state=42)

# network = LGBM_w_Feature_Selector(model="lgbm",problem_type="regression",param_grid=lgbm_param_grid,X_train=X_train, X_test=X_test, slack=0.0, 
#                                         X_val_1=X_val1, y_val_1=y_val1,X_val_2=X_val2, y_val_2=y_val2, y_train=y_train,y_test=y_test,iterations=10) 
# # gbmo_score, flbmo_score = network.test_on_synth_data(slacks=[0.001, 0.005], loss_ts=[1,2], f_numbers=[10, 20, 30])
# # print(gbmo_score)
# # print(flbmo_score)
# _,all_score, flbmo_score, mo_score, gbm_score, rfe_score, mi_score = network.feature_extraction(5, seed=11 , method="convergence", loss_tolerance=1,run_CV=True, include_RFE=True, f_numbers=[6, 10, 15], loss_ts=[1,2], slacks=[ 0.001, 0.005, 0.02, 0.05]) #slacks=[0.005, 0.01])
