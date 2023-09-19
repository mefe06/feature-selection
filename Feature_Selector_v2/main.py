import os
from synthetic_dataloader import create_regression_Dataset, create_Classification_Dataset, create_reg_Dataset, create_breast_cancer_class
#from feature_selector_model import LGBM_w_Feature_Selector
from model import LGBM_w_Feature_Selector 
import logging
import warnings
from sklearn.exceptions import DataConversionWarning,ConvergenceWarning
from sklearn.preprocessing import StandardScaler


# Your code that triggers the warning
# For example:
# model.fit(X, y)

# Suppress the warning temporarily

warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Your code that triggers the warning
    # For example:
    # model.fit(X, y)

if __name__ == '__main__':
    logging.basicConfig(filename='console.log', level=logging.DEBUG)
    logging.info('Feature Selector LGBM Log')
    #X_train,X_test,X_val,y_val,y_train,y_test = create_regression_Dataset(n_samples=100, n_features=30, n_informative=3)
    
    #X_train,X_test,X_val,y_val,y_train,y_test = create_Classification_Dataset(n_samples=300, n_features=15, n_informative=3, n_redundant=0)
    #X_train,X_test,X_val,y_val,y_train,y_test,g_t = create_reg_Dataset(n_samples=200, val_ratio=0.3, test_ratio=0.1, n_features=20, n_informative=12 )#,shuffle=True)
    X_train,X_test,X_val,y_val,y_train,y_test = create_breast_cancer_class(n_samples=None, val_ratio=0.3, test_ratio=0.1, n_features=20, n_informative=12 )
    #X_train,X_test,X_val,y_val,y_train,y_test, g_t = create_reg_Dataset(n_samples=100, n_features=20, n_informative=5 )#,shuffle=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #y_train = scaler.fit_transform(y_train.reshape(-1,1))
    X_test = scaler.fit_transform(X_test)
    #y_test = scaler.fit_transform(y_test.reshape(-1,1))
    X_val = scaler.fit_transform(X_val)
    #y_val = scaler.fit_transform(y_val.reshape(-1,1))
    #print(y_val)
    #X_train,X_test,X_val,y_val,y_train,y_test = create_reg_Dataset(n_samples=400, n_features=15, n_informative=4)
    #X_train,X_test,X_val,y_val,y_train,y_test = create_Classification_Dataset(n_samples=300, n_features=15, n_informative=5, n_redundant=0)
    # network = LGBM_w_Feature_Selector(boosting_type = "gbdt", num_leaves= 15, max_depth=1,
    #                                   learning_rate=0.01, subsample=0.8,
    #                                   colsample_bytree=0.8,reg_alpha=0.0
    #                                   ,reg_lambda=0.1, slack =0.008,
    #                                   #, classifier_type=["binary"]
    #                                   n_estimators=10,
    #                                   random_state=42,early_stopping_rounds=100,X_train=X_train, X_test=X_test,
    #                                   X_val=X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10)
    
    # network = LGBM_w_Feature_Selector(boosting_type = "gbdt", num_leaves=7, max_depth=2,
    #                                     learning_rate=0.01, subsample=0.8,
    #                                     colsample_bytree=0.8,reg_alpha=0.0
    #                                     ,reg_lambda=0.1, objective= "regression",
    #                                     problem_type="regression", #, classifier_type=["binary"]
    #                                     n_estimators=5,
    #                                     random_state=42,early_stopping_rounds=100,X_train=X_train, X_test=X_test,
    #                                     X_val=X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10)
        #network.fit_with_fix_params(objective="regression") 40

    # network = LGBM_w_Feature_Selector(model="mlp", problem_type="regression",layer_sizes=(20,10),X_train=X_train, X_test=X_test, slack=0.05, 
    #                                      X_val=X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10)

    # network = LGBM_w_Feature_Selector(model="mlp", problem_type="regression",layer_sizes=(20),X_train=X_train, X_test=X_test, slack=0.05, 
    #                                      X_val=X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10)    
    network = LGBM_w_Feature_Selector(model="mlp",layer_sizes=(40),X_train=X_train, X_test=X_test, slack=0.05, 
                                         X_val=X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10) 
    # network = LGBM_w_Feature_Selector(model="lgbm",  problem_type="regression", max_depth=2,X_train=X_train, X_test=X_test, slack=0.1, 
    #                                      X_val=X_val, y_val=y_val,y_train=y_train,y_test=y_test,iterations=10)

        
    #features, importance = network.general_feature_importance_calc()
    #print(features, importance)
    #network.main_search_2()
    #
    network.feature_extraction(5, seed=42 , method="number of features")
    #network.TestBench(5)
    #network.feature_extraction(6, seed=42 , method="random search")
    #print("Ground truth feature importances: {} and weights {}".format(str(g_t.argsort()[::-1]), str(g_t)))
    #network.test()
