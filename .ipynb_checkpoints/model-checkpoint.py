import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, mean_squared_error,mean_absolute_error, log_loss
from tqdm import tqdm
import plotly.graph_objects as go
#import pickle
#import pygmo as pg
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import random
from sklearn.svm import SVR, SVC


class LGBM_w_Feature_Selector():

    def __init__(self, layer_sizes = None, boosting_type=None, num_leaves=None, max_depth=None, learning_rate=0.01, subsample=None, colsample_bytree=None, reg_alpha=None,
                 reg_lambda=None, model="lgbm",problem_type="Classifier", objective="binary", param_grid=None,n_estimators=1000, random_state=42,
                 early_stopping_rounds=100, X_train=None, X_test=None, X_val=None, y_val=None, y_train=None, show_loss_plot = False,
                 y_test=None, iterations=1000, slack=0.01):

        self.problem_type = problem_type
        if model=="lgbm":
            self.model_type = "lgbm"
            if problem_type == "Classifier":
                self.params = param_grid
                self.model = lgb.LGBMClassifier(max_depth=max_depth,verbose= -1)
                self.initial_model=lgb.LGBMClassifier(random_state=42, verbose= -1)
                self.criterion = log_loss
                self.cv_scoring = "roc_auc_ovr"

            else:
                self.params = param_grid

                self.model = lgb.LGBMRegressor(max_depth=max_depth,verbose= -1)
                self.initial_model=lgb.LGBMClassifier(random_state=42,verbose= -1)

                self.criterion = mean_squared_error
                self.cv_scoring = "neg_mean_squared_error"
        else:
            self.model_type = "mlp"

            if problem_type == "Classifier":
                self.model = MLPClassifier(hidden_layer_sizes= layer_sizes, activation='relu')
                self.params = param_grid
                self.initial_model = MLPClassifier(random_state=42)

                #self.params ={
                #    "hidden_layer_sizes": layer_sizes, 
                #    "activation":['relu','sigmoid'], 
                #    "solver":['adam',], 
                #    "alpha":[0.0001, 0.001, 0.01], 
                #    "learning_rate_init":[0.001, 0.01], 
                #    "power_t":[0.5],
                    #"max_iter":200, 
                    #"random_state":None, 
                    #"tol":0.0001, 
                #    "warm_start":[False], 
                #    "momentum":[0.9], 
                    #"validation_fraction":[0.1], 
                #    "beta_1":[0.9], 
                #    "beta_2":[0.999],
                #}                
                self.criterion = log_loss
                self.cv_scoring = "roc_auc_ovr"

            else:
            #self.model = MLPRegressor(hidden_layer_sizes=layer_sizes, warm_start=False, max_iter=500)
                self.model = MLPRegressor(hidden_layer_sizes= layer_sizes, activation='relu', random_state=42)
                self.initial_model = MLPRegressor(random_state=42)
                self.params = param_grid
                self.criterion = mean_squared_error
                self.cv_scoring = "neg_mean_squared_error"

        self.early_stopping_rounds = early_stopping_rounds
        self.initial_model = self.model
        self.iterations = iterations
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.slack = slack
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.plot_loss = show_loss_plot
        self.num_of_features = self.X_train.shape[1]
        self.mask = np.ones(self.num_of_features)
        #if problem_type == "Classifier":
        #    self.criterion = log_loss
        #else:
        #    self.criterion = mean_squared_error

    def shuffle_column(self,arr, i):
        # Copy the array to avoid modifying the original array
        arr_copy = np.copy(arr)
        
        # Get the ith column
        column = arr_copy[:, i]
        
        # Shuffle the values in the column
        np.random.shuffle(column)
        
        # Update the ith column in the copied array
        arr_copy[:, i] = column
        
        return arr_copy 
    def set_column_to_zero(self, input_array, i):
        if i < 0 or i >= input_array.shape[1]:
            raise ValueError("Invalid column index")

        result_array = input_array.copy()
        result_array[:, i] = 0
        return result_array
    def general_feature_importance_calc(self):
        self.model.fit(self.X_train, self.y_train)
        inital_error = mean_squared_error(self.model.predict(self.X_val), self.y_val)

        importance_dict = {}
        for feat in range(self.X_train.shape[1]):
            importance_dict[feat] = mean_squared_error(self.model.predict(self.shuffle_column(self.X_val, feat)), self.y_val) - inital_error

        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1],reverse=True)

        sorted_keys = [item[0] for item in sorted_items]
        sorted_values = [item[1] for item in sorted_items]        
        return sorted_keys, sorted_values



    def search(self,mask,x_val, y_val, shuffle = False):
        min_loss = np.inf
        least_useful_feature_ind = None
        ret_mask = mask.copy()
        losses = []
        for i in range(len(self.mask)):
            temp_mask = mask.copy()
            if mask[i] != 0:
                if shuffle:
                    x = self.shuffle_column(x_val, i)
                else:
                    x = self.set_column_to_zero(x_val, i)
                # try: 
                #     temp_loss = self.criterion(self.best_model_opt_1.predict(x*temp_mask), y_val)
                # except: 
                if self.problem_type=="Classifier":
                    temp_loss = self.criterion(self.model.predict(x*temp_mask), y_val, labels=(0,1))
                else: 
                    temp_loss = self.criterion(self.model.predict(x*temp_mask), y_val)

                losses.append(temp_loss)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    least_useful_feature_ind = i
        
        ret_mask[least_useful_feature_ind] = 0
        return min_loss, ret_mask, losses
        
    def main_search_1(self, tolerance):
        self.model.fit(self.X_train,self.y_train)
        cur_mask = self.mask
        mask = cur_mask-1
        if self.problem_type == "Classifier":
            initial_loss = log_loss(self.model.predict(self.X_val), self.y_val)

        else: 

            initial_loss = mean_squared_error(self.model.predict(self.X_val), self.y_val)
        cur_loss = initial_loss
        print(cur_loss)
        same_mask = 0
        while (cur_loss < initial_loss + self.slack)&(tolerance>same_mask):
            if np.array_equal(mask,cur_mask):
                same_mask+=1
            else:
                same_mask=0
            mask = cur_mask
            print(mask)
            self.model.fit(self.X_train*mask,self.y_train)
            cur_loss, cur_mask = self.search_1(mask, self.X_val, self.y_val)
            print(cur_loss)
        ### optimal mask
        print(mask)
        return mask


    def plot_losses(self, losses):
        plt.plot(losses)

        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('Loss')
        plt.show()

    def main_search_2(self, run_cv = False, p =0.1, loss_tolerance = 2):
        ## get best hyperparameters on val
        if run_cv:
            random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
                                                random_state=42, verbose=0,scoring=self.cv_scoring)
            # Fit the model
            random_search.fit(self.X_train, self.y_train)

            # Update the best model and best parameters
            best_params_opt_1 = random_search.best_params_
            self.best_model_opt_1 = random_search.best_estimator_
            print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
            self.model = self.best_model_opt_1

        self.model.fit(self.X_train, self.y_train)
        cur_loss = 20
        feat_nb = self.X_val.shape[1]
        mask = np.ones(feat_nb)
        #loss_tolerance = 2 #int(feat_nb/tol_param)
        update_tolerance = 3
        prev_loss = np.inf
        loss_patience, update_patience = 0, 0
        while (loss_patience < loss_tolerance)&(np.sum(mask)>feat_nb/4):
        #while np.sum(mask)>nb_features:
            prev_loss = cur_loss
            
            #cur_loss, cur_mask, losses = self.search(mask, self.X_val, self.y_val)
            cur_loss, cur_mask, losses = self.search(mask, self.X_val, self.y_val)
            #print(cur_loss)
            #print(mask)
            # if self.plot_loss:
            #     self.plot_losses(losses)
            # if np.array_equal(cur_mask,mask):
            #     update_patience+=1
            # else:
            #     update_patience = 0

            if update_patience>update_tolerance:
                break
            if (cur_loss<prev_loss*(1+self.slack)): ## update mask, introduce stochasticity
                loss_patience = 0
                #if random.random()<p:
                mask = cur_mask            #print(mask)
            else:
                loss_patience+=1
        
        ### optimal mask
        #print(mask)
        return mask
    def main_search_3(self, f_number,  run_cv = False,shuffle=False):
            ## get best hyperparameters on val
            if run_cv:
                random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
                                                    random_state=42, verbose=0,scoring=self.cv_scoring)
                # Fit the model
                random_search.fit(self.X_train, self.y_train)
    
                # Update the best model and best parameters
                best_params_opt_1 = random_search.best_params_
                self.best_model_opt_1 = random_search.best_estimator_
                print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
                self.model = self.best_model_opt_1
    
            self.model.fit(self.X_train, self.y_train)
            feat_nb = self.X_val.shape[1]
            mask = np.ones(feat_nb)
            while (np.sum(mask) > f_number):
                cur_loss, cur_mask, losses = self.search(mask, self.X_val, self.y_val, shuffle = shuffle)
                mask = cur_mask            #print(mask)

            ### optimal mask
            #print(mask)
            return mask

    def main_search_4(self, f_number, run_cv = False, tolerance=10):
            ## get best hyperparameters on val
            if run_cv:
                random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
                                                    random_state=42, verbose=0,scoring=self.cv_scoring)
                # Fit the model
                random_search.fit(self.X_train, self.y_train)

                # Update the best model and best parameters
                best_params_opt_1 = random_search.best_params_
                self.best_model_opt_1 = random_search.best_estimator_
                print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
                cur_model = self.best_model_opt_1
            else:
                self.model.fit(self.X_train, self.y_train)
            feat_nb = self.X_val.shape[1]
            mask = np.ones(feat_nb)
            patience =0
            index_is_unique = False
            cur_loss = 200
            while (np.sum(mask)>f_number) : #&(patience<tolerance):
                temp_mask = mask.copy()
                prev_loss = cur_loss
                while not(index_is_unique):
                    cur_index = np.random.randint(0, feat_nb)
                    if mask[cur_index]==1:
                        index_is_unique = True
                temp_mask[cur_index] = 0
                index_is_unique = False
                cur_loss=self.criterion((self.model.predict(self.X_val*np.expand_dims(temp_mask,axis=0))), self.y_val)
                if (cur_loss<prev_loss):
                    mask[cur_index] = 0
                else:
                    patience+=1
                #cur_loss, cur_mask, losses = self.search(mask, self.X_val, self.y_val)
                #print(cur_loss)
                #print(mask)
            ### optimal mask
            #print(mask)
            return mask


    def CV_test(self, X,y):
        '''
        StratifiedKFold Cross-Validation with KNeighborsClassifier
        '''
        score = []
        for i_n in [11,23,3]:
            if self.problem_type=="Classifier":
                cv = StratifiedKFold(5, random_state=i_n*10, shuffle = True)
            else:
                cv = KFold(5, random_state=i_n*10, shuffle = True)
            #reg = MLPRegressor()
            if self.problem_type == "Classifier":
                #reg = SVC()
                score.append(cross_val_score(self.model, X, y, cv=cv, scoring=make_scorer(roc_auc_score)).mean())
            else: 
                #reg = SVR()
                score.append(cross_val_score(self.model, X, y, cv=cv, scoring=make_scorer(mean_squared_error)).mean())
        return np.mean(score)


    def create_mask(self, selected_indices, n):
        result_array = [0] * n  # Initialize the array with zeros
        
        for index in selected_indices:
            if 0 <= index < n:
                result_array[index] = 1
        
        return result_array

    def compare_fi(self, X,y,d, debug=False):
        '''
        Compare feature selection methods
        '''
        nn_lst = []
        gb_list = []
        acc_nn, acc_gg = [], []
        
        for f_nn, f_gb in zip(d['nn'], d['gb']):
            nn_lst.append(f_nn)
            gb_list.append(f_gb)
            if debug:
                print(nn_lst)
                print(gb_list)
            acc_nn.append(self.CV_test(X[:,nn_lst], y).mean())
            acc_gg.append(self.CV_test(X[:,gb_list], y).mean())
        return acc_nn, acc_gg

    def compare_fi_f_nb(self, X,y,d, debug=False):
        '''
        Compare feature selection methods
        '''
        nn_lst = []
        gb_list = []
        acc_nn, acc_gg = [], []
        
        for f_nn, f_gb in zip(d['nn'], d['gb']):
            nn_lst.append(f_nn)
            gb_list.append(f_gb)
            if debug:
                print(nn_lst)
                print(gb_list)
            acc_nn.append(self.CV_test(X[:,nn_lst], y).mean())
            acc_gg.append(self.CV_test(X[:,gb_list], y).mean())
        return acc_nn, acc_gg

    def feature_extraction(self, f_number, method="number of features", seed =88, include_RFE=False, loss_tolerance = 3, shuffle=False, run_CV=False):
        
        feature_weights = {}
        # Gradient Boosting FS
        np.random.seed(seed)
        if self.problem_type=="Classifier":
            forest=lgb.LGBMClassifier(n_estimators=50,random_state=0,importance_type='gain', verbose=-1)
        else:
            forest = lgb.LGBMRegressor(n_estimators=50,random_state=0,importance_type='gain', verbose=-1)
        forest.fit(self.X_train, self.y_train.ravel())
        gb_importances = forest.feature_importances_
        
        # NN FS
        #cancelout_weights_importance = self.main_search_1(tolerance=0.5)
        if method == "number of features":
            cancelout_weights_importance = self.main_search_3(f_number= f_number, shuffle=shuffle, run_cv=run_CV)
        elif method == "convergence": 
            cancelout_weights_importance = self.main_search_2(loss_tolerance = loss_tolerance, run_cv=run_CV)
        elif method == "random search":
            cancelout_weights_importance = self.main_search_4(f_number= f_number, run_cv=run_CV)
        else:
            raise ValueError("choose a valid feature selection method.")

        #print("CancelOut weights after the activation function:")
        #print(cancelout_weights_importance,'\n')
        # selecting first 5 features 
        if method == "number of features":
            feature_weights['nn'] = cancelout_weights_importance.argsort()[::-1]
            feature_weights['gb'] = gb_importances.argsort()[::-1]
            nn_fi = cancelout_weights_importance.argsort()[-f_number:][::-1]
            gb_fi = gb_importances.argsort()[-f_number:][::-1]
            #print(self.model.coefs_)
            #print('Features selected using mask optimization', sorted(nn_fi, reverse=True))
            #print('Features selected using LigthGBM feature importance ',sorted(gb_fi, reverse=True))
            mask_len = self.X_test.shape[1]        
            #print(f'CV score from all features: {self.CV_test( self.X_train, self.y_train.ravel())}')
            #print(f'CV score GB FS: {self.CV_test(self.X_train*np.expand_dims(self.create_mask(gb_fi, mask_len),axis=0), self.y_train.ravel())}')
            #print(f'CV score MO FS: {self.CV_test(self.X_train*np.expand_dims(self.create_mask(nn_fi, mask_len),axis=0), self.y_train.ravel())}')
            all_score = self.test_with_mask( np.expand_dims(np.ones(mask_len),axis=0))
            gb_score = self.test_with_mask( np.expand_dims(self.create_mask(gb_fi, mask_len),axis=0))
            mo_score = self.test_with_mask( np.expand_dims(self.create_mask(nn_fi, mask_len),axis=0) )
            #print(f'Test score from all features: {all_score}')
            #print(f'Test score GB FS: {gb_score}')
            #print(f'Test score MO FS: {mo_score}')   
            if include_RFE&(self.model_type=="lgbm"):
                rfe = RFE(self.initial_model, n_features_to_select=f_number) 
                rfe.fit(self.X_train, self.y_train)
                rfe_mask = 1*rfe.support_
                rfe_score = self.test_with_mask( np.expand_dims(rfe_mask,axis=0))
                #print(f'Test score RFE FS: {rfe_score}')
                return feature_weights,all_score, gb_score, mo_score, rfe_score
            
            return feature_weights,all_score, gb_score, mo_score
        
        else:
            #nn_fi = cancelout_weights_importance.argsort()[:][::-1]
            nn_fi = [i for i, x in enumerate(cancelout_weights_importance) if x == 1]
            mask_len = self.X_test.shape[1]        
            mo_score = self.test_with_mask(  np.expand_dims(self.create_mask(nn_fi, mask_len),axis=0) )
            all_score = self.test_with_mask( np.expand_dims(np.ones(mask_len),axis=0))
            return feature_weights,all_score, mo_score



    def test_with_mask(self,mask):
        zero_columns = np.where(mask == 0)
        #print(zero_columns)
        masked_train_x = np.delete(self.X_train, zero_columns, axis=1)
        masked_test_x = np.delete(self.X_test, zero_columns, axis=1)
        self.model.fit(masked_train_x, self.y_train)    
        predicted_values = self.model.predict(masked_test_x)

        if self.problem_type == "Classifier":
            loss = roc_auc_score(self.y_test ,predicted_values)
        else:  
            loss = mean_squared_error(self.y_test  ,predicted_values)
            #print("Test MSE: {},Test MAE: {}".format(mse, mae))
        
        return loss

    def TestBench(self,f):
        d = self.feature_extraction(f)
        acc_nn, acc_gg = self.compare_fi( self.X_test, self.y_test,d, debug=False)
        plt.plot(np.array(acc_gg), 'r', label='FS using GradientBoosting')
        plt.plot(np.array(acc_nn), 'g', label='FS using mask optimization')
        plt.legend(loc='best')
        plt.xlabel('Number of Features')
        plt.ylabel('AUC')
        plt.xlim(1, self.X_train.shape[1]-1)
        plt.tight_layout()
        plt.show()


