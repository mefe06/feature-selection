import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, mean_squared_error,mean_absolute_error, log_loss
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import random
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd
import time

class LGBM_w_Feature_Selector():

    def __init__(self, layer_sizes = None, boosting_type=None, num_leaves=None, max_depth=None, learning_rate=0.01, subsample=None, colsample_bytree=None, reg_alpha=None,
                 reg_lambda=None, model="lgbm",problem_type="Classifier", objective="binary", param_grid=None,n_estimators=1000, random_state=42, target_name="Y",
                 early_stopping_rounds=100, X_train=None, X_test=None, X_val_1=None, X_val_2=None, y_val_1=None, y_val_2=None, y_train=None, show_loss_plot = False,
                 y_test=None, iterations=1000, slack=0.01):

        self.problem_type = problem_type
        if model=="lgbm":
            self.model_type = "lgbm"
            if problem_type == "Classifier":
                self.model = lgb.LGBMClassifier(max_depth=max_depth,verbose= -1)
                self.initial_model=lgb.LGBMClassifier(random_state=42, verbose= -1)
                self.criterion = log_loss
                self.cv_scoring = "neg_log_loss"

            else:

                self.model = lgb.LGBMRegressor(max_depth=max_depth,verbose= -1)
                self.initial_model=lgb.LGBMRegressor(random_state=42,verbose= -1)

                self.criterion = mean_squared_error
                self.cv_scoring = "neg_mean_squared_error"
        else:
            self.model_type = "mlp"

            if problem_type == "Classifier":
                self.model = MLPClassifier(hidden_layer_sizes= layer_sizes, activation='relu', random_state=42)
                self.initial_model = MLPClassifier(random_state=42)
                self.criterion = log_loss
                self.cv_scoring = "neg_log_loss"

            else:
            #self.model = MLPRegressor(hidden_layer_sizes=layer_sizes, warm_start=False, max_iter=500)
                self.model = MLPRegressor(hidden_layer_sizes= layer_sizes, activation='relu', random_state=42)
                self.initial_model = MLPRegressor(random_state=42)
                self.criterion = mean_squared_error
                self.cv_scoring = "neg_mean_squared_error"
        self.params = param_grid
        self.val_losses = []
        self.selected_feature_nbs_gbmo=[]
        self.best_ft_nbs = []
        self.early_stopping_rounds = early_stopping_rounds
        self.initial_model = self.model
        self.iterations = iterations
        self.X_train = X_train
        self.X_test = X_test
        self.X_val_1 = X_val_1
        self.X_val_2 = X_val_2
        self.X_full_train = np.vstack((X_train,X_val_1))
        self.slack = slack
        self.y_train = y_train
        self.y_val_1 = y_val_1
        self.y_val_2 = y_val_2
        self.y_test = y_test
        self.y_full_train = np.vstack((y_train,y_val_1))
        self.plot_loss = show_loss_plot
        self.num_of_features = self.X_train.shape[1]
        self.mask = np.ones(self.num_of_features)
        self.target_name=target_name
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
    # def general_feature_importance_calc(self):
    #     self.model.fit(self.X_train, self.y_train)
    #     inital_error = mean_squared_error(self.model.predict(self.X_val), self.y_val)

    #     importance_dict = {}
    #     for feat in range(self.X_train.shape[1]):
    #         importance_dict[feat] = mean_squared_error(self.model.predict(self.shuffle_column(self.X_val, feat)), self.y_val) - inital_error

    #     sorted_items = sorted(importance_dict.items(), key=lambda x: x[1],reverse=True)

    #     sorted_keys = [item[0] for item in sorted_items]
    #     sorted_values = [item[1] for item in sorted_items]        
    #     return sorted_keys, sorted_values



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
                    temp_loss = self.criterion( y_val, self.model.predict_proba(x*temp_mask)[:,1])
                else: 
                    temp_loss = self.criterion(self.model.predict(x*temp_mask), y_val)

                losses.append(temp_loss)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    least_useful_feature_ind = i
        
        ret_mask[least_useful_feature_ind] = 0
        return min_loss, ret_mask, losses
        
    def main_search_1(self, run_cv = False, p =0.1, loss_tolerance = 2, lamda=15):
        ## get best hyperparameters on val
        # if run_cv:
        #     random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
        #                                         random_state=42, verbose=0,scoring=self.cv_scoring)
        #     # Fit the model
        #     random_search.fit(self.X_train, self.y_train)

        #     # Update the best model and best parameters
        #     best_params_opt_1 = random_search.best_params_
        #     self.best_model_opt_1 = random_search.best_estimator_
        #     print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
        #     self.model = self.best_model_opt_1

        self.model.fit(self.X_train, self.y_train)
        cur_loss = 20
        feat_nb = self.X_val_1.shape[1]
        mask = np.ones(feat_nb)
        #loss_tolerance = 2 #int(feat_nb/tol_param)
        self.selected_feature_nbs_gbmo=[]
        self.val_losses = []

        update_tolerance = 3
        prev_loss = np.inf
        loss_patience, update_patience = 0, 0
        cur_loss, cur_mask, losses = self.search(mask, self.X_val_1, self.y_val_1)
        self.val_losses.append(cur_loss)
        prev_mask = np.zeros(feat_nb)
        while (loss_patience < loss_tolerance) and (np.sum(mask)>2): #and (cur_loss<=self.val_losses[0]): #and (np.sum(mask)>2): #&(np.sum(mask)>feat_nb/lamda):
            self.selected_feature_nbs_gbmo.append(np.sum(mask))
            prev_loss = cur_loss
            prev_mask = cur_mask
            cur_loss, cur_mask, losses = self.search(mask, self.X_val_1, self.y_val_1)
            self.val_losses.append(cur_loss)

            if (cur_loss<prev_loss*(1+self.slack)): 
                loss_patience = 0
                #if random.random()<p:
                mask = cur_mask            #print(mask)
            else:
                loss_patience+=1
            # if np.array_equal(cur_mask, prev_mask):
            #     print("same feature selected")
        
        ### optimal mask
        #print(mask)
        return mask
    def main_search_2(self, f_number,  run_cv = False,shuffle=False):
            ## get best hyperparameters on val
            # if run_cv:
            #     random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
            #                                         random_state=42, verbose=0,scoring=self.cv_scoring)
            #     # Fit the model
            #     random_search.fit(self.X_train, self.y_train)
    
            #     # Update the best model and best parameters
            #     best_params_opt_1 = random_search.best_params_
            #     self.best_model_opt_1 = random_search.best_estimator_
            #     print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
            #     self.model = self.best_model_opt_1
    
            self.model.fit(self.X_train, self.y_train)
            feat_nb = self.X_val_1.shape[1]
            mask = np.ones(feat_nb)
            self.selected_feature_nbs_flbmo=[]
            while (np.sum(mask) > f_number):
                self.selected_feature_nbs_flbmo.append(np.sum(mask))
                
                cur_loss, cur_mask, losses = self.search(mask, self.X_val_1, self.y_val_1, shuffle = shuffle)
                mask = cur_mask            #print(mask)

            ### optimal mask
            #print(mask)
            return mask


    # def CV_test(self, X,y):
    #     '''
    #     StratifiedKFold Cross-Validation with KNeighborsClassifier
    #     '''
    #     score = []
    #     if self.problem_type=="Classifier":
    #         cv = StratifiedKFold(5, random_state=42, shuffle = True)
    #     else:
    #         cv = KFold(5, random_state=42, shuffle = True)
    #     #reg = MLPRegressor()
    #     if self.problem_type == "Classifier":
    #         #reg = SVC()
    #         score.append(cross_val_score(self.model, X, y, cv=cv, scoring=make_scorer(roc_auc_score)).mean())
    #     else: 
    #         #reg = SVR()
    #         score.append(cross_val_score(self.model, X, y, cv=cv, scoring=make_scorer(mean_squared_error)).mean())
    # return np.mean(score)


    def create_mask(self, selected_indices, n):
        result_array = [0] * n  # Initialize the array with zeros
        
        for index in selected_indices:
            if 0 <= index < n:
                result_array[index] = 1
        
        return result_array

    # def compare_fi(self, X,y,d, debug=False):
    #     '''
    #     Compare feature selection methods
    #     '''
    #     nn_lst = []
    #     gb_list = []
    #     acc_nn, acc_gg = [], []
        
    #     for f_nn, f_gb in zip(d['nn'], d['gb']):
    #         nn_lst.append(f_nn)
    #         gb_list.append(f_gb)
    #         if debug:
    #             print(nn_lst)
    #             print(gb_list)
    #         acc_nn.append(self.CV_test(X[:,nn_lst], y).mean())
    #         acc_gg.append(self.CV_test(X[:,gb_list], y).mean())
    #     return acc_nn, acc_gg

    # def compare_fi_f_nb(self, X,y,d, debug=False):
    #     '''
    #     Compare feature selection methods
    #     '''
    #     nn_lst = []
    #     gb_list = []
    #     acc_nn, acc_gg = [], []
        
    #     for f_nn, f_gb in zip(d['nn'], d['gb']):
    #         nn_lst.append(f_nn)
    #         gb_list.append(f_gb)
    #         if debug:
    #             print(nn_lst)
    #             print(gb_list)
    #         acc_nn.append(self.CV_test(X[:,nn_lst], y).mean())
    #         acc_gg.append(self.CV_test(X[:,gb_list], y).mean())
    #     return acc_nn, acc_gg

    def cv_for_other_fs_methods(self,f_numbers ):
        best_gbm=1000000
        best_mi=1000000
        if self.problem_type=="Classifier":
            mi_importances= mutual_info_classif(self.X_full_train, self.y_full_train)
        else:
            mi_importances= mutual_info_regression(self.X_full_train, self.y_full_train)
        temp = np.hstack((self.X_full_train, self.y_full_train))#pd.concat([self.X_full_train, self.y_full_train], axis=1)
        corr = np.corrcoef(temp, rowvar=False)
        target_corr = corr[:-1, -1]
        for f_number in f_numbers:

            mi_fi = mi_importances.argsort()[-f_number:][::-1]
            gb_fi = np.argsort(np.abs(target_corr))[-f_number:]
            mi_score = self.val_with_mask( np.expand_dims(self.create_mask(mi_fi, self.mask_len),axis=0))
            gb_score = self.val_with_mask( np.expand_dims(self.create_mask(gb_fi, self.mask_len),axis=0))
            if mi_score<best_mi:
                best_mi = mi_score
                print("f number:{}".format(str(f_number)))
                print(mi_fi)
                mi = self.val_with_mask( np.expand_dims(self.create_mask(mi_fi, self.mask_len),axis=0))

            if gb_score<best_gbm:
                best_gbm = gb_score            
                print("f number:{}".format(str(f_number)))
                print(gb_fi)
                gb = self.test_with_mask( np.expand_dims(self.create_mask(gb_fi, self.mask_len),axis=0))
        return gb, mi
    def cv_for_rfe(self,f_numbers):
        best_rfe = 1000000
        t0=time.process_time()
        for f_number in f_numbers:
            self.model.fit(self.X_full_train, self.y_full_train)
            rfe = RFE(self.model, n_features_to_select=f_number) 
            rfe.fit(self.X_full_train, self.y_full_train)
            rfe_mask = 1*rfe.support_
            rfe_score = self.val_with_mask( np.expand_dims(rfe_mask,axis=0))
            if rfe_score<best_rfe:
                best_rfe= rfe_score   
                print("f number:{}".format(str(f_number)))
                print(rfe_mask)
                rfe = self.test_with_mask( np.expand_dims(rfe_mask ,axis=0))
        t1=time.process_time()
        print("rfe time:{}".format(str(t1-t0)))    
        return best_rfe 
    def cv_gbmo(self,loss_ts, slacks):
        best_gbmo=1000000
        t0=time.process_time()
        for loss_t in loss_ts:
            for slack in slacks:
                self.slack=slack
                cancelout_weights_importance = self.main_search_1(loss_tolerance = loss_t, run_cv=True)
                nn_fi = [i for i, x in enumerate(cancelout_weights_importance) if x == 1]
                mo_score = self.val_with_mask( np.expand_dims(self.create_mask(nn_fi, self.mask_len),axis=0))
                if mo_score<best_gbmo:
                    self.best_ft_nbs=self.selected_feature_nbs_gbmo
                    self.best_ft_val_losses = self.val_losses[1:]
                    best_gbmo= mo_score           
                    print("Loss t:{}, slack: {}, nb of selected feat:{}".format(str(loss_t), str(slack), str(len(nn_fi))))
                    gbmo = self.test_with_mask( np.expand_dims(self.create_mask(nn_fi, self.mask_len),axis=0))
        t1=time.process_time()
        print(self.best_ft_nbs)

        print("gbmo time:{}".format(str(t1-t0)))   
        return gbmo 
    
    def cv_flbmo(self, f_numbers):
        best_flbmo = 1000000
        t0=time.process_time()
        for f_number in f_numbers:
            cancelout_weights_importance = self.main_search_2(f_number = f_number, run_cv=True)
            nn_fi = [i for i, x in enumerate(cancelout_weights_importance) if x == 1]
            mo_score = self.val_with_mask( np.expand_dims(self.create_mask(nn_fi, self.mask_len),axis=0))
            if mo_score<best_flbmo:
                self.flbmo_selected_ft_nbs=self.selected_feature_nbs_flbmo
                best_flbmo= mo_score           
                print("nb of selected feat:{}".format(str(f_number)))
                print(nn_fi)
                flbmo = self.test_with_mask( np.expand_dims(self.create_mask(nn_fi, self.mask_len),axis=0))
        t1=time.process_time()
        print("flbmo time:{}".format(str(t1-t0)))   
        return flbmo 
    def test_on_synth_data(self,  f_numbers, loss_ts, slacks,seed=42):
        np.random.seed(seed)
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
                                            random_state=42, verbose=0,scoring=self.cv_scoring)
        # Fit the model
        random_search.fit(self.X_train, self.y_train)
        self.mask_len = self.X_test.shape[1]        

        # Update the best model and best parameters
        best_params_opt_1 = random_search.best_params_
        print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
        self.model = random_search.best_estimator_
        #nn_fi = [i for i, x in enumerate(cancelout_weights_importance) if x == 1]
        mask_len = self.X_test.shape[1]        
        gbmo_score = self.cv_gbmo(loss_ts, slacks)
        flbmo_score = self.cv_flbmo(f_numbers)
        return gbmo_score, flbmo_score


    def feature_extraction(self, f_number, method="number of features", seed =88, include_RFE=False, loss_tolerance = 3, shuffle=False, run_CV=False, f_numbers=[10,20, 30, 40], loss_ts=[1,2,3,4], slacks=[0, 0.05, 0.1, 0.15, 0.2]):
        
        feature_weights = {}
        # Gradient Boosting FS
        np.random.seed(seed)
        # if self.problem_type=="Classifier":
        #     forest=lgb.LGBMClassifier(max_depth = 3, n_estimators=10,random_state=0,importance_type='gain', verbose=-1)
        #     mi_importances= mutual_info_classif(self.X_train, self.y_train)
        # else:
        #     forest = lgb.LGBMRegressor(max_depth = 3, n_estimators=10,random_state=0,importance_type='gain', verbose=-1)
        #     mi_importances= mutual_info_regression(self.X_train, self.y_train)
        # forest.fit(self.X_train, self.y_train.ravel())
        # gb_importances = forest.feature_importances_
        self.mask_len = self.X_test.shape[1]        

        # if method == "number of features":
        #     cancelout_weights_importance = self.main_search_2(f_number= f_number, shuffle=shuffle, run_cv=run_CV)
        # elif method == "convergence": 
        #     cancelout_weights_importance = self.main_search_1(loss_tolerance = loss_tolerance, run_cv=run_CV)
        # else:
        #     raise ValueError("choose a valid feature selection method.")
        #mi_fi = mi_importances.argsort()[-f_number:][::-1]

        if method == "number of features":
            feature_weights['nn'] = cancelout_weights_importance.argsort()[::-1]
            feature_weights['gb'] = gb_importances.argsort()[::-1]
            nn_fi = cancelout_weights_importance.argsort()[-f_number:][::-1]
            gb_fi = gb_importances.argsort()[-f_number:][::-1]
            mask_len = self.X_test.shape[1]        
            all_score = self.test_with_mask( np.expand_dims(np.ones(mask_len),axis=0))
            gb_score = self.test_with_mask( np.expand_dims(self.create_mask(gb_fi, mask_len),axis=0))
            mi_score = self.test_with_mask( np.expand_dims(self.create_mask(mi_fi, mask_len),axis=0))
            mo_score = self.test_with_mask( np.expand_dims(self.create_mask(nn_fi, mask_len),axis=0))

            if include_RFE:#&(self.model_type=="lgbm"):
                rfe = RFE(self.model, n_features_to_select=f_number) 
                rfe.fit(self.X_train, self.y_train)
                rfe_mask = 1*rfe.support_
                rfe_score = self.test_with_mask( np.expand_dims(rfe_mask,axis=0))
                return feature_weights,all_score, gb_score, mo_score, rfe_score, mi_score
            
            return feature_weights,all_score, gb_score, mo_score, mi_score
        
        else:
            random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
                                                random_state=42, verbose=0,scoring=self.cv_scoring)
            # Fit the model
            random_search.fit(self.X_train, self.y_train)

            # Update the best model and best parameters
            best_params_opt_1 = random_search.best_params_
            print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
            self.model = random_search.best_estimator_
            #nn_fi = [i for i, x in enumerate(cancelout_weights_importance) if x == 1]
            mask_len = self.X_test.shape[1]        
            gbmo_score = self.cv_gbmo(loss_ts, slacks)
            flbmo_score = self.cv_flbmo(f_numbers)
            cc_score, mi_score = self.cv_for_other_fs_methods(f_numbers)
            all_score = self.test_with_mask( np.expand_dims(np.ones(mask_len),axis=0))
            if include_RFE: #(self.model_type=="lgbm"):
                rfe_score=self.cv_for_rfe(f_numbers)
                return feature_weights,all_score, flbmo_score, gbmo_score, cc_score, rfe_score, mi_score

            return feature_weights,all_score, flbmo_score, gbmo_score, cc_score, mi_score


    def val_with_mask(self,mask):
        zero_columns = np.where(mask == 0)
        masked_train_x = np.delete(self.X_train, zero_columns, axis=1)
        masked_test_x = np.delete(self.X_val_2, zero_columns, axis=1)
        #print(np.shape(masked_train_x))
        self.model.fit(masked_train_x, self.y_train)    
        if self.problem_type == "Classifier":
            #loss = roc_auc_score(self.y_test ,self.model.predict_proba(masked_test_x)[:,1])
            loss = log_loss(self.y_val_2 ,self.model.predict_proba(masked_test_x)[:,1])
        else:  
            loss = mean_squared_error(self.y_val_2  ,self.model.predict(masked_test_x))
        return loss

    def test_with_mask(self,mask):
        zero_columns = np.where(mask == 0)
        masked_train_x = np.delete(self.X_train, zero_columns, axis=1)
        masked_test_x = np.delete(self.X_test, zero_columns, axis=1)
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=20, cv=5,
                                            random_state=42, verbose=0,scoring=self.cv_scoring)
        # Fit the model
        random_search.fit(masked_train_x, self.y_train)
        #self.model.fit(masked_train_x, self.y_train)    
        if self.problem_type == "Classifier":
            #loss = roc_auc_score(self.y_test ,self.model.predict_proba(masked_test_x)[:,1])
            loss = log_loss(self.y_test ,random_search.best_estimator_.predict_proba(masked_test_x)[:,1])
        else:  
            loss = mean_squared_error(self.y_test  ,random_search.best_estimator_.predict(masked_test_x))
        
        return loss


