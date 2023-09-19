import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, mean_squared_error,mean_absolute_error, log_loss 
from tqdm import tqdm
import plotly.graph_objects as go
import pickle
#import pygmo as pg
from gekko import GEKKO
from sklearn.neural_network import MLPRegressor, MLPClassifier
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, value
import pyomo
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import random
from sklearn.svm import SVR, SVC
class LGBM_w_Feature_Selector():

    def __init__(self, layer_sizes = None, boosting_type=None, num_leaves=None, max_depth=None, learning_rate=0.01, subsample=None, colsample_bytree=None, reg_alpha=None,
                 reg_lambda=None, model="lgbm",problem_type="Classifier", objective="binary", n_estimators=1000, random_state=42,
                 early_stopping_rounds=100, X_train=None, X_test=None, X_val=None, y_val=None, y_train=None, show_loss_plot = False,
                 y_test=None, iterations=1000, slack=0.01):

        self.problem_type = problem_type
        if model=="lgbm":
            if problem_type == "Classifier":
                self.params = {'objective': objective,
                            'boosting_type': boosting_type,
                            'num_leaves': num_leaves,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree,
                            'reg_alpha': reg_alpha,
                            'reg_lambda': reg_lambda,
                            'n_estimators': n_estimators,
                            'random_state': random_state}
                self.model = lgb.LGBMClassifier()

            else:
                self.params = {'objective': objective,
                            'boosting_type': boosting_type,
                            'num_leaves': num_leaves,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree,
                            'reg_alpha': reg_alpha,
                            'reg_lambda': reg_lambda,
                            'n_estimators': n_estimators,
                            'random_state': random_state}

                self.model = lgb.LGBMRegressor()
        else:
            if problem_type == "Classifier":
                self.model = MLPClassifier(hidden_layer_sizes= layer_sizes, activation='logistic')

                self.params ={
                    "hidden_layer_sizes": layer_sizes, 
                    "activation":['relu','sigmoid'], 
                    "solver":['adam',], 
                    "alpha":[0.0001, 0.001, 0.01], 
                    "learning_rate_init":[0.001, 0.01], 
                    "power_t":[0.5],
                    #"max_iter":200, 
                    #"random_state":None, 
                    #"tol":0.0001, 
                    "warm_start":[False], 
                    "momentum":[0.9], 
                    #"validation_fraction":[0.1], 
                    "beta_1":[0.9], 
                    "beta_2":[0.999],
                }                

            else:
            #self.model = MLPRegressor(hidden_layer_sizes=layer_sizes, warm_start=False, max_iter=500)
                self.model = MLPRegressor(hidden_layer_sizes= layer_sizes, activation='relu')

                self.params ={
                    "hidden_layer_sizes": layer_sizes, 
                    "activation":['relu','sigmoid'], 
                    "solver":['adam',], 
                    "alpha":[0.0001, 0.001, 0.01], 
                    "learning_rate_init":[0.001, 0.01], 
                    "power_t":[0.5],
                    #"max_iter":200, 
                    #"random_state":None, 
                    #"tol":0.0001, 
                    "warm_start":[False], 
                    "momentum":[0.9], 
                    #"validation_fraction":[0.1], 
                    "beta_1":[0.9], 
                    "beta_2":[0.999],
                } 
        self.early_stopping_rounds = early_stopping_rounds
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


    def mlp_fit_with_mask(self):
        self.selected_X_train_val = np.concatenate([self.X_train,self.X_val])* self.mask
        self.y_train_val = np.concatenate([self.y_train, self.y_val])
        self.train_data = lgb.Dataset(self.selected_X_train_val, label=self.y_train_val)

        for iter in tqdm(range(self.iterations)):
            # Optimize the model
            random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=1, cv=5,
                                               random_state=42, verbose=4,scoring='neg_mean_squared_error')
            # Fit the model
            random_search.fit(self.train_data.data, self.train_data.label)

            # Update the best model and best parameters
            best_params_opt_1 = random_search.best_params_
            self.best_model_opt_1 = random_search.best_estimator_
            print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
            # Optimize the mask
            self.mask = self.MINLP_Solver(self.mask)
            #self.mask = self.pyomo_Solver(self.mask)
            # bounds = [(0.0, 1.0)] * self.num_of_features
            # constraints = {'type': 'eq', 'fun': self.binary_constraint}
            # result = minimize(self.objective_2_classifier, self.mask, bounds=bounds, constraints=constraints,
            #                   method='SLSQP')
            # self.mask = np.round(result.x)
            print("Mask for iteration {} is: {}".format(iter, self.mask))
            # Evaluate the model
            #accuracy = accuracy_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #recall = recall_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #precision = precision_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #f1 = f1_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #roc_auc = roc_auc_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            mse = mean_squared_error(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))

            print(
                "Validation Mean Squared Error: {}".format(
                    mse))

            if iter % 10 == 9:
                trace = go.Scatter(x=np.arange(self.num_of_features),
                                   y=np.cumsum(self.mask), mode="lines")
                layout = go.Layout(title="Feature Selection Layer Cumulative Sum of Weights",
                                   xaxis_title="Feature Index",
                                   yaxis_title="Cumulative Sum of Weights")
                fig = go.Figure(data=[trace], layout=layout)
                fig.show()       


    def search_1(self,mask,x_val, y_val):
        min_loss = np.inf
        least_useful_feature_ind = None
        ret_mask = mask.copy()
        for i in range(len(self.mask)):
            temp_mask = mask.copy()
            if mask[i] != 0:
                temp_mask[i] = 0
                if self.problem_type == "Classifier":
                    temp_loss = log_loss(self.model.predict(x_val*temp_mask), y_val)

                else:

                    temp_loss = mean_squared_error(self.model.predict(x_val*temp_mask), y_val)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    least_useful_feature_ind = i
        
        ret_mask[least_useful_feature_ind] = 0
        return min_loss, ret_mask
    def search(self,mask,x_val, y_val):
        min_loss = np.inf
        least_useful_feature_ind = None
        ret_mask = mask.copy()
        losses = []
        for i in range(len(self.mask)):
            temp_mask = mask.copy()
            if mask[i] != 0:
                temp_mask[i] = 0
                try: 
                    temp_loss = mean_squared_error(self.best_model_opt_1.predict(x_val*temp_mask), y_val)
                except: 
                    temp_loss = mean_squared_error(self.model.predict(x_val*temp_mask), y_val)
                
                losses.append(temp_loss)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    least_useful_feature_ind = i
        
        ret_mask[least_useful_feature_ind] = 0
        return min_loss, ret_mask, losses

    def search_shuffle(self,mask,x_val, y_val):
        min_loss = np.inf
        least_useful_feature_ind = None
        ret_mask = mask.copy()
        losses = []
        for i in range(len(self.mask)):
            temp_mask = mask.copy()
            if mask[i] != 0:
                #temp_mask = self.shuffle_column(temp_mask, i)
                x = self.shuffle_column(x_val, i)
                try: 
                    temp_loss = mean_squared_error(self.best_model_opt_1.predict(x*temp_mask), y_val)
                except: 
                    temp_loss = mean_squared_error(self.model.predict(x*temp_mask), y_val)

                losses.append(temp_loss)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    least_useful_feature_ind = i
        
        ret_mask[least_useful_feature_ind] = 0
        return min_loss, ret_mask, losses
        
    def main_search_1(self, tolerance):
        self.model.fit(self.X_train,self.y_train)
        #lgb.plot_importance(self.model, importance_type='gain')
        #plt.show()
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
            #initial_loss = cur_loss
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

    def main_search_2(self, p =0.1):
        ## get best hyperparameters on val
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=1, cv=5,
                                            random_state=42, verbose=4,scoring='neg_mean_squared_error')
        # Fit the model
        random_search.fit(self.X_train, self.y_train)

        # Update the best model and best parameters
        best_params_opt_1 = random_search.best_params_
        self.best_model_opt_1 = random_search.best_estimator_
        print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
        cur_loss = 1000000
        mask = np.ones(self.X_val.shape[1])
        prev_loss = np.inf
        while cur_loss<prev_loss+self.slack:
            prev_loss = cur_loss
            cur_loss, cur_mask, losses = self.search_shuffle(mask, self.X_train, self.y_train)
            print(cur_loss)
            if self.plot_loss:
                self.plot_losses(losses)
            if random.random()<p: ## update mask, introduce stochasticity
                mask = cur_mask
            #print(mask)
        
        ### optimal mask
        print(mask)
        return mask
    
    def search_2(self, p=0.1, tol_param = 4 ):
        ## get best hyperparameters on val
        self.model.fit(self.X_train, self.y_train)

        # Update the best model and best parameters
        # best_params_opt_1 = random_search.best_params_
        # self.best_model_opt_1 = random_search.best_estimator_
        # print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
        cur_loss = 20
        feat_nb = self.X_val.shape[1]
        mask = np.ones(feat_nb)
        loss_tolerance = 3 #int(feat_nb/tol_param)
        update_tolerance = 3
        prev_loss = np.inf
        loss_patience, update_patience = 0, 0
        while (loss_patience < loss_tolerance):
        #while np.sum(mask)>nb_features:
            prev_loss = cur_loss
            
            #cur_loss, cur_mask, losses = self.search(mask, self.X_val, self.y_val)
            cur_loss, cur_mask, losses = self.search(mask, self.X_val, self.y_val)
            print(cur_loss)
            print(mask)
            # if self.plot_loss:
            #     self.plot_losses(losses)
            if np.array_equal(cur_mask,mask):
                update_patience+=1
            else:
                update_patience = 0

            if update_patience>update_tolerance:
                break
            if (cur_loss<prev_loss*(1+self.slack)): ## update mask, introduce stochasticity
                patience = 0
                #if random.random()<p:
                mask = cur_mask            #print(mask)
            else:
                loss_patience+=1
        
        ### optimal mask
        print(mask)
        return mask
    def search_new(self, p=0.1):
        ## get best hyperparameters on val
        self.model.fit(self.X_train, self.y_train)

        # Update the best model and best parameters
        # best_params_opt_1 = random_search.best_params_
        # self.best_model_opt_1 = random_search.best_estimator_
        # print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
        cur_loss = 20
        mask = np.ones(self.X_val.shape[1])
        prev_loss = np.inf
        loss_cache = []
        while cur_loss<prev_loss*(1+self.slack):
        #while np.sum(mask)>nb_features:
            prev_loss = cur_loss
            
            cur_loss, cur_mask, losses = self.search_shuffle(mask, self.X_val, self.y_val)
            loss_cache.append(cur_loss)
            print(cur_loss)
            print(mask)
            # if self.plot_loss:
            #     self.plot_losses(losses)
            
            if ((cur_loss<prev_loss*(1+self.slack))or (loss_cache[0] >cur_loss))&(random.random()<p): ## update mask, introduce stochasticity
                mask = cur_mask            #print(mask)
        
        ### optimal mask
        print(mask)
        return mask

    def CV_test(self, X,y):
        '''
        StratifiedKFold Cross-Validation with KNeighborsClassifier
        '''
        score = []
        for i_n in [11,23,3]:
            cv = KFold(5, random_state=i_n*10, shuffle = True)
            #reg = MLPRegressor()
            if self.problem_type == "Classifier":
                #reg = SVC()
                score.append(cross_val_score(self.model, X, y, cv=cv, scoring=make_scorer(roc_auc_score)).mean())
            else: 
                #reg = SVR()
                score.append(cross_val_score(self.model, X, y, cv=cv, scoring=make_scorer(mean_squared_error)).mean())
        return np.mean(score)

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

    def feature_extraction(self, f_number):
        
        feature_weights = {}
        # Gradient Boosting FS
        if self.problem_type=="Classifier":
            forest=lgb.LGBMClassifier(n_estimators=50,random_state=0,importance_type='gain')
        else:
            forest = lgb.LGBMRegressor(n_estimators=50,random_state=0,importance_type='gain')
        forest.fit(self.X_train, self.y_train.ravel())
        gb_importances = forest.feature_importances_
        
        # NN FS
        #cancelout_weights_importance = self.main_search_1(tolerance=0.5)
        cancelout_weights_importance = self.search_2(p = 0.03)

        print("CancelOut weights after the activation function:")
        print(cancelout_weights_importance,'\n')
        # selecting first 5 features 
        feature_weights['nn'] = cancelout_weights_importance.argsort()[::-1]
        feature_weights['gb'] = gb_importances.argsort()[::-1]
        nn_fi = cancelout_weights_importance.argsort()[-f_number:][::-1]
        gb_fi = gb_importances.argsort()[-f_number:][::-1]
        
        print('Features selected using mask optimization', sorted(nn_fi))
        print('Features selected using LigthGBM feature importance ',sorted(gb_fi))
        
        print(f'CV score from all features: {self.CV_test( self.X_val, self.y_val.ravel())}')
        print(f'CV score GB FS: {self.CV_test(self.X_val[:,gb_fi], self.y_val.ravel())}')
        print(f'CV score MO FS: {self.CV_test(self.X_val[:,nn_fi], self.y_val.ravel())}')
        
        return feature_weights


    def TestBench(self,f):
        d = self.feature_extraction(f)
        #acc_nn, acc_gg = self.compare_fi( self.X_train, self.y_train,d, debug=False)
        # plt.plot(np.array(acc_gg), 'r', label='FS using GradientBoosting')
        # plt.plot(np.array(acc_nn), 'g', label='FS using masl optimization')
        # plt.legend(loc='best')
        # plt.xlabel('Number of Features')
        # plt.ylabel('AUC')
        # plt.xlim(1, self.X_train.shape[1]-1)
        # plt.tight_layout()
        #plt.show()



    def lgbm_fit_with_mask(self):
        self.selected_X_train_val = np.concatenate([self.X_train,self.X_val])* self.mask
        self.y_train_val = np.concatenate([self.y_train, self.y_val])
        self.train_data = lgb.Dataset(self.selected_X_train_val, label=self.y_train_val)

        for iter in tqdm(range(self.iterations)):
            # Optimize the model
            random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.params, n_iter=1, cv=5,
                                               random_state=42, verbose=4,scoring='neg_mean_squared_error')
            # Fit the model
            random_search.fit(self.train_data.data, self.train_data.label)

            # Update the best model and best parameters
            best_params_opt_1 = random_search.best_params_
            self.best_model_opt_1 = random_search.best_estimator_
            print("Best parameters for iteration {} are: {}".format(iter, best_params_opt_1))
            # Optimize the mask
            self.mask = self.MINLP_Solver(self.mask)
            #self.mask = self.pyomo_Solver(self.mask)
            # bounds = [(0.0, 1.0)] * self.num_of_features
            # constraints = {'type': 'eq', 'fun': self.binary_constraint}
            # result = minimize(self.objective_2_classifier, self.mask, bounds=bounds, constraints=constraints,
            #                   method='SLSQP')
            # self.mask = np.round(result.x)
            print("Mask for iteration {} is: {}".format(iter, self.mask))
            # Evaluate the model
            #accuracy = accuracy_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #recall = recall_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #precision = precision_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #f1 = f1_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            #roc_auc = roc_auc_score(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))
            mse = mean_squared_error(self.y_val, self.best_model_opt_1.predict(self.X_val * self.mask))

            print(
                "Validation Mean Squared Error: {}".format(
                    mse))

            if iter % 10 == 9:
                trace = go.Scatter(x=np.arange(self.num_of_features),
                                   y=np.cumsum(self.mask), mode="lines")
                layout = go.Layout(title="Feature Selection Layer Cumulative Sum of Weights",
                                   xaxis_title="Feature Index",
                                   yaxis_title="Cumulative Sum of Weights")
                fig = go.Figure(data=[trace], layout=layout)
                fig.show()

    def save_model(self):
        # Save the model
        with open("best_model_lgbm.pkl", "wb") as f:
            pickle.dump(self.best_model_opt_1, f)
        # Save the mask
        with open("mask.pkl", "wb") as f:
            pickle.dump(self.mask, f)

    def test(self):
        # Predict the values
        predicted_values = self.best_model_opt_1.predict(self.X_test*self.mask)
        print(self.mask)
        true_values = self.y_test
        # Evaluate the model
        if self.problem_type == "Classifier":
            accuracy = accuracy_score(true_values, predicted_values)
            recall = recall_score(true_values, predicted_values)
            precision = precision_score(true_values, predicted_values)
            f1 = f1_score(true_values, predicted_values)
            roc_auc = roc_auc_score(true_values, predicted_values)
            print("Test Accuracy: {},Test Recall: {},Test Precision: {}, Test F1: {}, Test ROC_AUC: {}".format(accuracy,
                                                                                                               recall,
                                                                                                               precision,
                                                                                                               f1,
                                                                                                               roc_auc))
            return accuracy, recall, precision, f1, roc_auc
        # Evaluate the model
        else:
            mse = mean_squared_error(true_values ,predicted_values)
            mae = mean_absolute_error(true_values ,predicted_values)
            print("Test MSE: {},Test MAE: {}".format(mse, mae))
            return mse, mae
