import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler




def create_breast_cancer_class(n_samples=1200, n_features=30, n_informative=5, n_redundant=2, n_repeated=1,
                                  n_classes=2, n_clusters_per_class=16, weights=None, flip_y=0.01, class_sep=2,
                                  hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None, val_ratio=0.2,
                                  test_ratio=0.2):
    scaler = StandardScaler()
    if n_samples:
        X, y = load_breast_cancer()['data'][:n_samples, :], load_breast_cancer()['target'][:n_samples]
    else: 
        X, y = load_breast_cancer()['data'], load_breast_cancer()['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio + test_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio / (val_ratio + test_ratio),
                                                    random_state=42)

    return X_train, X_test, X_val, y_val, y_train, y_test


def create_Classification_Dataset(n_samples=1200, n_features=30, n_informative=5, n_redundant=2, n_repeated=1,
                                  n_classes=2, n_clusters_per_class=16, weights=None, flip_y=0.01, class_sep=2,
                                  hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None, val_ratio=0.2,
                                  test_ratio=0.2):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated,
                               n_classes=n_classes, weights=weights,
                               flip_y=flip_y, class_sep=class_sep,
                               hypercube=hypercube, shift=shift, scale=scale, shuffle=False,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio + test_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio / (val_ratio + test_ratio),
                                                    random_state=42)

    return X_train, X_test, X_val, y_val, y_train, y_test


def create_reg_Dataset(n_samples=250, n_features=15, n_informative=5, 
                                 random_state=None, val_ratio=0.2, shuffle=True,
                                  test_ratio=0.2):
    X, y, g_t = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               random_state=random_state, coef=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio + test_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio / (val_ratio + test_ratio),
                                                    random_state=42)

    return X_train, X_test, X_val, y_val, y_train, y_test, g_t

def create_regression_Dataset(n_samples=250, n_features=23, n_informative=2, val_ratio=0.2,
                                  test_ratio=0.2):
    X = np.random.rand(n_samples, n_features)
    x_1=np.expand_dims(X[:,0], axis=1)
    x_2=np.expand_dims(X[:,4], axis=1)
    x_3=np.expand_dims(X[:,6], axis=1)

    #noise_1 = np.random.normal(0,0.1,size=(n_samples, 1)) 

    #X = np.concatenate((X,2*x_1),axis=1 )
    #X = np.concatenate((X,x_1+x_2),axis=1 )
    #noise = np.random.normal(0,0.3,size=(n_samples, 1))
    #y = x_1#+noise
    #y=np.power(x_1+x_2,2)+noise
    y=np.power(x_1,3)+np.power(x_2,2)+x_1*x_2+np.sin(x_2)*4+np.power(x_2,2)*np.power(x_3,2)+x_2*x_3 #+noise
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio + test_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio / (val_ratio + test_ratio),
                                                    random_state=42)

    return X_train, X_test, X_val, y_val, y_train, y_test

