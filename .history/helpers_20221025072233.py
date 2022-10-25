import csv
import numpy as np
import matplotlib.pyplot as plt
from implementations import *

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis = 0)
    std_x = np.std(x,  axis = 0)
    x = (x - mean_x) / std_x
    return x, mean_x, std_x

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_model_data(label, data):
    """Form (y,tX) to get regression data in matrix form."""
    y = label
    x = data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def predict_logistic(w, X):
    
    y_pred = sigmoid(np.dot(X, w))
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def predict(w, X):
    
    y_pred = np.dot(X)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def build_poly_cross_terms_up_to_deg_4(X, d):
    
    if (d==1):
        return X
    
    d_w = X.shape[1]
    n = X.shape[0]
    poly = X.copy()
    
    if (d==2):
        for i in range(d_w):
            for j in range(i, d_w):
                poly = np.hstack((poly, X[:, i].reshape((n,1))*X[:,j:]))
    elif (d==3):
        for i in range(d_w):
            for j in range(i, d_w):   
                for k in range(j,d_w):
                    poly = np.hstack((poly, X[:,i].reshape((n,1))*X[:,j].reshape((n,1))*X[:,k:]))
    elif (d>=4):
        for i in range(d_w):
            for j in range(i, d_w):   
                for k in range(j,d_w):
                    for l in range(k,d_w):
                        poly = np.c_[poly, X[:, i].reshape((n,1))*X[:,j].reshape((n,1))*X[:,k].reshape((n,1))*X[:,l:]]

    return poly


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def accuracy(y, y_pred):
    return np.mean(y_pred == y)
