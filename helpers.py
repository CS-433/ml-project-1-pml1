import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools
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


def standardize(x):
    """Standardize the original data set."""
    #mean_x = np.mean(x,axis = 0)
    std_x = np.std(x,  axis = 0)
    x = (x - mean_x) / std_x
    return x, mean_x, std_x

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
    
    y_pred = np.dot(X, w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def sin_cos(X, frequence, degree,monome=False,d_m=0):
    
    poly = []
    for f in frequence:
        for i in range(X.shape[1]):
            poly.append(np.sin(f*X[:,i]))
    return build_poly_cross_terms(np.array(poly).T,degree,monome,d_m)


def build_derived_quantities(X,model,concatenate = True):
    
    DER = [X[:,13]*np.cos(X[:,15]),X[:,13]*np.sin(X[:,15]),
           X[:,13]*np.cosh(X[:,14]),X[:,13]*np.sinh(X[:,14]),
           X[:,16]*np.cos(X[:,18]),X[:,16]*np.sin(X[:,18]),
           X[:,16]*np.cosh(X[:,17]),X[:,16]*np.sinh(X[:,17]),
           X[:,13]*np.cos(X[:,15]),X[:,13]*np.sin(X[:,15]),
           X[:,13]*np.cosh(X[:,14]),X[:,13]*np.sinh(X[:,14]),
           X[:,19]*np.cos(X[:,20]),X[:,19]*np.sin(X[:,20]),
           2*np.arctan(np.exp(-X[:,14])),
           2*np.arctan(np.exp(-X[:,17]))
           ]
    
    if model != 0:
        DER.append(X[:,23]*np.cos(X[:,25]))#45
        DER.append(X[:,23]*np.sin(X[:,25]))
        DER.append(X[:,23]*np.cosh(X[:,24]))
        DER.append(X[:,23]*np.sinh(X[:,24]))
        DER.append(2*np.arctan(np.exp(-X[:,24])))#49
        
    if model>1:
        DER.append(X[:,26]*np.cos(X[:,28]))#50
        DER.append(X[:,26]*np.sin(X[:,28]))
        DER.append(X[:,26]*np.cosh(X[:,27]))
        DER.append(X[:,26]*np.sinh(X[:,27]))
        DER.append(2*np.arctan(np.exp(-X[:,27])))#54
    
    for i in range(len(DER)):
        DER[i] = DER[i].reshape(X.shape[0],1)
    if concatenate:   
        return np.concatenate([X,np.concatenate(DER,axis = 1)],axis = 1)
    else:
        return np.concatenate(DER,axis = 1)


def build_add_minus_term(X):
    
    X = np.concatenate([np.zeros((X.shape[0],1)),X],axis =1)
    poly = []
    
    for ind in itertools.combinations(X.T,2):
        poly.append(functools.reduce(lambda x,y:x+y,ind))
        poly.append(functools.reduce(lambda x,y:x-y,ind))

    return np.array(poly).T[:,1:]


def build_poly_cross_terms(X, d, add_higher_monome = False, d_monome = 2):
    
    X = np.concatenate([np.ones((X.shape[0],1)),X],axis =1)
    poly = []
    
    if d != 0:
        for ind in itertools.combinations_with_replacement(X.T,d):
            poly.append(functools.reduce(lambda x,y:x*y,ind))

    if add_higher_monome:
        for i in range(d+1,d+d_monome+1):
            for j in range(1,X.shape[1]):
                poly.append(X[:,j]**i)
    return np.array(poly).T[:,1:]


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

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    return dimension: k-fold * interval
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)] 
    return np.array(k_indices)

def hyperparameter_tuning_GD(tx, y, k_fold, seed, initial_w, max_iters, gammas):
    """
    select hyperparameter for stochastic gradient descent

    Args:
        tx: numpy array of shape (N,D), D is the number of features.
        y: numpy array of shape (N,), N is the number of samples.
        k-fold: integer.
        seed: integer.
        initial_w: initial weight, shape=(D,).
        max_iters: number of iteration.
        gammas: set of tested step size.

    Returns:
        gamma: optimal step size to achive high accuracy for testing set
    """
    acc = []
    for ind, gamma_ in enumerate(gammas):
        k_indices = build_k_indices(y, k_fold, seed)

        val_acc = []
        for k in range(k_fold):
            val_y = y[k_indices[k, :]]
            val_x = tx[k_indices[k, :]]
            train_idx = np.vstack([k_indices[:k, :], k_indices[k+1:, :]]).flatten()
            train_y = y[train_idx]
            train_x = tx[train_idx]

            w, _ = least_squares_GD(train_y, train_x, initial_w, max_iters, gamma_)
            pred = val_x @ w
            pred[pred > 0] = 1
            pred[pred < 0] = -1
            val_acc.append(np.mean(pred == val_y))

        acc.append(sum(val_acc)/len(val_acc))
    
    index_min = np.argmin(acc)
    gamma_ = gammas[index_min]
    return gamma_
