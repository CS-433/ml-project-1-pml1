import numpy as np
from helpers import *
from tqdm import tqdm

def compute_mse(y, tx, w):
    """
    Compute the loss for linear regression (Mean Square Error)

    :param y: labels for prediction
    :param tx: features (input) for prediction
    :param w: weights of the model
    :return:
        out: the loss for the current w
    """
    return 0.5 * np.mean((y - tx@w)**2)

def compute_gradient(y, tx, w):
    """
    Compute the gradient for linear regression
    
    :param y: labels for prediction
    :param tx: features (input) for prediction
    :param w: weights of the model
    :return:
        out: the gradient for the current w
    """
    e = y - np.dot(tx, w)
    return -np.dot(tx.T, e) / len(y)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    :param y: labels for prediction
    :param tx: features (input) for prediction
    :param initial_w: initial weights of the model
    :param max_iters: number of iterations
    :param gamma: learning rate
    :return:
        w: the best weights (resulting the smallest loss) during training
        loss: training loss (MSE) resulted from the best weights
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)
    loss = compute_mse(y, tx, w)
    return w, loss

def batch_iter(y, tx, batch_size=1, num_batches=2500, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    
    :param y: labels for prediction
    :param tx: features (input) for prediction
    :param batch_size: number of samples per batch
    :param num_batches: number of batches per iteration
    :param shuffle: whether or not shuffle the dataset before yielding batches
    :yield: [y_batch, tx_batch]: 
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    :param y: labels for prediction
    :param tx: features (input) for prediction
    :param initial_w: initial weights of the model
    :param max_iters: number of iterations (= the number of epochs here)
    :param gamma: learning rate
    :return:
        w: the best weights (resulting the smallest loss) during training
        loss: training loss (MSE) resulted from the best weights
    """

    w = initial_w
    for n_iters in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=tx.shape[0]):
            w = w - gamma * compute_gradient(y_batch, tx_batch, w)
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares_SGD_batch(y, tx, initial_w, max_iters, gamma,batch_size ):
    """
    Linear regression using stochastic gradient descent.

    :param y: labels for prediction
    :param tx: features (input) for prediction
    :param initial_w: initial weights of the model
    :param max_iters: number of iterations (= the number of epochs here)
    :param gamma: learning rate
        :param batch_size: number of samples per batch
    :return:
        w: the best weights (resulting the smallest loss) during training
        loss: training loss (MSE) resulted from the best weights
    """
    w = initial_w
    for n_iters in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=int(tx.shape[0]/batch_size)):
            w = w - gamma * compute_gradient(y_batch, tx_batch, w)
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    mse = compute_mse(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    a = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    mse = compute_mse(y, tx, w)
    return w, mse

def split_data(x, y, ratio, seed):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
        
    """
    # set seed
    np.random.seed(seed)
    # split the data based on the given ratio
    ind = np.random.permutation(len(y))
    num_tr = int(np.floor(len(y) * ratio))
    x_tr = x[ind[:num_tr]]
    x_te = x[ind[num_tr:]]
    y_tr = y[ind[:num_tr]]
    y_te = y[ind[num_tr:]]
    return x_tr, x_te, y_tr, y_te

def hyperparameter_tuning_ridge_regression(x, y, degree, ratio, seed, lambdas):
    """select hyperparameter for ridge regression"""

    # split the data, and return train and test data
    x_tr, x_te, y_tr, y_te = split_data(x, y, ratio, seed)
    
    # form train and test data with polynomial basis function
    # tx_tr = build_poly(x_tr, degree)
    # tx_te = build_poly(x_te, degree)
    tx_tr = x_tr
    tx_te = x_te

    # rmse_tr = []
    # rmse_te = []
    mse_te = []
    for ind, lambda_ in enumerate(lambdas):
        w, _ = ridge_regression(y_tr, tx_tr, lambda_ )
        mse_te.append(compute_mse(y_te, tx_te, w))
    
    index_min = np.argmin(mse_te)
    lambda_ = lambdas[index_min]
    return lambda_

def sigmoid(x):
    #sup = 1.*(x>=0)
    #inf = 1-sup
    #return sup*1./(1.+np.exp(-x*sup)) + inf*np.exp(inf*x)/(np.exp(inf*x)+1.)
    exp_x = np.exp(x)
    return exp_x / (1 + exp_x)


def logit_loss(y,tx,w):
    #X_tx = tx@w
   # return ((X_tx<=100)*np.log(1+np.exp(X_tx)) + ((X_tx>100) - y)*X_tx).mean()
    pred = tx @ w
    return np.mean(np.log(1 + np.exp(pred)) - y * pred)

def compute_gradient_logit(y,tx,w):
    #return tx.T@(sigmoid(tx@w)-y)/len(y) 
    return np.mean(tx * (sigmoid(tx@w) - y)[:, np.newaxis], axis=0)


def logistic_regression(y, tx, initial_w ,max_iters,gamma):
    w = initial_w
    for n_iters in range(max_iters):
        w = w - gamma * compute_gradient_logit(y, tx, w)
    loss = logit_loss(y, tx, w)
    return w, loss

def compute_gradient_reg_logit(y,tx,lambda_,w):
    return tx.T@(sigmoid(tx@w)-y) + lambda_*w

def reg_logit_loss(y,tx,lambda_,w):
    X_tx = tx@w 
    return ((X_tx<=100)*np.log(1+np.exp((X_tx<=100)*X_tx)) + ((X_tx>100) - (1+y)/2)*X_tx).sum() + lambda_/2*np.linalg.norm(w)


def reg_logistic_regression(y, tx,lambda_, initial_w ,max_iters , gamma):
    w = initial_w
    for n_iters in range(max_iters):
        w = w - gamma * compute_gradient_reg_logit(y, tx,lambda_, w)
    loss = reg_logit_loss(y, tx,lambda_, w)

    return w, loss


def training_procedure(y, tx, initial_gamma, initial_w = "Gaussian", type_ = "GD", num_iterations = 10, increase_limit = 0, verbose = True,lambda_ = None,batch_size=500):
    
    gamma = initial_gamma
    
    if initial_w == "Gaussian":
        w = np.random.normal(scale = 1/np.sqrt(tx.shape[0]), size=(tx.shape[1]))
    
    losses = []
    best_loss = 1e10
    best_w = w
    num_increases = 0
    for i in tqdm(range(num_iterations)):
        
        if type_ == "GD":
            w, loss = least_squares_GD(y, tx, w, 1, gamma)
        elif type_ == "logistic":
            w, loss = logistic_regression(y, tx, w, 1, gamma)
        elif type_ == "SGD_batch":
            w, loss = least_squares_SGD_batch(y, tx, w, 1, gamma,batch_size )
            
        if i > 0 and losses[-1] < loss:
            num_increases += 1
            if num_increases > increase_limit :
                gamma = gamma / 5
        else:
            num_increases = 0
            if loss < best_loss:
                best_loss = loss
                best_w = w
        losses.append(loss)
        if verbose and i%500 == 0:
            print("----------------------------------------------------")
            print(f"At iteration {i}, the loss is {loss:.6f} while the best loss is {best_loss:.6f}.")
            print("----------------------------------------------------")
            if i%500==0:
                y_pred = tx @ w
                y_pred[y_pred > 0] = 1
                y_pred[y_pred < 0] = -1
                print('Accuracy:', np.mean(y_pred == y))
            
    return best_w, w, losses


