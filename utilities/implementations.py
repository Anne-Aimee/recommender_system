# -*- coding: utf-8 -*-
"""some matrix factorization functions """

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from math import sqrt
from helpers import *

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    
    # returns initialized with random values :
    #     user_features: shape = num_features, num_user
    #     item_features: shape = num_features, num_item

    
    max_initial_value = 2*sqrt(np.mean(train.data)/num_features)
    
    user_features = max_initial_value*np.random.rand(num_features, train.shape[1])
    item_features = max_initial_value*np.random.rand(num_features, train.shape[0])

    
    return user_features,item_features

#... other models require a different initialization

def init_MF_baseline(train, num_features):
    """init the parameter for matrix factorization."""
    
    # returns initialized with random values :
    #     user_features: shape = num_features, num_user
    #     item_features: shape = num_features, num_item

    max_initial_value = 0.01
    user_features = max_initial_value*(np.random.rand(num_features, train.shape[1])-0.5)
    item_features = max_initial_value*(np.random.rand(num_features, train.shape[0])-0.5)
    
    return user_features,item_features

#SGD
def matrix_factorization_SGD(train, test):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.05
    num_features = 25   # K in the lecture notes
    num_epochs = 15     # number of full iterations through the train set
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col, train.data))

    print("learn the matrix factorization using SGD...")
    rmse_tr = compute_error(train.data, user_features, item_features, train.nonzero())
    rmse_te = compute_error(test.data, user_features, item_features, test.nonzero())
    print("initial RMSE on training set: {}, RMSE on testing set: {}.".format(rmse_tr,rmse_te))
    
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n, x_dn in nz_train:
        # update matrix factorization.     

            item_features[:,d] += gamma*(x_dn - np.inner(item_features[:,d],user_features[:,n]))*user_features[:,n]
            user_features[:,n] += gamma*(x_dn - np.inner(item_features[:,d],user_features[:,n]))*item_features[:,d]
        
        rmse_tr = compute_error(train.data, user_features, item_features, train.nonzero())
        rmse_te = compute_error(test.data, user_features, item_features, test.nonzero())
        #print("iter: {}, RMSE on training set: {}, RMSE on testing set: {}.".format(it, rmse_tr,rmse_te))
        
        errors.append(rmse_te)

    # evaluate the test error.
    rmse = compute_error(test.data, user_features, item_features, test.nonzero())
    print("RMSE on test data: {}.".format(rmse))

def matrix_factorization_SGD_regularized(train, test, num_features, lambda_user, lambda_item, gamma, 
                                         gamma_dec_step_size, num_epochs, seed, stop_criterion, baseline=False):
    """matrix factorization by SGD."""
    
    # set seed
    np.random.seed(seed)

    # init matrix
    if (baseline):
        user_features, item_features = init_MF_baseline(train, num_features)
    else :
        user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices  
    nz_train = list(zip(train.row, train.col, train.data))
    
    print("learn the matrix factorization using SGD...")
    rmse_tr = [compute_error(train.data, user_features, item_features, train.nonzero())]
    rmse_te = [compute_error(test.data, user_features, item_features, test.nonzero())]
    print("initial RMSE on training set: {}, RMSE on testing set: {}.".format(rmse_tr[0],rmse_te[0]))
    
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= gamma_dec_step_size
        
        for d, n, x_dn in nz_train:
        # update matrix factorization.

            item_features[:,d] += gamma*((x_dn - np.inner(item_features[:,d],user_features[:,n]))*user_features[:,n]-lambda_item*item_features[:,d])
            user_features[:,n] += gamma*((x_dn - np.inner(item_features[:,d],user_features[:,n]))*item_features[:,d]-lambda_user*user_features[:,n])
        
            """
            item_features[:,d] += gamma*((x_dn - np.inner(item_features[:,d],user_features[:,n]))*user_features[:,n]-lambda_item*item_features[:,d])
            user_features[:,n] += gamma*((x_dn - np.inner(item_features[:,d],user_features[:,n]))*item_features[:,d]-lambda_user*user_features[:,n])
            """
        
        rmse_tr.append(compute_error(train.data, user_features, item_features, train.nonzero()))
        rmse_te.append(compute_error(test.data, user_features, item_features, test.nonzero()))
        print("iter: {}, RMSE on training set: {}, RMSE on testing set: {}.".format(it, rmse_tr[-1],rmse_te[-1]))
        
        #if np.isclose(rmse_te[-1],rmse_te[-2],atol=stop_criterion,rtol=stop_criterion) or rmse_tr[-1] > rmse_tr[0]:
        if rmse_te[-1] > rmse_te[-2]:    
            break
            
    # evaluate the test error.
    min_rmse_te = min(rmse_te)
    print("RMSE on test data: {}.".format(min_rmse_te))
    
    return min_rmse_te


#ALS

def update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""

    # update and return user feature.
    user_features = np.zeros((item_features.shape[0],train.shape[1]))
    
    for n in range(train.shape[1]):
        
        item_features_n = np.zeros(item_features.shape)
        item_features_n[:,nz_user_itemindices[n]] = item_features[:,nz_user_itemindices[n]]
        user_features[:,n] = np.linalg.solve(np.dot(item_features_n,item_features.T) + lambda_user * nnz_items_per_user[n] * np.identity(user_features.shape[0]), np.dot(item_features,np.squeeze(np.asarray(train.getcol(n).todense()))))
    
    return user_features

def update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""

    # update and return item feature.
    item_features = np.zeros((user_features.shape[0],train.shape[0]))
    
    for d in range(train.shape[0]):
        
        user_features_d = np.zeros(user_features.shape)
        user_features_d[:,nz_item_userindices[d]] = user_features[:,nz_item_userindices[d]]
        item_features[:,d] = np.linalg.solve(np.dot(user_features_d,user_features.T) + lambda_item * nnz_users_per_item[d] * np.identity(user_features.shape[0]), np.dot(user_features,np.squeeze(np.asarray(train.getrow(d).todense()))))
    
    return item_features

def matrix_factorization_SGD_regularized_predict(train, test, num_features, lambda_user, lambda_item, gamma, gamma_dec_step_size, num_epochs, seed, stop_criterion,baseline=False):
    """matrix factorization by SGD."""
    
    # set seed
    np.random.seed(seed)

    # init matrix
    if (baseline) :
        user_features, item_features = init_MF_baseline(train, num_features)
    else :
        user_features, item_features = init_MF(train, num_features)
    best_user_features = np.copy(user_features)
    best_item_features = np.copy(item_features)
    
    # find the non-zero ratings indices  
    nz_train = list(zip(train.row, train.col, train.data))
    
    print("learn the matrix factorization using SGD...")
    rmse_tr = [compute_error(train.data, user_features, item_features, (train.row,train.col))]
    rmse_te = [compute_error(test.data, user_features, item_features, (test.row,test.col))]
    print("initial RMSE on training set: {}, RMSE on testing set: {}.".format(rmse_tr[0],rmse_te[0]))
    
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= gamma_dec_step_size
        
        for d, n, x_dn in nz_train:
        # update matrix factorization.

            item_features[:,d] += gamma*((x_dn - np.inner(item_features[:,d],user_features[:,n]))*user_features[:,n]-lambda_item*item_features[:,d])
            user_features[:,n] += gamma*((x_dn - np.inner(item_features[:,d],user_features[:,n]))*item_features[:,d]-lambda_user*user_features[:,n])
        
        rmse_tr.append(compute_error(train.data, user_features, item_features, (train.row,train.col)))
        rmse_te.append(compute_error(test.data, user_features, item_features, (test.row,test.col)))
        print("iter: {}, RMSE on training set: {}, RMSE on testing set: {}.".format(it, rmse_tr[-1],rmse_te[-1]))
        
        if rmse_te[-1] < min(rmse_te[:-1]):
            best_user_features = np.copy(user_features)
            best_item_features = np.copy(item_features)
            
        if np.isclose(rmse_te[-1],rmse_te[-2],atol=stop_criterion,rtol=stop_criterion) or rmse_tr[-1] > rmse_tr[0]:
            break
            
    # evaluate the test error.
    min_rmse_te = min(rmse_te)
    print("RMSE on test data: {}.".format(min_rmse_te))
    
    return min_rmse_te, best_user_features, best_item_features

def ALS(train, test, num_features, lambda_user, lambda_item, max_iter, seed, stop_criterion, baseline=False):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    
    # set seed
    np.random.seed(seed)

    # init ALS
    if (baseline) :
        user_features, item_features = init_MF_baseline(train, num_features)
    else :
        user_features, item_features = init_MF(train, num_features)
    # start you ALS-WR algorithm. 
    
    nz_row, nz_col = train.nonzero()
    
    nz_user_itemindices = [nz_row[nz_col==n] for n in range(train.shape[1])]
    nnz_items_per_user = np.array([len(nz_user_itemindice) for nz_user_itemindice in nz_user_itemindices])
    nz_item_userindices = [nz_col[nz_row==d] for d in range(train.shape[0])]
    nnz_users_per_item = np.array([len(nz_item_userindice) for nz_item_userindice in nz_item_userindices])
    
    rmse_tr = [compute_error(train.data, user_features, item_features, train.nonzero())]
    rmse_te = [compute_error(test.data, user_features, item_features, test.nonzero())]
    print("initial: RMSE on training set: {}, RMSE on testing set: {}.".format(rmse_tr[0],rmse_te[0]))
    

    for it in range(max_iter):
        
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        rmse_tr.append(compute_error(train.data, user_features, item_features, train.nonzero()))
        rmse_te.append(compute_error(test.data, user_features, item_features, test.nonzero()))
        print("iter: {}, RMSE on training set: {}, RMSE on testing set: {}.".format(it, rmse_tr[-1],rmse_te[-1]))
        
        if np.isclose(rmse_tr[-1],rmse_tr[-2],stop_criterion) or rmse_tr[-1] > rmse_tr[0]:
            break
        
    min_rmse_te = min(rmse_te)
    print("RMSE on test data: {}.".format(min_rmse_te))
    
    return min_rmse_te

def ALS_pred(train, test, num_features, lambda_user, lambda_item, max_iter, seed, stop_criterion):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    
    # set seed
    np.random.seed(seed)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    best_user_features = np.copy(user_features)
    best_item_features = np.copy(item_features)
    
    # start you ALS-WR algorithm. 
    
    nz_row, nz_col = train.nonzero()
    
    nz_user_itemindices = [nz_row[nz_col==n] for n in range(train.shape[1])]
    nnz_items_per_user = np.array([len(nz_user_itemindice) for nz_user_itemindice in nz_user_itemindices])
    nz_item_userindices = [nz_col[nz_row==d] for d in range(train.shape[0])]
    nnz_users_per_item = np.array([len(nz_item_userindice) for nz_item_userindice in nz_item_userindices])
    
    rmse_tr = [compute_error(train.data, user_features, item_features, train.nonzero())]
    rmse_te = [compute_error(test.data, user_features, item_features, test.nonzero())]
    print("initial: RMSE on training set: {}, RMSE on testing set: {}.".format(rmse_tr[0],rmse_te[0]))
    

    for it in range(max_iter):
        
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        
        
        rmse_tr.append(compute_error(train.data, user_features, item_features, train.nonzero()))
        rmse_te.append(compute_error(test.data, user_features, item_features, test.nonzero()))
        print("iter: {}, RMSE on training set: {}, RMSE on testing set: {}.".format(it, rmse_tr[-1],rmse_te[-1]))
        
        if rmse_te[-1] < min(rmse_te[:-1]):
            best_user_features = np.copy(user_features)
            best_item_features = np.copy(item_features)
        
        if np.isclose(rmse_tr[-1],rmse_tr[-2], atol=stop_criterion, rtol=stop_criterion) or rmse_tr[-1] > rmse_tr[0]:
            break
        
    min_rmse_te = min(rmse_te)
    print("RMSE on test data: {}.".format(min_rmse_te))
    
    return min_rmse_te, best_user_features, best_item_features

# PARAMETER TUNING

def gradient_descent_tuning_parameters_MF_SGD(train, test, num_features_init, lambda_user_init, lambda_item_init,
                                              gamma_init, gamma_dec_step_size_init, seed, stop_criterion, lr, dec_lr,
                                              dx, max_iter, num_epochs,baseline=False):
    """gradient descent to determine hyperparameters"""

    # set initial conditions
    num_features = [num_features_init]   # K in the lecture notes
    lambda_user = [lambda_user_init]
    lambda_item = [lambda_item_init]
    gamma = [gamma_init]
    gamma_dec_step_size = [gamma_dec_step_size_init]

    min_rmse_te = [matrix_factorization_SGD_regularized(train, test, num_features[-1], lambda_user[-1], 
                                                        lambda_item[-1], gamma[-1], gamma_dec_step_size[-1], 
                                                        num_epochs, seed, stop_criterion,baseline=False)]
    
    # num_features takes discrete values and is in a different order of magnitude

    for it in range(max_iter):
    
        #compute approximation of gradient
        min_rmse_te_num_features = matrix_factorization_SGD_regularized(train, test, int(num_features[-1] + np.rint(5000*dx)), 
                                                                        lambda_user[-1], lambda_item[-1], gamma[-1], 
                                                                        gamma_dec_step_size[-1], num_epochs, seed, 
                                                                        stop_criterion,baseline=False)
        min_rmse_te_lambda_user = matrix_factorization_SGD_regularized(train, test, num_features[-1], lambda_user[-1]+dx, 
                                                                       lambda_item[-1], gamma[-1], gamma_dec_step_size[-1], 
                                                                       num_epochs, seed, stop_criterion,baseline)
        min_rmse_te_lambda_item = matrix_factorization_SGD_regularized(train, test, num_features[-1], lambda_user[-1],
                                                                       lambda_item[-1]+dx, gamma[-1], 
                                                                       gamma_dec_step_size[-1], num_epochs, seed,
                                                                       stop_criterion,baseline=False)
        min_rmse_te_gamma = matrix_factorization_SGD_regularized(train, test, num_features[-1], lambda_user[-1], 
                                                                 lambda_item[-1], gamma[-1]+dx, gamma_dec_step_size[-1],
                                                                 num_epochs, seed,stop_criterion,baseline=False)
                                                             
        min_rmse_te_dec_step_size = matrix_factorization_SGD_regularized(train, test, num_features[-1], lambda_user[-1], 
                                                                         lambda_item[-1], gamma[-1], 
                                                                         gamma_dec_step_size[-1]+dx, num_epochs,
                                                                         seed, stop_criterion,baseline=False)
    
        min_rmse_te_delta_num_features = min_rmse_te_num_features - min_rmse_te[-1]
        min_rmse_te_delta_lambda_user = min_rmse_te_lambda_user - min_rmse_te[-1]
        min_rmse_te_delta_lambda_item = min_rmse_te_lambda_item - min_rmse_te[-1]
        min_rmse_te_delta_gamma = min_rmse_te_gamma - min_rmse_te[-1]
        min_rmse_te_delta_dec_step_size = min_rmse_te_dec_step_size - min_rmse_te[-1]
    
        # Normalization of gradient
        length_gradient = sqrt(min_rmse_te_delta_num_features**2 + min_rmse_te_delta_lambda_user**2 +
                               min_rmse_te_delta_lambda_item**2 + min_rmse_te_delta_gamma**2 + 
                               min_rmse_te_delta_dec_step_size**2)
    
        # Update parameters
        num_features.append(max(0,int(num_features[-1] - np.rint(2000*lr*min_rmse_te_delta_num_features/length_gradient))))
        lambda_user.append(lambda_user[-1] - lr*min_rmse_te_delta_lambda_user/length_gradient) 
        lambda_item.append(lambda_item[-1] - lr*min_rmse_te_delta_lambda_item/length_gradient) 
        gamma.append(gamma[-1] - lr*min_rmse_te_delta_gamma/length_gradient)
        gamma_dec_step_size.append(gamma_dec_step_size[-1] - lr*min_rmse_te_delta_dec_step_size/length_gradient) 
    
        # compute new loss
        min_rmse_te.append(matrix_factorization_SGD_regularized(train, test, num_features[-1], lambda_user[-1], 
                                                                lambda_item[-1], gamma[-1], gamma_dec_step_size[-1],
                                                                num_epochs, seed, stop_criterion,baseline=False))
    
        print("it = ({}/{}), rmse = {}, num_features = {}, lambda_user = {}, lambda_item = {}, gamma = {}, dec_step_size = {}"
              .format(it+1,max_iter,min_rmse_te[-1],num_features[-1],lambda_user[-1],lambda_item[-1],gamma[-1],
                      gamma_dec_step_size[-1]))
    
        # Decrease learning rate
        lr /= dec_lr
        dx /= dec_lr
    
        if np.isclose(min_rmse_te[-1],min_rmse_te[-2],stop_criterion):
            break
    return min_rmse_te, num_features, lambda_user, lambda_item, gamma, gamma_dec_step_size

# SGD sigmoid

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.nan_to_num(np.exp(-t)))

def sigmoid_inv(t):
    """apply the inverse sigmoid function on t."""
    return -np.log(-4/(1-t)-1)

def sigmoid_customized(t,shift):
    """apply sigmoid function on t."""
    return 4*sigmoid(t+shift)+1
    
def grad_sigmoid_customized(t,shift):
    """return the gradient of sigmoid on t."""
    return 4*sigmoid(t+shift)*(1-sigmoid(t+shift))

def init_MF_sigmoid(train, num_features):
    """init the parameter for matrix factorization."""
    
    # returns initialized with random values :
    #     user_features: shape = num_features, num_user
    #     item_features: shape = num_features, num_item
    
    
    max_initial_value = 0.01
    user_features = max_initial_value*(np.random.rand(num_features, train.shape[1])-0.5)
    item_features = max_initial_value*(np.random.rand(num_features, train.shape[0])-0.5)

    return user_features,item_features

def compute_error_sigmoid(data, user_features, item_features, nz,shift):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # calculate rmse (we only consider nonzero entries.)
    approx_data_matrix = sigmoid_customized(np.dot(item_features.T,user_features),shift)
    return sqrt(calculate_mse(data,approx_data_matrix[nz])/(len(data)))

def matrix_factorization_SGD_regularized_sigmoid_predict(train, test, num_features, lambda_user, lambda_item, gamma, gamma_dec_step_size, num_epochs, seed, stop_criterion, shift):
    """matrix factorization by SGD with sigmoid on output."""
    
    # set seed
    np.random.seed(seed)

    # init matrix
    user_features, item_features = init_MF_sigmoid(train, num_features)
    best_user_features = np.copy(user_features)
    best_item_features = np.copy(item_features)
    
    # find the non-zero ratings indices  
    nz_train = list(zip(train.row, train.col, train.data))
    
    print("learn the matrix factorization using SGD...")
    rmse_tr = [compute_error_sigmoid(train.data, user_features, item_features, (train.row,train.col),shift)]
    rmse_te = [compute_error_sigmoid(test.data, user_features, item_features, (test.row,test.col),shift)]
    print("initial RMSE on training set: {}, RMSE on testing set: {}.".format(rmse_tr[0],rmse_te[0]))
    
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= gamma_dec_step_size
        
        for d, n, x_dn in nz_train:
        # update matrix factorization.
               
            item_features_d = np.copy(item_features[:, d])
            err = x_dn - sigmoid_customized(np.inner(item_features[:,d],user_features[:,n]),shift)
            item_features[:,d] += gamma*(err*grad_sigmoid_customized(err,shift)*user_features[:,n] - lambda_item*item_features[:,d])
            user_features[:,n] += gamma*(err*grad_sigmoid_customized(err,shift)*item_features_d - lambda_user*user_features[:,n])
            
            """
            item_features[:,d] += gamma*((x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*grad_sigmoid(x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*user_features[:,n] - lambda_item*item_features[:,d])
            user_features[:,n] += gamma*((x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*grad_sigmoid(x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*item_features[:,d] - lambda_user*user_features[:,n])
            """ 
            
        rmse_tr.append(compute_error_sigmoid(train.data, user_features, item_features, (train.row,train.col),shift))
        rmse_te.append(compute_error_sigmoid(test.data, user_features, item_features, (test.row,test.col),shift))
        print("iter: {}, RMSE on training set: {}, RMSE on testing set: {}.".format(it, rmse_tr[-1],rmse_te[-1]))
        
        if rmse_te[-1] < min(rmse_te[:-1]):
            best_user_features = np.copy(user_features)
            best_item_features = np.copy(item_features)
            
        if np.isclose(rmse_te[-1],rmse_te[-2],atol=stop_criterion,rtol=stop_criterion) or rmse_tr[-1] > rmse_tr[0]:
            break
            
    # evaluate the test error.
    min_rmse_te = min(rmse_te)
    print("RMSE on test data: {}.".format(min_rmse_te))
    
    return min_rmse_te, best_user_features, best_item_features

def matrix_factorization_SGD_regularized_sigmoid(train, test, num_features, lambda_user, lambda_item, gamma, gamma_dec_step_size, num_epochs, seed, stop_criterion, shift):
    # matrix factorization by SGD with sigmoid on output.
    
    # set seed
    np.random.seed(seed)

    # init matrix
    user_features, item_features = init_MF_sigmoid(train, num_features)
    
    # find the non-zero ratings indices  
    nz_train = list(zip(train.row, train.col, train.data))
    
    print("learn the matrix factorization using SGD...")
    rmse_tr = [compute_error_sigmoid(train.data, user_features, item_features, (train.row,train.col),shift)]
    rmse_te = [compute_error_sigmoid(test.data, user_features, item_features, (test.row,test.col),shift)]
    print("initial RMSE on training set: {}, RMSE on testing set: {}.".format(rmse_tr[0],rmse_te[0]))
    
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= gamma_dec_step_size
        
        for d, n, x_dn in nz_train:
        # update matrix factorization.
               
            item_features_d = np.copy(item_features[:, d])
            err = x_dn - sigmoid_customized(np.inner(item_features[:,d],user_features[:,n]),shift)
            item_features[:,d] += gamma*(err*grad_sigmoid_customized(err,shift)*user_features[:,n] - lambda_item*item_features[:,d])
            user_features[:,n] += gamma*(err*grad_sigmoid_customized(err,shift)*item_features_d - lambda_user*user_features[:,n])
            
            
            #item_features[:,d] += gamma*((x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*grad_sigmoid(x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*user_features[:,n] - lambda_item*item_features[:,d])
            #user_features[:,n] += gamma*((x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*grad_sigmoid(x_dn - sigmoid(np.inner(item_features[:,d],user_features[:,n])))*item_features[:,d] - lambda_user*user_features[:,n])
            
            
        rmse_tr.append(compute_error_sigmoid(train.data, user_features, item_features, (train.row,train.col),shift))
        rmse_te.append(compute_error_sigmoid(test.data, user_features, item_features, (test.row,test.col),shift))
        print("iter: {}, RMSE on training set: {}, RMSE on testing set: {}.".format(it, rmse_tr[-1],rmse_te[-1]))
        
        if np.isclose(rmse_te[-1],rmse_te[-2],atol=stop_criterion,rtol=stop_criterion) or rmse_te[-1] > rmse_te[-2]:
            break
            
    # evaluate the test error.
    min_rmse_te = min(rmse_te)
    print("RMSE on test data: {}.".format(min_rmse_te))
    
    return min_rmse_te

def gradient_descent_tuning_parameters_MF_SGD_sigmoid(train, test, num_features_init, lambda_user_init, lambda_item_init,
                                                      gamma_init, gamma_dec_step_size_init, seed, stop_criterion, lr, dec_lr,
                                                      dx, max_iter, num_epochs):
    """gradient descent to determine hyperparameters"""

    # set initial conditions
    num_features = [num_features_init]   # K in the lecture notes
    lambda_user = [lambda_user_init]
    lambda_item = [lambda_item_init]
    gamma = [gamma_init]
    gamma_dec_step_size = [gamma_dec_step_size_init]

    min_rmse_te = [matrix_factorization_SGD_regularized_sigmoid(train, test, num_features[-1], lambda_user[-1], 
                                                        lambda_item[-1], gamma[-1], gamma_dec_step_size[-1], 
                                                        num_epochs, seed, stop_criterion,shift)]
    
    # num_features takes discrete values and is in a different order of magnitude

    for it in range(max_iter):
    
        #compute approximation of gradient
        min_rmse_te_num_features = matrix_factorization_SGD_regularized_sigmoid(train, test, int(num_features[-1] + np.rint(5000*dx)), 
                                                                        lambda_user[-1], lambda_item[-1], gamma[-1], 
                                                                        gamma_dec_step_size[-1], num_epochs, seed, 
                                                                        stop_criterion,shift)
        min_rmse_te_lambda_user = matrix_factorization_SGD_regularized_sigmoid(train, test, num_features[-1], lambda_user[-1]+dx, 
                                                                       lambda_item[-1], gamma[-1], gamma_dec_step_size[-1], 
                                                                       num_epochs, seed, stop_criterion,shift)
        min_rmse_te_lambda_item = matrix_factorization_SGD_regularized_sigmoid(train, test, num_features[-1], lambda_user[-1],
                                                                       lambda_item[-1]+dx, gamma[-1], 
                                                                       gamma_dec_step_size[-1], num_epochs, seed,
                                                                       stop_criterion,shift)
        min_rmse_te_gamma = matrix_factorization_SGD_regularized_sigmoid(train, test, num_features[-1], lambda_user[-1], 
                                                                 lambda_item[-1], gamma[-1]+dx, gamma_dec_step_size[-1],
                                                                 num_epochs, seed,stop_criterion,shift)
                                                             
        min_rmse_te_dec_step_size = matrix_factorization_SGD_regularized_sigmoid(train, test, num_features[-1], lambda_user[-1], 
                                                                         lambda_item[-1], gamma[-1], 
                                                                         gamma_dec_step_size[-1]+dx, num_epochs,
                                                                         seed, stop_criterion,shift)
    
        min_rmse_te_delta_num_features = min_rmse_te_num_features - min_rmse_te[-1]
        min_rmse_te_delta_lambda_user = min_rmse_te_lambda_user - min_rmse_te[-1]
        min_rmse_te_delta_lambda_item = min_rmse_te_lambda_item - min_rmse_te[-1]
        min_rmse_te_delta_gamma = min_rmse_te_gamma - min_rmse_te[-1]
        min_rmse_te_delta_dec_step_size = min_rmse_te_dec_step_size - min_rmse_te[-1]
    
        # Normalization of gradient
        length_gradient = sqrt(min_rmse_te_delta_num_features**2 + min_rmse_te_delta_lambda_user**2 +
                               min_rmse_te_delta_lambda_item**2 + min_rmse_te_delta_gamma**2 + 
                               min_rmse_te_delta_dec_step_size**2)
    
        # Update parameters
        num_features.append(max(0,int(num_features[-1] - np.rint(2000*lr*min_rmse_te_delta_num_features/length_gradient))))
        lambda_user.append(lambda_user[-1] - lr*min_rmse_te_delta_lambda_user/length_gradient) 
        lambda_item.append(lambda_item[-1] - lr*min_rmse_te_delta_lambda_item/length_gradient) 
        gamma.append(gamma[-1] - lr*min_rmse_te_delta_gamma/length_gradient)
        gamma_dec_step_size.append(gamma_dec_step_size[-1] - lr*min_rmse_te_delta_dec_step_size/length_gradient) 
    
        # compute new loss
        min_rmse_te.append(matrix_factorization_SGD_regularized_sigmoid(train, test, num_features[-1], lambda_user[-1], 
                                                                lambda_item[-1], gamma[-1], gamma_dec_step_size[-1],
                                                                num_epochs, seed, stop_criterion,shift))
    
        print("it = ({}/{}), rmse = {}, num_features = {}, lambda_user = {}, lambda_item = {}, gamma = {}, dec_step_size = {}"
              .format(it+1,max_iter,min_rmse_te[-1],num_features[-1],lambda_user[-1],lambda_item[-1],gamma[-1],
                      gamma_dec_step_size[-1]))
    
        # Decrease learning rate
        lr /= dec_lr
        dx /= dec_lr
    
        if np.isclose(min_rmse_te[-1],min_rmse_te[-2],stop_criterion):
            break
    return min_rmse_te, num_features, lambda_user, lambda_item, gamma, gamma_dec_step_size

