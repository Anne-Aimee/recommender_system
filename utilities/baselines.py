# -*- coding: utf-8 -*-
"""some baselines functions """
import scipy
import scipy.io
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from scipy import stats # to fit linear regression
from math import sqrt
from helpers import *


def baseline_global_mean(train, test):
    """baseline method: use the global mean."""  
    
    return sqrt(calculate_mse(test.data,np.mean(train.data))/(test.nnz))

def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    #Sum over nth user
    sum_ratings_movie = np.squeeze(np.asarray(train.sum(0)))    # sum of the nonzero elements, for each row
    count_ratings_movie = np.diff(train.tocsc().indptr)         # count of the nonzero elements, for each row
    mean_rating_movie = sum_ratings_movie/count_ratings_movie
    return sqrt(calculate_mse(test.data,mean_rating_movie[test.col])/(test.nnz))


def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    
    #Sum over dth movie
    sum_ratings_user = np.squeeze(np.asarray(train.sum(1)))    # sum of the nonzero elements, for each row
    count_ratings_user = np.diff(train.tocsr().indptr)         # count of the nonzero elements, for each row
    mean_rating_user = sum_ratings_user/count_ratings_user
    rmse_item_mean = sqrt(calculate_mse(test.data,mean_rating_user[test.row])/(test.nnz))

    return rmse_item_mean, mean_rating_user


def baseline_item_mean_blending(train, test, blending_constant):
    """baseline method: use item means as the prediction."""
    
    sum_ratings_movie = np.squeeze(np.asarray(train.sum(1)))    # sum of the nonzero elements, for each row
    count_ratings_movie = np.diff(train.tocsr().indptr)         # count of the nonzero elements, for each row
    average_movie_ratings = sum_ratings_movie/count_ratings_movie

    #fitting linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(count_ratings_movie,average_movie_ratings)
    better_average_movie_ratings = ((slope*count_ratings_movie+intercept)*blending_constant + sum_ratings_movie)/(blending_constant + count_ratings_movie)
    #better_average_movie_ratings = (np.mean(train.data)*blending_constant + sum_ratings_movie)/(blending_constant + count_ratings_movie)
    
    rmse_item_mean_blending = sqrt(calculate_mse(test.data, better_average_movie_ratings[test.row])/(test.nnz))

    return rmse_item_mean_blending, better_average_movie_ratings

def baseline_user_offset_blending(train, test, blending_constant):
    """baseline method: use item means as the prediction."""
    train_normalized = train.copy()
    train_normalized.data -= np.mean(train_normalized.data)
    sum_offsets_user = np.squeeze(np.asarray(train_normalized.sum(0)))    # sum of the nonzero elements, for each row
    count_ratings_user = np.diff(train.tocsc().indptr)         # count of the nonzero elements, for each row
    average_user_offsets = sum_offsets_user/count_ratings_user
    better_average_user_offsets = sum_offsets_user/(blending_constant + count_ratings_user)

    rmse_user_offset_blending = sqrt(calculate_mse(test.data, better_average_user_offsets[test.col]+np.mean(train.data))/(test.nnz))

    return rmse_user_offset_blending, better_average_user_offsets

def baseline_average_item_user_offset(train, test, blending_constant_item, blending_constant_user):
    """baseline method: use item means as the prediction."""
    rmse_item_mean_blending, better_average_movie_ratings = baseline_item_mean_blending(train, test, 
                                                                                        blending_constant_item)
    rmse_user_offset_blending, better_average_user_offsets = baseline_user_offset_blending(train, test, 
                                                                                           blending_constant_item)
    prediction = better_average_movie_ratings[test.row] + better_average_user_offsets[test.col]
    rmse_average_item_offset_user = sqrt(calculate_mse(test.data, prediction)/(test.nnz))
    
    return rmse_average_item_offset_user, better_average_movie_ratings, better_average_user_offsets

def baseline_item_user(train, test):
    """baseline method: find best parameters for the model y_dn = w_0 + w_item[d] + w_user[n] (D+N+1) parameters
       and make a prediction."""
    
    global_mean = np.mean(train.data)
    
    #Sum over nth user
    sum_ratings_movie = np.squeeze(np.asarray(train.sum(0)))    # sum of the nonzero elements, for each row
    count_ratings_movie = np.diff(train.tocsc().indptr)         # count of the nonzero elements, for each row
    
    #Sum over dth movie
    sum_ratings_user = np.squeeze(np.asarray(train.sum(1)))    # sum of the nonzero elements, for each col
    count_ratings_user = np.diff(train.tocsr().indptr)         # count of the nonzero elements, for each col
    
    num_items, num_users = train.shape
    
    # Constructing linear system defining the model's optimal parameters in form of a matrix
    
    # Matrix of the same shape as ratings, 1 if rating present, 0 otherwise
    mask_train = sp.coo_matrix((np.ones(train.nnz), (train.row, train.col)), shape=train.shape) 
    
    A = sp.hstack((sp.diags(count_ratings_user), mask_train))
    A = sp.vstack((A, sp.hstack((mask_train.T, sp.diags(count_ratings_movie)))))
    A = sp.hstack((A, sp.coo_matrix(np.concatenate((count_ratings_movie,count_ratings_user))).T))
    A = sp.vstack((A, sp.coo_matrix(np.ones(num_items+num_users+1))))
    
    b = np.append(np.concatenate((sum_ratings_user, sum_ratings_movie)),global_mean)
    
    # Solving the system
    x = spsolve(A.tocsc(),b)
    
    # Extracting the parameters w_0, w_item[d] and w_user[n] 
    w_item, w_user, w_0 = np.split(x,np.array([num_items,num_items+num_users]))
    

    
    rmse_te = sqrt(calculate_mse(test.data, w_item[test.row] + w_user[test.col] + w_0)/(test.nnz))
    
    return rmse_te, w_item, w_user, w_0


