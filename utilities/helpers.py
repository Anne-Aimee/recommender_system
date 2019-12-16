# -*- coding: utf-8 -*-
"""some helper functions """
import csv
import numpy as np
import scipy.sparse as sp
from math import sqrt


def create_csv_submission(ids, pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each rating prediction (user and item))
               pred (predicted rating)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, pred):
            #writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            writer.writerow({'Id':r1,'Prediction':str(r2)})
            

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_col,max_row))

    # build rating matrix.
    ratings = sp.lil_matrix((max_col,max_row))
    for row, col, rating in data:
        ratings[col - 1,row - 1] = rating
    return ratings

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1, seed = 988):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(seed)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users]  
    
    # split the data and return train and test data.
    # we only consider users and movies that have more than 10 ratings

    ind_test = np.random.choice(valid_ratings.nnz, int(valid_ratings.nnz*p_test), replace=False)
    ind_train = np.delete(np.arange(valid_ratings.nnz),ind_test)
    
    valid_ratings_coo = valid_ratings.tocoo()
    data = valid_ratings_coo.data
    row = valid_ratings_coo.row
    col = valid_ratings_coo.col
    
    test = sp.coo_matrix((data[ind_test], (row[ind_test], col[ind_test])), shape=valid_ratings.get_shape())
    train = sp.coo_matrix((data[ind_train], (row[ind_train], col[ind_train])), shape=valid_ratings.get_shape()) 
    
    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""

    # calculate rmse (we only consider nonzero entries.)
    approx_data_matrix = np.dot(item_features.T,user_features)
    return sqrt(calculate_mse(data,approx_data_matrix[nz])/(len(data)))

def compute_prediction_baseline_average_item_user_offset(ratings, shape_valid_ratings, item_features_baseline_corr,
                                                         user_features_baseline_corr, num_items_per_user,
                                                         num_users_per_item, min_num_ratings, 
                                                         better_average_movie_ratings, better_average_user_offsets):

    # Prediction of the valid part
    rows, cols = np.indices(shape_valid_ratings)
    baseline = better_average_movie_ratings[rows] + better_average_user_offsets[cols]
    matrix_factorisation = np.dot(item_features_baseline_corr.T,user_features_baseline_corr)
    
    prediction_valid =  baseline + matrix_factorisation
    
    
    # clip values in an invalid range
    prediction_valid[prediction_valid > 5] = 5;
    prediction_valid[prediction_valid < 1] = 1;
    
    # Fill the Matrix's invalid entries
    prediction = np.zeros(ratings.shape)
    
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    
    invalid_users = np.where(num_items_per_user < min_num_ratings)[0]
    invalid_items = np.where(num_users_per_item < min_num_ratings)[0]
                        
    #prediction[valid_items, :][: , valid_users] = prediction_valid # Fancy notation dosn't work as it returns a copy
    prediction[np.ix_(valid_items,valid_users)] = prediction_valid # This line achieves what the previous can't do
    
    sum_ratings_user = np.squeeze(np.asarray(ratings.sum(0)))    # sum of the nonzero elements, for each row
    count_ratings_user = np.diff(ratings.tocsc().indptr)         # count of the nonzero elements, for each row
    average_ratings_user = sum_ratings_user/count_ratings_user
    
    for invalid_item in invalid_items:
        prediction[invalid_item, :] = average_ratings_user
    
    sum_ratings_movie = np.squeeze(np.asarray(ratings.sum(1)))    # sum of the nonzero elements, for each row
    count_ratings_movie = np.diff(ratings.tocsr().indptr)         # count of the nonzero elements, for each row
    average_movie_ratings = sum_ratings_movie/count_ratings_movie
    
    for invalid_user in invalid_users:
        prediction[: , invalid_user] = average_movie_ratings
    
    global_mean = np.mean(ratings.tocoo().data)
    
    for invalid_item in invalid_items:
        for invalid_item in invalid_items:
            prediction[invalid_items , invalid_users] = global_mean
            
    #prediction[ratings.nonzero()] = ratings.tocoo().data
    prediction = np.rint(prediction)
    
    return prediction
 
def compute_prediction_baseline_linear(ratings, shape_valid_ratings, item_features_baseline_corr,
                                                         user_features_baseline_corr, num_items_per_user,
                                                         num_users_per_item, min_num_ratings, 
                                                         w_item, w_user, w_0):

   


    # Prediction of the valid part
    rows, cols = np.indices(shape_valid_ratings)
    baseline = w_item[rows] + w_user[cols] + w_0
    matrix_factorisation = np.dot(item_features_baseline_corr.T,user_features_baseline_corr)
    
    prediction_valid =  baseline + matrix_factorisation
    
    
    # clip values in an invalid range
    prediction_valid[prediction_valid > 5] = 5;
    prediction_valid[prediction_valid < 1] = 1;
    
    # Fill the Matrix's invalid entries
    prediction = np.zeros(ratings.shape)
    
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    
    invalid_users = np.where(num_items_per_user < min_num_ratings)[0]
    invalid_items = np.where(num_users_per_item < min_num_ratings)[0]
                        
    #prediction[valid_items, :][: , valid_users] = prediction_valid # Fancy notation dosn't work as it returns a copy
    prediction[np.ix_(valid_items,valid_users)] = prediction_valid # This line achieves what the previous can't do
    
    sum_ratings_user = np.squeeze(np.asarray(ratings.sum(0)))    # sum of the nonzero elements, for each row
    count_ratings_user = np.diff(ratings.tocsc().indptr)         # count of the nonzero elements, for each row
    average_ratings_user = sum_ratings_user/count_ratings_user
    
    for invalid_item in invalid_items:
        prediction[invalid_item, :] = average_ratings_user
    
    sum_ratings_movie = np.squeeze(np.asarray(ratings.sum(1)))    # sum of the nonzero elements, for each row
    count_ratings_movie = np.diff(ratings.tocsr().indptr)         # count of the nonzero elements, for each row
    average_movie_ratings = sum_ratings_movie/count_ratings_movie
    
    for invalid_user in invalid_users:
        prediction[: , invalid_user] = average_movie_ratings
    
    global_mean = np.mean(ratings.tocoo().data)
    
    for invalid_item in invalid_items:
        for invalid_item in invalid_items:
            prediction[invalid_items , invalid_users] = global_mean
            
    #prediction[ratings.nonzero()] = ratings.tocoo().data
    prediction = np.rint(prediction)
    
    return prediction
 


