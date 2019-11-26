# -*- coding: utf-8 -*-
"""some helper functions """
import csv
import numpy as np
import scipy.sparse as sp

"""
def load_data(data_path, sub_sample=False):

    #Loads data and returns a matrix with 1000 rows corresponding to items and 10 000 colums to users. 

    X = np.zeros((1000,10000))
    ids= np.genfromtxt(data_path, delimiter =",",skip_header=1, dtype=str,usecols=0)
    users=np.zeros(ids.shape[0])
    items=np.zeros(ids.shape[0])
    rates= np.genfromtxt(data_path, delimiter =",",skip_header=1, dtype=int,usecols=1)

    for i in range (ids.shape[0]) :
        users[i] = (ids[i])[1:ids[i].find("_")]
        items[i] = (ids[i])[ids[i].find("c")+1:]
        
    for i in range(rates.shape[0]) :
        X[int(items[i])-1,int(users[i])-1]=rates[i]
        
    return X
"""

def create_csv_submission(ids, pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each rating prediction (user and item))
               pred (predicted rating)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
# Helper functions from exercise 10

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

"""
# Functions from Exercise 10 that were not used 

#from itertools import groupby
def group_by(data, index):
    #group list of list by a specific index.
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    #build groups for nnz rows and cols.
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

"""

