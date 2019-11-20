# -*- coding: utf-8 -*-
"""some helper functions """
import csv
import numpy as np


def load_data(data_path, sub_sample=False):
    """
    Loads data and returns a matrix with 1000 rows corresponding to items and 10 000 colums to users. 
    """
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
