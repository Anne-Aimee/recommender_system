# -*- coding: utf-8 -*-
"""some helper functions """
import csv
import numpy as np


def load_data(data_path, sub_sample=False):
    """
    Loads data and returns r (user labels), c (item labels) and rating.
    rating[n] is the rating of user[n] for item[n]
    """
    raise NotImplementedError
    return r, c, rating

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
