# -*- coding: utf-8 -*-
"""
@author: fornax
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import os
import json

import prepare_data1 as prep

p_cols = [u'NPWD2372', u'NPWD2401', u'NPWD2402', u'NPWD2451', u'NPWD2471',
          u'NPWD2472', u'NPWD2481', u'NPWD2482', u'NPWD2491', u'NPWD2501',
          u'NPWD2531', u'NPWD2532', u'NPWD2551', u'NPWD2552', u'NPWD2561',
          u'NPWD2562', u'NPWD2691', u'NPWD2692', u'NPWD2721', u'NPWD2722',
          u'NPWD2742', u'NPWD2771', u'NPWD2791', u'NPWD2792', u'NPWD2801',
          u'NPWD2802', u'NPWD2821', u'NPWD2851', u'NPWD2852', u'NPWD2871',
          u'NPWD2872', u'NPWD2881', u'NPWD2882']


def load_data(filename='prepared', interpolate=True):
    """
    Function loads the data from a CSV file and interpolates missing values.
    
    :param filename: path to the CSV file
    :param interpolate: a flag whether to interpolate missing values
    :return: read and processed data frame, and a list of features
    """
    df = pd.read_csv(os.path.join(prep.DATA_PATH, filename + '.csv'), index_col=0)
    if interpolate:
        df = linear_interpolate(df.mission_time, df)
        
    with open(os.path.join(prep.DATA_PATH, filename + '.features'), 'r') as f:
        features = json.load(f)
        
    return df, features


def linear_interpolate(ref_time, df):
    """
    Function fills missing data with linearly interpolated values.
    
    :param ref_time: vector representing the time axis of the data frame
    :param df: data frame with missing values
    :return: data frame with linearly interpolated missing values
    """
    r_data = np.zeros([ref_time.size,df.columns.size])
    for c in range(df.columns.size):
        col = df.columns[c]
        r_time = ref_time.values[np.where(np.isfinite(df[col]))[0]]
        r_value = df[col].values[np.where(np.isfinite(df[col]))[0]]
        r_data[:,c] = np.interp(ref_time,r_time,r_value)
    return pd.DataFrame(r_data,columns=df.columns,index=df.index)


def RMSE(true, pred):
    """
    Function computes RMSE.
    
    :param true: expected values
    :param pred: predicted values
    :return: computed RMSE
    """
    if hasattr(true, 'values'):
        true = true.values
    if hasattr(pred, 'values'):
        pred = pred.values
    diff = true-pred
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    return rmse


def correct_cols_with_median(Y_train, Y_hat, to_correct):
    """
    Function creates a constant value prediction for selected power lines on the 
    basis of past median current.
    
    :param Y_train: training data
    :param Y_hat: predictions
    :param to_correct: a list of power lines selected for filling in with median currents
    :return: data frame with constant value predictions for selected power lines
    """
    Y_hat_m = Y_hat.copy()
    medians = Y_train.median()
    for p_col in to_correct:
        Y_hat_m.ix[:,p_col] = medians[p_col]
    return Y_hat_m


def correct_min_max(Y_train, Y_hat):
    """
    Function clamps predictions to min-max values as observed in the training 
    data.
    
    :param Y_train: training data
    :param Y_hat: predictions
    :return: predictions with clamped min-max values
    """
    Y_hat[Y_hat<np.min(Y_train)] = np.min(Y_train)
    Y_hat[Y_hat>np.max(Y_train)] = np.max(Y_train)
    return Y_hat

