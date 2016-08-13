# -*- coding: utf-8 -*-
"""
@author: fornax
"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import xgboost

import utils
from preprocessing import dmop_analysis

np.random.seed(5)
# default model parameters
N_ESTIMATORS = 200
MAX_DEPTH = 3
LAMBDA = 1
# percentage of features to randomly select in each bagged model
FEATS_PERC = 0.7
# columns to correct by median 
cols_to_correct = [u'NPWD2401', u'NPWD2402', u'NPWD2481', u'NPWD2482', u'NPWD2501',
                   u'NPWD2531', u'NPWD2691', u'NPWD2722', u'NPWD2771', u'NPWD2801']


def create_model(X_train, Y_train, X_test, 
                 num_models=1, 
                 n_estimators=[N_ESTIMATORS], max_depths=[MAX_DEPTH], lambdas=[LAMBDA]):
    """
    Builds a (bagged) xgboost model for a single power line and creates predictions
    on the test data. 
    
    :param X_train: features to train on
    :param Y_train: training target
    :param X_test: features from which to create predictions
    :param num_models: number of bagging models to train
    :param n_estimators: a list with numbers of estimators from which to draw 
        in each bag
    :param max_depths: as above, but for max_depth
    :param lambdas: as above, but for lambda
    :return: prediction, clamped into min-max values as seen in the training data
    """
    preds = []
    for m in range(num_models):
        max_depth = np.random.choice(max_depths)
        trees = np.random.choice(n_estimators)
        reg_lambda = np.random.choice(lambdas)
        mdl = xgboost.XGBRegressor(max_depth=max_depth, n_estimators=trees, reg_lambda=reg_lambda)
        
        if num_models > 1:
            feats = np.random.choice(X_train.columns, size=int(FEATS_PERC*X_train.shape[1]), replace=False)
        else:
            feats = X_train.columns
        dummy = mdl.fit(X_train[feats], Y_train)
        preds.append(mdl.predict(X_test[feats]))
    
    pred = np.vstack(preds).T
    pred = np.mean(pred, axis=1)
    pred = utils.correct_min_max(Y_train, pred)
    return pred


###############################################################################
############################# LOAD DATA #######################################
###############################################################################
df, features = utils.load_data('dataset1')
df2, features2 = utils.load_data('dataset2')
aooo = dmop_analysis.get_npwd2881_features(df)

p_cols = utils.p_cols

Y = df[p_cols]
X = df.drop(p_cols + features['aux_time'], axis=1)
X2 = df2.drop(p_cols + features2['aux_time'], axis=1)

###############################################################################
############################ SUBMISSION #######################################
###############################################################################
trainset = (df.m_year <= 2) & (df.m_year > 0)
testset = df.m_year == 3
X_train, Y_train = X[trainset], Y[trainset]
X_test, Y_test = X[testset], Y[testset]
X_train2 = X2[trainset]
X_test2 = X2[testset]

Y_test_hat = []

for p_col in p_cols:
    print('%s...' % p_col)
    if p_col == 'NPWD2881':
        pred = create_model(aooo[trainset.values], Y_train[p_col], aooo[testset.values])
    elif p_col == 'NPWD2451' or p_col == 'NPWD2532':
        pred = create_model(X_train2, Y_train[p_col], X_test2, num_models=200)
    elif p_col == 'NPWD2851':
        pred = create_model(X_train, Y_train[p_col], X_test, num_models=200)
    elif p_col == 'NPWD2551':
        pred = create_model(X_train, Y_train[p_col], X_test, num_models=200, 
                            max_depths=[3,5,7,9], n_estimators=[200,400,600], lambdas=[1,2])
    else:
        pred = create_model(X_train, Y_train[p_col], X_test)
    Y_test_hat.append(pred)
    
Y_test_hat = np.vstack(Y_test_hat).T

# median correction
Y_test_hat = pd.DataFrame(Y_test_hat, columns = p_cols)
Y_test_hat_corrected = utils.correct_cols_with_median(Y_train, Y_test_hat, cols_to_correct)
Y_test_hat = Y_test_hat_corrected.values

#Preparing the submission file:
Y_test_hat=pd.DataFrame(Y_test_hat, index=X_test.index, columns=p_cols)
Y_test_hat.index = pd.to_datetime(Y_test_hat.index)
Y_test_hat['ut_ms'] = (Y_test_hat.index.astype(np.int64) * 1e-6).astype(int)
Y_test_hat[['ut_ms'] + p_cols].to_csv('results_4y/xgb.csv', index=False)
