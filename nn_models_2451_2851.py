# coding: utf-8
"""
@author: fornax
"""
from __future__ import print_function, division
import os
import re
import json
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

import keras.backend as K
from keras.metrics import mean_squared_error
from keras import optimizers
from keras.layers.core import Dense, Activation, Flatten, TimeDistributedDense, Merge
from keras.layers.advanced_activations import SReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential, model_from_json
from keras.regularizers import l2

import sys
sys.path.append('../')
import utils  

#########################################################
##################  HELPER METHODS ######################
#########################################################


def rmse_nn(y_true, y_pred, scaling=1):
    """
    Function computes the RMSE.

    :param y_true: expected values
    :param y_pred: predicted values
    :return: computed RMSE
    """
    return K.sqrt(mean_squared_error(y_true/scaling, y_pred/scaling))


def create_submission(model_factory, dataset, y_scaling, epochs, batch_size, optim_lr, optim_mu, model_target): 
    """
    Function loads and processes the dataset, and creates a submission using the given model and learning paramters.

    :param model_factory: function taking data input and output sizes and creating a keras model ready for training
    :param dataset: name of dataset to be loaded
    :param y_scaling: scaling factor of the target values
    :param epochs: number of learning epochs
    :param batch_size: size of learning batch
    :param optim_lr: learning rate paramter passed to the Adam optimizer
    :param optim_mu: learning mu parameter passed to the Adam optimizer
    :param model_target: target lines to make predictions
    :return: predictions for year 4
    """

    df, features = utils.load_data(dataset)
    df = utils.linear_interpolate(df.mission_time, df)

    trainset = (df['m_year'] > 0) & (df['m_year'] < 3)
    testset = df['m_year'] == 3
    
    df.drop(features['aux_time'], axis=1, inplace=True)

    Y = df[model_target]
    X = df.drop(utils.p_cols, axis=1)
    X_train, Y_train = X[trainset], Y[trainset]
    X_test = X[testset]

    # remove cols with 0 std (their presence blocks learning)
    x_std = X_train.std()
    X_train = X_train.drop(X_train.columns[np.where(x_std==0)[0]], axis=1)
    X_test = X_test.drop(X_test.columns[np.where(x_std==0)[0]], axis=1)

    # scale y_values
    Y_train = Y_train * y_scaling

    # normalize data
    means_x = X_train.mean()
    x_std = X_train.std()
    X_train = (X_train-means_x)/x_std
    X_test = (X_test-means_x)/x_std

    means_y = Y_train.mean()
    Y_train = (Y_train-means_y)

    optim = optimizers.Adam(lr=optim_lr, beta_1=optim_mu, beta_2=0.999, epsilon=1e-08)
    p_rmse_nn = partial(rmse_nn, scaling=y_scaling)
    p_rmse_nn.__name__ = 'RMSE'

    mdl = model_factory(X_train.shape[1], Y_train.shape[1])
    mdl.compile(loss='mse', optimizer=optim, metrics=[p_rmse_nn])
    mdl.fit(X_train.as_matrix(), Y_train.as_matrix(), batch_size=batch_size, nb_epoch=epochs, verbose=2)

    Y_test_hat = (mdl.predict(X_test.as_matrix()) + means_y.values) / y_scaling
    Y_test_hat = pd.DataFrame(Y_test_hat, columns=model_target)
    Y_test_hat = Y_test_hat.fillna(0)

    return Y_test_hat


#########################################################
#########################################################
##############  SUBMISSION FOR LINE 2451 ################
#########################################################
#########################################################


def feed_forward_2451(input_size, output_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size))
    model.add(SReLU())
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(SReLU())
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(SReLU())
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(SReLU())
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(SReLU())
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(SReLU())
    model.add(Dropout(0.2))

    model.add(Dense(output_size))
    model.add(Activation('linear'))
    return model


# RUN SETTINGS
dataset = 'dataset2'
y_scaling = 1000
epochs = 150
batch_size = 128
optim_lr = 1e-05
optim_mu = 0.85
model_target = ['NPWD2451']

print('Creating submission for NPWD2451...')
Y_hat_4y = create_submission(feed_forward_2451, dataset, y_scaling, epochs, batch_size, optim_lr, optim_mu, model_target)
Y_hat_4y.to_csv('results_4y/nn_2451.csv')

#########################################################
#########################################################
##############  SUBMISSION FOR LINE 2851 ################
#########################################################
#########################################################


def feed_forward_2851(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size))
    model.add(SReLU())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(SReLU())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(SReLU())
    model.add(Dropout(0.5))

    model.add(Dense(output_size))
    model.add(Activation('linear'))
    return model


# RUN SETTINGS
dataset = 'dataset1'
y_scaling = 1000
epochs = 200
batch_size = 128
optim_lr = 2.5e-05
optim_mu = 0.85
model_target = ['NPWD2851']

print('Creating submission for NPWD2851...')
Y_hat_4y = create_submission(feed_forward_2851, dataset, y_scaling, epochs, batch_size, optim_lr, optim_mu, model_target)
Y_hat_4y.to_csv('results_4y/nn_2851.csv')

