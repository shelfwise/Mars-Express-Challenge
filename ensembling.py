# -*- coding: utf-8 -*-
"""
@author: fornax
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd

import utils


def sub(df, d):
    """
    Function computes and substitutes ensembles of models for designated power lines.
    Ensembling is done via a simply averaging of predictions.
    
    :param df: data frame with predictions for each power line
    :param d: dictionary having power line names as keys, predictions to ensemble
    as values
    :return: data frame with predictions ensembled
    """
    for p_col in d.keys():
        cols = [d[p_col][i][p_col] for i in range(len(d[p_col]))]
        df[p_col] = np.mean(cols, axis=0)
    return df


# whether to use a "mask" for NPWD2551 (see README for an explanation)
USE_MASK_2551 = False

# read all predictions
xgb = pd.read_csv('results_4y/xgb.csv', index_col=0)
nn_2451 = pd.read_csv('results_4y/nn_2451.csv', index_col=0)
nn_2851 = pd.read_csv('results_4y/nn_2851.csv', index_col=0)

# ensemble XGB and NN for two power lines
to_sub = {}
to_sub['NPWD2451'] = [xgb, nn_2451]
to_sub['NPWD2851'] = [xgb, nn_2851]
pout = xgb.copy()
pout = sub(pout, to_sub)

# apply mask for NPWD2551
if USE_MASK_2551:
    Y = pd.read_csv('merged_data/dataset1.csv', index_col=0, 
                usecols=['ut_ms', 'm_year', 'mission_time', 'ATTT_current_305C_305O_305P_306C_306P'])
    Y = utils.linear_interpolate(Y.mission_time, Y)
    Y_test = Y.loc[Y.m_year == 3]

    mask = 1-Y_test['ATTT_current_305C_305O_305P_306C_306P'].values
    pout.NPWD2551 = pout.NPWD2551 * mask

# save the result
pout.to_csv('results_4y/ensemble.csv')
