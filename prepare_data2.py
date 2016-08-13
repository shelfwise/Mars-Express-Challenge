# -*- coding: utf-8 -*-
"""
@author: fornax
"""
import os
import pandas as pd
import json

import prepare_data1 as prep

DATA_PATH = prep.DATA_PATH
INTERVALS = ['2H', '6H', '12H', '1D']
FUNCTIONS = ['mean', 'std', 'count']


def resample(df, df_name='generic', intervals=INTERVALS, functions=FUNCTIONS, cols=None):
    """
    Function computes summary statistics over past values of each feature for a series
    of intervals, as well as a 'count' feature which represents the number of 
    changes/activities in the past across all features (i.e. describes how many 
    measurements were made in the data frame in the last X hours).
    
    :df: dataset from which to create past-based features
    :df_name: name of the dataset
    :intervals: a list of intervals for which to compute summary statistics w.r.t.
    each time-point
    :functions: a list of summary statistics to compute
    :cols: a list of features on which to compute past statistics (if None, 
    then all features will be processed)
    :return: input data frame with new features joined
    """
    if cols is None:
        cols = df.columns
        
    if 'm_year' in cols:
        cols = list(cols)
        cols.remove('m_year')
    
    for interval in intervals:
        for function in functions:
            if function == 'count':
                resampler = df[cols[:1]].resample(interval, label='right')
                df_resampled = resampler.count()
                df_resampled.columns = ['_'.join([df_name, function, 'past', interval])]
                df = df.join(df_resampled, how='outer')
            else:
                resampler = df[cols].resample(interval, label='right')
                df_resampled = eval('resampler.' + function + '()')
                df_resampled.columns = ['_'.join([c, function, 'past', interval]) for c in cols]
                df = df.join(df_resampled, how='outer')
    return df


def parse_ltdata():
    print('Preparing LTDATA...')
    f = os.path.join(DATA_PATH, 'ltdata.csv')
    df_raw = prep.parse_data(f)
    
    intervals = ['3D', '7D']
    functions = ['mean', 'std']
    df = resample(df_raw, 'ltdata', intervals=intervals, functions=functions)
    df = df.resample('1H').mean()
    return df


def parse_saaf():
    print('Preparing SAAF...')
    f = os.path.join(DATA_PATH, 'saaf.csv')
    df_raw = prep.parse_data(f)
    
    df = resample(df_raw, 'saaf')
    df = df.resample('1H').mean()
    return df
    

def parse_ftl():
    print('Preparing FTL...')
    f = os.path.join(DATA_PATH, 'ftl_processed.csv')
    df = pd.read_csv(f, index_col=0)
    df['ut_ms'] = df['utb_ms']
    df.drop(['utb_ms', 'ute_ms'], axis=1, inplace=True)
    df = prep.to_datetime(df)
    df = df.set_index('ut_ms')
    
    df.drop(['type', 'is_d5pphb', 'is_d7plts', 'is_d8pltp', 'is_d9pspo', 
                      'is_nadir_lander', 'is_radio_science', 'is_specular', 'is_spot'], axis=1, inplace=True)
    
    df = resample(df, 'ftl')
    df = df.resample('1H').mean()
    return df


def parse_evtf():
    print('Preparing EVTF...')
    f = os.path.join(DATA_PATH, 'evtf_processed.csv')
    df = prep.parse_data(f)
    
    to_remove = map(lambda x: filter(lambda y: x in y, df.columns), 
                    ['OCC_DEIMOS', 'OCC_PHOBOS', 'PHO_UMBRA', 'PHO_PENUMBRA', 'DEI_PENUMBRA', 'DEI_UMBRA'])
    to_remove = [i for sub in to_remove for i in sub]
    df.drop(to_remove, axis=1, inplace=True)
    
    to_expand=['OCC', 'MAR_PENUMBRA', 'MAR_UMBRA', 'OCC_MARS']
    df = resample(df, 'evtf', cols=to_expand)
    df = df.resample('1H').mean()
    return df


if __name__ == '__main__':
    filename = 'dataset2'
    
    print('Preparing powers...')
    powers_train = prep.parse_power(os.path.join(DATA_PATH, 'power.csv'), intervals='1H')
    powers_test = prep.parse_power('test_set/power-prediction-sample-2014-04-14_2016-03-01.csv', intervals='1H', dropna=False)
    powers_test['m_year'] = 3
    powers = pd.concat([powers_train, powers_test])

    filling_method = 'nearest'
    saaf = parse_saaf()
    saaf = prep.align_to_power(saaf, powers, method=filling_method)
    
    evtf = parse_evtf()
    evtf = prep.align_to_power(evtf, powers, method=filling_method)
    
    ftl = parse_ftl()
    ftl = prep.align_to_power(ftl, powers, method=filling_method)
    
    ltdata = parse_ltdata()
    ltdata = prep.align_to_power(ltdata, powers, method=filling_method)

    print('Creating the dataframe...')
    df = powers.copy()
    df = df.join(saaf)
    df = df.join(evtf)
    df = df.join(ftl)
    df = df.join(ltdata)
    
    df['mission_time'] = prep.to_utms(df.index)
    
    path_to_save = os.path.join(DATA_PATH, filename + '.csv')
    print('Saving to %s' % path_to_save)
    df.to_csv(path_to_save)    
    
    features = {}
    features['NPWD'] = list(powers.columns)
    features['saaf'] = list(saaf.columns)
    features['evtf'] = list(evtf.columns)
    features['ftl'] = list(ftl.columns)
    features['ltdata'] = list(ltdata.columns)
    features['aux_time'] = ['m_year', 'mission_time']
    
    with open(os.path.join(DATA_PATH, filename + '.features'), 'w') as f:
        json.dump(features, f)
    print('Done!')
