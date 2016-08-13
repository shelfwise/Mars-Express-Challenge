# -*- coding: utf-8 -*-
"""
@author: fornax
"""
from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import re
import json

from preprocessing import dmop_analysis

DATA_PATH = 'merged_data'


def to_datetime(df):
    """
    Converts UCT timestamp [ms] to datetime in a dataframe
    """
    df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
    return df


def to_utms(ut):
    """
    Converts datetime to UTC timestamp [ms]
    """
    return (ut.astype(np.int64) * 1e-6).astype(int)


def resample(df, intervals='1H'):
    """
    Resamples the data frame to a given interval
    """
    df = df.resample(intervals).mean()
    return df


def parse_data(filename):
    """
    Read a dataframe and prepare the time axis
    """
    df = pd.read_csv(filename)
    df = to_datetime(df)
    df = df.set_index('ut_ms')
    return df


def parse_power(filename, intervals='1H', dropna=True):
    """
    Prepares the power data, with resampling
    """
    df = parse_data(filename)
    df = resample(df, intervals)
    if dropna:
        df = df.dropna()
    return df


def parse_dmop(filename):
    """
    Prepares the DMOP data, along with cleaning and merging of commands
    """
    df = parse_data(filename)
    df.fillna('', inplace=True)
    df.drop(['subsystem'], axis=1, inplace=True)
    
    cols_numeric = [i for i in df if re.search('^[A-Z]{4}_curr', i) is None]
    cols_nonnumeric = [i for i in df if re.search('^[A-Z]{4}_curr', i) is not None]
    
    dmop_all = df[cols_numeric]
    dmop_all = dmop_all.join(pd.get_dummies(df[cols_nonnumeric]))
    dmop_all = dmop_analysis.correct_dmop(dmop_all)
    
    return dmop_all


def parse_ftl(filename):
    """
    Prepares FTL data
    """
    df = pd.read_csv(filename, index_col=0)
    df['ut_ms'] = df['utb_ms']
    df.drop(['utb_ms', 'ute_ms'], axis=1, inplace=True)
    df = to_datetime(df)
    df = df.set_index('ut_ms')
    return df
    

def align_to_power(df, powers, method='nearest'):
    """
    Aligns dataframe's time axis to a common ground
    """
    df = df.reindex(powers.index, method=method)
    if 'm_year' in df.columns:
        df.drop(['m_year'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    filename = 'dataset1'
    intervals = '60min'
    
    print('Preparing powers...')
    powers = parse_power(os.path.join(DATA_PATH, 'power.csv'), intervals=intervals)
    powers_test = parse_power('test_set/power-prediction-sample-2014-04-14_2016-03-01.csv', intervals=intervals, dropna=False)
    powers_test['m_year'] = 3
    powers_all = pd.concat([powers, powers_test])
    
    print('Preparing SAAF...')
    saaf_all = parse_data(os.path.join(DATA_PATH, 'saaf_processed.csv'))

    print('Preparing LTDATA...')
    ltdata_all = parse_data(os.path.join(DATA_PATH, 'ltdata.csv'))
    
    print('Preparing EVTF...')
    evtf_all = parse_data(os.path.join(DATA_PATH, 'evtf_processed.csv'))
    evtf_all = evtf_all.groupby(level=0).last()
    print('Resampling EVTF...')
    evtf_all = resample(evtf_all, intervals)
    
    print('Preparing DMOP...')
    dmop_all = parse_dmop(os.path.join(DATA_PATH, 'dmop_processed.csv'))
    print('Resampling DMOP...')
    dmop_all = resample(dmop_all, intervals)
    
    print('Preparing FTL...')
    ftl_all = parse_ftl(os.path.join(DATA_PATH, 'ftl_processed.csv'))
    print('Resampling FTL...')
    ftl_all = resample(ftl_all, intervals)
    
    print('Aligning time to powers...')
    filling_method = 'nearest'
    saaf_all = align_to_power(saaf_all, powers_all, method=filling_method)
    ltdata_all = align_to_power(ltdata_all, powers_all, method=filling_method)
    dmop_all = align_to_power(dmop_all, powers_all, method=filling_method)
    ftl_all = align_to_power(ftl_all, powers_all, method=filling_method)
    evtf_all = align_to_power(evtf_all, powers_all, method=filling_method)
    
    print('Creating the dataframe...')
    df = powers_all.copy()
    df = df.join(saaf_all)
    df = df.join(ltdata_all)
    df = df.join(dmop_all)
    df = df.join(ftl_all)
    df = df.join(evtf_all)
    
    df['mission_time'] = to_utms(df.index)
    
    path_to_save = os.path.join(DATA_PATH, filename + '.csv')
    print('Saving to %s' % path_to_save)
    df.to_csv(path_to_save)
    
    features = {}
    features['NPWD'] = list(powers_all.columns)
    features['saaf'] = list(saaf_all.columns)
    features['ltdata'] = list(ltdata_all.columns)
    features['dmop'] = list(dmop_all.columns)
    features['ftl'] = list(ftl_all.columns)
    features['evtf'] = list(evtf_all.columns)
    features['aux_time'] = ['m_year', 'mission_time']
    
    with open(os.path.join(DATA_PATH, filename + '.features'), 'w') as f:
        json.dump(features, f)
    print('Done!')
