# -*- coding: utf-8 -*-
"""
@author: fornax
"""
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.dirname(os.getcwd()))
import prepare_data1 as prep
DATA_PATH = os.path.join('..', prep.DATA_PATH)

dates = ['2008-08-22_2010-07-10', '2010-07-10_2012-05-27', '2012-05-27_2014-04-14', '2014-04-14_2016-03-01']
contexts = ['dmop', 'evtf', 'ftl', 'saaf', 'ltdata']

for context in contexts:
    aggr = []
    for m_year, date in enumerate(dates):
        if m_year < 3:
            folder = '../train_set'
        else:
            folder = '../test_set'
        df = pd.read_csv(os.path.join(folder, 'context--%s--%s.csv' % (date, context)))
        df['m_year'] = m_year
        aggr.append(df)
    pd.concat(aggr).to_csv(os.path.join(DATA_PATH, context + '.csv'), index=False)

# power files
aggr = []
for m_year, date in enumerate(dates[:-1]):
    df = pd.read_csv(os.path.join('../train_set', 'power--%s.csv' % (date)))
    df['m_year'] = m_year
    aggr.append(df)
pd.concat(aggr).to_csv(os.path.join(DATA_PATH, 'power.csv'), index=False)
