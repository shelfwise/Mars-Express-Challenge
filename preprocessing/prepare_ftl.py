# coding: utf-8
"""
@author: fornax
"""
from __future__ import print_function, division
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.dirname(os.getcwd()))
import prepare_data1 as prep
DATA_PATH = os.path.join('..', prep.DATA_PATH)


# get ohe of columns
def get_ohe(example, column_names):
    return (column_names == example).astype(int)


# load data
print('Loading FTL data...')
ftl_file_path = os.path.join(DATA_PATH, 'ftl.csv')
ftl_df = pd.read_csv(ftl_file_path)

# apply one-hot encoding to columns
print('One-hot encoding columns...')
point_types = np.unique(ftl_df.type)
ohe_point_type_cols = ['is_{}'.format(pt.lower()) for pt in point_types]
ftl_df[ohe_point_type_cols] = ftl_df.type.apply(lambda x: pd.Series(get_ohe(x, point_types)))

point_types = np.append(np.unique(ftl_df.type), ['flagcomms'])
ohe_point_type_cols = np.append(ohe_point_type_cols, ['flagcomms'])
ftl_df['flagcomms'] = ftl_df.flagcomms.apply(lambda x: 1 if x else 0)

# save processed FTL data
print('Saving processed FTL data...')
processed_file_path = os.path.join(DATA_PATH, 'ftl_processed.csv')
ftl_df.to_csv(processed_file_path)