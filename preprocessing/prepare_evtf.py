# -*- coding: utf-8 -*-
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

# load EVTF
print('Loading EVTF...')
evtf = pd.read_csv(os.path.join(DATA_PATH, 'evtf.csv'))
###############################################################################
############################# OCCULTATIONS ####################################
###############################################################################
feats_mars_occultations = ['OCC_PHOBOS', 'PHO_PENUMBRA', 'PHO_UMBRA',
                           'MAR_PENUMBRA', 'MAR_UMBRA', 'OCC_MARS_200KM', 'OCC_MARS',
                           'OCC_DEIMOS', 'DEI_PENUMBRA']
#, 'DEI_UMBRA'] # DEI_UMBRA has only 2 occurences and they don't match in the 
# data well (end->start->end->start)

for feat in feats_mars_occultations:
    evtf['%s' % feat] = 0

for feat in feats_mars_occultations:
    print('Processing %s' % feat)
    if feat == 'OCC_MARS':
        rule_start = lambda x: feat in x and 'START' in x and 'OCC_MARS_200KM' not in x
        rule_end = lambda x: feat in x and 'END' in x and 'OCC_MARS_200KM' not in x
    else:
        rule_start = lambda x: feat in x and 'START' in x
        rule_end = lambda x: feat in x and 'END' in x
    starts = np.where(map(rule_start, evtf.description.values))[0]
    ends = np.where(map(rule_end, evtf.description.values))[0]
    assert(len(starts) == len(ends))
    assert(starts[0] < ends[0])
    # indicate ongoing events
    for start, end in zip(starts, ends):
        evtf.ix[start:end, '%s' % feat] = 1

###### ALL OCCULTATIONS COMBINED ######

def merge_embedded_occ(occ_idx_list):
    prev_start, prev_end = occ_idx_list[0]
    approved_list = []
    for start, end in occ_idx_list:
        if start > prev_end:
            approved_list.append((prev_start, prev_end))
            prev_start, prev_end = start, end
        else:
            prev_end = end  
    approved_list.append((prev_start, prev_end))
    return approved_list

print('Processing all occultations')
evtf['OCC'] = 0

rule_start = lambda x: any(map(lambda y: y in x, feats_mars_occultations)) and 'START' in x
rule_end = lambda x: any(map(lambda y: y in x, feats_mars_occultations)) and 'END' in x

starts = np.where(map(rule_start, evtf.description.values))[0]
ends = np.where(map(rule_end, evtf.description.values))[0]
assert(len(starts) == len(ends))
assert(starts[0] < ends[0])

new_list = merge_embedded_occ(zip(starts, ends))
starts, ends = zip(*new_list)

for start, end in zip(starts, ends):
    evtf.ix[start:end, 'OCC'] = 1

###############################################################################
############################# X/Y POINTING ####################################
###############################################################################
'''
Types NPSS and NPNS indicate the times in the mission, when the pointing
of the x axis has to switch from North to South (NPSS) or from South to North
(NPNS) in order to avoid Sun incidence on the S/C -x face in nadir pointing
mode around Mars.
In nadir pointing mode, with the x axis perpendicular to the ground track, the
angle between the S/C -x axis and the Sun direction varies around the peri-
centre by some degrees (e.g. at the switching time around mid March 2004
about  5  degrees).  This  means  that  there  is  not  a  single  date  and  time  to
switch to the correct x axis pointing or, conversely, depending on the duration
of the nadir pointing, it might therefore not be possible, to avoid Sun incidence
on  the  S/C  -x  face  during  a  complete  pericentre  passage  in  nadir  pointing
mode (neither with North nor with South pointing option). Instead, the dura-
tion of the nadir pointing has to be reduced or a small Sun incidence must be
tolerated.
'''
feats_pos_changes = ['NADIR_POINTING_X_N_TO_S_SWITCH', 'NADIR_POINTING_X_S_TO_N_SWITCH'
                    'EARTH_POINTING_Y_N_TO_S_SWITCH', 'EARTH_POINTING_Y_S_TO_N_SWITCH']

evtf['NADIR_POINTING_X'] = 0
evtf['EARTH_POINTING_Y'] = 0

for feat in ['NADIR_POINTING_X', 'EARTH_POINTING_Y']:
    print('Processing %s' % feat)
    changes = np.where(map(lambda x: feat in x, evtf.description.values))[0]
    for start, end in zip(changes, np.concatenate([changes[1:], [len(evtf)]])):
        evtf.ix[start:end, '%s' % feat] = 1 if 'N_TO_S' in evtf.description.values[start] else -1
    evtf.ix[0:changes[0], '%s' % feat] = evtf.ix[changes[0], '%s' % feat] * -1


###############################################################################
########################## TRAJECTORY EVENTS ##################################
###############################################################################
'''
'x km descend’ and ‘x km ascend’, refer to the event
when the height of the S/C position above the Mars reference ellipsoid drops
below or rises above x km.
'''
feats_trajectory = np.unique(
                            filter(lambda x: x.endswith('SCEND'), 
                                   np.unique(evtf.description)))

evtf['trajectory_position_above_reference'] = 0
evtf['trajectory_direction'] = 0
changes_trajectory = np.where(map(lambda x: x.endswith('SCEND'), evtf.description.values))[0]
print('Processing trajectory changes')
for start, end in zip(changes_trajectory, np.concatenate([changes_trajectory[1:], [len(evtf)]])):
    splits = evtf.description.iloc[start].split('_')
    pos = int(splits[0])
    updown = 1 if splits[-1] == 'ASCEND' else -1
    evtf.ix[start:end, 'trajectory_position_above_reference'] = pos
    evtf.ix[start:end, 'trajectory_direction'] = updown

###############################################################################
################################ SAVING #######################################
###############################################################################
evtf.drop(['description'], axis=1, inplace=True)

filename = 'evtf_processed'
savepath = os.path.join(DATA_PATH, filename + '.csv')
print('Saving to %s' % savepath)
evtf.to_csv(savepath, index=False)
