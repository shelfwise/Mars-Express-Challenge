# -*- coding: utf-8 -*-
"""
@author: fornax
"""
from __future__ import print_function, division
import os
import re
import time
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.dirname(os.getcwd()))
import prepare_data1 as prep
DATA_PATH = os.path.join('..', prep.DATA_PATH)


def get_subsystem_command(sc):
    if 'trigger' in sc.lower():
        subsystem = sc
        command = ''
    elif '.' in sc:
        subsystem = sc.split('.')[0]
        command = ''
    else:
        subsystem = sc[:4]
        command = sc[4:]
    return subsystem, command


# load DMOP
print('Loading DMOP...')
dmop = pd.read_csv(os.path.join(DATA_PATH, 'dmop.csv'))

# all possible subsystem+command combinations
all_commands = np.unique(dmop.subsystem)

# extract flight commands of the form "XXX.00000000"
flight_commands = filter(
                        lambda x: re.search('\.', x) is not None, all_commands)
# extract subsystem actions of the form "XXXXXXXXX"
subsystems_commands = filter(
                            lambda x: re.search('\.', x) is None, all_commands)
# get two trigger events
triggers = [subsystems_commands[-7], subsystems_commands[-1]]
# remove trigger events from the subsystem_commands list
subsystems_commands = np.concatenate(
                        [subsystems_commands[:-7], subsystems_commands[-6:-1]])

###############################################################################
############################ Processing #######################################
###############################################################################
# separating subsystems and their commands into a dictionary
# {subsystem_name: list_of_commands}
subsystems_commands = map(get_subsystem_command, all_commands)
subs, comms = zip(*subsystems_commands)
subs, comms = np.asarray(subs), np.asarray(comms)
subsystems = {}
for subsys in np.unique(subs):
    subsystems[subsys] = list(np.unique(comms[subs == subsys]))

# creating features for each subsystem (XXX - name of the subsystem)
# XXX_current - currently running or last ran command for that subsystem
for subsys in subsystems.keys():
    if len(subsystems[subsys]) > 1:
        dmop['%s_current' % subsys] = ''
    dmop['%s_changed' % subsys] = 0

# filling up
temp = dmop.subsystem.apply(lambda x: get_subsystem_command(x))
dmop['subsystem_name'] = map(lambda x: x[0], temp)
dmop['subsystem_command'] = map(lambda x: x[1], temp)

start = time.clock()
for subsys_num, subsys in enumerate(subsystems.keys()):
    print('Processing subsystem %s (%d/%d)...' % (subsys, subsys_num+1, len(subsystems.keys())))
    time_subsys_start = time.clock()
    # get all indices where there was a new command issued for the subsystem
    idx = np.where(dmop['subsystem_name'].values == subsys)[0]
    print('Number of changes: %d' % len(idx))
    
    # mark the time steps where the commands were issued
    dmop.ix[idx, '%s_changed' % subsys] = 1
    # calculate number of commands issued in the last X [mins/hours/days]
    changes = dmop['%s_changed' % subsys].values
    
    # fill up current and past command names
    for i in range(len(idx)):
        # select a slice from now till next change, or from now till the end of
        # data if this is the last command change
        if i+1 < len(idx):
            slice_ = slice(idx[i], idx[i+1])
        else:
            slice_ = slice(idx[i], len(dmop))

        # fill up currently running command and past ones
        if '%s_current' % subsys in dmop.columns:
            dmop.ix[slice_, '%s_current' % subsys] = dmop.ix[idx[i], 'subsystem_command']
    print('Done. Time taken: %f' % (time.clock() - time_subsys_start) )
print('Total time: %f' % (time.clock() - start) )

###############################################################################
############################## Saving #########################################
###############################################################################
dmop.drop(['subsystem_name', 'subsystem_command'], axis=1, inplace=True)
filename = 'dmop_processed'
savepath = os.path.join(DATA_PATH, filename + '.csv')
print('Saving to %s' % savepath)
dmop.to_csv(savepath, index=False)