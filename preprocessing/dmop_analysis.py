# -*- coding: utf-8 -*-
"""
@author: fornax
"""
import numpy as np
import pandas as pd

def get_npwd2881_features(df):
    """
    Extracts AOOO commands from pandas file which are 
    correlated with NPWD2881 line. Those commands are
    then used as a train features for final predictions.
    
    :param df: a data frame contains all features with added merged DMOP columns.
    :return: a data frame containing selected features used for training of 
        NPWD2881 power line. 
        
    """
    aooo_list = [    
        ['AOOO_current_F03A1',3,True], 
        ['AOOO_current_F04A0',0,True], 
        ['AOOO_current_F05A0',3,False], 
        ['AOOO_current_F06A0',0,False], 
        ['AOOO_current_F100A',0,False], 
        ['AOOO_current_F100C',0,False], 
        ['AOOO_current_F20A1',2,True], 
        ['AOOO_current_F20D1',2,True], 
        ['AOOO_current_F22A0',2,True], 
        ['AOOO_current_F32A0',2,True], 
        ['AOOO_current_F62A0',3,False], 
        ['AOOO_current_F63A0',3,False], 
        ['AOOO_current_F64A0',3,False], 
        ['AOOO_current_F65A0',3,False], 
        ['AOOO_current_F66A0',3,False], 
        ['AOOO_current_F67A0',3,False], 
        ['AOOO_current_F68A0',3,False], 
        ['AOOO_current_F77A0',3,False], 
        ['AOOO_current_F02A1',3,False], 
        ['AOOO_current_F01D0',3,True], 
        ['AOOO_current_F01D1',3,True], 
        ['AOOO_current_F02A0',3,False], 
        ['AOOO_current_F03A0',3,False], 
        ['AOOO_current_F32R0',3,False], 
        ['AOOO_current_F33A0',3,False], 
        ['AOOO_current_F34A0',4,False], 
        ['AOOO_current_F15A0',3,True], 
        ['AOOO_current_F100B',1,True], 
        ['AOOO_current_F15B0',1,True], 
        ['AOOO_current_F22A1',1,True], 
        ['AOOO_current_F22R1',0,False], 
        ['AOOO_current_F23A0',0,True], 
        ['AOOO_current_F24A0',1,False], 
        ['AOOO_current_F100D',0,False], 
        ['ATTT_current_F321D_F321R',1,False], 
    ]


    n_cols    = np.shape(aooo_list)[0]
    aooo_data = np.zeros([df.mission_time.size,n_cols])
    aooo_cols = []
    it        = 0
    # iterate over all selected features    
    for l in aooo_list:
        offset = l[1]        
        v = df[l[0]].values
        if(l[2] == True):  # offset the signal and pad it while translating
            for k in range(1,offset+1):
                v2 = np.append(df[l[0]].values[k:],[0]*k)
                v = v + v2
        else: # just offset the signal
            v = np.append(df[l[0]].values[offset:],[0]*offset)    
        
        aooo_cols.append(l[0])
        aooo_data[:,it] = v
        it += 1
    
    pd_aooo = pd.DataFrame(aooo_data,columns=aooo_cols)
    return pd_aooo


def correct_dmop(df):
    """
    Function removes unwanted subsystem/command combinations, and merges commands
    within a single subsystem as indicated for merging by manual inspection (see README).
    
    :param df: data frame
    :return: input data frame with features 
    """
    trash, merge, atmb_vals = correction_list(df)    
    to_delete = [i for i in trash.values() for i in i]

    # merging
    to_merge = [i for i in merge.values() if len(i)>1]
    to_merge = [i for i in to_merge for i in i]
    
    for cols in to_merge:
        name = '_'.join(cols[0].split('_')[:-1] + [i.split('_')[-1] for i in cols])
        df[name] = 0
        for col in cols:
            df[name] += df[col]
            to_delete.append(col)
    
    # deleting
    for col in to_delete:
        if col in df.columns:
            subsys = col.split('_')[0]
            command = col.split('_')[-1]
            col_related = filter(lambda x: subsys in x and command in x and len(x.split('_')) == 3, df.columns)
            df.drop(col_related, axis=1, inplace=True)
    
    # ATMB
    df['ATMB_temp'] = atmb_vals
    
    return df


def correction_list(pd_data):
    """
    Function returns dictionaries of subsystem/command combinations that should
    be removed from the data, or tuples of commands to merge together into 
    a single feature. Also, a temperature-based feature is extracted from
    the ATMB subsystem.
    
    :param pd_data: data frame with ATMB subsystem
    :return: dictionaries
    """
    trash = {}  
    merge  = {}
    # --------------------------------------------------
    trash['ATTT'] = ['ATTT_current_260A',
             'ATTT_current_F301A',
             'ATTT_current_F301B',
             'ATTT_current_F301E',
             'ATTT_current_F301F',
             'ATTT_current_F301I',
             'ATTT_current_F301J',
             'ATTT_current_F310A',
             'ATTT_current_F310B',
             'ATTT_current_F410B',
             'ATTT_current_F420B']
             
    merge['ATTT'] = [['ATTT_current_305C','ATTT_current_305O','ATTT_current_305P','ATTT_current_306C','ATTT_current_306P'],
             ['ATTT_current_309A','ATTT_current_309B','ATTT_current_309P','ATTT_current_309Q'],
             ['ATTT_current_F321A','ATTT_current_F321P'],
             ['ATTT_current_F321D','ATTT_current_F321R']
             ]
    # --------------------------------------------------        
    trash['ASXX']  = []
    merge['ASXX'] = [['ASXX_current_303A','ASXX_current_304A'],
                     ['ASXX_current_307A','ASXX_current_308A'],
                     ['ASXX_current_382C','ASXX_current_383C','ASXX_current_382S','ASXX_current_383S','ASXX_current_382R']]
    # --------------------------------------------------
    trash['AVVV']  = ['AVVV_current_01A0', 
                 'AVVV_current_02A0', 
                 'AVVV_current_03A0', 
                 'AVVV_current_03B0', 
                 'AVVV_current_05A0', 
                 'AVVV_current_06A0', 
                 'AVVV_current_07A0']
             
    merge['AVVV'] = [[]]
    # --------------------------------------------------
    trash['AHHH']  = ['AHHH_current_C05A1',
                       'AHHH_current_C25A1',
                    'AHHH_current_C532E',
                    'AHHH_current_F04P3',
                    'AHHH_current_F095B',
                    'AHHH_current_F095C',
                    'AHHH_current_F11A2',
                    'AHHH_current_F20A1',
                    'AHHH_current_F23P1',
                    'AHHH_current_F50A2'
                    ]
             
    merge['AHHH'] = [['AHHH_current_F01A2','AHHH_current_F01P1','AHHH_current_F01R1'],
            ['AHHH_current_F01S0'],
            ['AHHH_current_F02A1','AHHH_current_F02P1'],
            ['AHHH_current_F03A2'],
            ['AHHH_current_F04A3'],
            ['AHHH_current_F05A2'],
            ['AHHH_current_F06A1','AHHH_current_F06P1','AHHH_current_F06R1'],
            ['AHHH_current_F06S0'],
            ['AHHH_current_F11A1'],
            ['AHHH_current_F13A1'],
            ['AHHH_current_F17A1','AHHH_current_F17B1','AHHH_current_F17C2'],
            ['AHHH_current_F19A1']]
    # --------------------------------------------------
    trash['AOOO']  = []  
    merge['AOOO'] = [[]]
    # --------------------------------------------------
    trash['AMMM']  = [
    'AMMM_current_F01A0',
    'AMMM_current_F01B0',
    'AMMM_current_F01R0',
    'AMMM_current_F71A0',
    'AMMM_current_F71AF',
    'AMMM_current_F73A0',
    'AMMM_current_F21A0',
    'AMMM_current_F22A0',
    'AMMM_current_F06B0',
    'AMMM_current_F06R0',
    'AMMM_current_F13A0',
    'AMMM_current_F14A0',
    'AMMM_current_F26A0',
    'AMMM_current_F32A0'
    ]         
    merge['AMMM'] = [
    ['AMMM_current_F04A0','AMMM_current_F40A0'],
    ['AMMM_current_F05A0','AMMM_current_F40C0'],
    ['AMMM_current_F19A0'],
    ['AMMM_current_F51A0','AMMM_current_F52A0','AMMM_current_F52D1','AMMM_current_F52D2','AMMM_current_F52D3','AMMM_current_F52D4'],
    ['AMMM_current_F10A0','AMMM_current_F11A0','AMMM_current_F12A0','AMMM_current_F18A0','AMMM_current_F20A0','AMMM_current_F23A0','AMMM_current_F24A0','AMMM_current_F40B0']
    ]
    # --------------------------------------------------
    trash['APSF']  = [
    'APSF_current_12B1',
    'APSF_current_12C1',
    'APSF_current_12D1',
    'APSF_current_12E1',
    'APSF_current_12G1',
    'APSF_current_82B1',
    'APSF_current_83A1',
    'APSF_current_83B1',
    'APSF_current_88A1',
    'APSF_current_29B1',
    'APSF_current_15A2',
    'APSF_current_16A2',
    'APSF_current_22A1',
    'APSF_current_01A2',
    'APSF_current_02A1',
    'APSF_current_03A3',
    'APSF_current_13A3',
    'APSF_current_14A2',
    'APSF_current_23B1',
    'APSF_current_28A1',
    'APSF_current_30A1',
    'APSF_current_30B2',
    'APSF_current_30C2',
    'APSF_current_31A1',
    'APSF_current_31B1',
    'APSF_current_32A1',
    'APSF_current_33A1',
    'APSF_current_35A1',
    'APSF_current_37A1',
    'APSF_current_38A1',
    'APSF_current_40A1',
    'APSF_current_82A1',
    'APSF_current_89A1'
    ]         
    merge['APSF'] = [
    ['APSF_current_06A1','APSF_current_06A2','APSF_current_60B0'],
    ['APSF_current_50A2'],
    ['APSF_current_12H1'],
    ['APSF_current_28A1','APSF_current_60A0','APSF_current_60D0']]
    # --------------------------------------------------
    trash['ASSS']  = [
    'ASSS_current_F57A0',
    'ASSS_current_F58A0',
    'ASSS_current_F59A0',
    'ASSS_current_F60A0',
    'ASSS_current_F63A0',
    ]         
    merge['ASSS'] = [
        ['ASSS_current_F01A0','ASSS_current_F01P0'],
        ['ASSS_current_F06A0','ASSS_current_F06P0'],
        ['ASSS_current_F62A0'],
        ['ASSS_current_F53A0','ASSS_current_F55A0','ASSS_current_F56A0']
    ]
    # --------------------------------------------------
    trash['AXXX']  = [
    'AXXX_current_301A',
    'AXXX_current_301B',
    'AXXX_current_301C',
    'AXXX_current_301E',
    'AXXX_current_302E',
    'AXXX_current_305A',
    'AXXX_current_305B',
    'AXXX_current_380A',
    'AXXX_current_380B',
    'AXXX_current_380C',
    'AXXX_current_380R',
    'AXXX_current_381A',
    'AXXX_current_381B',
    'AXXX_current_381C'
    ]
    merge['AXXX'] = [[]]
    # --------------------------------------------------
    trash['AACF']  = [
    'AACF_current_319O',
    'AACF_current_325B',
    'AACF_current_E90A',
    'AACF_current_E90B',
    'AACF_current_U07D',
    'AACF_current_M13A',
    'AACF_current_E92A',
    'AACF_current_325E',
    'AACF_current_325C',
    'AACF_current_325D',
    'AACF_current_M03A',
    ]         
    merge['AACF'] = [
    ['AACF_current_M21A','AACF_current_M22A','AACF_current_E70A',],
    ['AACF_current_M02A'],
    ['AACF_current_M06A'],
    ['AACF_current_M07A'],
    ['AACF_current_E03A'],
    ['AACF_current_E05A']
    ]
    # --------------------------------------------------   
    # --------------------------------------------------
    # This subsystem looks like a temperature indicator
    # We delete all commands and process it into a single "temperature" feature
    # --------------------------------------------------
    trash['ATMB']  = ['ATMB_current_003K'
                    ,'ATMB_current_022K'
                    ,'ATMB_current_045K'
                    ,'ATMB_current_057K'
                    ,'ATMB_current_076K'
                    ,'ATMB_current_091K'
                    ,'ATMB_current_114K'
                    ,'ATMB_current_152K'
                    ,'ATMB_current_182K'
                    ,'ATMB_current_228K']         
    merge['ATMB'] = [[]]
    
    # Creating a single signal
    atmb_cols = [i for i in pd_data.columns if i.startswith('ATMB_current_')]
    atmb_vals = np.copy(pd_data[atmb_cols[0]])*0
    temps = [3,22,45,57,76,91,114,152,182,228] 
    for i in range(1, np.size(atmb_cols)):
        atmb_vals += pd_data[atmb_cols[i]]*temps[i-1]
    # --------------------------------------------------    
    return trash, merge, atmb_vals
