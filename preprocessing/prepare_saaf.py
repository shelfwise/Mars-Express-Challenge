# -*- coding: utf-8 -*-
"""
@author: fornax
"""
import os
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.dirname(os.getcwd()))
import prepare_data1 as prep
DATA_PATH = os.path.join('..', prep.DATA_PATH)


def convert_time(df):
    df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
    return df

    
def signal_reconstruct(timeT, df, time_diff_thresh, value_diff_thresh, interpolate = True):
    
    time = np.ceil(timeT)
    xt = np.arange(time[0],time[-1],1000)
                
    r_sx = np.zeros(xt.size)
    r_sy = np.zeros(xt.size)
    r_sz = np.zeros(xt.size)
    r_sa = np.zeros(xt.size)
    
    density = np.zeros(xt.size)
    
    for i,sample in enumerate(time[:-1]):
        
        density[int(sample - time[0])/1000] = 1
    
        for signal, r_signal in zip([df["sx"].values, df["sy"].values, df["sz"].values, df["sa"].values],
                                        [r_sx, r_sy, r_sz, r_sa]):    
    
        # if time step is smaller than threshold    
            if (time[i+1] - time[i])/1000 < time_diff_thresh:   
                    # if value diff is smaller than threshold
                    if np.abs(signal[i+1]-signal[i]) < value_diff_thresh:
                        if interpolate:
                            r_signal[(time[i]-time[0])/1000:(time[i+1]-time[0])/1000] \
                                = np.linspace(signal[i],signal[i+1],(time[i+1]-time[i])/1000)
                        else:
                            r_signal[(sample-time[0])/1000:(time[i+1]-time[0])/1000] = signal[i]
            else:
                r_signal[(time[i]-time[0])/1000:(time[i+1]-time[0])/1000] = 0
    

    print "xt.size: ", xt.size
        
    return xt, r_sx, r_sy, r_sz, r_sa, density#, reconstructed_density


def signal_statistics(df, time_stamp_size):
    df_cos = (df[["sa","sx","sy","sz"]]*np.pi/180).apply(np.cos)
    df_cos.columns  = ["sa_cos", "sx_cos", "sy_cos", "sz_cos"]
    df_cos = df_cos.resample(time_stamp_size).mean()  
    df = df.resample(time_stamp_size).mean()
    df_final = pd.concat([df, df_cos], axis=1)
    return df_final
    

saaf = pd.read_csv(os.path.join(DATA_PATH, 'saaf.csv'))
t, x, y, z, a, d = signal_reconstruct(saaf["ut_ms"].values, saaf, 1e+7, 90, interpolate = True)

r_saaf = pd.DataFrame({"ut_ms" : t, "sx": x, "sy": y, "sz": z, "sa": a, "density": d})

r_saaf = convert_time(r_saaf)
r_saaf = r_saaf.set_index('ut_ms')

final_re = signal_statistics(r_saaf, "60S")

final_re.to_csv(os.path.join(DATA_PATH, 'saaf_processed.csv'))
