import re
from datetime import time

import numpy as np
import pandas as pd
import os

regexp_time_list = [
    r'(?P<days>.+?(?=-))-(?P<hours>.+?(?=:)):(?P<minutes>.+?(?=:)):(?P<seconds>\d+)',
    r'(?P<hours>.+?(?=:)):(?P<minutes>.+?(?=:)):(?P<seconds>\d+)'
]
compiled_regexps = [re.compile(regexp) for regexp in regexp_time_list]

def convert_time_to_seconds(element):
    for rcompile in compiled_regexps:
        rsearch = rcompile.search(element)
        if rsearch:
            try:
                return (int(rsearch.group('days')) * 24 + int(rsearch.group('hours'))) * 3600 + int(
                    rsearch.group('minutes')) * 60 + int(rsearch.group('seconds'))
            except:
                return int(rsearch.group('hours')) * 3600 + int(rsearch.group('minutes')) * 60 + int(
                    rsearch.group('seconds'))

def convert_tl_el(df):
    df['Timelimit'] = df['Timelimit'].apply(convert_time_to_seconds)
    df['Elapsed'] = df['Elapsed'].apply(convert_time_to_seconds)
    return df

def add_weekend_features(df):
    df['Submit'] = pd.to_datetime(df['Submit'])
    df['dayofweek'] = df['Submit'].dt.dayofweek
    import numpy as np
    df['dayofweek_name'] = df['Submit'].dt.day_name()
    df['is_weekend'] = np.where(df['dayofweek_name'].isin(['Sunday', 'Saturday']), 1, 0)
    df['year'] = df['Submit'].dt.year
    df['quarter'] = df['Submit'].dt.quarter
    df['month'] = df['Submit'].dt.month
    df['week'] = df['Submit'].dt.weekofyear
    df['day'] = df['Submit'].dt.day
    return df

def encode_area(df):
    df['Area'].replace({'geophys': 0,
    'radiophys': 1,
    'phys': 2,
    'bioinf': 4,
    'mach': 5,
    'biophys': 6,
    'it': 7,
    'mech': 8,
    'energ': 9,
    'astrophys': 10}, inplace=True)
    #df['Area'] = df['Area'].astype(int)
    return df

def encode_partition(df):
    df['Partition'].replace({'tornado': 0,
 'g2': 1,
 'cascade': 2,
 'tornado-k40': 3,
 'nv': 4}, inplace=True)
    df['Partition'] = df['Partition'].astype(int)
    return df

def encode_state(df):
    df['State'].replace({
    'COMPLETED': 0,
    'FAILED': 1,
    'TIMEOUT': 2,
    'NODE_FAIL': 3,
    'OUT_OF_MEMORY': 4
}, inplace=True)
    df['State'] = df['State'].replace(r'(CANCELLED.+)|(CANCELLED)', 5, regex=True)
    #df['State'] = df['State'].astype(int)
    return df

def get_name_len(df):
    df['NameLen'] = df['JobName'].apply(len)
    return df

def calculate_features_uid(group, suffix='uid', windows=[1,5,10], quant=[0.1, 0.3, 0.7, 0.9]):
    for k in windows:
        group['mean_elapsed'+'_'+suffix+str(k)] = group['Elapsed'].expanding(min_periods=k).mean()
        group['mean_timelimit'+'_'+suffix+str(k)] = group['Timelimit'].expanding(min_periods=k).mean()
        group['succeeded_task_count'+'_'+suffix+str(k)] = group['State'].expanding(min_periods=group.shape[0]//(100//k)).apply(lambda x: (x == 0).sum())
        group['failed_task_count'+'_'+suffix+str(k)] = group['State'].expanding(min_periods=group.shape[0]//(100//k)).apply(lambda x: (x == 1).sum())
        group['timeout_task_count'+'_'+suffix+str(k)] = group['State'].expanding(min_periods=group.shape[0]//(100//k)).apply(lambda x: (x == 2).sum())
    for q in quant:
        group['q_elapsed'+'_'+suffix+str(q)] = group['Elapsed'].expanding(min_periods=15).quantile(q)
        group['q_timelimit'+'_'+suffix+str(q)] = group['Timelimit'].expanding(min_periods=15).quantile(q)
    return group

def calculate_features_area(group, suffix='area', windows=[1,5,10], quant=[0.1, 0.3, 0.7, 0.9]):
    for k in windows:
        group['mean_elapsed'+'_'+suffix+str(k)] = group['Elapsed'].expanding(min_periods=k).mean()
        group['mean_timelimit'+'_'+suffix+str(k)] = group['Timelimit'].expanding(min_periods=k).mean()
        group['succeeded_task_count'+'_'+suffix+str(k)] = group['State'].expanding(min_periods=group.shape[0]//(100//k)).apply(lambda x: (x == 0).sum())
        group['failed_task_count'+'_'+suffix+str(k)] = group['State'].expanding(min_periods=group.shape[0]//(100//k)).apply(lambda x: (x == 1).sum())
        group['timeout_task_count'+'_'+suffix+str(k)] = group['State'].expanding(min_periods=group.shape[0]//(100//k)).apply(lambda x: (x == 2).sum())
    for q in quant:
        group['q_elapsed'+'_'+suffix+str(q)] = group['Elapsed'].expanding(min_periods=15).quantile(q)
        group['q_timelimit'+'_'+suffix+str(q)] = group['Timelimit'].expanding(min_periods=15).quantile(q)
    return group




#######################
def extract_numbers_from_filename(filename):
    match = re.search(r'(\d+)_(\d+)\.h5$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None, None


def find_h5_files(directory):
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    return h5_files

def split_datasets(df, lims):
    dsets = []
    for limit in lims:
        d = df[(df['Timelimit'] < limit[1]) & (df['Timelimit'] >= limit[0])].copy()
        dsets.append(d)
    return dsets

