import pandas as pd
import re
# import pickle

from typing import List
import re
from datetime import time

import numpy as np
import pandas as pd
import catboost as cb
from utilitis import *


class SKPredModel:
    def __init__(self, models_folder_path: str):
        """
        Here you initialize your model
        """
        h5_files = find_h5_files(models_folder_path)
        self.models = []
        for m_file in h5_files:
            m = cb.CatBoostRegressor()
            lb, hb = extract_numbers_from_filename(m_file)
            self.models.append({'model': m.load_model(fname=m_file),
                           'lb': lb,
                           'hb': hb
                           })

        self.limits = []
        for model in self.models:
            self.limits.append((model['lb'], model['hb']))


    def prepare_df(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Here you put any feature generation algorithms that you use in your model

        :param test_df:
        :return: test_df extended with generated features
        """
        test_df.fillna(0, inplace=True)
        test_df.Submit = pd.DatetimeIndex(test_df.Submit)
        test_df.Start = pd.DatetimeIndex(test_df.Start)
        test_df = convert_tl_el(test_df)
        test_df = add_weekend_features(test_df)
        test_df = encode_area(test_df)
        test_df = encode_partition(test_df)
        test_df = encode_state(test_df)
        test_df = get_name_len(test_df)
        test_df.drop(columns=['JobName', 'ExitCode'], inplace=True)
        test_df.sort_values(by=['UID', 'Submit'], inplace=True)
        test_df = test_df.groupby('UID').apply(calculate_features_uid)
        test_df.sort_values(by=['Area', 'Submit'], inplace=True)
        test_df = test_df.groupby('Area').apply(calculate_features_area)
        test_df.drop(columns=['Submit', 'Start', 'UID', 'State', 'dayofweek_name', 'Elapsed'])
        datasets = split_datasets(test_df, self.limits)
        return datasets

    def predict(self, datasets: pd.DataFrame) -> pd.Series:
        """
        Here you implement inference for your model

        :param test_df: dataframe to predict
        :return: vector of estimated times in milliseconds
        """
        import numpy as np
        predicts = []
        for m, d in zip(self.models, datasets):
            model = m['model']
            X = d.drop(columns=['Submit', 'Start', 'UID', 'State', 'dayofweek_name', 'Elapsed'])
            y = d['Elapsed']
            pred = model.predict(X)
            predicts.append(pred)
        result = np.concatenate(predicts)

        return pd.Series(result)

# Usage example
'''
test_df = pd.read_csv('data/train_w_areas_st_till_june.csv',index_col=0)
test_df = test_df[7000:12000]
model = SKPredModel('models_catboost2')
prepared = model.prepare_df(test_df.copy())
predictions = model.predict(prepared)

from sklearn.metrics import r2_score
r2_score(y_pred=predictions, y_true=test_df['Elapsed'].apply(convert_time_to_seconds))
'''
