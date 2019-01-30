import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, OneHotEncoder, CategoricalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# This class creates the Training and Test set using random sampling and stratified sampling
class Create_Train_Test:
    def __init__(self, data):
        self.data = data

    def split_train_test(self, test_ratio):
        shuffled_indices = np.random.permutation(len(self.data))
        test_set_size = int(len(self.data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        return self.data.iloc[train_indices], self.data.iloc[test_indices]

    def test_set_check(self, identifier, test_ratio, hash):
        return hash(np.int64(identifier)).digest()[-1] < (256 * test_ratio)

    def split_train_test_by_id(self, test_ratio, hash=hashlib.md5):
        data_with_id = self.data.reset_index()
        ids = data_with_id['index']
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio, hash))

        return data_with_id.loc[~in_test_set], data_with_id.loc[in_test_set]

    def stratified_sampling(self, test_ratio, category_strata_name, category_name):
        self.data[category_strata_name] = np.ceil(self.data[category_name] / 1.5)
        self.data[category_strata_name].where(self.data[category_strata_name] < 5, 5.0, inplace=True)

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        for train_index, test_index in split.split(self.data, self.data[category_strata_name]):
            strat_train_set = self.data.loc[train_index]
            strat_test_set = self.data.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop(category_strata_name, axis=1, inplace=True)

        return strat_train_set, strat_test_set


# this class prepares the data by droping text attributes, cleaning data (median),
# categorizing the text attributes, combining number attributes (new columns)
# feature scaling (MinMax)
class Data_Prepare:
    def __init__(self, data):
        self.data = data

    def drop_text_attribute(self, column_name, data):
        self.data = data.drop(column_name, axis=1)
        return self.data

    def data_clean(self, column_name, data, missing_value_option=3):
        if missing_value_option is 1:
            self.data = data
            self.data.dropna(subset=column_name)
        elif missing_value_option is 2:
            self.data = data
            self.data.drop(column_name, axis=1)
        elif missing_value_option is 3:
            median = self.data[column_name].median()
            self.data = data
            self.data[column_name].fillna(median, inplace=True)

            imputer = SimpleImputer(strategy='median')
            imputer.fit(self.data)
            X = imputer.transform(self.data)

            self.data = pd.DataFrame(X, columns=data.columns)
        else:
            pass

        return self.data

    def handling_text(self, column_name, data):
        data_cat_encoded, data_cat = data[column_name].factorize()
        data_cat_encoded_reshape = data_cat_encoded.reshape(-1,1)

        encoder = OneHotEncoder()
        data_cat_encoded_reshape_1hot = encoder.fit_transform(data_cat_encoded_reshape)

        return data_cat_encoded_reshape_1hot

    def combine_num_attribute(self, data, new_column_name, numerator_col_name, denominator_col_name, type='fit'):
        if type is 'fit':
            return self
        elif type is 'transform':
            self.data = data
            self.data[new_column_name] = data[numerator_col_name] / data[denominator_col_name]
        return self.data

    def feature_scaling(self, data, type, min=0, max=1):
        self.data = data.as_matrix()
        if type is 'MinMax':
            scaler = MinMaxScaler(feature_range=(min,max))
            data_scaled = scaler.fit_transform(self.data)


        return data_scaled

