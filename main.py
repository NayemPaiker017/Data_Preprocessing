import os
import tarfile
from six.moves import urllib
import pandas as pd
from data_preparation.data_preparation import Create_Train_Test, Data_Prepare

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

# fetch the data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# load the data
def load_houding_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def Main():
    #1 ---------- getting the data ---------------
    print('Fetching and extracting the data ...')
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    print('Loading the data from datasets/housing folder')
    housing = load_houding_data(HOUSING_PATH)


    #2 ---------- creating training and test set ----------------
    data = Create_Train_Test(housing)

    print('Creating Training and Test set using stratified sampling')
    #train_set, test_set = data.split_train_test_by_id(0.2)
    strat_train_set, strat_test_set = data.stratified_sampling(0.2, 'income_cat', 'median_income')

    housing = strat_test_set.copy()


    #3 --------- Copy Training and Test Set ----------
    strat_train_copy = strat_train_set.copy()
    strat_test_copy = strat_test_set.copy()


    #4 --------- Data Preparation ----------
    data_preparation = Data_Prepare(strat_train_copy)


    # drop text attribute
    print('Droping the text attribute called ocean_proximity')
    strat_train_no_text = data_preparation.drop_text_attribute('ocean_proximity', strat_train_copy)

    # data clean
    print('Data cleaning using median to take care of the missing data')
    strat_train_no_text_clean = data_preparation.data_clean(column_name='total_bedrooms', data=strat_train_no_text, missing_value_option=3)

    # handling text
    print('Categorizing the text attribute')
    strat_train_text_handled = data_preparation.handling_text(column_name='ocean_proximity', data=strat_train_copy)


    # combine number attribute
    print('creating new columns called rooms_per_household, bedrooms_per_room, population_per_household using columns called '
        'total_rooms, households, total_bedrooms, population, households')
    strat_train_no_text_clean_comb_num_attr_1 = data_preparation.combine_num_attribute(data=strat_train_no_text_clean,
                                                                                     new_column_name='rooms_per_household',
                                                                                     numerator_col_name='total_rooms',
                                                                                     denominator_col_name='households',
                                                                                     type='transform')

    strat_train_no_text_clean_comb_num_attr_2 = data_preparation.combine_num_attribute(data=strat_train_no_text_clean_comb_num_attr_1,
                                                                                       new_column_name='bedrooms_per_room',
                                                                                       numerator_col_name='total_bedrooms',
                                                                                       denominator_col_name='total_rooms',
                                                                                       type='transform')


    strat_train_no_text_clean_comb_num_attr_3 = data_preparation.combine_num_attribute(data=strat_train_no_text_clean_comb_num_attr_2,
                                                                                       new_column_name='population_per_household',
                                                                                       numerator_col_name='population',
                                                                                       denominator_col_name='households',
                                                                                       type='transform')


    #5 ---------- Print ----------
    print('-'*20)
    print(strat_train_no_text_clean_comb_num_attr_3[:1])
    strat_train_no_text_clean_comb_num_attr_3_minScaled = data_preparation.feature_scaling(data=strat_train_no_text_clean_comb_num_attr_3,
                                                                                           type='MinMax', min=0, max=1)


    print(strat_train_no_text_clean_comb_num_attr_3.info())
    print('-' * 20, '\n', strat_train_no_text_clean_comb_num_attr_3.head())
    print('-'*20,'\n', strat_train_no_text_clean_comb_num_attr_3_minScaled[:5])









if __name__ == '__main__':
    Main()