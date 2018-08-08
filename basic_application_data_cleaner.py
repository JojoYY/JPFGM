"""This module performs basic cleaning for the application training and testing data."""

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer


def read_raw_application_data(path_to_kaggle_data='~/kaggle_JPFGM/Data/'):
    """Read in raw csv files of application train and test data."""
    app_train = pd.read_csv(path_to_kaggle_data + 'application_train.csv')
    print('Raw training data shape: ', app_train.shape)

    app_test = pd.read_csv(path_to_kaggle_data + 'application_test.csv')
    print('Raw testing data shape: ', app_test.shape)
    return app_train, app_test


def wrangle_application_train_test_data(app_train, app_test):
    """Wrangle application data for given train and test subset.

    Parameters:
        app_train: training data, with columns in application_train.csv
        app_test: testing data, with columns in application_test.csv
    """
    df_train = app_train.copy()
    df_test = app_test.copy()

    """Transformations that are the same for train and test set"""
    df_train = fix_application_data_anomalies(df_train)
    df_test = fix_application_data_anomalies(df_test)

    """Impute missing values for numerical columns based on train set"""
    imputer = Imputer(strategy='median')  # Median after fixing anomalies

    train_num_cols = list(df_train.drop('TARGET', axis=1)._get_numeric_data().columns)
    test_num_cols = list(df_test._get_numeric_data().columns)

    df_train[train_num_cols] = imputer.fit_transform(df_train[train_num_cols])
    df_test[test_num_cols] = imputer.transform(df_test[test_num_cols])

    """Encode categorical attributes based on train set"""
    df_train = pd.get_dummies(df_train, dummy_na=True, drop_first=True)
    df_test = pd.get_dummies(df_test, dummy_na=True, drop_first=True)

    # align data frames according to training data
    train_labels = df_train['TARGET']

    df_train, df_test = df_train.drop('TARGET', axis=1).align(
        df_test, join='left', fill_value=0, axis=1
    )  # df_test may have columns full of 0, which is ok for prediction

    # Add the target back in
    df_train['TARGET'] = train_labels

    print('Cleaned training data shape: ', df_train.shape)
    print('Cleaned testing data shape: ', df_test.shape)

    return df_train, df_test


def fix_application_data_anomalies(app_data):
    """Wrangle application data column anomalies.

    Modifies column values and adds and deletes columns, only if it pertains exactly as stated
    to any subset of training and testing data.
    """
    df = app_data.copy()

    """Fix DAYS_EMPLOYED anomalies"""
    # Create an anomalous flag column
    df['DAYS_EMPLOYED_ANOM'] = (df["DAYS_EMPLOYED"] == 365243)

    # Replace the anomalous values with nan
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    """make DAYS_BIRTH positive"""
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])

    """Impute missing values for categorical columns"""
    categ_cols = list(df.select_dtypes(include=['object']).columns)
    df[categ_cols].fillna(value='Unknown', inplace=True)

    return df
