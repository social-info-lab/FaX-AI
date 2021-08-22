import numpy as np
import pandas as pd
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_compas(prot_attr = 'race',subsample=0):
    """
    Loads and prepocesses the compas dataset using AIF.
    Args:
        prot_attr: the protected attribute
        subsample: how many positive labels for underprivledged group to remove
    """
    X, y = fetch_compas(binary_race=True)
    X = X.drop(columns=['age_cat', 'c_charge_desc'])

    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
    y = 1 - pd.Series(y.factorize(sort=True)[0], index=y.index)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train = X_train.rename(columns={"race_Caucasian": "race", "sex_Female": "sex"})
    X_test = X_test.rename(columns={"race_Caucasian": "race", "sex_Female": "sex"})
    X_train = X_train[ [prot_attr] + [ col for col in X_train.columns if col not in [prot_attr,"c_charge_degree_M","race_Caucasian","race_African-American","sex_Male"] ] ]
    X_test = X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr,"c_charge_degree_M","race_Caucasian","race_African-American","sex_Male"] ] ]

    if subsample:
        rep_ind = y_train.iloc[y_train.index.get_level_values(prot_attr) == 0].loc[lambda x : x==1][:subsample].index
        y_train = y_train.drop(rep_ind)
        X_train = X_train.drop(rep_ind)

    return X_train, X_test, y_train, y_test

def load_census(prot_attr = 'sex'):
    """
    Loads and prepocesses the census dataset using AIF.
    Args:
        prot_attr: the protected attribute
    """
    X, y, sample_weight = fetch_adult()

    X = X.drop(columns=['native-country'])

    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

    y = pd.Series(y.factorize(sort=True)[0], index=y.index)
    # there is one unused category ('Never-worked') that was dropped during dropna
    X.workclass.cat.remove_unused_categories(inplace=True)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train = X_train.rename(columns={"race_White": "race", "sex_Male": "sex"})
    X_test = X_test.rename(columns={"race_White": "race", "sex_Male": "sex"})

    X_train = X_train[ [prot_attr] + [ col for col in X_train.columns if col not in [prot_attr,"race_White","race_Non-white","sex_Female","sex_Male"] ] ]
    X_test =X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr,"race_White","race_Non-white","sex_Female","sex_Male"] ] ]

    return X_train, X_test, y_train, y_test

def load_german(prot_attr = 'sex'):
    """
    Loads and prepocesses the German credit dataset using AIF.
    Args:
        prot_attr: the protected attribute
    """
    X, y = fetch_german(numeric_only=True)

    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

    y = pd.Series(y.factorize(sort=True)[0], index=y.index)

    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)
    X_train = X_train[[prot_attr] + [ col for col in X_train.columns if col not in [prot_attr] ] ]
    X_test =X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr] ] ]

    return X_train, X_test, y_train, y_test
