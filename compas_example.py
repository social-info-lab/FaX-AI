import fair_mass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from aif360.sklearn.datasets import fetch_compas


def load_compas(prot_attr = 'race'):
    """
    Loads the COMPAS dataset from the AIF library and prepares it for use.
    Note: reqiures AIF library to be installed
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

    return X_train, X_test, y_train, y_test

class OptimizationAIF():
    """
    Layer on top of Optimization method for using dataframes.
    Layer is not explicitly needed but shows general way on how to convert
    dataframes from common ML-fairness libraries to the expected numpy arrays.

    This allows for AIF data to be used with Optimization method.
    """
    def __init__(self, X_df, Y_df, prot_attr, influence='shap', params=None):
        X = X_df[X_df.columns.drop(prot_attr)].values
        Z = X_df[prot_attr].values.reshape(-1, 1)
        Y = Y_df.values
        self.prot_attr = prot_attr
        self.model = fair_mass.Optimization(X, Z, Y, influence=influence, params=params)

    def predict_proba(self, X_df):
        X = X_df[X_df.columns.drop(self.prot_attr)].values
        return self.model.predict_proba(X)

    def predict(self, X_df):
        X = X_df[X_df.columns.drop(self.prot_attr)].values
        return self.model.predict(X)

X_train_compas, X_test_compas, y_train_compas, y_test_compas = load_compas()
prot_attr = 'race'
#create an optimization model with the COMPAS data
#note that we already had X and Z seperated
model_aif = OptimizationAIF(X_train_compas, y_train_compas, prot_attr, params = {'num_epochs': 10, 'learning_rate': 1e-3})

#predict the test data using the model
print(model_aif.predict_proba(X_test_compas))
print(model_aif.predict(X_test_compas))
