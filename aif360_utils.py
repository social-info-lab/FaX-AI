import numpy as np
import pandas as pd
import FaX_methods
import shap
from sklearn.model_selection import train_test_split
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction

class FaXAIF():
    """
    Layer on top of FaX AI methods for using dataframes.
    Layer is not explicitly needed but shows general way on how to convert
    dataframes from common ML-fairness libraries to the expected numpy arrays.

    This allows for AIF data to be used with FaX AI methods.

    Args:
        X_df: dataframe of training samples
        Y_df: dataframe of training outcomes
        prot_attr: name of protected attribute
        model_type: (MIM/optimization) which FaX AI algorithm to use
        influence: (shap/mde) which influence measure to use for OPT approach
        params: a dictionary of the parameters to be used for OPT approach
    """
    def __init__(self, X_df, Y_df, prot_attr, model_type='MIM', influence='shap', params=None):
        X = X_df[X_df.columns.drop(prot_attr)].values
        Z = X_df[prot_attr].values.reshape(-1, 1)
        Y = Y_df.values
        self.prot_attr = prot_attr
        if model_type == 'MIM':
            self.model = FaX_methods.MIM(X, Z, Y)
        elif model_type in ['optimization','OPT']:
            self.model = FaX_methods.Optimization(X, Z, Y, influence=influence, params=params)

    def predict_proba(self, X_df):
        X = X_df[X_df.columns.drop(self.prot_attr)].values
        return self.model.predict_proba(X)

    def predict(self, X_df):
        X = X_df[X_df.columns.drop(self.prot_attr)].values
        return self.model.predict(X)

def load_compas(prot_attr = 'race',subsample=0,train_size=0.7):
    """
    Loads and preprocesses the COMPAS dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        subsample: number of positive outcomes to remove for the disadvantaged group
        train_size: percentage of data to be used for training
    """
    X, y = fetch_compas(binary_race=True)
    X = X.drop(columns=['age_cat', 'c_charge_desc'])

    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
    y = 1 - pd.Series(y.factorize(sort=True)[0], index=y.index)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=train_size, random_state=1234567)

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

def load_census(prot_attr = 'sex',train_size=0.7):
    """
    Loads and preprocesses the Adult Census Income dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        train_size: percentage of data to be used for training
    """
    X, y, sample_weight = fetch_adult()

    X = X.drop(columns=['native-country'])

    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

    y = pd.Series(y.factorize(sort=True)[0], index=y.index)
    # there is one unused category ('Never-worked') that was dropped during dropna
    X.workclass.cat.remove_unused_categories(inplace=True)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=train_size, random_state=1234567)
    print(X_train.columns)

    X_train = pd.get_dummies(X_train)
    print(X_train.columns.values)
    X_test = pd.get_dummies(X_test)

    X_train = X_train.rename(columns={"race_White": "race", "sex_Male": "sex"})
    X_test = X_test.rename(columns={"race_White": "race", "sex_Male": "sex"})

    X_train = X_train[ [prot_attr] + [ col for col in X_train.columns if col not in [prot_attr,"race_White","race_Non-white","sex_Female","sex_Male"] ] ]
    X_test =X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr,"race_White","race_Non-white","sex_Female","sex_Male"] ] ]

    return X_train, X_test, y_train, y_test

def load_german(prot_attr = 'sex',train_size=0.7):
    """
    Loads and preprocesses the German Credit dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        train_size: percentage of data to be used for training
    """
    X, y = fetch_german(numeric_only=True)

    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

    y = pd.Series(y.factorize(sort=True)[0], index=y.index)

    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=train_size, random_state=1234567)
    X_train = X_train[[prot_attr] + [ col for col in X_train.columns if col not in [prot_attr] ] ]
    X_test =X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr] ] ]

    return X_train, X_test, y_train, y_test

def shap_aif(model, train_data, test_data, explainer_samples=100):
    """
    Middle layer to make AIF360 formated dataframes and methods work with the
    SHAP library.

    Args:
        model: trained AIF360 or sklearn compatible model
        train_data: data for integrating out features to generate explainer
        test_data: data to calculate the SHAP values for
        explainer_samples: number of samples to use when generating explainer
    """
    def transform_shap(x,Xt):
        df= pd.DataFrame(x, columns=Xt.columns)
        index_names = list(Xt.index.names)
        if len(index_names)==1:
            df = df.set_index(index_names[0], drop=False)
        else:
            df.index = pd.MultiIndex.from_arrays(df[index_names[1:]].values.T, names=index_names[1:])
        return df
    f = lambda x, Xt= test_data: model.predict_proba(transform_shap(x, Xt))[:,1]
    explainer = shap.KernelExplainer(f, shap.kmeans(train_data, 100))
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(test_data)
    return shap_values, expected_value, explainer

class ExponentiatedGradientReductionFix(ExponentiatedGradientReduction):
    """
    Fix for exponentiated gradient AIF implementation to work with SHAP.
    Original implementation errors on data following SHAP integrating out features.
    """
    def predict(self, X):
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.classes_[self.model.predict(X)]


    def predict_proba(self, X):
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.model._pmf_predict(X)
