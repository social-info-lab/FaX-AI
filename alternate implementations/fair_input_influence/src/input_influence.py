import numpy as np
import pandas as pd
import shap

def shap_aif(model, train_data, test_data):
    """
    Calculates SHAP values for models in the AIF360 sklearn packages.
    Args:
        model: the model to get the shap values of
        train_data: training samples
        test_data: test samples
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
    return shap_values, expected_value

def shap_results(shap_values,expected_value,test_data):
    """
    Aggregates the given SHAP values.
    Args:
        shap_values: shap values for a given model
        expected_value: expected value of predictions for a given model
        test_data: test samples
    """
    shap.summary_plot(shap_values, test_data)
    cols = test_data.columns.values
    #shap.decision_plot(expected_value, shap_values, cols)
    shap_mean = np.mean(np.abs(shap_values), axis=0)
    print("Influences:")
    for i, c in enumerate(cols):
        print(c, shap_mean[i])
    print()
    return shap_mean

def ate(model, estimator, X_test, prot_attr, prob=True, nsamp = 1000):
    """
    Computes ATE as defined in the text
    Args:
        model: the given model with a predict or predict_proba function
        estimator: base estimator, e.g., LogisticRegression from sklearn
        X_test: test samples
        prot_attr: string denoting the protected attribute
        prob: indicates whether to use predict or predict_proba function
        nsamp: number of samples
    """
    nde_li = [] #holds the nde for each sample for each feature
    X_test = X_test.sample(n=nsamp, replace=False)
    for col in X_test.columns.values:
        #X_test = Xi, Xj, Z
        X_test_col = X_test[col].values #Xi
        samp_col = X_test[col].sample(frac=1.0, replace=True).values #Xi''
        nde_col = [] #list to hold averaged nde for each N samples of Xi
        for j in range(X_test_col.shape[0]):
            X_test_samp = X_test.copy().sample(frac=1.0, replace=True) #Xi', Xj', Z' (sampling just Xj, Z would give same)
            X_test_samp[col] = samp_col[j] #replace Xi' to get Xi'',Xj', Z'
            X_test_samp_ncol = X_test_samp.copy() #Xi'', Xj', Z'
            X_test_samp_ncol[col] = X_test_col[j] #replace Xi'' to get Xi, Xj',Z'

            a = X_test_samp_ncol.copy()
            b = X_test_samp.copy()

            #since some models might rely on index = prot_attr
            index_names = list(X_test.index.names)
            if len(index_names)==1:
                a = a.set_index(index_names[0], drop=False)
                b = b.set_index(index_names[0], drop=False)
            else:
                a.index = pd.MultiIndex.from_arrays(a[index_names[1:]].values.T, names=index_names[1:])
                b.index = pd.MultiIndex.from_arrays(b[index_names[1:]].values.T, names=index_names[1:])

            if prob:
                Y = estimator.predict_proba(a)[:,1].flatten() #Y_{Xi,Xj',Z'}
                Yp = estimator.predict_proba(b)[:,1].flatten() #Y_{Xi'',Xj',Z'}
                Yhat = model.predict_proba(a)[:,1].flatten() #\hat{Y}_{Xi,Xj',Z'}
                Yhatp = model.predict_proba(b)[:,1].flatten() #\hat{Y}_{Xi'',Xj',Z'}
            else:
                Y = estimator.predict(a).flatten()
                Yp = estimator.predict(b).flatten()
                Yhat = model.predict(a).flatten()
                Yhatp = model.predict(b).flatten()
            #nde_j = np.array((Y - Yp) - (Yhat - Yhatp)).reshape(-1, 1)
            nde_j = np.array(Yhat - Yhatp).reshape(-1, 1)
            nde_col.append(np.mean(nde_j, axis=0))
        nde_li.append(np.array(nde_col)) #add the nde values for the feature to list
    nde_li = np.hstack(nde_li) #turn list into N X num col matrix
    return np.mean(np.abs(nde_li), axis=0) #absolute average over rows
