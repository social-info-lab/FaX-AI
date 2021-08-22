import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from aif360.sklearn.preprocessing import ReweighingMeta
from aif360.sklearn.inprocessing import AdversarialDebiasing, ExponentiatedGradientReduction, GridSearchReduction
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
from aif360.sklearn.metrics import generalized_fnr, difference, statistical_parity_difference,equal_opportunity_difference
import matplotlib.pyplot as plt
import seaborn as sns

import cvxpy
import shap
from ast import literal_eval
from load_irl import load_compas, load_census, load_german
from models import ExponentiatedGradientReductionFix, mixture_model, marginal_mixture_model
from input_influence import shap_aif, shap_results, ate

def model_results(model, model_name, X_train_syn, X_test_syn, y_train_syn, y_test_syn, prot_attr,X1_ind, X2_ind,nsamples=1000,estimator=None,ii='shap'):
    """
    Generates the results for all measures for a given model and training+test data.
    """
    df_results = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er'])

    X_test_syn_sample = shap.sample(X_test_syn.copy(), nsamples)
    model.fit(X_train_syn, y_train_syn)
    if ii=='shap':
        model_shap, model_expected = shap_aif(model, X_train_syn, X_test_syn_sample)
        model_shap_values = shap_results(model_shap,model_expected, X_test_syn_sample)
    elif ii=='ate':
        model_shap_values = ate(model, estimator, X_test_syn.copy(), prot_attr, nsamp=nsamples)
    else:
        raise ValueError('Invalid input influence measure.')
    pred = model.predict(X_test_syn)

    acc = accuracy_score(y_test_syn, pred)
    z = np.sort(X_test_syn[prot_attr].unique())

    X_test_syn_0 = X_test_syn.iloc[X_test_syn.index.get_level_values(prot_attr) == z[0]].copy()
    X_test_syn_1 = X_test_syn.iloc[X_test_syn.index.get_level_values(prot_attr) == z[1]].copy()
    y_test_syn_0 = y_test_syn.iloc[y_test_syn.index.get_level_values(prot_attr) == z[0]].copy()

    pred0 = model.predict(X_test_syn_0)
    er0 = (1.0-accuracy_score(y_test_syn_0, pred0))/(1-acc)

    di = disparate_impact_ratio(y_test_syn, pred, prot_attr=prot_attr)
    dd = statistical_parity_difference(y_test_syn, pred, prot_attr=prot_attr)
    eqopp = equal_opportunity_difference(y_test_syn, pred, prot_attr=prot_attr)
    eqodd = average_odds_error(y_test_syn, pred, prot_attr=prot_attr)
    return [model_name, model_shap_values, model_shap_values[0], model_shap_values[X1_ind], model_shap_values[X2_ind], \
                                       acc, di, dd, eqopp, eqodd, er0]

def full_results(X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples=1000, ii='shap'):
    """
    Generates all results for all models.
    """
    df_results = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er'])

    print("Full")
    logReg = LogisticRegression(solver='lbfgs', max_iter=10000)
    df_results.loc[len(df_results)] = model_results(logReg, "full", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("Marg")
    X_train_marginal = X_train.copy()
    X_train_marginal[prot_attr] = 0
    marg_logReg = LogisticRegression(solver='lbfgs', max_iter=10000)
    df_results.loc[len(df_results)] = model_results(marg_logReg, "marginal", X_train_marginal, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("MIM")
    mm = marginal_mixture_model(prot_attr,LogisticRegression(solver='lbfgs', max_iter=10000))
    df_results.loc[len(df_results)] = model_results(mm, "mixt-probnorm", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("ReWeighing")
    rew = ReweighingMeta(estimator=LogisticRegression(solver='lbfgs', max_iter=10000))
    params = {'estimator__C': [1, 10], 'reweigher__prot_attr': [prot_attr]}
    clf = GridSearchCV(rew, params, scoring='accuracy', cv=5)
    df_results.loc[len(df_results)] = model_results(clf, "Reweighing", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("Adv Deb")
    adv_deb = AdversarialDebiasing(prot_attr=prot_attr, random_state=1234567, num_epochs=10)
    df_results.loc[len(df_results)] = model_results(adv_deb, "Adv Deb", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("Exp Grad Reduction DP")
    exp_grad =  ExponentiatedGradientReductionFix(prot_attr=prot_attr, estimator=LogisticRegression(solver='lbfgs', max_iter=10000),constraints='DemographicParity')
    df_results.loc[len(df_results)] = model_results(exp_grad, "Exp Grad DP", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("Exp Grad Reduction EO")
    exp_grad =  ExponentiatedGradientReductionFix(prot_attr=prot_attr, estimator=LogisticRegression(solver='lbfgs', max_iter=10000),constraints='EqualizedOdds')
    df_results.loc[len(df_results)] = model_results(exp_grad, "Exp Grad EO", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("Exp Grad Reduction TPR")
    exp_grad =  ExponentiatedGradientReductionFix(prot_attr=prot_attr, estimator=LogisticRegression(solver='lbfgs', max_iter=10000),constraints='TruePositiveRateDifference')
    df_results.loc[len(df_results)] = model_results(exp_grad, "Exp Grad TPR", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("Exp Grad Reduction ER")
    exp_grad =  ExponentiatedGradientReductionFix(prot_attr=prot_attr, estimator=LogisticRegression(solver='lbfgs', max_iter=10000),constraints='ErrorRateRatio')
    df_results.loc[len(df_results)] = model_results(exp_grad, "Exp Grad ER", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    print("CalEqOdd")
    cal_eq_odds = CalibratedEqualizedOdds(prot_attr, cost_constraint='fnr', random_state=1234567)
    log_reg = LogisticRegression(solver='lbfgs', max_iter= 10000)
    postproc = PostProcessingMeta(estimator=log_reg, postprocessor=cal_eq_odds, random_state=1234567)
    df_results.loc[len(df_results)] = model_results(postproc, "CalEqOdd", X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind,nsamples,estimator=logReg,ii=ii)

    df_results['influence diff']= df_results['influence'].copy()
    df_results.loc[:, "influence diff"] = df_results["influence diff"].apply(lambda x: x - df_results.iloc[0]['influence'])

    return df_results

def generate_results(X_train, X_test, y_train, y_test, prot_attr, X1_name, X2_name, path, nsamples, ii='shap'):
    """
    Generates all results and saves it to a csv
    """
    X1_ind = X_train.columns.get_loc(X1_name)
    X2_ind = X_train.columns.get_loc(X2_name)
    results = full_results(X_train, X_test, y_train, y_test, prot_attr, X1_ind, X2_ind, nsamples=nsamples, ii=ii)
    results.to_csv(path+'_results.csv', index=False)
    return results

def trial_compas(num_trials=10, path = 'irl/compas/compas_',nsamples = 100, ii='shap'):
    """
    Runs n tirals for the compas data and save the results and each trial to a csv.
    """
    boot_results = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er'])
    for i in range(num_trials):
        print('Trial: ', i)
        prot_attr = 'race'
        X, y = fetch_compas(binary_race=True)
        X = X.drop(columns=['age_cat', 'c_charge_desc'])

        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
        y = 1 - pd.Series(y.factorize(sort=True)[0], index=y.index)

        X = pd.get_dummies(X)

        X = X.rename(columns={"race_Caucasian": "race", "sex_Female": "sex"})
        X = X[ [prot_attr] + [ col for col in X.columns if col not in [prot_attr,"c_charge_degree_M","race_Caucasian","race_African-American","sex_Male"] ] ]

        rep_ind = y.iloc[y.index.get_level_values('race') == 0].loc[lambda x : x==1][:500].index
        y = y.drop(rep_ind)
        X = X.drop(rep_ind)

        X_train = X.sample(frac=1,replace=True,random_state=i)
        y_train = y[X_train.index]

        X_test = X.drop(X_train.index)
        y_test = y.drop(y_train.index)

        prot_attr = 'race'
        path1 = path+ str(i)
        X1 = "age"
        X2 = "priors_count"

        compas_results = generate_results(X_train, X_test, y_train, y_test, prot_attr, X1, X2, path1, nsamples, ii=ii)
        boot_results = boot_results.append(compas_results, ignore_index=True)
    boot_results.to_csv(path+'_results.csv', index=False)
    return boot_results

def trial_census(num_trials=10, path = 'irl/census/census_',nsamples = 100, ii='shap'):
    """
    Runs n tirals for the census data and save the results and each trial to a csv.
    """
    census_boot_results = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er'])
    for i in range(num_trials):
        print('Trial: ', i)
        prot_attr = 'sex'
        path1 = path +str(i)
        X1 = "relationship_Wife"
        X2 = "relationship_Unmarried"
        X, y, sample_weight = fetch_adult()

        X = X.drop(columns=['native-country'])

        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

        y = pd.Series(y.factorize(sort=True)[0], index=y.index)
        # there is one unused category ('Never-worked') that was dropped during dropna
        X.workclass.cat.remove_unused_categories(inplace=True)
        X = pd.get_dummies(X)
        X = X.rename(columns={"race_White": "race", "sex_Male": "sex"})

        X = X[ [prot_attr] + [ col for col in X.columns if col not in [prot_attr,"race_White","race_Non-white","sex_Female","sex_Male"] ] ]

        X_train = X.sample(frac=1,replace=True,random_state=i)
        y_train = y[X_train.index]

        X_test = X.drop(X_train.index)
        y_test = y.drop(y_train.index)

        census_results = generate_results(X_train, X_test, y_train, y_test, prot_attr, X1, X2, path1, nsamples, ii=ii)
        census_boot_results = census_boot_results.append(census_results, ignore_index=True)
    census_boot_results.to_csv(path+'_results.csv', index=False)
    return census_boot_results

def trial_german(num_trials=10, path = 'irl/german/german_',nsamples = 100, ii='shap'):
    """
    Runs n tirals for the German data and save the results and each trial to a csv.
    """
    german_boot_results = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er'])
    for i in range(num_trials):
        print('Trial: ', i)
        prot_attr = 'sex'
        pathi = path +str(i)
        X1 = "num_dependents"
        X2 = "age"
        X, y = fetch_german(numeric_only=True)

        X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
        y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)

        y = pd.Series(y.factorize(sort=True)[0], index=y.index)
        X = pd.get_dummies(X)
        X = X[[prot_attr] + [ col for col in X.columns if col not in [prot_attr] ] ]

        X_train = X.sample(frac=1,replace=True,random_state=i)
        y_train = y[X_train.index]

        X_test = X.drop(X_train.index)
        y_test = y.drop(y_train.index)

        german_results = generate_results(X_train, X_test, y_train, y_test, prot_attr, X1, X2, pathi, nsamples, ii=ii)
        german_boot_results = german_boot_results.append(german_results, ignore_index=True)
    german_boot_results.to_csv(path+'_results.csv', index=False)
    return german_boot_results

def generate_results_synthetic(path, nsamples, cor_code='ll', yname='log_disc', coeff_code='111',ii='shap'):
    """
    Generates results for the synthetic data and save the results to a csv.
    Synthetic data is provided in csv format.
    """
    f_li = []
    f_li.append("full_zzz_"+coeff_code+".csv")
    for i in range(10):
            f= "full_"+str(i)+cor_code+'_'+coeff_code+".csv"
            f_li.append(f)
    yname += "_"+coeff_code

    results_corrs = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er','ind'])
    for i,fname in enumerate(f_li):
        print("--------------------------------------------------------------------------")
        print(fname)
        Xt = pd.read_csv("..\data\\"+yname+"\Xt_"+fname, names=['Z','X1', 'X2', "Xint"], header=None)
        Y = pd.read_csv("..\data\\"+yname+"\Y_"+fname, names=['Y'], header=None)
        Xt = Xt.set_index("Z", drop=False)
        Y = pd.Series(Y['Y'].values, index=Xt['Z'])
        (X_train, X_test, y_train, y_test) = train_test_split(Xt, Y, train_size=0.7, random_state=1234567)
        results = full_results(X_train, X_test, y_train, y_test,'Z',X_train.columns.get_loc('X1'),X_train.columns.get_loc('X2'), nsamples=nsamples,ii=ii)
        results['ind']=i*0.1
        results_corrs= results_corrs.append(results, ignore_index=True)

    results_corrs.to_csv(path+'_results.csv', index=False)
    return results_corrs

def generate_results_synthetic_trials(path, nsamples, cor_code='ll', yname='log_disc', coeff_code='111', num_trials=20,ii='shap'):
    """
    Generates results for n trials for the synthetic data and save the results to a csv.
    Synthetic data is provided in csv format.
    """
    f_li = []
    f_li.append("full_zzz_"+coeff_code+".csv")
    for i in range(10):
            f= "full_"+str(i)+cor_code+'_'+coeff_code+".csv"
            f_li.append(f)
    yname += "_"+coeff_code

    results_corrs = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er','ind'])
    for n in range(num_trials):
        print('Trial: ', n)
        results_corrs_i = pd.DataFrame(columns = ['model','influence','influence_prot_attr','influence_X1','influence_X2','acc','di','dd','eqopp','eqodd','er','ind'])
        for i,fname in enumerate(f_li):
            print("--------------------------------------------------------------------------")
            print(fname)
            Xt = pd.read_csv("..\data\\"+yname+"\Xt_"+fname, names=['Z','X1', 'X2', "Xint"], header=None)
            Y = pd.read_csv("..\data\\"+yname+"\Y_"+fname, names=['Y'], header=None)
            X_train = Xt.sample(frac=1,replace=True,random_state=n)
            y_train = Y.iloc[X_train.index]
            X_test = Xt.drop(X_train.index)
            y_test = Y.drop(y_train.index)

            X_train = X_train.set_index("Z", drop=False)
            X_test = X_test.set_index("Z", drop=False)
            y_train = pd.Series(y_train['Y'].values, index=X_train['Z'])
            y_test = pd.Series(y_test['Y'].values, index=X_test['Z'])
            results = full_results(X_train, X_test, y_train, y_test,'Z',X_train.columns.get_loc('X1'),X_train.columns.get_loc('X2'), nsamples=nsamples,ii=ii)
            results['ind']=i*0.1
            results_corrs= results_corrs.append(results, ignore_index=True)
            results_corrs_i= results_corrs_i.append(results, ignore_index=True)
        results_corrs_i.to_csv(path+'trial_'+str(n)+'_results.csv', index=False)

    results_corrs.to_csv(path+'_results.csv', index=False)
    return results_corrs

def transform_result_df(data):
    """
    Converts strings in dataframes to lists
    """
    for s in ['influence', 'influence diff']:
        data[s] = data[s].apply(lambda x: x.replace('\n',''))
        for i in range(10):
            data[s] = data[s].apply(lambda x: x.replace('[ ', '['))
            data[s] = data[s].apply(lambda x: x.replace(' '+str(i)+'.', ','+str(i)+'.'))
            data[s] = data[s].apply(lambda x: x.replace(' -'+str(i)+'.', ',-'+str(i)+'.'))
        data[s] = data[s].apply(lambda x: literal_eval(x))
    return data

def read_trial_results(path, num_trials):
    """
    Reads, converts, and combines the results for each trial
    """
    li = []
    for i in range(num_trials):
        df = pd.read_csv(path +str(i)+'_results.csv')
        li.append(df)
    data = pd.concat(li, axis=0, ignore_index=True)
    data = transform_result_df(data.copy())
    return data

def gen_errors(data):
    """
    Finds mean and error from dataframes that have results for n trials.
    """
    df_err = pd.DataFrame(columns = data.columns)
    models = [ c for c in data['model'].unique() if c not in ['mixt-probnorm','full','marginal'] ] + ['mixt-probnorm','full','marginal']
    for m in models:#models:
        m_results = data.loc[data['model'] == m].copy()
        n = m_results.shape[0]
        m_results.loc['mean'] = m_results.mean()
        m_results.loc['ci'] = 1.96*m_results.std()/np.sqrt(n)
        for s in ['influence', 'influence diff']:
            inf = np.array(m_results[:-2][s].tolist())
            m_inf = inf.mean(axis=0)
            m_std = inf.std(axis=0)
            m_results.at['mean', s] = m_inf
            m_results.at['ci', s] = 1.96*m_std/np.sqrt(n)
        m_results.at['mean', 'model'] = m
        m_results.at['ci', 'model'] = m
        mci_results = m_results[-2:].copy()
        df_err = df_err.append(mci_results)
    return df_err
