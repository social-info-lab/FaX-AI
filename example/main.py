import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import fair_mass

import helper, aif_methods, aif_metrics
import numpy as np
import time, torch, pathlib
from sklearn.linear_model import LogisticRegression as LR
from multiprocessing.pool import Pool
import tensorflow as tf

# aif requires this
tf.compat.v1.disable_eager_execution()


def get_method(struct, data):

    # Full Method
    if struct.name == 'full':
        model = LR().fit(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    if struct.name == 'partial':
        model = LR().fit(data['X_train'], data['Y_train'])
        test = {'input': data['X_test'], 'output': data['Y_test']}

     # Optimization method
    elif struct.name[:3] == 'opt':
        hyper = helper.load_obj(struct.hyper, True)
        model = fair_mass.Optimization(data['X_train'], data['Z_train'], data['Y_train'], influence=struct.name[4:], params=hyper)
        test = {'input': data['X_test'], 'output': data['Y_test']}

    # MIM method
    elif struct.name == 'mim':
        model = fair_mass.MIM(data['X_train'], data['Z_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    # AIF models
    elif struct.name == 'adv_deb':
        model = aif_methods.AdvDeb(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    elif struct.name == 're_weigh':
        model = aif_methods.ReWeigh(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    elif struct.name == 'eo_cal':
        model = aif_methods.EOCal(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    elif struct.name == 'dp_exp':
        model = aif_methods.DPExp(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    elif struct.name == 'eo_exp':
        model = aif_methods.EOExp(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    elif struct.name == 'tpr_exp':
        model = aif_methods.TPRExp(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}

    elif struct.name == 'er_exp':
        model = aif_methods.ERExp(data['XZ_train'], data['Y_train'])
        test = {'input': data['XZ_test'], 'output': data['Y_test']}


    if 'method' not in data:
        data['method'] = {}
    if 'test' not in data:
        data['test'] = {}
    data['method'][struct.name] = model
    data['test'][struct.name] = test


def get_metrics(data):
    if 'metrics' in data:
        return data['metrics']
    di = aif_metrics.DisparateImpactRatio()
    dd = aif_metrics.StatisticalParityDifference()
    eqopp = aif_metrics.EqualOpportunityDifference()
    eqodd = aif_metrics.AverageOddsError()
    metrics = {'di':di, 'dd':dd, 'eqopp': eqopp, 'eqodd':eqodd}
    data['metrics'] = metrics
    return metrics


def get_iis(data):
    if 'iis' in data:
        return data['iis']
    shap = fair_mass.SHAP()
    ate = fair_mass.ObjBI(data['XZ_test'])
    iis = {'shap':shap, 'ate':ate}
    data['iis'] = iis
    return iis


def performance_metrics(struct, data):
    metrics = get_metrics(data)
    method = data['method'][struct.name]
    x = data['test'][struct.name]['input']
    y = data['test'][struct.name]['output']
    z = data['Z_test']
    for metric in struct.metrics:
        if metric not in data:
            data[metric] = {}
        data[metric][struct.name] = metrics[metric].metric(method, x, z, y)


def performance_ii(struct, data):
    iis = get_iis(data)
    method = data['method'][struct.name]
    x = data['test'][struct.name]['input']
    for ii in struct.iis:
        if ii not in data:
            data[ii] = {}
        data[ii][struct.name] = iis[ii].influence(method, x)


def performance_error(struct, data):
    y = data['test'][struct.name]['output']
    x = data['test'][struct.name]['input']
    method = data['method'][struct.name]
    y_pred = method.predict(x)
    error = helper.error_class(y, y_pred)
    if 'error' not in data:
        data['error'] = {}
    data['error'][struct.name] = error


def main(filename, structs, verbose=1):
    data = helper.get_data(filename)
    begin = time.time()
    for struct in structs:
        start = time.time()
        get_method(struct, data)
        performance_error(struct, data)
        performance_ii(struct, data)
        performance_metrics(struct, data)
        if verbose > 0:
            print('{} ends in {:5.0f} ms\n'.format(struct.name, (time.time() - start) * 1000))
    if verbose > 0:
        print('*** Filename: {} ends in {:5.0f} ms ***\n'.format(filename, (time.time() - begin) * 1000))
    return data


# run to be used when we are trying to compute confidence intervals
def run_single(path, filename, verbose=1, run=None):
    identifier = filename.split('.')[0]
    structs = get_structs(path, identifier)

    # assuming all structs have same model
    if run is None:
        file_data = path + filename
        file_results = path + 'results/'
    else:
        file_data = path + 'instances/' + str(run) + '/' + filename
        file_results = path + 'results/instances/' + str(run) + '/'

    data = main(file_data, structs, verbose)
    pathlib.Path(file_results).mkdir(parents=True, exist_ok=True)
    helper.save_results(data, file_results + identifier)

    if verbose > 2:
        helper.print_error(data)
        helper.print_ii(data)
        helper.print_metrics(data)


def run_synthetic(path, count, runs=1):
    start = time.time()
    pool = Pool(min(20, count * runs))#count * runs)
    verbose = 1
    inputs = []
    for run in range(1, runs+1):
        inputs = inputs + [(path, str(i + 1) + '.csv', verbose, run) for i in range(count)]
    pool.starmap(run_single, inputs)
    print('Total time: {:5.0f} ms'.format((time.time() - start) * 1000))


# decide methods and ii
def get_structs(path, id=''):
    hyper_path = path + 'hyper/'
    iis = ['shap', 'ate']
    metrics = ['di', 'dd', 'eqopp', 'eqodd']
    structs = [
        helper.Struct(name='full', iis=iis, metrics=metrics, hyper=hyper_path + 'full/' + id),
        helper.Struct(name='partial', iis=iis, metrics=metrics, hyper=hyper_path + 'partial/' + id),
        helper.Struct(name='opt_shap', iis=iis, metrics=metrics, hyper=hyper_path + 'opt_shap/' + id),
        helper.Struct(name='opt_ate', iis=iis, metrics=metrics, hyper=hyper_path + 'opt_ate/' + id),

        helper.Struct(name='adv_deb', iis=iis, metrics=metrics), # can drop this too
        helper.Struct(name='eo_cal', iis=iis, metrics=metrics),
        helper.Struct(name='re_weigh', iis=iis, metrics=metrics), # same
        helper.Struct(name='dp_exp', iis=iis, metrics=metrics), # same
        helper.Struct(name='eo_exp', iis=iis, metrics=metrics),
        helper.Struct(name='tpr_exp', iis=iis, metrics=metrics),
        helper.Struct(name='er_exp', iis=iis, metrics=metrics),
    ]
    return structs


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    path = 'datasets/synthetic/multivariate_normal/scenario1/'
    # path = 'datasets/german/'
    # path = 'datasets/bertrand/'
    # path = 'datasets/lipton'
    # run_synthetic(path, 2, 2)
    run_single(path, '1.csv', 1, 1)
