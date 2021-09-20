from numpy.lib.arraysetops import isin
import torch
from enum import Enum
from dataclasses import dataclass
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json
import copy



class State(Enum):
    LOAD = 0  # simply load, don't train
    TRAIN = 1  # train the model from scratch
    TUNE = 2  # tune hyperparams using optimization library
    MANUAL = 3  # manual search of hyperparams, plot the loss curve


# requires python 3.7
@dataclass
class Struct:
    iis: list
    metrics: list
    name: list
    hyper: str = ''


def error_class(Y, Y_out):
    return (Y != Y_out).sum() / Y.shape[0] * 100


def split(X, Y):
    # 80:20 split
    split = int(X.shape[0] * 0.8)  # todo: make it random
    return X[:split], X[split:], Y[:split], Y[split:]


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def pre_process(X, Y, Z):
    assert (X.ndim == 2 and Z.ndim == 2 and Y.ndim == 1)
    assert (X.shape[0] == Z.shape[0] and X.shape[0] == Y.shape[0])
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    # disabled normalization of Z so that metrics can work
    # Z = (Z - torch.mean(Z, 0)) / torch.std(Z, 0)
    size = X.shape[0]
    # 80:20 split, todo: make it random
    split = int(size * 0.8)
    return X[:split], X[split:], Y[:split], Y[split:], Z[:split], Z[split:]


# second last column is Z and last column is Y
def read_csv(filename):
    df = pd.read_csv(filename)
    X = df[df.columns[:-2]].to_numpy()
    Z = df[df.columns[-2:-1]].to_numpy()
    Y = df[df.columns[-1]].to_numpy(dtype=np.int32)
    return X, Y, Z, df.columns


def get_data(filename):
    # loading data
    X, Y, Z, columns = read_csv(filename)

    # pre-processing data
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = pre_process(X, Y, Z)

    XZ_train = np.hstack((X_train, Z_train))
    XZ_test = np.hstack((X_test, Z_test))
    labels = np.max(Y + 1)
    data = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test,
            'Z_train': Z_train, 'Z_test': Z_test, 'XZ_train': XZ_train, 'XZ_test': XZ_test,
            'labels': labels}
    return data


def plot(y, x, xlabel, ylabel, show, names=None, path=''):
    plt.figure()
    if names is not None:
        for id, name in enumerate(names):
            plt.plot(x, y[:, id], label=name)
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(path + '.png')


def save_obj(obj, name, is_json=False):
    if not is_json:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(name + '.json', 'w') as f:
            json.dump(jsoning(obj), f, indent=4)


def load_obj(name, is_json=False):
    if not is_json:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.json', 'r') as fp:
            return unjsoning(json.load(fp))


def mse(x1, x2):
    return np.mean((x1 - x2) ** 2)


def impute(data, marginal, order):
    dims = np.nonzero(marginal)[0]
    new_data = data.repeat(order.shape[0], 1, 1)
    for i in range(order.shape[0]):
        new_data[i][:, dims] = new_data[i][:, dims][order[i]]
    return new_data


def print_error(data):
    if 'error' in data:
        print('Error: ')
        s = pd.Series(data['error'])
        print(s)


def print_ii(data):
    iis = data['iis']
    print('Input Influence: ')
    ii_dict = {key: data[key] for key in iis if key in data}
    df = pd.DataFrame.from_dict(ii_dict)
    print(df)

def print_metrics(data):
    metrics = data['metrics']
    print('Metrics: ')
    metrics_dict = {key: data[key] for key in metrics if key in data}
    df = pd.DataFrame.from_dict(metrics_dict)
    print(df)


def plot_error_bar(data, loc):
    plt.bar(data['error'].keys(), data['error'].values())
    ylabel = 'error'
    plt.ylabel(ylabel)
    plt.xlabel('methods')
    plt.savefig(loc + '.' + ylabel + '.bar.png')


# we need 'full' as a method else it breaks
def get_ii_error(data, ii):
    ii_err = {}
    for ii_name in ii:
        if ii_name not in data:
            continue
        ii_err[ii_name] = {}
        for method in data[ii_name]:
            ii_err[ii_name][method] = np.mean(np.abs(np.array(data[ii_name][method]) - np.array(data[ii_name]['full'])))
    return ii_err


def plot_ii_bar(data, ii, loc):
    ii_err = get_ii_error(data, ii)
    for ii_name in ii:
        plt.figure()
        plt.bar(ii_err[ii_name].keys(), ii_err[ii_name].values())
        plt.ylabel(ii_name + ' error')
        plt.xlabel('methods')
        plt.savefig(loc + '.' + ii_name + '.bar.png')


def plot_ii_synthetic(results, ii, loc):
    ii_syn_errs = [get_ii_error(data, ii) for data in results]
    num_files = len(results)
    for ii_name in ii:
        ii_syn_err = [ii_err[ii_name] for ii_err in ii_syn_errs]
        num_methods = len(ii_syn_err[0])
        y = np.empty((num_files, num_methods))
        names = []
        for i, method in enumerate(ii_syn_err[0]):
            y[:, i] = [ii_err[method] for ii_err in ii_syn_err]
            names.append(method)
        x = np.linspace(0, 1, num_files)
        ylabel = 'IIL' + ' error'
        xlabel = 'corr(x1, z)'
        path = loc + ii_name
        plot(y, x, xlabel, ylabel, False, names=names, path=path)


def plot_ii_synthetic2(results, ii, loc, runs):
    ii_syn_errs = [get_ii_error(data, ii) for data in results]
    num_files = int(len(results) / runs)
    for ii_name in ii:
        ii_syn_err = [ii_err[ii_name] for ii_err in ii_syn_errs]
        num_methods = len(ii_syn_err[0])
        y = np.empty((num_methods, int(num_files * runs)))
        for i, method in enumerate(ii_syn_err[0]):
            y[i, :] = [ii_err[method] for ii_err in ii_syn_err]
        y = y.reshape((num_methods, runs, num_files))
        x = np.linspace(0, 0.7, num_files)  # todo
        for i, method in enumerate(ii_syn_err[0]):
            mean = np.mean(y[i], 0)
            std = np.std(y[i], 0)
            plt.plot(x, mean, label=method)
            plt.fill_between(x, mean + std, mean - std, alpha=0.3)
        plt.xlabel('corr(x1, z)')
        plt.ylabel('IIL ' + ii_name)  # todo
        plt.legend()
        plt.show()


def plot_ii_synthetic3(results, ii, loc, runs, dim=0):
    ii_syn_errs = [get_ii_error(data, ii) for data in results]
    num_files = int(len(results) / runs)
    for ii_name in ii:
        ii_syn_err = [ii_err[ii_name] for ii_err in ii_syn_errs]
        num_methods = len(ii_syn_err[0])
        y = np.empty((num_methods, runs, num_files))
        for i, method in enumerate(ii_syn_err[0]):
            y[i, :] = np.array([r[ii_name][method][dim] for r in results]).reshape(runs, num_files)
        x = np.linspace(0, 0.7, num_files)  # todo
        for i, method in enumerate(ii_syn_err[0]):
            # if method == 'eo' or method == 'dp':
            #     continue
            mean = np.mean(y[i], 0)
            std = np.std(y[i], 0)
            print(method + ': ', mean)
            plt.plot(x, mean, label=method)
            plt.fill_between(x, mean + std, mean - std, alpha=0.3)
        plt.xlabel('corr(x1, z)')
        if dim == 0:
            ylabel = ii_name + '_x1'
        else:
            ylabel = ii_name + '_x2'
        plt.ylabel(ylabel)  # todo
        plt.legend()
        if loc:
            plt.savefig(loc + ylabel + '.png')
        else:
            plt.show()


def plot_err_synthetic(results, loc):
    err_syn = [data['error'] for data in results]
    num_files = len(results)
    num_methods = len(err_syn[0].keys())
    y = np.empty((num_files, num_methods))
    names = []
    for i, method in enumerate(err_syn[0]):
        y[:, i] = [err[method] for err in err_syn]
        names.append(method)
    x = np.linspace(0, 1, num_files)
    ylabel = 'error'
    xlabel = 'corr(x1, z)'
    path = loc + 'error'
    plot(y, x, xlabel, ylabel, False, names=names, path=path)


def plot_err_synthetic2(results, loc, runs):
    err_syn = [data['error'] for data in results]
    num_files = int(len(results) / runs)
    num_methods = len(err_syn[0])
    y = np.empty((num_methods, int(num_files * runs)))
    for i, method in enumerate(err_syn[0]):
        y[i, :] = [err[method] for err in err_syn]
    y = y.reshape((num_methods, runs, num_files))
    x = np.linspace(0, 0.7, num_files)  # todo
    for i, method in enumerate(err_syn[0]):
        mean = np.mean(y[i], 0)
        std = np.std(y[i], 0)
        plt.plot(x, mean, label=method)
        plt.fill_between(x, mean + std, mean - std, alpha=0.2)
    plt.xlabel('corr(x1, z)')
    plt.ylabel('error')
    plt.legend()
    if loc:
        plt.savefig(loc + 'error.png')
    else:
        plt.show()


def plot_ii_full_partial(results_dis, results_ndis, ii, runs):
    num_files = int(len(results_dis) / runs)
    for ii_name in ii:
        two_d = lambda results, method: np.array([r[ii_name][method][0] for r in results]).reshape((runs, num_files))
        ii_dis_full = two_d(results_dis, 'full')
        ii_ndis_full = two_d(results_ndis, 'full')
        ii_dis_partial = two_d(results_dis, 'partial')
        ii_ndis_partial = two_d(results_ndis, 'partial')
        x = np.linspace(0, 0.7, num_files)  # todo

        def plot_data_method(ii_data_method, label):
            mean = np.mean(ii_data_method, 0)
            print(label + ': ', mean)
            std = np.mean(ii_data_method, 0)
            plt.plot(x, mean, label=label)
            plt.fill_between(x, mean + std, mean - std, alpha=0.2)

        plot_data_method(ii_dis_full, 'full discriminatory')
        plot_data_method(ii_ndis_full, 'full non-discriminatory')
        plot_data_method(ii_dis_partial, 'partial discriminatory')
        plot_data_method(ii_ndis_partial, 'partial non-discriminatory')

        plt.xlabel('corr(x1, z)')
        plt.ylabel(ii_name + ' x1')
        plt.legend()
        plt.show()


# we can't serialize np arrays and floats
def jsoning(data):
    if isinstance(data, dict):
        for i in data:
            data[i] = jsoning(data[i])
    elif isinstance(data, np.floating):
        data = float(data)
    elif isinstance(data, np.integer):
        data = int(data)
    elif isinstance(data, np.ndarray):
        data = list(data)
        for i in range(len(data)):
            data[i] = jsoning(data[i])
    return data


def unjsoning(data):
    if isinstance(data, dict):
        for i in data:
            data[i] = jsoning(data[i])
    elif isinstance(data, list):
        data = np.ndarray(data)
    return data


def save_results(data, loc):
    iis = data['iis']
    metrics = data['metrics']
    results = {'error': copy.deepcopy(data['error'])}
    for ii in iis:
        if ii in data:
            results[ii] = copy.deepcopy(data[ii])
    for metric in metrics:
        if metric in data:
            results[metric] = copy.deepcopy(data[metric])
    save_obj(results, loc, is_json=True)


def dataframe(XZ):
    columns = ['X_' + str(i) for i in range(1, XZ.shape[1])] + ['Z']
    df = pd.DataFrame(XZ, columns = columns)
    df.set_index('Z', drop=False, append=True, inplace=True)
    return df


def series(y, z):
    Y = y.numpy()
    Z = z.numpy()
    s = pd.Series(Y, index=pd.Index(Z, name='Z'))
    return s


# mip_x2 = [0.03383599, 0.03541867, 0.03367857, 0.0340074,  0.03474125, 0.03454214, 0.03405202, 0.03347666]
# full_x2 = [0.02975559, 0.02805819, 0.02927172, 0.0292545,  0.028401  , 0.02828493, 0.02892713, 0.02987106]
#
#
# full_x1 = [0.00086074, 0.00070576, 0.0008643 , 0.00088569, 0.00129428, 0.00079718, 0.00129252, 0.00237415]
# mip_x1 = [0.00099134, 0.00084258, 0.00097887, 0.00098911, 0.00113417, 0.00092351, 0.00116321, 0.00087726]


