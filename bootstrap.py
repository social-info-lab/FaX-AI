import numpy as np
import pathlib
from shutil import copyfile
import pandas as pd


def boot(filename, data_path, hyper_path_full, hyper_path_opt, num_trials, dst_path):
    folders = ['full', 'partial', 'opt_shap', 'opt_objBI']
    # hyper and saved
    for folder in folders:
        curr_path = dst_path + 'hyper/logistic/' + folder
        pathlib.Path(curr_path).mkdir(parents=True, exist_ok=True)
        if folder == 'full' or folder == 'partial':
            copyfile(hyper_path_full, curr_path + '/' + filename + '.json')
        else:
            copyfile(hyper_path_opt, curr_path + '/' + filename + '.json')
        for i in range(1, num_trials+1):
            pathlib.Path(dst_path + 'saved/logistic/' + folder + '/instances/' + str(i)).mkdir(parents=True, exist_ok=True)

    # bootstrap
    df = pd.read_csv(data_path)
    size = df.shape[0]
    for i in range(1, num_trials+1):
        samples = np.random.choice(size, size, replace=True)
        df_curr = df.iloc[samples]
        # instance
        pathlib.Path(dst_path + 'instances/' + str(i)).mkdir(parents=True, exist_ok=True)
        df_curr.to_csv(dst_path + 'instances/' + str(i) + '/' + filename + '.csv', index=False)
        # results
        pathlib.Path(dst_path + 'results/logistic/instances/' + str(i)).mkdir(parents=True, exist_ok=True)




if __name__ == '__main__':

    filename = '1'
    data_path = 'datasets/others/compas.csv'
    hyper_path_full = 'datasets/others/hyper_full.json'
    hyper_path_opt = 'datasets/others/hyper_opt.json'
    num_trials = 30
    dst_path = 'datasets/compas2/'
    boot(filename, data_path, hyper_path_full, hyper_path_opt, num_trials, dst_path)



    # for i in range(10):
    #     filename = str(i+1)
    #     data_path = 'datasets/others/nick_scenario2/' + str(i+1) + '.csv'
    #     hyper_path_full = 'datasets/others/hyper_full.json'
    #     hyper_path_opt = 'datasets/others/hyper_opt.json'
    #     num_trials = 30
    #     dst_path = 'datasets/synthetic/nick/scenario2/'
    #     boot(filename, data_path, hyper_path_full, hyper_path_opt, num_trials, dst_path) 