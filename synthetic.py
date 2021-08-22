import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pathlib
from shutil import copyfile

def skew_normal(mean, cov, size):
    count = 0
    XZ = np.zeros((size, mean.shape[0]))
    while count < size:
        data = np.random.multivariate_normal(mean, cov, size)
        for xz in data:
            if xz[0] < mean[0] - 1 and np.random.uniform() <= .05:
                XZ[count] = xz
            elif xz[0] < mean[0] and np.random.uniform() <= .15:
                XZ[count] = xz
            elif xz[0] >= mean[0]:
                XZ[count] = xz
            else:
                count -= 1
            count += 1
            if count == size:
                break
    return XZ


def beta_normal_marginal(coeff, size, w, x2z_corr=False, alpha=2, beta=5, ratio=0.4, binary_Y=True, intercept=True):
    # x1 = beta(alpha=2, beta=5)
    # x2 = np.random.normal()
    # z = count(zeros) = size * ratio, count(ones) = size - count(zeros)
    XZ = np.zeros((size, 3))
    XZ[:, 0] = np.random.beta(a=alpha, b=beta, size=size)
    XZ[:, 1] = np.random.normal(size=size)
    count_ones = int(size - size * ratio)
    XZ[:, 2][:count_ones] = np.ones(count_ones)
    XZ[:, [0, 2]] = skew_marginal(XZ[:, 0], XZ[:, 2], coeff)
    if x2z_corr:
        XZ[:, [1, 2]] = skew_marginal(XZ[:, 1], XZ[:, 2], x2z_corr)
    # adding 1-array for intercept
    if intercept:
        XZ = np.hstack((XZ, np.ones((size, 1))))
    # Y_i in {0, 1}
    np.random.shuffle(XZ)
    if binary_Y:
        Y = np.random.binomial(1, 1 / (1 + np.exp(-XZ @ w)))
    # continuous Y
    else:
        Y = XZ @ w
    # removing intercept
    if intercept:
        XZ = XZ[:, :-1]
    return XZ, Y


# given continuous and binary marginal distributions along with correlation
# ... coefficient it computes the joint distribution
def skew_marginal(marginal_c, marginal_b, coeff):
    # ensure order of b remains unchanged
    order_b = defaultdict(list)
    for i in range(marginal_b.shape[0]):
        order_b[marginal_b[i]].append(i)

    E_c = np.mean(marginal_c)
    E_b = np.mean(marginal_b)
    E_c2 = np.mean(np.square(marginal_c))
    E_b2 = np.mean(np.square(marginal_b))
    # pearson's correlation coefficient:
    # coeff = (E[XY] - E[X]E[Y])/sqrt((E[X^2] - E[X]^2) * (E[Y^2] - E[Y]^2))
    # E[XY] = den * coeff + E[X]E[Y]
    den1 = np.sqrt(E_c2 - E_c ** 2)
    den2 = np.sqrt(E_b2 - E_b ** 2)
    den = den1 * den2
    E_cb = den * coeff + E_c * E_b
    n = marginal_b.shape[0]
    # what is target
    target = E_cb * n

    # initial distribution
    count = dict(sorted(Counter(marginal_b).items(), key=lambda i: i[0]))
    keys = list(count.keys())
    select_indices = np.random.choice(n, count[keys[0]], replace=False)
    others = [i for i in np.arange(n) if i not in select_indices]
    elements = {keys[0]: sorted(marginal_c[select_indices]), keys[1]: sorted(marginal_c[others])}
    current = np.sum(elements[keys[0]]) * keys[0] + np.sum(elements[keys[1]]) * keys[1]

    # if current is less than target we switch elements, until switch happens...
    # ...otherwise vice-versa
    delta = current - target
    if delta > 0:
        elements[keys[1]] = elements[keys[1]][::-1]
        error = lambda current: current - target
    else:
        elements[keys[0]] = elements[keys[0]][::-1]
        error = lambda current: target - current
    index = 0
    # possible bug: infinite for loop
    # goes upto 0.7-0.8 coeff
    while error(current) > 0:
        # swap
        current = current + keys[0] * (elements[keys[1]][index] - elements[keys[0]][index])
        current = current + keys[1] * (elements[keys[0]][index] - elements[keys[1]][index])
        elements[keys[0]][index], elements[keys[1]][index] = elements[keys[1]][index], elements[keys[0]][index]
        index += 1
    joint_cb = np.empty((n, 2))
    joint_cb[:count[keys[0]]] = np.vstack((elements[keys[0]], keys[0] * np.ones(count[keys[0]]))).T
    joint_cb[count[keys[0]]:] = np.vstack((elements[keys[1]], keys[1] * np.ones(count[keys[1]]))).T

    # reshuffling such that order of b remains fixed
    output = np.zeros(joint_cb.shape)
    for i in joint_cb:
        output[order_b[i[1]][0]] = i
        order_b[i[1]] = order_b[i[1]][1:]
    # np.random.shuffle(joint_cb)
    return output


def normal_linear(size, mean, cov, w, skewed, intercept, binary_Z, binary_Y):
    if not skewed:
        XZ = np.random.multivariate_normal(mean, cov, size)
    else:
        XZ = skew_normal(mean, cov, size)
    # setting z to -1 or +1
    if binary_Z:
        XZ[:, -1] = 2 * ((XZ[:, -1] > mean[1]) - 0.5)
    # adding 1-array for intercept
    if intercept:
        XZ = np.hstack((XZ, np.ones((size, 1))))
    # Y_i in {0, 1}
    if binary_Y:
        Y = np.random.binomial(1, 1 / (1 + np.exp(-XZ @ w)))
    # continuous Y
    else:
        Y = XZ @ w
    # removing bias 1
    if intercept:
        XZ = XZ[:, :-1]
    return XZ, Y


def to_dict(XZ, Y):
    keys = ["X" + str(i + 1) for i in range(XZ.shape[1] - 1)]
    values = [list(x) for x in XZ.T[:-1]]
    data = dict(zip(keys, values))
    data['Z'] = list(XZ.T[-1])
    data['Y'] = list(Y)
    return data


def plot(data, binary_Z=True, binary_Y=True, show=False):
    keys = data.keys()
    fig, axs = plt.subplots(len(keys), figsize=(5, 5 * len(keys)))
    for i, key in enumerate(keys):
        if key == 'Y' and binary_Y:
            bins = 2
        elif key == 'Z' and binary_Z:
            bins = 2
        else:
            bins = 10
        axs[i].hist(data[key], bins=bins, density='True', ec='white')
        axs[i].set(ylabel='Probability', xlabel=key)
    if show:
        plt.show()


def multivariate_scenarios():
    size = 10000
    mean = np.zeros(3)
    # weights are corresponding to x1, x2, ..., xn, z, 1(if intercept)

    # non-discriminatory(scenario 1)
    x2z_corr = 0.5
    w = np.array([0, 1.0, 0, 1.0])
    
    # discriminatory(scenario 2)
    # x2z_corr = 0
    # w = np.ones(4)

    instances = 30
    target_corr = np.linspace(0, 0.8, 9)
    path = "datasets/synthetic/multivariate_normal/scenario1/"
    for instance in range(instances):
        corr = np.zeros(target_corr.shape[0])
        curr_path = path + 'instances/' + str(instance + 1) + '/'
        pathlib.Path(curr_path).mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(target_corr):
            cov = np.array([1, 0, c, 0, 1, x2z_corr, c, x2z_corr, 1]).reshape(3, 3)
            XZ, Y = normal_linear(size, mean, cov, w, False, True, True, True)
            corr[i] = np.corrcoef(XZ[:, 0], XZ[:, -1])[0, 1]
            data = to_dict(XZ, Y)
            df = pd.DataFrame(data, columns=data.keys())
            # no need to plot
            # pathlib.Path(curr_path + '/dist').mkdir(parents=True, exist_ok=True)
            # plot(data)
            # plt.savefig(curr_path + '/dist/' + str(i + 1) + ".png")
            df.to_csv(curr_path + str(i + 1) + ".csv", index=False)
        np.savetxt(curr_path + "corr.txt", corr)

    # rest of the folders
    folders = ['full', 'partial', 'opt_shap', 'opt_objBI']
    hyper_path_full = 'datasets/others/hyper_full.json'
    hyper_path_opt= 'datasets/others/hyper_opt.json'
    # hyper and saved
    for folder in folders:
        curr_path = path + 'hyper/logistic/' + folder
        pathlib.Path(curr_path).mkdir(parents=True, exist_ok=True)
        for i in range(1, target_corr.shape[0]+1):
            if folder == 'full' or folder == 'partial':
                copyfile(hyper_path_full, curr_path + '/' + str(i) + '.json')
            else:
                copyfile(hyper_path_opt, curr_path + '/' + str(i) + '.json')
        for i in range(1, instances+1):
            pathlib.Path(path + 'saved/logistic/' + folder + '/instances/' + str(i)).mkdir(parents=True, exist_ok=True)
    #results
    for i in range(1, instances+1):
        pathlib.Path(path + 'results/logistic/instances/' + str(i)).mkdir(parents=True, exist_ok=True)





def beta_normal_scenarios():
    size = 10000
    # weights are corresponding to x1, x2, ..., xn, z, 1(if intercept)

    # non-discriminatory(scenario 1)
    # x2z_corr = 0.5
    # w = np.array([0, 1.0, 0, 1.0])

    # discriminatory(scenario 2)
    x2z_corr = False
    w = np.ones(4)
    instances = 30
    target_corr = np.linspace(0, 0.7, 8)
    path = "datasets/synthetic/beta_normal/scenario2/instances/"
    for instance in range(instances):
        corr = np.zeros(target_corr.shape[0])
        curr_path = path + str(instance + 1) + '/'
        for i, c in enumerate(target_corr):
            XZ, Y = beta_normal_marginal(c, size, w, x2z_corr)
            corr[i] = np.corrcoef(XZ[:, 0], XZ[:, -1])[0, 1]
            data = to_dict(XZ, Y)
            df = pd.DataFrame(data, columns=data.keys())
            plot(data)
            pathlib.Path(curr_path + '/dist').mkdir(parents=True, exist_ok=True)
            plt.savefig(curr_path + '/dist/' + str(i + 1) + ".png")
            df.to_csv(curr_path + str(i + 1) + ".csv", index=False)
        np.savetxt(curr_path + "corr.txt", corr)



if __name__ == "__main__":
    # multivariate_scenarios()
    # beta_normal_scenarios()
    pass
