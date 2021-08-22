from collections import defaultdict
import helper
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def names(s):
    if s == 'corr': return r'Correlation$(X_1, Z)$'
    elif s == 'di': return 'Disparate Impact Ratio'
    elif s == 'dd': return 'Demographic Disparity'
    elif s == 'eqopp': return 'Equal Opportunity Difference'
    elif s == 'eqodd': return 'Average Odds Error'
    elif s == 'adv_deb': return 'Adv Deb'
    elif s == 're_weigh': return 'Reweighing'
    elif s == 'eo_cal': return 'Cal Eq Odds'
    elif s == 'dp_exp': return 'Exp Grad DP'
    elif s == 'eo_exp': return 'Exp Grad EO'
    elif s == 'tpr_exp': return 'Exp Grad TPR'
    elif s == 'er_exp': return 'Exp Grad ER'
    elif s == 'method': return 'Method'
    elif s == "full": return "Trad. with $Z$"
    elif s == "partial": return "Trad. w/o $Z$"
    elif s == "mip": return "MIM"
    elif s == "opt_shap": return "OPT-SHAP"
    elif s == "opt_objBI": return "OPT-ATE"
    elif s[:4] == 'shap': return "$\mathbb{E}[|$SHAP("+ s[4:] + "$)|]$"
    elif s[:5] == 'objBI': return "$\mathbb{E}[|$ATE("+ s[5:] + "$)|]$"
    elif s == 'error': return 'Accuracy'
    else: return s

def palette():
    tab10 = sns.color_palette('tab10')
    d = {
        'adv_deb': tab10[2],
        're_weigh': tab10[9],
        'eo_cal': tab10[4],
        'dp_exp': tab10[5],
        'eo_exp': tab10[6],
        'tpr_exp': tab10[8],
        'er_exp': tab10[7],
        'full': 'tab:blue',
        'partial': 'tab:blue',
        'mip': 'tab:red',
        'opt_shap': 'tab:red',
        'opt_objBI': 'tab:red',
    }
    p = {}
    for k in d.keys():
        p[names(k)] = d[k]
    return p

def palette_bar():
    tab10 = sns.color_palette('tab10')
    d = {
        'adv_deb': 'tab:orange',
        're_weigh': 'tab:orange',
        'eo_cal': 'tab:orange',
        'dp_exp': 'tab:orange',
        'eo_exp': 'tab:orange',
        'tpr_exp': 'tab:orange',
        'er_exp': 'tab:orange',
        'full': 'tab:blue',
        'partial': 'tab:blue',
        'mip': 'tab:red',
        'opt_shap': 'tab:red',
        'opt_objBI': 'tab:red',
    }
    p = {}
    for k in d.keys():
        p[names(k)] = d[k]
    return p

def style():
    d = {
        'adv_deb': '',
        're_weigh': '',
        'eo_cal': '',
        'dp_exp': '',
        'eo_exp': '',
        'tpr_exp': '',
        'er_exp': '',
        'full': '',
        'partial': (1, 1),
        'mip': '',
        'opt_shap': '',
        'opt_objBI': (1, 1)
    }
    s = {}
    for k in d.keys():
        s[names(k)] = d[k]
    return s

def linewidth():
    d = {
        'adv_deb': 2,
        're_weigh': 2,
        'eo_cal': 2,
        'dp_exp': 2,
        'eo_exp': 2,
        'tpr_exp': 2,
        'er_exp': 2,
        'full': 4,
        'partial': 2,
        'mip': 2,
        'opt_shap': 2,
        'opt_objBI': 2
    }
    s = {}
    for k in d.keys():
        s[names(k)] = d[k]
    return s

def order():
    l = ['adv_deb', 're_weigh', 'eo_cal', 'dp_exp', 'eo_exp', 'tpr_exp', 'er_exp', 'full', 'partial', 'opt_shap', 'opt_objBI']
    l = [names(i) for i in l]
    return l

def dropped():
    l = ['mip']
    return l


def plot_metrics(path, count, runs, metrics):
    d = {}
    for c in range(1, count+1):
        for r in range(1, runs+1):
            data = helper.load_obj(path + 'instances/' + str(r) + '/' + str(c), is_json=True)
            for metric in metrics:
                if metric not in d:
                    d[metric] = defaultdict(list)
                for method in data[metric]:
                    if method in dropped():
                        continue
                    d[metric][names('corr')].append((c - 1) * 0.1)
                    d[metric][names('method')].append(names(method))
                    if metric == 'error':
                        d[metric][names(metric)].append(1 - data[metric][method]/100)
                    else:
                        d[metric][names(metric)].append(np.abs(data[metric][method]))
    
    for metric in metrics:
        plt.figure()
        df = pd.DataFrame(d[metric])
        print(df.head())
        if count > 1:
            sns.set(font_scale = 1.6)
            sns.set_style("white")
            ax = sns.lineplot(data=df, x=names('corr'), y=names(metric), style=names('method'), hue=names('method'), 
            size=names('method'), dashes=style(), palette=palette(), sizes=linewidth(), err_style="bars", ci=95, legend=True)
            ax.legend_.remove()
        else:
            sns.set(font_scale = 1.6)
            sns.set_style("whitegrid")
            ax = sns.barplot(data=df, x=names('method'), y=names(metric), palette=palette_bar(), order=order(), ci=95)
            plt.xticks(rotation=90)
            ax.set(xlabel=None)
        plt.tight_layout()
        plt.savefig(path + metric + '.pdf', bbox_inches='tight')
        # plt.show()
        # return
        
        
        


def plot_influence(path, count, runs, ii, features):
    d = {}
    for c in range(1, count+1):
        for r in range(1, runs+1):
            data = helper.load_obj(path + 'instances/' + str(r) + '/' + str(c), is_json=True)
            for ii_name in ii:
                for idx, feature in enumerate(features):
                    column = ii_name + feature
                    if column  not in d:
                        d[column] = defaultdict(list)
                    for method in data[ii_name]:
                        if method in dropped():
                            continue
                        d[column][names('corr')].append((c - 1) * 0.1)
                        d[column][names('method')].append(names(method))
                        # value not available
                        if len(data[ii_name][method]) <= idx:
                            d[column][names(column)].append(0)
                        else:
                            d[column][names(column)].append(data[ii_name][method][idx])
                        

    for column in d.keys():
        plt.figure()
        df = pd.DataFrame(d[column])
        print(df.head())
        
        if count > 1:
            sns.set(font_scale = 1.6)
            sns.set_style("white")
            ax = sns.lineplot(data=df, x=names('corr'), y=names(column), style=names('method'), hue=names('method'), 
            size=names('method'), dashes=style(), sizes=linewidth(), palette=palette(), err_style="bars", ci=95)
            # Put the legend out of the figure
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text size
            ax.legend_.remove()
        else:
            sns.set(font_scale = 1.6)
            sns.set_style("whitegrid")

            ax = sns.barplot(data=df, x=names('method'), y=names(column), palette=palette_bar(), order=order(), ci=95)
            plt.xticks(rotation=90)
            ax.set(xlabel=None)
        
        plt.tight_layout()
        plt.savefig(path + column + '.pdf', bbox_inches='tight')
        # plt.show()
        # return
        
        
        
        


# path = 'datasets/synthetic/multivariate_normal/scenario2/results/logistic/'
# features = ['X_1', 'X_2', 'Z']

path = 'datasets/compas/results/logistic/'
features = ['c_charge_degree_F','age','juv_fel_count','juv_misd_count','juv_other_count','priors_count','sex','race']

# path = 'datasets/german/results/logistic/'
# features = ['foreign_worker', 'duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents', 'sex']

metrics = ['di', 'dd', 'eqopp', 'eqodd', 'error']
ii = ['shap', 'objBI']

plot_influence(path, 1, 30, ii, features)
plot_metrics(path, 1, 30, metrics)