import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def nmt(s):
    """
    Name conversions for various models.
    """
    if s=="bilal1": return "Zafar (2015)"
    elif s=="bilal2": return "Zafar OMR (2017)"
    elif s=="bilal3": return "Zafar EF (2017)"
    elif s=="eqopp": return "Hardt (2016)"
    elif s=="mixt-probnorm": return "MIM (proposed)"
    elif s=="donini": return "Donini (2018)"
    elif s=="full": return "Trad. with $Z$"
    elif s=="nn-full": return "Trad NN"
    elif s=="nn-marginal": return "Without $Z$ NN"
    elif s=="nn-mixt-probnorm": return "OIP NN"
    elif s=="nn-eqopp": return "Hardt (2016) NN"
    elif s=="marginal": return "Trad. w/o $Z$"
    elif s == "capuchin-IC": return "Salimi IC (2019)"
    elif s == "capuchin-MF": return "Salimi MF (2019)"
    else: return s

def plot_irl(df, y_axis, y_label, path='irl/',x_col="", err_bar=False, ii = 'SHAP'):
    """
    Plot results for measurements for irl datasets
    """
    data = df.copy()
    plt.rcParams['font.size'] = 22
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(10,6))
    models = [ c for c in data['model'].unique() if c not in ['mixt-probnorm','full','marginal'] ] + ['mixt-probnorm','full','marginal']
    x=[]
    y=[]
    yerr=[]
    for m in models:
        x.append(nmt(m))
        y.append(abs(data.loc[data['model'] == m][y_axis].values[0]))
        if err_bar:
            yerr.append(data.loc[data['model'] == m][y_axis].values[1])

    if err_bar:
        barlist = plt.bar(x, y, color ='tab:orange', width = 0.4, yerr=yerr)
    else:
        barlist = plt.bar(x, y, color ='tab:orange', width = 0.4)
    barlist[-1].set_color('tab:blue')
    barlist[-2].set_color('tab:blue')
    barlist[-3].set_color('tab:red')
    plt.grid(color='black', which='major', axis='y', linestyle='solid',zorder=0)
    plt.xticks(fontsize='medium',rotation='vertical')
    plt.yticks(fontsize='large')
    #plt.xlabel("Model", size="x-large")
    plt.ylabel(y_label, size="large")
    plt.savefig(path+"_"+y_axis+".pdf", bbox_inches = "tight")
    plt.show()

def plot_irl_inf_diff_feats(df, feat_name, feat_ind, path='irl/', err_bar=False, delta = True, ii = 'SHAP'):
    """
    Plot influence for each feature for irl datasets
    """
    data = df.copy()
    plt.rcParams['font.size'] = 22
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(10,6))
    models = [ c for c in data['model'].unique() if c not in ['mixt-probnorm','full','marginal'] ] + ['mixt-probnorm','full','marginal']
    x=[]
    y=[]
    yerr=[]
    if delta:
        col_s = 'influence diff'
    else:
        col_s = 'influence'
    for m in models:
        x.append(nmt(m))
        y.append(data.loc[data['model'] == m][col_s].values[0][feat_ind])
        if err_bar:
            yerr.append(data.loc[data['model'] == m][col_s].values[1][feat_ind])

    if err_bar:
        barlist = plt.bar(x, y, color ='tab:orange', width = 0.4, yerr=yerr)
    else:
        barlist = plt.bar(x, y, color ='tab:orange', width = 0.4)
    barlist[-1].set_color('tab:blue')
    barlist[-2].set_color('tab:blue')
    barlist[-3].set_color('tab:red')
    plt.grid(color='black', which='major', axis='y', linestyle='solid',zorder=0)
    plt.xticks(fontsize='medium',rotation='vertical')
    plt.yticks(fontsize='large')
    #plt.xlabel("Model", size="x-large")
    if delta:
        plt.ylabel("$\Delta\mathbb{E}[|"+ii+"("+feat_name.replace("_", "\_")+")|]$", size="large")
        plt.savefig(path+"_diff_"+feat_name+ii+".pdf", bbox_inches = "tight")
    else:
        plt.ylabel("$\mathbb{E}[|"+ii+"("+feat_name.replace("_", "\_")+")|]$", size="large")
        plt.savefig(path+"_inf_"+feat_name+ii+".pdf", bbox_inches = "tight")
    plt.show()

def plot_irl_inf_diff_models(df, cols, path='irl/',figsize=(10,6),err_bar=False, ii = 'SHAP'):
    """
    Plot influence for each model seperately for irl datasets
    """
    data = df.copy()
    plt.rcParams['font.size'] = 22
    plt.rc('axes', axisbelow=True)
    yerr=None
    models = [ c for c in data['model'].unique() if c not in ['mixt-probnorm','full','marginal'] ] + ['mixt-probnorm','full','marginal']
    for m in models:
        fig = plt.figure(figsize=figsize)
        if err_bar:
            yerr = data.loc[data['model'] == m]['influence diff'].values[1]
        barlist = plt.bar(cols, data.loc[data['model'] == m]['influence diff'].values[0], yerr=yerr, color ='tab:orange', width = 0.4)
        plt.grid(color='black', which='major', axis='y', linestyle='solid',zorder=0)

        plt.xticks(fontsize='medium',rotation='vertical')
        plt.yticks(fontsize='large')
        #plt.xlabel("Model", size="x-large")
        plt.title(nmt(m))
        plt.ylabel("$\Delta\mathbb{E}[|"+ii+"|]$", size="large")
        plt.savefig(path+"_diff_"+m+".pdf", bbox_inches = "tight")
        plt.show()
    for m in models:
        fig = plt.figure(figsize=figsize)
        if err_bar:
            yerr = data.loc[data['model'] == m]['influence'].values[1]
        barlist = plt.bar(cols, data.loc[data['model'] == m]['influence'].values[0], yerr=yerr, color ='tab:orange', width = 0.4)
        plt.grid(color='black', which='major', axis='y', linestyle='solid',zorder=0)

        plt.xticks(fontsize='medium',rotation='vertical')
        plt.yticks(fontsize='large')
        #plt.xlabel("Model", size="x-large")
        plt.title(nmt(m))
        plt.ylabel("$\mathbb{E}[|"+ii+"|]$", size="large")
        plt.savefig(path+"_nodiff_"+m+ii+".pdf", bbox_inches = "tight")
        plt.show()

def plot_irl_inf_MIM_comp(df, cols, path='irl/',figsize=(10,6),diff=False, err_bar= False, ii = 'SHAP'):
    """
    Plot influence comparison between MIM model and traditional models
    """
    data = df.copy()
    plt.rcParams['font.size'] = 22
    plt.rc('axes', axisbelow=True)
    yerr1, yerr2, yerr3= (None, None, None)
    models = [ c for c in data['model'].unique() if c not in ['mixt-probnorm','full','marginal'] ] + ['mixt-probnorm','full','marginal']
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(cols))  # the label locations
    width = 0.25
    if diff:
        if err_bar:
            yerr1 = data.loc[data['model'] == 'mixt-probnorm']['influence diff'].values[1]
            yerr3 = data.loc[data['model'] == 'marginal']['influence diff'].values[1]
        rects1 = ax.bar(x - width/2, data.loc[data['model'] == 'mixt-probnorm']['influence diff'].values[0], yerr=yerr1, color ='tab:red',  width=width, label=nmt('mixt-probnorm'))
        rects3 = ax.bar(x + width/2, data.loc[data['model'] == 'marginal']['influence diff'].values[0], yerr=yerr3, color ='tab:blue',  width=width, label=nmt('marginal'))
    else:
        if err_bar:
            yerr1 = data.loc[data['model'] == 'mixt-probnorm']['influence'].values[1]
            yerr2 = data.loc[data['model'] == 'full']['influence'].values[1]
            yerr3 = data.loc[data['model'] == 'marginal']['influence'].values[1]
        rects1 = ax.bar(x - width, data.loc[data['model'] == 'mixt-probnorm']['influence'].values[0], yerr=yerr1, color ='tab:red',  width=width, label=nmt('mixt-probnorm'))
        rects2 = ax.bar(x, data.loc[data['model'] == 'full']['influence'].values[0], yerr=yerr2, color ='tab:green', width=width, label=nmt('full'))
        rects3 = ax.bar(x + width, data.loc[data['model'] == 'marginal']['influence'].values[0], yerr=yerr3, color ='tab:blue',  width=width, label=nmt('marginal'))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(color='black', which='major', axis='y', linestyle='solid',zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    plt.xticks(fontsize='medium',rotation='vertical')
    plt.yticks(fontsize='large')
    ax.autoscale(tight=True)
    #plt.xlabel("Model", size="x-large")
    if not diff:
        plt.title("MIM vs Full vs Marginal")
        plt.ylabel("$\mathbb{E}[|"+ii+"|]$", size="large")
        plt.savefig(path+ii+"_mimvsmarginf.pdf", bbox_inches = "tight")
    else:
        plt.title("MIM vs Marginal")
        plt.ylabel("$\Delta\mathbb{E}[|"+ii+"|]$", size="large")
        plt.savefig(path+ii+"_mimvsmargdiff.pdf", bbox_inches = "tight")
    plt.show()

def plot_syn(df_results,inf_col,name,y_lim=None,path = "synthetic/", ii='SHAP', yerr=None, show_legend=False):
    """
    Plots the synthetic results
    """
    linestyleMapping = {nmt("bilal1"): ((0, (1, 1)),'tab:orange'),
        nmt("bilal2"):((0, (3, 2, 1, 2)),'tab:orange'),
        nmt("bilal3"):('solid','tab:orange'),
        nmt("eqopp"):((0, (5, 1)),'tab:orange'),#(0, (3, 5, 1, 5)),
        nmt("donini"):((0, (3, 1, 1, 1)),'tab:orange'),
        nmt("full"): ('solid','tab:blue'),
        nmt("marginal"): ((0, (1, 1)),'tab:blue'),
        nmt("mixt-probnorm"): ('solid','tab:red'),
        nmt("nn-full"): ('solid','tab:green'),
        nmt("nn-marginal"): ((0, (1, 1)),'tab:green'),
        nmt("nn-mixt-probnorm"): ((0, (1, 1)),'tab:green'),
        'Adv Deb':((0, (3, 5, 1, 5, 1, 5)),'tab:orange'),
        'Reweighing':((0, (3, 2, 1, 2)),'tab:orange'),
        'Cal Eq Odds':((0, (1, 1)),'tab:orange')}
    plt.rcParams['font.size'] = 22
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(12,9))
    i=2

    for m in df_results['model'].unique():
        res_df = df_results.loc[(df_results['model'] == m)&(df_results['ind'] != 0.0)]
        if yerr is not None:
            yerr_m = yerr.loc[(yerr['model'] == m)&(yerr['ind'] != 0.0)][inf_col].copy()
        m=nmt(m)
        if m in ["Trad. with $Z$","MIM (proposed)","Trad. w/o $Z$"]:
            if yerr is None:
                plt.plot(res_df['ind'], np.abs(res_df[inf_col]), label=m, color=linestyleMapping[m][1],linestyle=linestyleMapping[m][0], linewidth=5.0)
            else:
                plt.errorbar(res_df['ind'], np.abs(res_df[inf_col]), yerr=yerr_m, label=m, color=linestyleMapping[m][1],linestyle=linestyleMapping[m][0], linewidth=5.0)
        else:
            color = cmap(i)
            if yerr is None:
                plt.plot(res_df['ind'], np.abs(res_df[inf_col]), label=m, color=color, linewidth=3.5)
            else:
                plt.errorbar(res_df['ind'], np.abs(res_df[inf_col]), yerr=yerr_m, label=m, color=color, linewidth=3.5)
            if i==2:
                i+=2
            else:
                i+=1
    #plt.xticks([0,0.25,0.5,0.75,1.0],fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.xlabel("Correlation$(X_1,Z)$", size="x-large")
    if 'Z' in inf_col or 'prot_attr' in inf_col:
        plt.ylabel("$\mathbb{E}[|"+ii+"(Z)|]$", size="x-large")
    elif inf_col=='acc':
        plt.ylabel("Accuracy", size="x-large")
    elif inf_col=='di':
        plt.ylabel("Disparate Impact Ratio", size="x-large")
    elif inf_col=='dd':
        plt.ylabel("Demographic Disparity", size="x-large")
    elif inf_col=='eqodd':
        plt.ylabel("Average Equalized Odds Error", size="x-large")
    elif inf_col=='eqopp':
        plt.ylabel("Equal Opportunity Difference", size="x-large")
    elif 'X' in inf_col:
        plt.ylabel("$\mathbb{E}[|"+ii+"(X_"+inf_col[-1]+")|]$", size="x-large")
    elif inf_col == 'er':
        plt.ylabel("Error Ratio", size="x-large")

    if y_lim is not None:
        plt.ylim(y_lim)
    def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    if show_legend:
        legend = plt.legend(fontsize='x-large',loc='center left', bbox_to_anchor=(1, 0.5),ncol=5)
        export_legend(legend, filename= path+'legend.pdf')
        plt.savefig(path+inf_col+'_abs_'+ii+'_'+name+'.pdf',bbox_inches='tight')
    plt.savefig(path+inf_col+'_abs_'+ii+'_'+name+'noleg.pdf',bbox_inches='tight')

def plot_all(results, X_test, prot_attr, x1_name, x2_name, path, err_bar=False, figsizemodels=(10,6)):
    """
    Plot all possible plots for the results of a irl dataset
    """
    plot_irl(results, 'influence_prot_attr', "$\mathbb{E}[|SHAP("+prot_attr.replace("_", "\_")+")|]$",path=path,err_bar=err_bar)
    plot_irl(results, 'influence_X1', "$\mathbb{E}[|SHAP("+x1_name.replace("_", "\_")+")|]$",path=path,err_bar=err_bar)
    plot_irl(results, 'influence_X2', "$\mathbb{E}[|SHAP("+x2_name.replace("_", "\_")+")|]$",path=path,err_bar=err_bar)
    plot_irl(results, 'acc', "Accuracy",path=path,err_bar=err_bar)
    plot_irl(results, "di", "Disparate Impact Ratio",path=path,err_bar=err_bar)
    plot_irl(results, "dd", "Demographic Disparity",path=path,err_bar=err_bar)
    plot_irl(results, "eqodd", "Equalized Odds",path=path,err_bar=err_bar)
    plot_irl(results, "eqopp", "Equal Opportunity Difference",path=path,err_bar=err_bar)
    plot_irl(results, "er", "Error ratio",path=path,err_bar=err_bar)
    for i, col in enumerate(X_test.columns.values):
        plot_irl_inf_diff_feats(results, col, i,path=path,err_bar=err_bar,delta=False)
    plot_irl_inf_diff_models(results, X_test.columns.values,path=path,err_bar=err_bar,figsize=figsizemodels)
    plot_irl_inf_MIM_comp(results, X_test.columns.values,figsize=(25,12),path=path,err_bar=err_bar)
    plot_irl_inf_MIM_comp(results, X_test.columns.values,figsize=(25,12),diff=True,path=path,err_bar=err_bar)
    X_test.corr()[prot_attr][1:].plot.bar(figsize=figsizemodels,ylabel="Correlation").grid(axis='y')
    plt.savefig(path+"_corr.pdf", bbox_inches = "tight")

def plot_ate(results, X_test, prot_attr, x1_name, x2_name, path, err_bar=False, figsizemodels=(10,6)):
    """
    Plot all ATE plots for the results of a irl dataset
    """
    plot_irl(results, 'influence_prot_attr', "$\mathbb{E}[|ATE("+prot_attr.replace("_", "\_")+")|]$",path=path,err_bar=err_bar)
    plot_irl(results, 'influence_X1', "$\mathbb{E}[|ATE("+x1_name.replace("_", "\_")+")|]$",path=path,err_bar=err_bar)
    plot_irl(results, 'influence_X2', "$\mathbb{E}[|ATE("+x2_name.replace("_", "\_")+")|]$",path=path,err_bar=err_bar)
    for i, col in enumerate(X_test.columns.values):
        plot_irl_inf_diff_feats(results, col, i,path=path,err_bar=err_bar,delta=False, ii = 'ATE')
    plot_irl_inf_diff_models(results, X_test.columns.values,path=path,err_bar=err_bar,figsize=figsizemodels, ii = 'ATE')
    plot_irl_inf_MIM_comp(results, X_test.columns.values,figsize=(25,12),path=path,err_bar=err_bar, ii = 'ATE')
    plot_irl_inf_MIM_comp(results, X_test.columns.values,figsize=(25,12),diff=True,path=path,err_bar=err_bar, ii = 'ATE')
    X_test.corr()[prot_attr][1:].plot.bar(figsize=figsizemodels,ylabel="Correlation").grid(axis='y')
    plt.savefig(path+"_corr.pdf", bbox_inches = "tight")

def plot_syn_all_err(syn_results, yname, coeff_code, path, num_trials, ii='SHAP'):
    """
    Plot all possible plots for the results of a synthetic dataset
    """
    df_mean = syn_results.groupby(['model','ind']).mean().reset_index()
    df_std = syn_results.groupby(['model','ind']).std().reset_index()
    df_std[df_std.columns[2:]] *= 1.96/np.sqrt(num_trials)
    plot_syn(df_mean,'influence_prot_attr', yname+'_'+coeff_code, y_lim=None,path = path, yerr=df_std,ii=ii)
    plot_syn(df_mean,'influence_X1', yname+'_'+coeff_code,path = path, yerr=df_std,ii=ii)
    plot_syn(df_mean,'acc', yname+'_'+coeff_code,(0.5,0.9),path = path, yerr=df_std)
    plot_syn(df_mean,'di', yname+'_'+coeff_code,path = path, yerr=df_std)
    plot_syn(df_mean,'dd', yname+'_'+coeff_code,path = path, yerr=df_std)
    plot_syn(df_mean,'eqopp', yname+'_'+coeff_code,path = path, yerr=df_std)
    plot_syn(df_mean,'eqodd', yname+'_'+coeff_code,path = path, yerr=df_std)
    plot_syn(df_mean,'influence_X2', yname+'_'+coeff_code,path = path, yerr=df_std,ii=ii)
    plot_syn(df_mean,'er', yname+'_'+coeff_code,(.9,1.75),path = path, yerr=df_std)
    plot_syn(df_mean,'er', yname+'_'+coeff_code,(.9,1.75),path = path, yerr=df_std, show_legend=True)
