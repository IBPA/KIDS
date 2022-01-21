import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind
import pandas as pd
import numpy as np

def load_pickle(filepath_str):
    """
    Load pickled results.

    Inputs:
        filepath_str: path to the pickle file to load

    Returns:
        loaded pickle file
    """
    with open(filepath_str, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def save_figure(fig, save_to):
    fig.savefig(save_to, bbox_inches='tight')


pra_precision_list = []
mlp_precision_list = []
stacked_precision_list = []

pra_recall_list = []
mlp_recall_list = []
stacked_recall_list = []

pra_aupr_list = []
mlp_aupr_list = []
stacked_aupr_list = []

pra_f1_list = []
mlp_f1_list = []
stacked_f1_list = []

for fold in ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']:
    pra_results = load_pickle(
        '../../hypothesis_generator/pra/model/model_instance/{}/instance/test/results/results.pkl'.format(fold))
    mlp_results = load_pickle(
        '../../hypothesis_generator/er_mlp/model/model_instance/{}/test/results/results.pkl'.format(fold))
    stacked_results = load_pickle(
        '../../hypothesis_generator/stacked/model_instance/{}/test/results/results.pkl'.format(fold))

    pra_precision_list.append(pra_results['overall']['precision'])
    mlp_precision_list.append(mlp_results['overall']['precision'])
    stacked_precision_list.append(stacked_results['overall']['precision'])

    pra_recall_list.append(pra_results['overall']['recall'])
    mlp_recall_list.append(mlp_results['overall']['recall'])
    stacked_recall_list.append(stacked_results['overall']['recall'])

    pra_aupr_list.append(pra_results['overall']['map'])
    mlp_aupr_list.append(mlp_results['overall']['map'])
    stacked_aupr_list.append(stacked_results['overall']['map'])

    pra_f1_list.append(pra_results['overall']['f1'])
    mlp_f1_list.append(mlp_results['overall']['f1'])
    stacked_f1_list.append(stacked_results['overall']['f1'])

transe_precision_list = [0.19607843137254902, 0.2033096926713948, 0.16795366795366795, 0.151033386327504, 0.165402124430956]
transe_recall_list = [0.25, 0.26959247648902823, 0.2727272727272727, 0.29780564263322884, 0.34169278996865204]
transe_aupr_list = [0.11255384018355645, 0.12426293637557527, 0.10919977074903114, 0.11037806383278484, 0.11761317520898823]
transe_f1_list = [0.2197802197802198, 0.23180592991913745, 0.2078853046594982, 0.20042194092827006, 0.2229038854805726]

transd_precision_list = [0.25925925925925924, 0.2980132450331126, 0.3008849557522124, 0.2789473684210526, 0.23919308357348704]
transd_recall_list = [0.153125, 0.28213166144200624, 0.21316614420062696, 0.16614420062695925, 0.2601880877742947]
transd_aupr_list = [0.13665278183463295, 0.19258632525922778, 0.15276210822199224, 0.15480085319179704, 0.16245139910942705]
transd_f1_list = [0.1925343811394892, 0.2898550724637681, 0.24954128440366974, 0.20825147347740672, 0.2492492492492493]

precision = [*pra_precision_list, *mlp_precision_list, *stacked_precision_list, *transe_precision_list, *transd_precision_list]
metric = ['Precision' for _ in range(len(precision))]
model = [*['PRA' for _ in range(5)], *['MLP' for _ in range(5)], *['Stacked' for _ in range(5)], *['TransE' for _ in range(5)], *['TransD' for _ in range(5)]]
pd_precision = pd.DataFrame({'Score': precision, 'Evaluation Metric': metric, 'Model': model})

recall = [*pra_recall_list, *mlp_recall_list, *stacked_recall_list, *transe_recall_list, *transd_recall_list]
metric = ['Recall' for _ in range(len(recall))]
pd_recall = pd.DataFrame({'Score': recall, 'Evaluation Metric': metric, 'Model': model})

aupr = [*pra_aupr_list, *mlp_aupr_list, *stacked_aupr_list, *transe_aupr_list, *transd_aupr_list]
metric = ['AUPR' for _ in range(len(aupr))]
pd_aupr = pd.DataFrame({'Score': aupr, 'Evaluation Metric': metric, 'Model': model})

f1 = [*pra_f1_list, *mlp_f1_list, *stacked_f1_list, *transe_f1_list, *transd_f1_list]
metric = ['F1' for _ in range(len(f1))]
pd_f1 = pd.DataFrame({'Score': f1, 'Evaluation Metric': metric, 'Model': model})

pd_result = pd.concat([pd_precision, pd_recall, pd_aupr, pd_f1]).reset_index(drop=True)

# f1
print('pra_aupr_list: ', np.mean(pra_aupr_list))
print('mlp_aupr_list: ', np.mean(mlp_aupr_list))
print('stacked_aupr_list: ', np.mean(stacked_aupr_list))
print('transe_aupr_list: ', np.mean(transe_aupr_list))
print('transd_aupr_list: ', np.mean(transd_aupr_list))

# p-value
print('Precision p-value')
_, pval = ttest_ind(pra_precision_list, mlp_precision_list)
print('  pra, mlp: ', pval)
_, pval = ttest_ind(mlp_precision_list, stacked_precision_list)
print('  mlp, stacked: ', pval)
_, pval = ttest_ind(pra_precision_list, stacked_precision_list)
print('  pra, stacked: ', pval)
print()

print('Recall p-value')
_, pval = ttest_ind(pra_recall_list, mlp_recall_list)
print('  pra, mlp: ', pval)
_, pval = ttest_ind(mlp_recall_list, stacked_recall_list)
print('  mlp, stacked: ', pval)
_, pval = ttest_ind(pra_recall_list, stacked_recall_list)
print('  pra, stacked: ', pval)
print()

print('AUPR p-value')
_, pval = ttest_ind(pra_aupr_list, mlp_aupr_list)
print('  pra, mlp: ', pval)
_, pval = ttest_ind(mlp_aupr_list, stacked_aupr_list)
print('  mlp, stacked: ', pval)
_, pval = ttest_ind(pra_aupr_list, stacked_aupr_list)
print('  pra, stacked: ', pval)
print()

print('F1 p-value')
_, pval = ttest_ind(pra_f1_list, mlp_f1_list)
print('  pra, mlp: ', pval)
_, pval = ttest_ind(mlp_f1_list, stacked_f1_list)
print('  mlp, stacked: ', pval)
_, pval = ttest_ind(pra_f1_list, stacked_f1_list)
print('  pra, stacked: ', pval)
print()

trucker_f1_list = [0.29411764705882354, 0.34991423670668953, 0.28901734104046245, 0.3029315960912053, 0.30153846153846153]
_, pval = ttest_ind(trucker_f1_list, stacked_f1_list)
print('  tucker, stacked: ', pval)
print()
sys.exit()

# figure
fig = plt.figure()

sns.set(style="whitegrid")
ax = sns.boxplot(x='Evaluation Metric', y='Score', hue='Model', data=pd_result, palette="Set3")
ax = sns.swarmplot(x='Evaluation Metric', y='Score', hue='Model', data=pd_result, color=".25", dodge=True)

plt.axis([None, None, 0.08, 0.57])
save_figure(fig, './models_metric.svg')
