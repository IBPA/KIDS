import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from scipy.stats import pearsonr

###########
# Cycle 1 #
###########

pd_validated = pd.read_csv('./all_validated_hypothesis.txt', sep='\t')
# pd_validated = pd.read_csv('./26_hypotheses.txt', sep='\t')
pd_hypotheses = pd.read_csv('./hypotheses_confidence.txt', sep='\t', names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'])
pd_merged = pd_hypotheses.merge(pd_validated, on=['Subject','Predicate', 'Object'], how='left', indicator=True)
pd_merged['Validated'] = pd_merged['_merge'].apply(lambda x: True if x == 'both' else False)

pd_plot_data = pd_merged[pd_merged['_merge'] == 'both'].copy()
pd_plot_data['Label'] = pd_plot_data['Resistance'].apply(lambda x: 1 if x == 'Yes' else 0)

# pr curve
precision, recall, _ = precision_recall_curve(pd_plot_data['Label'], pd_plot_data['Probability'])
ap = average_precision_score(pd_plot_data['Label'], pd_plot_data['Probability'])

print(ap)

plt.figure()
plt.step(recall, precision, where='post')
plt.hlines(y=0.02, xmin=0.0, xmax=1.0, linestyles='dashed')
plt.grid(True)

# roc
fpr, tpr, _ = roc_curve(pd_plot_data['Label'], pd_plot_data['Probability'])
roc_auc = auc(fpr, tpr)

print(roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC:{:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='k', linestyle='dashed', label='baseline (AUC:{:.3f})'.format(0.5))
plt.grid(True)

###########
# Cycle 2 #
###########

pd_plot_data = pd.read_csv('./all_validated_hypothesis_cycle_2.txt', sep='\t')
pd_plot_data['Label'] = pd_plot_data['Resistance'].apply(lambda x: 1 if x == 'Yes' else 0)

# pr curve
precision, recall, _ = precision_recall_curve(pd_plot_data['Label'], pd_plot_data['Probability'])
ap = average_precision_score(pd_plot_data['Label'], pd_plot_data['Probability'])

print(ap)

plt.figure()
plt.step(recall, precision, where='post')
plt.hlines(y=0.02, xmin=0.0, xmax=1.0, linestyles='dashed')
plt.grid(True)

# roc
fpr, tpr, _ = roc_curve(pd_plot_data['Label'], pd_plot_data['Probability'])
roc_auc = auc(fpr, tpr)

print(roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC:{:.3f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='k', linestyle='dashed', label='baseline (AUC:{:.3f})'.format(0.5))
plt.grid(True)



plt.show()

# # how many tested
# print(pd_merged.head())

# # bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# pd_merged['binned'] = pd.cut(pd_merged['Probability'], bins, include_lowest=True)
# pd_all_grouped = pd_merged.groupby(['binned']).size()
# print(pd_all_grouped)

# pd_validated_grouped = pd_merged[pd_merged['Validated'] == True].groupby(['binned']).size()
# print(pd_validated_grouped)

# pd_validated_positive_grouped = pd_merged[pd_merged['Resistance'] == 'Yes'].groupby(['binned']).size()
# print(pd_validated_positive_grouped)

# # pearsonr
# pd_temp = pd_merged[pd_merged['Validated'] == True]
# print(pd_temp.head())
# probabilities = pd_temp['Probability'].tolist()
# resistance = [1 if x == 'Yes' else 0 for x in pd_temp['Resistance'].tolist()]
# print(pearsonr(probabilities, resistance))
