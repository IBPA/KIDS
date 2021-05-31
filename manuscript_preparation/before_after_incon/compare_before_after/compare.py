import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import pearsonr
import sys

# read
pd_validated = pd.read_csv('./validated_first_cycle.txt', sep='\t')
pd_validated['Validated'] = True

columns = ['Subject', 'Predicate', 'Object', 'Label', 'Probability']
pd_hypotheses = pd.read_csv('./hypotheses_confidence.txt', sep='\t', names=columns)
pd_hypotheses_wo_resolved = pd.read_csv('./hypotheses_confidence_without_resolved_inconsistencies.txt', sep='\t', names=columns)

pd_hypotheses = pd_hypotheses.drop('Label', axis=1)
pd_hypotheses_wo_resolved = pd_hypotheses_wo_resolved.drop('Label', axis=1)

# merge
pd_hypotheses_merged = pd_hypotheses.merge(
	pd_hypotheses_wo_resolved,
	how='outer',
	on=['Subject', 'Predicate', 'Object'],
	suffixes=('_original', '_without_resolved'))

# extract
pd_extracted = pd_hypotheses_merged.merge(
	pd_validated,
	how='outer',
	on=['Subject', 'Predicate', 'Object'])
pd_extracted = pd_extracted[pd_extracted['Validated'] == True]

pd_extracted['Percent_change'] = (pd_extracted['Probability_original'] - pd_extracted['Probability_without_resolved'])*100/pd_extracted['Probability_without_resolved']
pd_extracted.to_csv('final.txt', index=False, sep='\t')

pd_extracted_without_nan = pd_extracted[~np.isnan(pd_extracted['Percent_change'])]
percent_change_list = pd_extracted_without_nan['Percent_change'].tolist()
cra_list = pd_extracted_without_nan['CRA'].tolist()

# pairwise pearson correlation
pearson_result = pearsonr(percent_change_list, cra_list)
print(pearson_result)
