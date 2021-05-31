import sys
import pandas as pd
import numpy as np

# read
pd_hypotheses_1 = pd.read_csv(
	'./hypotheses_confidence.txt',
	names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'],
	sep='\t')
pd_hypotheses_1 = pd_hypotheses_1[['Subject', 'Predicate', 'Object', 'Probability']]

pd_hypotheses_2 = pd.read_csv(
	'./hypotheses_confidence_2.txt',
	names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'],
	sep='\t')
pd_hypotheses_2 = pd_hypotheses_2[['Subject', 'Predicate', 'Object', 'Probability']]

pd_validated_hypotheses_1 = pd.read_csv(
	'./validated_hypotheses_1.txt',
	sep='\t')

antibiotics_in_stock = list(set(pd_validated_hypotheses_1['Object'].tolist()))

# merge
pd_hypotheses_merged = pd_hypotheses_1.merge(
	pd_hypotheses_2,
	how='outer',
	on=['Subject', 'Predicate', 'Object'],
	suffixes=('_1', '_2'))

pd_hypotheses_merged = pd_hypotheses_merged.merge(
	pd_validated_hypotheses_1,
	how='outer',
	on=['Subject', 'Predicate', 'Object'])

pd_hypotheses_merged['Antibiotics in stock'] = pd_hypotheses_merged['Object'].apply(
	lambda x: 'Yes' if x in antibiotics_in_stock else 'No')

pd_hypotheses_merged = pd_hypotheses_merged.sort_values(by=['Probability_2'], ascending=False)
# pd_hypotheses_merged.to_csv('~/Jason/VM_Shared/hypotheses_merged.txt', sep='\t', index=False)


pd_hypotheses_can_test = pd_hypotheses_merged[pd_hypotheses_merged['Antibiotics in stock'] == 'Yes']
pd_hypotheses_over_20 = pd_hypotheses_can_test[pd_hypotheses_can_test['Probability_2'] > 0.2]
pd_hypotheses_below_20 = pd_hypotheses_can_test[pd_hypotheses_can_test['Probability_2'] <= 0.2]

sampled_list = []
for antibiotic_in_stock in antibiotics_in_stock:
	pd_filtered = pd_hypotheses_below_20[pd_hypotheses_below_20['Object'] == antibiotic_in_stock]
	sampled_list.append(pd_filtered.sample(n=2))

pd_hypotheses_sampled_below_20 = pd.concat(sampled_list)

pd_hypotheses_to_test = pd.concat([pd_hypotheses_over_20, pd_hypotheses_sampled_below_20])
pd_hypotheses_to_test.to_csv('./hypotheses_to_test.txt', sep='\t', index=False)
