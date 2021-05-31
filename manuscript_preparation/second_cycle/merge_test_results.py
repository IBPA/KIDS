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

pd_validated_hypotheses_1 = pd.read_csv('./validated_hypotheses_1.txt', sep='\t')
pd_validated_hypotheses_2 = pd.read_csv('./validated_hypotheses_2.txt', sep='\t')

# antibiotics in stock
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

pd_hypotheses_merged = pd_hypotheses_merged.merge(
	pd_validated_hypotheses_2,
	how='outer',
	on=['Subject', 'Predicate', 'Object'],
	suffixes=('_1', '_2'))

pd_hypotheses_merged['Antibiotics in stock'] = pd_hypotheses_merged['Object'].apply(
	lambda x: 'Yes' if x in antibiotics_in_stock else 'No')

pd_hypotheses_merged = pd_hypotheses_merged.sort_values(by=['Probability_2'], ascending=False)
pd_hypotheses_merged.to_csv('./hypotheses_merged_including_second_cycle.txt', sep='\t', index=False)
