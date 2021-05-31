import sys
import pandas as pd
import numpy as np

# load
pd_hypotheses_merged = pd.read_csv(
	'./hypotheses_merged_including_second_cycle.txt',
	sep='\t',
	dtype={
		'Resistance_1': str,
		'Resistance_2': str})

pd_second_cycle = pd_hypotheses_merged[~pd_hypotheses_merged['Resistance_2'].isnull()]
print(pd_second_cycle.shape)
pd_second_cycle = pd_second_cycle[pd_second_cycle['Resistance_2'] != 'No strain'].reset_index(drop=True)
print(pd_second_cycle.head())
print(pd_second_cycle.shape)
print()

# fraction of positives
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
pd_second_cycle['binned'] = pd.cut(pd_second_cycle['Probability_2'], bins, include_lowest=True)
pd_all_grouped = pd_second_cycle.groupby(['binned']).size()
print(pd_all_grouped)

pd_validated_positive_grouped = pd_second_cycle[pd_second_cycle['Resistance_2'] == 'Yes'].groupby(['binned']).size()
print(pd_validated_positive_grouped)

# how much did the probability change?
