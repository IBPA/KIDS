import pandas as pd
import sys

# load correct data without inconsistencies
pd_correct = pd.read_csv(
	'./hypotheses_confidence_correct.txt',
	sep='\t',
	names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'])

pd_correct = pd_correct.sort_values(by=['Probability'], ascending=False)

# load previous wrong data with inconsistencies
pd_merged = pd.read_csv(
	'./hypotheses_merged_including_second_cycle.txt',
	sep='\t',
	dtype={'Resistance_1': str, 'Resistance_2': str})

# extract
pd_available_antibiotics = pd_merged[pd_merged['Antibiotics in stock'] == 'Yes']
antibiotics_in_stock = list(set(pd_available_antibiotics['Object'].tolist()))

tested_index = ~pd_merged['Resistance_1'].isnull()
tested_index |= ~pd_merged['Resistance_2'].isnull()
pd_tested = pd_merged[tested_index]

tested_dict = {}
for _, row in pd_tested.iterrows():
	pair = tuple([row['Subject'], row['Object']])
	options = [row['Resistance_1'], row['Resistance_2']]
	if 'Yes' in options:
		tested_dict[pair] = 'Yes'
	elif 'No' in options:
		tested_dict[pair] = 'No'
	elif 'No strain' in options:
		tested_dict[pair] = 'No strain'
	else:
		print(options)
		sys.exit('Something Wrong!')

# process correct file
pd_correct['Antibiotics in stock'] = pd_correct['Object'].apply(lambda x: 'Yes' if x in antibiotics_in_stock else 'No')

def _previous_result(row):
	pair = tuple([row['Subject'], row['Object']])
	if pair in tested_dict:
		return tested_dict[pair]
	else:
		return ''

pd_correct['Previous result'] = pd_correct.apply(lambda row: _previous_result(row), axis=1)
pd_correct.to_csv('./correct_hypotheses.txt', sep='\t', index=False)
