import collections
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# load
pd_hypotheses = pd.read_csv('./Supplementary_Data_3.csv')
all_resistance_index = (pd_hypotheses['Resistance_1'] == 'Yes') | (pd_hypotheses['Resistance_2'] == 'Yes')
pd_hypotheses = pd_hypotheses[all_resistance_index]

pd_validated = pd.read_csv('./Supplementary_Data_4.csv')
pd_validated = pd_validated.rename(columns={'Gene': 'Subject', 'Antibiotic': 'Object'})
pd_validated = pd_validated[['Subject', 'Object', 'Cycle', 'Discovery Type']]

# merge
pd_merged = pd_hypotheses.merge(pd_validated, on=['Subject', 'Object'], how='left', indicator=True)
pd_data = pd_merged[pd_merged['_merge'] == 'both'].copy()
print(pd_data.head())

def _merge_prob(row):
	if row['Resistance_1'] == 'Yes':
		return row['Probability_1']
	elif row['Resistance_2'] == 'Yes':
		return row['Probability_2']
	else:
		sys.exit('Something wrong!')

pd_data['Probability'] = pd_data.apply(lambda row: _merge_prob(row), axis=1)
pd_data['Probability'] = pd_data['Probability'].apply(lambda x: 0.00001 if x <= 0.00001 else x)
pd_data = pd_data[['Subject', 'Predicate', 'Object', 'Probability', 'Cycle', 'Discovery Type']]

# get genes
genes = list(set(pd_data['Subject'].to_numpy().tolist()))
pd_genes = pd.DataFrame(genes, columns=['Label'])
pd_genes['Category'] = 'gene'
print('gene:', pd_genes.shape)

# get antibiotics
antibiotics = list(set(pd_data['Object'].to_numpy().tolist()))
pd_antibiotics = pd.DataFrame(antibiotics, columns=['Label'])
pd_antibiotics['Category'] = 'antibiotic'
print('antibiotic:', pd_antibiotics.shape)

##############
# nodes file #
##############
pd_nodes = pd.concat([pd_genes, pd_antibiotics])
pd_nodes = pd_nodes.reset_index(drop=True)
pd_nodes['Id'] = pd_nodes.index
print(pd_nodes.head())
pd_nodes.to_csv('./nodes_two_cycles.csv', sep=',', index=False)

##############
# edges file #
##############
pd_edges = pd_data.copy()

lookup = pd_nodes['Label'].to_dict()
lookup = dict((v,k) for k,v in lookup.items())
pd_edges = pd_edges.replace({'Subject': lookup, 'Object': lookup})
pd_edges = pd_edges.rename(
	columns={
		'Subject': 'Source',
		'Predicate': 'Label',
		'Object': 'Target',
		'Probability': 'Weight'})

def map_func(label):
    if 'resistance to antibiotic' in label:
        return 'resistance to antibiotic'
    else:
        raise ValueError('Invalid label: {}'.format(label))

pd_edges['Category'] = pd_edges['Label'].apply(lambda x: map_func(x))
pd_edges['Type'] = 'Directed'
print(pd_edges.head())
pd_edges.to_csv('./edges_two_cycles.csv', sep=',', index=False)
