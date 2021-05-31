import collections
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

pd_validated = pd.read_csv('./all_validated_hypothesis.txt', sep='\t')
pd_hypotheses = pd.read_csv('./hypotheses_confidence.txt', sep='\t', names=['Subject', 'Predicate', 'Object', 'Label', 'Probability'])
pd_merged = pd_hypotheses.merge(pd_validated, on=['Subject','Predicate', 'Object'], how='left', indicator=True)
pd_data = pd_merged[pd_merged['_merge'] == 'both'].copy()

print(pd_data.head())
sys.exit()

pd_data = pd_data[pd_data['Resistance'] == 'Yes']
pd_data = pd_data[['Subject', 'Predicate', 'Object', 'Probability']]
pd_data['Probability'] = pd_data['Probability'].apply(lambda x: 0.00001 if x <= 0.00001 else x)

# select data by relation type
pd_cras = pd_data[pd_data['Predicate'].str.contains('resistance to antibiotic')]

# get genes
genes = []
genes.extend(pd_cras['Subject'].to_numpy().tolist())
genes = list(set(genes))

pd_genes = pd.DataFrame(genes, columns=['Label'])
pd_genes['Category'] = 'gene'
print('gene:', pd_genes.shape)

# get antibiotics
antibiotics = []
antibiotics.extend(pd_cras['Object'].to_numpy().tolist())
antibiotics = list(set(antibiotics))

pd_antibiotics = pd.DataFrame(antibiotics, columns=['Label'])
pd_antibiotics['Category'] = 'antibiotic'
print('antibiotic:', pd_antibiotics.shape)

##############
# nodes file #
##############
pd_nodes = pd.concat(
    [pd_genes,
    pd_antibiotics])

pd_nodes = pd_nodes.reset_index(drop=True)
pd_nodes['Id'] = pd_nodes.index
print(pd_nodes.head())
pd_nodes.to_csv('./nodes.csv', sep=',', index=False)

##############
# edges file #
##############
pd_edges = pd_data.copy()

lookup = pd_nodes['Label'].to_dict()
lookup = dict((v,k) for k,v in lookup.items())
pd_edges = pd_edges.replace({'Subject': lookup, 'Object': lookup})
pd_edges = pd_edges.rename(columns={'Subject': 'Source', 'Predicate': 'Label', 'Object': 'Target', 'Probability': 'Weight'})

def map_func(label):
    if 'resistance to antibiotic' in label:
        return 'resistance to antibiotic'
    else:
        raise ValueError('Invalid label: {}'.format(label))

pd_edges['Category'] = pd_edges['Label'].apply(lambda x: map_func(x))
pd_edges['Type'] = 'Directed'
print(pd_edges.head())
pd_edges.to_csv('./edges.csv', sep=',', index=False)
