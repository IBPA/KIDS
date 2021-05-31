import collections
import pandas as pd
import sys

pd_data = pd.read_csv('../other_models/all_validated_hypothesis.txt', sep='\t')
pd_data = pd_data[pd_data['Resistance'] == 'Yes']
pd_data = pd_data[['Subject', 'Predicate', 'Object']]

# get genes
genes = []
genes.extend(pd_data['Subject'].to_numpy().tolist())
genes = list(set(genes))

pd_genes = pd.DataFrame(genes, columns=['Label'])
pd_genes['Category'] = 'gene'
print('gene:', pd_genes.shape)

# get antibiotics
antibiotics = []
antibiotics.extend(pd_data['Object'].to_numpy().tolist())
antibiotics = list(set(antibiotics))

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
pd_nodes.to_csv('./nodes.csv', sep=',', index=False)

##############
# edges file #
##############
pd_edges = pd_data.copy()

lookup = pd_nodes['Label'].to_dict()
lookup = dict((v, k) for k, v in lookup.items())
pd_edges = pd_edges.replace({'Subject': lookup, 'Object': lookup})
pd_edges = pd_edges.rename(columns={'Subject': 'Source', 'Predicate': 'Label', 'Object': 'Target'})

def map_func(label):
    if 'resistance to antibiotic' in label:
        return 'confers resistance to antibiotic'
    else:
        raise ValueError('Invalid label: {}'.format(label))

pd_edges['Category'] = pd_edges['Label'].apply(lambda x: map_func(x))
pd_edges['Type'] = 'Directed'
pd_edges['Weight'] = 1.0

print(pd_edges.head())
pd_edges.to_csv('./edges.csv', sep=',', index=False)
