import collections
import pandas as pd
import sys

# load data
pd_data = pd.read_csv('./kg_final_with_temporal_data_and_validated_inconsistencies.txt', sep='\t')
pd_data = pd_data[['Subject', 'Predicate', 'Object']]

# filter positives only
neg_predicates = [
    'confers no resistance to antibiotic',
    'not upregulated by antibiotic',
    'no represses',
    'no activates',]

def _check_match(x, predicates):
    flag = False
    for predicate in predicates:
        if predicate in x:
            flag = True
    return flag

pd_data = pd_data[~pd_data['Predicate'].apply(lambda x: _check_match(x, neg_predicates))]


# select data by relation type
pd_cras = pd_data[pd_data['Predicate'].str.contains('resistance to antibiotic')]
pd_ubas = pd_data[pd_data['Predicate'].str.contains('upregulated by antibiotic')]
pd_represses = pd_data[pd_data['Predicate'].str.contains('represses')]
pd_activates = pd_data[pd_data['Predicate'].str.contains('activates')]
pd_tb = pd_data[pd_data['Predicate'].str.contains('targeted by')]

pd_data = pd.concat([pd_cras, pd_ubas, pd_represses, pd_activates, pd_tb]).reset_index(drop=True)

# get genes
genes = []
genes.extend(pd_cras['Subject'].to_numpy().tolist())
genes.extend(pd_ubas['Subject'].to_numpy().tolist())
genes.extend(pd_represses['Subject'].to_numpy().tolist())
genes.extend(pd_represses['Object'].to_numpy().tolist())
genes.extend(pd_activates['Subject'].to_numpy().tolist())
genes.extend(pd_activates['Object'].to_numpy().tolist())
genes.extend(pd_tb['Subject'].to_numpy().tolist())
genes = list(set(genes))

pd_genes = pd.DataFrame(genes, columns=['Label'])
pd_genes['Category'] = 'gene'
print('gene:', pd_genes.shape)

# get antibiotics
antibiotics = []
antibiotics.extend(pd_cras['Object'].to_numpy().tolist())
antibiotics.extend(pd_ubas['Object'].to_numpy().tolist())
antibiotics.extend(pd_tb['Object'].to_numpy().tolist())
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
pd_edges = pd_edges.rename(columns={'Subject': 'Source', 'Predicate': 'Label', 'Object': 'Target'})

def map_func(label):
    if 'resistance to antibiotic' in label:
        return 'resistance to antibiotic'
    elif 'upregulated by antibiotic' in label:
        return 'upregulated by antibiotic'
    elif 'represses' in label:
        return 'represses'
    elif 'activates' in label:
        return 'activates'
    elif 'targeted by' in label:
        return 'targeted by'
    else:
        raise ValueError('Invalid label: {}'.format(label))

pd_edges['Category'] = pd_edges['Label'].apply(lambda x: map_func(x))
pd_edges['Type'] = 'Directed'
pd_edges['Weight'] = 1.0
print(pd_edges.head())
pd_edges.to_csv('./edges.csv', sep=',', index=False)
