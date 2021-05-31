import collections
import pandas as pd
import sys

pd_data = pd.read_csv('./kg_final_with_temporal_data_and_validated_inconsistencies.txt', sep='\t')
pd_data = pd_data[['Subject', 'Predicate', 'Object']]

pd_confers = pd_data[pd_data['Predicate'].str.contains('confers')]
pd_confers = pd_confers.sample(frac=0.005)
pd_else = pd_data[~pd_data['Predicate'].str.contains('confers')].sample(frac=0.02)

pd_data = pd_confers.append(pd_else, ignore_index=True).reset_index()

pd_data = pd_data.sample(frac=0.10)

print(pd_data.shape)
print()

# select data by relation type
pd_cras = pd_data[pd_data['Predicate'].str.contains('resistance to antibiotic')]
pd_ubas = pd_data[pd_data['Predicate'].str.contains('upregulated by antibiotic')]
pd_represses = pd_data[pd_data['Predicate'].str.contains('represses')]
pd_activates = pd_data[pd_data['Predicate'].str.contains('activates')]
pd_has = pd_data[pd_data['Predicate'].str.contains('has')]
pd_iii = pd_data[pd_data['Predicate'].str.contains('is involved in')]
pd_ipo = pd_data[pd_data['Predicate'].str.contains('is part of')]
pd_tb = pd_data[pd_data['Predicate'].str.contains('targeted by')]

# get genes
genes = []
genes.extend(pd_cras['Subject'].to_numpy().tolist())
genes.extend(pd_ubas['Subject'].to_numpy().tolist())
genes.extend(pd_represses['Subject'].to_numpy().tolist())
genes.extend(pd_represses['Object'].to_numpy().tolist())
genes.extend(pd_activates['Subject'].to_numpy().tolist())
genes.extend(pd_activates['Object'].to_numpy().tolist())
genes.extend(pd_has['Subject'].to_numpy().tolist())
genes.extend(pd_iii['Subject'].to_numpy().tolist())
genes.extend(pd_ipo['Subject'].to_numpy().tolist())
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

# get molecular_function
molecular_functions = pd_has['Object'].to_numpy().tolist()
molecular_functions = list(set(molecular_functions))

pd_molecular_functions = pd.DataFrame(molecular_functions, columns=['Label'])
pd_molecular_functions['Category'] = 'molecular_function'
print('molecular_function:', pd_molecular_functions.shape)

# get molecular_function
biological_processes = pd_iii['Object'].to_numpy().tolist()
biological_processes = list(set(biological_processes))

pd_biological_processes = pd.DataFrame(biological_processes, columns=['Label'])
pd_biological_processes['Category'] = 'biological_process'
print('biological_process:', pd_biological_processes.shape)

# get molecular_function
cellular_components = pd_ipo['Object'].to_numpy().tolist()
cellular_components = list(set(cellular_components))

pd_cellular_components = pd.DataFrame(cellular_components, columns=['Label'])
pd_cellular_components['Category'] = 'cellular_component'
print('cellular_component:', pd_cellular_components.shape)

##############
# nodes file #
##############
pd_nodes = pd.concat(
    [pd_genes,
    pd_antibiotics,
    pd_molecular_functions,
    pd_biological_processes,
    pd_cellular_components])

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
    elif 'has' in label:
        return 'has'
    elif 'is involved in' in label:
        return 'is involved in'
    elif 'is part of' in label:
        return 'is part of'
    elif 'targeted by' in label:
        return 'targeted by'
    else:
        raise ValueError('Invalid label: {}'.format(label))

pd_edges['Category'] = pd_edges['Label'].apply(lambda x: map_func(x))
pd_edges['Type'] = 'Directed'
pd_edges['Weight'] = 1.0

print(pd_edges.head())
pd_edges.to_csv('./edges.csv', sep=',', index=False)
