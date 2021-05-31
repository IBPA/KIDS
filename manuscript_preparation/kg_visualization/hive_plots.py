import networkx as nx
import numpy as np
from hiveplot import HivePlot
import pandas as pd
import matplotlib.pyplot as plt

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

# remove the negatives
pd_data = pd_data[~pd_data['Predicate'].apply(lambda x: _check_match(x, neg_predicates))]

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

# get biological_process
biological_processes = pd_iii['Object'].to_numpy().tolist()
biological_processes = list(set(biological_processes))

pd_biological_processes = pd.DataFrame(biological_processes, columns=['Label'])
pd_biological_processes['Category'] = 'biological_process'
print('biological_process:', pd_biological_processes.shape)

# get cellular_component
cellular_components = pd_ipo['Object'].to_numpy().tolist()
cellular_components = list(set(cellular_components))

pd_cellular_components = pd.DataFrame(cellular_components, columns=['Label'])
pd_cellular_components['Category'] = 'cellular_component'
print('cellular_component:', pd_cellular_components.shape)

#########
# nodes #
#########
pd_nodes = pd.concat(
    [pd_genes,
    pd_antibiotics,
    pd_molecular_functions,
    pd_biological_processes,
    pd_cellular_components])

pd_nodes = pd_nodes.reset_index(drop=True)

#########
# edges #
#########
pd_edges = pd_data.copy()

def map_func(label):
    if label.startswith('confers resistance to antibiotic'):
        return 'confers resistance to antibiotic'
    elif label.startswith('confers no resistance to antibiotic'):
        return 'confers no resistance to antibiotic'
    elif label.startswith('upregulated by antibiotic'):
        return 'upregulated by antibiotic'
    elif label.startswith('not upregulated by antibiotic'):
        return 'not upregulated by antibiotic'
    elif 'represses' == label:
        return 'represses'
    elif 'no represses' == label:
        return 'no represses'
    elif 'activates' == label:
        return 'activates'
    elif 'no activates' == label:
        return 'no activates'
    elif 'has' == label:
        return 'has'
    elif 'is involved in' == label:
        return 'is involved in'
    elif 'is part of' == label:
        return 'is part of'
    elif 'targeted by' == label:
        return 'targeted by'
    else:
        raise ValueError('Invalid label: {}'.format(label))

pd_edges['Category'] = pd_edges['Predicate'].apply(lambda x: map_func(x))

# build graph
G = nx.MultiDiGraph()

for _, row in pd_nodes.iterrows():
	G.add_node(row['Label'], category=row['Category'])

for _, row in pd_edges.iterrows():
    G.add_edge(row['Subject'], row['Object'], category=row['Category'])

for n, d in G.nodes(data=True):
	print(n)
	print(d)
	break

for u, v, c in G.edges.data():
	print(u)
	print(v)
	print(c)
	break

# hive plot
nodes = dict()
# nodes['gene'] = [(n,d) for n, d in G.nodes(data=True) if d['category'] == 'gene']
# nodes['antibiotic'] = [(n,d) for n, d in G.nodes(data=True) if d['category'] == 'antibiotic']
# nodes['molecular_function'] = [(n,d) for n, d in G.nodes(data=True) if d['category'] == 'molecular_function']
# nodes['biological_process'] = [(n,d) for n, d in G.nodes(data=True) if d['category'] == 'biological_process']
# nodes['cellular_component'] = [(n,d) for n, d in G.nodes(data=True) if d['category'] == 'cellular_component']

nodes['gene'] = [n for n, d in G.nodes(data=True) if d['category'] == 'gene']
nodes['antibiotic'] = [n for n, d in G.nodes(data=True) if d['category'] == 'antibiotic']
nodes['molecular_function'] = [n for n, d in G.nodes(data=True) if d['category'] == 'molecular_function']
nodes['biological_process'] = [n for n, d in G.nodes(data=True) if d['category'] == 'biological_process']
nodes['cellular_component'] = [n for n, d in G.nodes(data=True) if d['category'] == 'cellular_component']

edges = dict()
edges['cra'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'confers resistance to antibiotic']
edges['uba'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'upregulated by antibiotic']
edges['represses'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'represses']
edges['activates'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'activates']
edges['has'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'has']
edges['iii'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'is involved in']
edges['ipo'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'is part of']
edges['tb'] = [(u,v,d) for u,v,d in G.edges.data() if d['category'] == 'targeted by']

nodes_cmap = dict()
nodes_cmap['gene'] = 'red'
nodes_cmap['antibiotic'] = 'green'
nodes_cmap['molecular_function'] = 'blue'
nodes_cmap['biological_process'] = 'pink'
nodes_cmap['cellular_component'] = 'magenta'

edges_cmap = dict()
edges_cmap['cra'] = 'gray'
edges_cmap['uba'] = 'gray'
edges_cmap['represses'] = 'gray'
edges_cmap['activates'] = 'gray'
edges_cmap['has'] = 'gray'
edges_cmap['iii'] = 'gray'
edges_cmap['ipo'] = 'gray'
edges_cmap['tb'] = 'gray'

h = HivePlot(nodes, edges, nodes_cmap, edges_cmap, linewidth=0.1, is_directed=True, scale=30)
h.draw()
plt.show()
