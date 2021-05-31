import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyveplot import Hiveplot, Axis, Node
import random

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
# create hiveplot object
h = Hiveplot()

# create three axes, spaced at 120 degrees from each other
h.axes = [
    Axis(start=20, angle=270), # gene
    Axis(start=20, angle=200), # antibiotic
    Axis(start=20, angle=350), # molecular_functions
    Axis(start=20, angle=40), # biological_processes
    Axis(start=20, angle=120), # cellular_components
]

# sort nodes by degree
k = list(nx.degree(G))
k.sort(key=lambda tup: tup[1])

# categorize them as high, medium and low degree
genes_sorted = [v[0] for v in k if v[0] in genes]
antibiotics_sorted = [v[0] for v in k if v[0] in antibiotics]
molecular_functions_sorted = [v[0] for v in k if v[0] in molecular_functions]
biological_processes_sorted = [v[0] for v in k if v[0] in biological_processes]
cellular_components_sorted = [v[0] for v in k if v[0] in cellular_components]

genes_sorted = genes_sorted[int(-0.05*len(genes_sorted)):]
antibiotics_sorted = antibiotics_sorted[int(-0.05*len(antibiotics_sorted)):]
molecular_functions_sorted = molecular_functions_sorted[int(-0.05*len(molecular_functions_sorted)):]
biological_processes_sorted = biological_processes_sorted[int(-0.05*len(biological_processes_sorted)):]
cellular_components_sorted = cellular_components_sorted[int(-0.05*len(cellular_components_sorted)):]

zipped = zip(
    h.axes,
    [genes_sorted,
     antibiotics_sorted,
     molecular_functions_sorted,
     biological_processes_sorted,
     cellular_components_sorted],
    ['gene',
     'antibiotic',
     'molecular_function',
     'biological_process',
     'cellular_components',])

# place these nodes into our three axes
for axis, nodes, node_type in zipped:
    if node_type == 'gene':
        circle_color = '#5ab3e5'
    elif node_type == 'antibiotic':
        circle_color = '#bd3886'
    elif node_type == 'molecular_function':
        circle_color = '#55d400'
    elif node_type == 'biological_process':
        circle_color = '#e7a025'
    elif node_type == 'cellular_components':
        circle_color = '#ff7045'
    else:
        raise ValueError('Invalid node type!')

    for idx, v in enumerate(nodes):
        # create node object
        node = Node(
            radius=G.degree(v),
            label="node %s degree=%s" % (v, G.degree(v)))

        # add it to axis
        axis.add_node(v, node)

        # once it has x, y coordinates, add a circle
        node.add_circle(
            fill=circle_color,
            # stroke_width=0.1,
            fill_opacity=0.7)

        if idx >= (len(nodes)-3):
            # also add a label
            if axis.angle < 180:
                orientation = -1
                scale = 0.6
            else:
                orientation = 1
                scale = 0.35

            if node_type == 'cellular_components':
                scale*=0.7

            node.add_label(
                "%s" % (v),
                angle=axis.angle + 90 * orientation,
                scale=0.15*G.degree(v)*scale)

# extract subgraphs
G_cra = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'confers resistance to antibiotic'))

G_uba = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'upregulated by antibiotic'))

G_represses = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'represses'))

G_activates = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'activates'))

G_has = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'has'))

G_iii = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'is involved in'))

G_ipo = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'is part of'))

G_tb = nx.MultiDiGraph(
    ((u, v, e) for u, v, e in G.edges(data=True) if e['category'] == 'targeted by'))

# draw curves between nodes connected by edges in network

# gene,cra,antibiotic
h.connect_axes(h.axes[0],
               h.axes[1],
               G_cra.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#607d8b')

# gene,uba,antibiotic
h.connect_axes(h.axes[0],
               h.axes[1],
               G_uba.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#9c5d99')

# gene,represses,gene
h.connect_axes(h.axes[0],
               h.axes[0],
               G_represses.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#00a014')

# gene,activates,gene
h.connect_axes(h.axes[0],
               h.axes[0],
               G_activates.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#b26c00')

# gene,has,molecular_function
h.connect_axes(h.axes[0],
               h.axes[2],
               G_has.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#e04d1c')

# gene,iii,biological_processes
h.connect_axes(h.axes[0],
               h.axes[3],
               G_iii.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#007a6b')

# gene,ipo,cellular_components
h.connect_axes(h.axes[4],
               h.axes[0],
               G_ipo.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#009d8b')

# gene,tb,antibiotic
h.connect_axes(h.axes[0],
               h.axes[1],
               G_tb.edges,
               stroke_width=1.5,
               stroke_opacity=0.05,
               stroke='#0081a5')

# save output
h.save('ba_hiveplot.svg')
