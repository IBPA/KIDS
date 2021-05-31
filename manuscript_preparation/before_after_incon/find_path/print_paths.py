import sys
import pandas as pd
import networkx as nx

# load train data
pd_train = pd.read_csv(
    './train_original.txt',
    sep='\t',
    names=['Subject', 'Predicate', 'Object', 'Label'])

pd_train = pd_train[['Subject', 'Predicate', 'Object']]

for column in ['Subject', 'Predicate', 'Object']:
    pd_train[column] = pd_train[column].apply(lambda x: x.replace('#SEMICOLON#', ':'))
    pd_train[column] = pd_train[column].apply(lambda x: x.replace('#SPACE#', ' '))
    pd_train[column] = pd_train[column].apply(lambda x: x.replace('#COMMA#', ','))

# load test data
# pd_test = pd.read_csv(
#     './test.txt',
#     sep='\t',
#     names=['Subject', 'Predicate', 'Object', 'Label'])

pd_test = pd.read_csv('./all_validated_hypothesis.txt', sep='\t')
pd_test = pd_test.dropna(subset=['Resistance'])
pd_test = pd_test[pd_test['Resistance'] == 'Yes']

pd_test = pd_test[['Subject', 'Predicate', 'Object']]

# for column in ['Subject', 'Predicate', 'Object']:
#     pd_test[column] = pd_test[column].apply(lambda x: x.replace('#SEMICOLON#', ':'))
#     pd_test[column] = pd_test[column].apply(lambda x: x.replace('#SPACE#', ' '))
#     pd_test[column] = pd_test[column].apply(lambda x: x.replace('#COMMA#', ','))

# build entity lookup
entities = []
entities.extend(pd_train['Subject'].tolist())
entities.extend(pd_train['Object'].tolist())
entities = list(set(entities))
entities_dict = {k: v for v, k in enumerate(entities)}
entities_dict_flip = {v: k for v, k in enumerate(entities)}

# build graph
G = nx.MultiGraph()

for _, row in pd_train.iterrows():
    node_from = entities_dict[row['Subject']]
    node_to = entities_dict[row['Object']]
    G.add_edge(node_from, node_to)

match = [('cydX', 'Vancomycin'),
         ('elfD', 'Vancomycin'),
         ('ylcG', 'Vancomycin'),
         ('rpsU', 'Sulfamethoxazole'),
         ('ydjI', 'Sulfamethoxazole'),
         ('yjjY', 'Sulfamethoxazole'),
         ('ymfI', 'Sulfamethoxazole')]

result = []

for index, row in pd_test.iterrows():
    if (index % 1000) == 0:
        print('Processing index {}/{}'.format(index, pd_test.shape[0]))

    if (row['Subject'] not in entities_dict) or (row['Object'] not in entities_dict):
        continue

    node_from = entities_dict[row['Subject']]
    node_to = entities_dict[row['Object']]

    for path in nx.all_simple_paths(G, source=node_from, target=node_to, cutoff=3):
        translated_path = [entities_dict_flip[p] for p in path]

        for sub, obj in match:
            translated_path_copy = translated_path.copy()

            if (sub in translated_path) and (obj in translated_path):
                sub_index = translated_path.index(sub)
                obj_index = translated_path.index(obj)

                if (sub_index + 1 == obj_index) or (obj_index + 1 == sub_index):
                    translated_path_copy.insert(0, str(index))
                    result.append(translated_path_copy)

with open('./result.txt', 'w') as file:
    for line in result:
        file.write('\t'.join(line) + '\n')
