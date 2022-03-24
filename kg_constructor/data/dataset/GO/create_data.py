import sys
import pandas as pd

# read data
with open('data.txt') as f:
    data = f.readlines()

# process data
predicate = {'C' : 'is part of', 'P' : 'is involved in', 'F': 'has'}
triples = []

for line in data:
    line = line[:-1]
    element = line.split('\t')
    gene = element[2]
    terms = element[5].split('|')
    for term in terms:
        triples.append([gene, predicate[element[9]], term])

# save data
pd_data = pd.DataFrame(triples, columns=['Subject', 'Predicate', 'Object'])
pd_data = pd_data.drop_duplicates()
pd_data.to_csv('GO.txt', sep='\t', index=False)
