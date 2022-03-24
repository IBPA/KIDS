import sys
import pandas as pd

# read raw data
with open('./raw_data.txt') as f:
    raw_data = f.readlines()

# process data
genes = []
antibiotics = []
knowledge = []

for line in raw_data:
    line = line[:-1]
    element = line.split('\t')
    if element[0] not in genes:
        genes.append(element[0])
    if element[1] != '' and element[1] not in antibiotics:
        antibiotics.append(element[1])
    if element[1] != '':
        knowledge.append(element[0]+'\t'+element[1])

triples = []
for gene in genes:
    for antibiotic in antibiotics:
        if gene+'\t'+antibiotic not in knowledge:
            triples.append([gene, 'confers no resistance to antibiotic after 36 hours', antibiotic])
        else:
            triples.append([gene, 'confers resistance to antibiotic after 36 hours', antibiotic])

# save data
pd_data = pd.DataFrame(triples, columns=['Subject', 'Predicate', 'Object'])
pd_data = pd_data.drop_duplicates()
pd_data.to_csv('Zhou_et_al.txt', sep='\t', index=False)
