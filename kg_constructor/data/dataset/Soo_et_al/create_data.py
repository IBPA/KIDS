import sys
import pandas as pd
import numpy as np

# read data
genes = pd.unique(pd.read_csv('./genes.txt')['gene'])
pd_raw = pd.read_csv('raw_data.txt', sep='\t', names=['Antibiotic', 'Gene'])
pd_raw = pd_raw.dropna()

# process data
antibiotics = []
knowledge = []
for _, row in pd_raw.iterrows():
   knowledge.append(row['Gene'] + ' ' + row['Antibiotic'])
   if row['Antibiotic'] not in antibiotics:
      antibiotics.append(row['Antibiotic'])

triples = []
for antibiotic in antibiotics:
   for gene in genes:
      if gene + ' ' + antibiotic in knowledge:
         triples.append([gene, 'confers resistance to antibiotic after 7 days', antibiotic])
      else:
         triples.append([gene, 'confers no resistance to antibiotic after 7 days', antibiotic])

# save data
pd_data = pd.DataFrame(triples, columns=['Subject', 'Predicate', 'Object'])
pd_data = pd_data.drop_duplicates()
pd_data.to_csv('./Soo_et_al.txt', sep='\t', index=False)
