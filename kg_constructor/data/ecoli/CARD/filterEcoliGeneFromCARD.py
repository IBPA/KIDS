#!/usr/bin/python

import sys
import pandas as pd
import numpy as np

gene_file = sys.argv[1]
card_file = sys.argv[2]

all_genes=pd.read_csv(gene_file,'\t')['name2']
card_kb=pd.read_csv(card_file,'\t')

print('Subject'+'\t'+'Predicate'+'\t'+'Object')
for index, row in card_kb.iterrows():
	if np.any(row['Subject'] == all_genes): # it is E. coli gene
		print(row['Subject']+'\t'+row['Predicate']+'\t'+row['Object'])
