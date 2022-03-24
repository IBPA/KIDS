import sys
import pandas as pd

not_included_gene_file = './not_included_genes.txt'
gene_file = '../../ecoli_refgene/ecoli_refgene.txt'
raw_file = './raw_data.txt'

# load data
with open(not_included_gene_file) as f:
    not_included_genes = f.readlines()
not_included_genes = [row.strip() for row in not_included_genes]

pd_refgenes = pd.read_csv(gene_file, sep='\t')
genes = []
for name, name2 in zip(pd_refgenes['name'], pd_refgenes['name2']):
    if name not in not_included_genes or name2 not in not_included_genes:
        genes.append(name2)

antibiotics = []
knowledge = []
triples = []
first = True

pd_raw_data = pd.read_csv(raw_file, sep='\t')
for row in pd_raw_data.itertuples(index=False):
    if row[2] not in antibiotics:
        antibiotics.append(row[2])
    triples.append(list(row))

for antibiotic in antibiotics:
    for gene in genes:
        pos_triple = [gene, 'upregulated by antibiotic after 30 mins', antibiotic]
        neg_triple = [gene, 'not upregulated by antibiotic after 30 mins', antibiotic]
        if pos_triple not in triples:
            triples.append(neg_triple)

pd_data = pd.DataFrame(triples, columns=['Subject', 'Predicate', 'Object'])
pd_data = pd_data.drop_duplicates()
pd_data.to_csv('./Shaw_et_al_intermediate.txt', sep='\t', index=False)
