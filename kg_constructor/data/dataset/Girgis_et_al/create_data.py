import sys
import pandas as pd

gene_file = '../../ecoli_refgene/ecoli_refgene.txt'
raw_file = './raw_data.txt'

# load data
pd_gene = pd.read_csv(gene_file, sep='\t')
dict_gene = dict(zip(pd_gene['name'], pd_gene['name2']))

# create data
triples = []
pd_data = pd.read_csv(raw_file, sep='\t')
for antibiotic in list(pd_data.columns)[1:]:
    for idx in range(pd_data[antibiotic].shape[0]):
        if pd_data['Gene'][idx] not in dict_gene:
            continue

        gene_name = dict_gene[pd_data['Gene'][idx]]
        if pd_data[antibiotic][idx] == 'ND':
            continue
        elif float(pd_data[antibiotic][idx]) < 0:
            triples.append([gene_name, 'confers resistance to antibiotic after 3 days', antibiotic])
        else:
            triples.append([gene_name, 'confers no resistance to antibiotic after 3 days', antibiotic])

# save data
pd_final = pd.DataFrame(triples, columns=['Subject', 'Predicate', 'Object'])
pd_final = pd_final.drop_duplicates()
pd_final.to_csv('Girgis_et_al.txt', sep='\t', index=False)
