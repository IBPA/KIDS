import sys
import pandas as pd
import matplotlib.pyplot as plt

# extract genes and number of positive antibiotics
pd_kg = pd.read_csv('./kg_final_with_temporal_data_and_validated_inconsistencies.txt', sep='\t')
pd_pos_cra = pd_kg[pd_kg['Predicate'].str.contains('confers resistance')]

antibiotic_gene = []
for antibiotic in list(set(pd_pos_cra['Object'].tolist())):
	pd_subset = pd_pos_cra[pd_pos_cra['Object'] == antibiotic]
	genes = list(set(pd_subset['Subject'].tolist()))

	antibiotic_gene.append([antibiotic, len(genes)])

pd_antibiotic_gene = pd.DataFrame(antibiotic_gene, columns=['antibiotic', 'num_genes'])
pd_antibiotic_gene = pd_antibiotic_gene.sort_values(by=['num_genes'], ascending=False, ignore_index=True)
pd_antibiotic_gene = pd_antibiotic_gene.iloc[0:30]
pd_antibiotic_gene = pd_antibiotic_gene.set_index('antibiotic')
print(pd_antibiotic_gene)
print()

pd_antibiotic_gene.plot.bar()
plt.gca().yaxis.grid(True)
plt.show()
