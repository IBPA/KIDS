import sys
import pandas as pd
import matplotlib.pyplot as plt

# extract antibiotic class
antibiotic_classes = [
	'Organoheterocyclic compounds',
	'Organic oxygen compounds',
	'Organic acids and derivatives',
	'Benzenoids',
	'Phenylpropanoids and polyketides']

def extract_class(taxonomy):
    if len(taxonomy.split(':')) == 1:
        return 'Other'
    elif taxonomy.split(':')[1] not in antibiotic_classes:
        return 'Other'
    else:
    	return taxonomy.split(':')[1]

pd_antibiotic_group = pd.read_csv('../antibiotics_figure/result.txt', sep='\t', names=['Antibiotic', 'Taxonomy'])
pd_antibiotic_group['Class'] = pd_antibiotic_group['Taxonomy'].apply(lambda x: extract_class(x))
antibiotic_class_lookup = dict(zip(pd_antibiotic_group['Antibiotic'], pd_antibiotic_group['Class']))

# extract genes and number of positive antibiotics
pd_kg = pd.read_csv('./kg_final_with_temporal_data_and_validated_inconsistencies.txt', sep='\t')
pd_pos_cra = pd_kg[pd_kg['Predicate'].str.contains('confers resistance')]

gene_antibiotic = []
for gene in list(set(pd_pos_cra['Subject'].tolist())):
	pd_subset = pd_pos_cra[pd_pos_cra['Subject'] == gene]
	antibiotics = list(set(pd_subset['Object'].tolist()))
	antibiotics_class = [antibiotic_class_lookup[i] for i in antibiotics]

	gene_antibiotic.append([gene, len(antibiotics), antibiotics_class])

pd_gene_antibiotic = pd.DataFrame(gene_antibiotic, columns=['gene', 'num_antibiotics', 'antibiotics_class'])
pd_gene_antibiotic = pd_gene_antibiotic.sort_values(by=['num_antibiotics'], ascending=False, ignore_index=True)
pd_gene_antibiotic = pd_gene_antibiotic.iloc[0:30]
print(pd_gene_antibiotic)
print()

antibiotics_column = pd_gene_antibiotic['antibiotics_class'].tolist()

organoheterocyclic_compounds = [i.count('Organoheterocyclic compounds') for i in antibiotics_column]
organic_oxygen_compounds = [i.count('Organic oxygen compounds') for i in antibiotics_column]
organic_acids_and_derivatives = [i.count('Organic acids and derivatives') for i in antibiotics_column]
benzenoids = [i.count('Benzenoids') for i in antibiotics_column]
phenylpropanoids_and_polyketides = [i.count('Phenylpropanoids and polyketides') for i in antibiotics_column]
other = [i.count('Other') for i in antibiotics_column]

pd_gene_antibiotic['Organoheterocyclic compounds'] = organoheterocyclic_compounds
pd_gene_antibiotic['Organic oxygen compounds'] = organic_oxygen_compounds
pd_gene_antibiotic['Organic acids and derivatives'] = organic_acids_and_derivatives
pd_gene_antibiotic['Benzenoids'] = benzenoids
pd_gene_antibiotic['Phenylpropanoids and polyketides'] = phenylpropanoids_and_polyketides
pd_gene_antibiotic['Other'] = other

pd_gene_antibiotic = pd_gene_antibiotic.set_index('gene')
pd_gene_antibiotic = pd_gene_antibiotic.drop(columns=['num_antibiotics', 'antibiotics_class'])

pd_gene_antibiotic.plot.bar(stacked=True)
plt.show()

