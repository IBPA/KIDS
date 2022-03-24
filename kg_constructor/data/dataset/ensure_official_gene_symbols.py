import sys
import pandas as pd

map_file = '../name_map.txt'
gene_file = '../keio_mutants/KEIO_KO_List.txt'

input_filepath = sys.argv[1]
output_filepath = sys.argv[2]
pos_predicate = sys.argv[3]
neg_predicate = sys.argv[4]
gene_idx = 0

# create gene name mapping lookup table
pd_name_map = pd.read_csv(map_file, sep='\t')
map_dict = dict(zip(pd_name_map['Source'], pd_name_map['Target']))

# create list of genes
with open(gene_file) as f:
    keio_genes = f.readlines()
keio_genes = [row.strip() for row in keio_genes]

genes = []
for kg in keio_genes[1:]:
    if kg in map_dict and map_dict[kg] not in genes:
        # this gene is from the 'Source' column
        # add the 'Target' gene name into genes
        genes.append(map_dict[kg])
    elif kg not in map_dict and kg not in genes:
        # name mapping for this gene cannot be found
        # add it to genes as-is
        genes.append(kg)

#
pd_input = pd.read_csv(input_filepath, sep='\t')
data_lines = zip(pd_input['Subject'], pd_input['Predicate'], pd_input['Object'])

mapped_lines = []
for line in data_lines:
    if line[gene_idx] in genes:
        mapped_lines.append(list(line))
    elif line[gene_idx] in map_dict:
        mapped_lines.append([map_dict[line[gene_idx]], line[1], line[2-gene_idx]])

#
final_data = []
for line in mapped_lines:
    if line[1] == neg_predicate and [line[0], pos_predicate, line[2]] in mapped_lines:
        continue
    if line[1] == pos_predicate and [line[0], neg_predicate, line[2]] in mapped_lines:
        continue
    final_data.append(line)

pd_final = pd.DataFrame(final_data, columns=['Subject', 'Predicate', 'Object'])
pd_final = pd_final.drop_duplicates()
pd_final.to_csv(output_filepath, sep='\t', index=False)
