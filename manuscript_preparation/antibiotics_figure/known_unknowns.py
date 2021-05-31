import pandas as pd
import sys

# known data
pd_known = pd.read_csv(
    '/home/jyoun/Jason/Research/KIDS/hypothesis_generator/data/kb/folds/data.txt',
    names=['Subject', 'Predicate', 'Object', 'Label'],
    sep='\t')

for column in ['Subject', 'Predicate', 'Object']:
    pd_known[column] = pd_known[column].apply(lambda x: x.replace('#SEMICOLON#', ':'))
    pd_known[column] = pd_known[column].apply(lambda x: x.replace('#SPACE#', ' '))
    pd_known[column] = pd_known[column].apply(lambda x: x.replace('#COMMA#', ','))

pd_known_cra = pd_known[pd_known['Predicate'] == 'confers resistance to antibiotic']
pd_known_pos_cra = pd_known_cra[pd_known_cra['Label'] == 1]
pd_known_neg_cra = pd_known_cra[pd_known_cra['Label'] == -1]

# unknown data
pd_unknown = pd.read_csv(
    '/home/jyoun/Jason/Research/KIDS/hypothesis_generator/data/kb/final/test.txt',
    names=['Subject', 'Predicate', 'Object', 'Label'],
    sep='\t')

for column in ['Subject', 'Predicate', 'Object']:
    pd_unknown[column] = pd_unknown[column].apply(lambda x: x.replace('#SEMICOLON#', ':'))
    pd_unknown[column] = pd_unknown[column].apply(lambda x: x.replace('#SPACE#', ' '))
    pd_unknown[column] = pd_unknown[column].apply(lambda x: x.replace('#COMMA#', ','))

# antibiotics
pd_antibiotic = pd.read_csv(
    './result.txt',
    names=['antibiotic', 'classification'],
    sep='\t')

pd_antibiotic['group'] = pd_antibiotic['classification'].apply(
    lambda x: x.split(':')[1] if len(x.split(':')) > 1 else '')

antibiotic_groups = [
    'Organoheterocyclic compounds',
    'Organic oxygen compounds',
    'Organic acids and derivatives',
    'Benzenoids',
    'Phenylpropanoids and polyketides',
    'Others']

pd_antibiotic['group'] = pd_antibiotic['group'].apply(
    lambda x: 'Others' if x not in antibiotic_groups else x)

antibiotics_dict = {}
for group in antibiotic_groups:
    pd_group = pd_antibiotic[pd_antibiotic['group'] == group]
    antibiotics_dict[group] = pd_group['antibiotic'].tolist()

# how many knowns for each antibiotic
print(len(set(pd_known_pos_cra['Object'].tolist())))
for group, antibiotics in antibiotics_dict.items():
    pd_known_selected = pd_known_cra[pd_known_cra['Object'].apply(lambda x: True if x in antibiotics else False)]
    pd_unknown_selected = pd_unknown[pd_unknown['Object'].apply(lambda x: True if x in antibiotics else False)]
    print(group, pd_known_selected.shape[0], pd_unknown_selected.shape[0])
