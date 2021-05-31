import pandas as pd

pd_data = pd.read_csv('./result.txt', sep='\t', names=['Antibiotic', 'Class'])

antibiotics = pd_data['Antibiotic'].tolist()
taxonomies = pd_data['Class'].tolist()
groups = []

for taxonomy in taxonomies:
    if len(taxonomy.split(':')) == 1:
        groups.append('Unknown')
    else:
        groups.append(taxonomy.split(':')[1])

antibiotic_group = dict(zip(antibiotics, groups))

query_list = [
    'Clarythromycin',
    'Doxycycline hyclate',
    'Doxycycline',
    'Novobiocin',
    'Streptonigrin',
    'Cerulenin']

for query in query_list:
    if query == '':
        print()
    else:
        print(query, 'haha', antibiotic_group[query])
