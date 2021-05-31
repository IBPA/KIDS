import sys
import numpy as np
import pandas as pd

# potential connection by resolved inconsistencies
pd_result = pd.read_csv(
    '/home/jyoun/Jason/Research/KIDS/manuscript_preparation/before_after_incon/find_path/result.txt',
    sep='\t',
    index_col=False,
    names=['index', 'node1', 'node2', 'node3', 'node4'])

# train data
pd_train = pd.read_csv(
    '/home/jyoun/Jason/Research/KIDS/manuscript_preparation/before_after_incon/find_path/train_original.txt',
    sep='\t',
    names=['Subject', 'Predicate', 'Object', 'Label'])

for column in ['Subject', 'Predicate', 'Object']:
    pd_train[column] = pd_train[column].apply(lambda x: x.replace('#SEMICOLON#', ':'))
    pd_train[column] = pd_train[column].apply(lambda x: x.replace('#SPACE#', ' '))
    pd_train[column] = pd_train[column].apply(lambda x: x.replace('#COMMA#', ','))

# entities
pd_entities = pd.read_csv(
    '/home/jyoun/Jason/Research/KIDS/kg_constructor/output/entity_full_names.txt',
    sep='\t',
    names=['entity_full_name'])

entities = pd_entities['entity_full_name'].tolist()

molecular_function = [entity.split(':')[-1] for entity in entities if 'concept:molecular_function' in entity]
biological_process = [entity.split(':')[-1] for entity in entities if 'concept:biological_process' in entity]
antibiotic = [entity.split(':')[-1] for entity in entities if 'concept:antibiotic' in entity]
cellular_component = [entity.split(':')[-1] for entity in entities if 'concept:cellular_component' in entity]

# scores
pd_original_score = pd.read_csv(
    '/home/jyoun/Jason/Research/KIDS/manuscript_preparation/before_after_incon/compare_pra/final_original/instance/test/scores/confers#SPACE#resistance#SPACE#to#SPACE#antibiotic',
    sep='\t',
    names=['score', 'path_exists'])

pd_without_resolved_score = pd.read_csv(
    '/home/jyoun/Jason/Research/KIDS/manuscript_preparation/before_after_incon/compare_pra/final_without_resolved/instance/test/scores/confers#SPACE#resistance#SPACE#to#SPACE#antibiotic',
    sep='\t',
    names=['score', 'path_exists'])

# reason
reason = []
with open('/home/jyoun/Jason/Research/KIDS/manuscript_preparation/before_after_incon/compare_pra/final_original/instance/test/predictions/confers#SPACE#resistance#SPACE#to#SPACE#antibiotic.reasons') as file:
    for line in file:
        line = line.replace('#SPACE#', ' ')
        line = line.replace('#SEMICOLON#', ':')
        line = line.replace('#COMMA#', ',')

        if line == '\n':
            continue

        line = line.replace('\n', '')
        if '\t' not in line:
            sub = line.split('$')[-1]
        else:
            items = line.split('\t')
            obj = items[1].split('$')[-1]
            models = sorted([item.split('=')[0] for item in items[2:]])
            models = ', '.join(models)

            reason.append([sub, obj, models])

pd_reason = pd.DataFrame(reason, columns=['Subject', 'Object', 'models'])

###########
# process #
###########

def check_model(node1, node2, node3, node4):
    index = (pd_train['Subject'] == node1)
    index &= (pd_train['Object'] == node2)
    selected = pd_train[index]
    predicates = selected['Predicate'].tolist()

    if not (('has' in predicates) or ('is involved in' in predicates) or ('upregulated by antibiotic' in predicates)):
        return ''

    if node2 in molecular_function:
        return '0'
    elif node2 in biological_process:
        return '1'
    elif node2 in antibiotic:
        index = (pd_train['Subject'] == node3)
        index &= (pd_train['Object'] == node2)
        selected = pd_train[index]

        predicates = selected['Predicate'].tolist()

        model_str = []

        if 'upregulated by antibiotic' in predicates:
            index = (pd_train['Subject'] == node3)
            index &= (pd_train['Object'] == node4)
            selected = pd_train[index]

            if selected['Predicate'].tolist() == 'confers resistance to antibiotic':
                model_str.append('3')

        if 'confers resistance to antibiotic' in predicates:
            index = (pd_train['Subject'] == node3)
            index &= (pd_train['Object'] == node4)
            selected = pd_train[index]

            if 'upregulated by antibiotic' in selected['Predicate'].tolist():
                model_str.append('4')

            if 'confers resistance to antibiotic' in selected['Predicate'].tolist():
                model_str.append('5')

        return ' '.join(model_str)


pd_result['model'] = ''
pd_result['model'] = pd_result.apply(lambda row: check_model(row['node1'], row['node2'], row['node3'], row['node4']), axis=1)
pd_result = pd_result[pd_result['model'] != '']

subject_dict = pd_result.groupby('index')['node1'].apply(lambda x: list(set(x))[0]).to_dict()
object_dict = pd_result.groupby('index')['node4'].apply(lambda x: list(set(x))[0]).to_dict()

pd_result_models = pd_result.groupby('index')['model'].apply(lambda x: ', '.join(sorted(list(set(x)))))
pd_result_models = pd_result_models.reset_index()

pd_result_models['Subject'] = pd_result_models['index'].apply(lambda x: subject_dict[x])
pd_result_models['Object'] = pd_result_models['index'].apply(lambda x: object_dict[x])

chosen_index = []

for index, row in pd_result_models.iterrows():
    select_index = (pd_reason['Subject'] == row['Subject'])
    select_index &= (pd_reason['Object'] == row['Object'])

    selected = pd_reason[select_index]

    all_models = selected['models'].tolist()[0].split(', ')
    result_models = row['model'].split(', ')

    flag = False
    for model in result_models:
        if model in all_models:
            flag = True

    if flag:
        chosen_index.append(row['index'])

pd_original_score_selected = pd_original_score.iloc[chosen_index]['score']
pd_without_resolved_score_selected = pd_without_resolved_score.iloc[chosen_index]['score']

percent_increase = pd_original_score_selected.sub(pd_without_resolved_score_selected)
percent_increase = percent_increase.divide(pd_without_resolved_score_selected)
percent_increase = percent_increase.multiply(100)
percent_increase = percent_increase.replace([np.inf, -np.inf], np.nan)

percent_increase.to_csv('percent_increase.txt', index=False)

print(percent_increase.tail(50))
print()
print(percent_increase.mean())
print(percent_increase.std())
print()


pd_original_score_selected = pd_original_score.drop(chosen_index)['score']
pd_without_resolved_score_selected = pd_without_resolved_score.drop(chosen_index)['score']

percent_increase = pd_original_score_selected.sub(pd_without_resolved_score_selected)
percent_increase = percent_increase.divide(pd_without_resolved_score_selected)
percent_increase = percent_increase.multiply(100)
percent_increase = percent_increase.replace([np.inf, -np.inf], np.nan)

print(percent_increase.mean())
print(percent_increase.std())
