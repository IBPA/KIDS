import pandas as pd

pd_val_incon = pd.read_csv('./validated_inconsistencies.txt', sep='\t')
pd_val_incon = pd_val_incon[['Subject', 'Predicate', 'Object']]

pd_val_incon['Label'] = -1
pos_index = pd_val_incon['Predicate'].str.match('confers resistance to antibiotic')
pd_val_incon.loc[pos_index, 'Label'] = 1
neg_index = pd_val_incon['Predicate'].str.match('confers no resistance to antibiotic')
pd_val_incon.loc[neg_index, 'Predicate'] = 'confers resistance to antibiotic'

for column in ['Subject', 'Predicate', 'Object']:
	pd_val_incon[column] = pd_val_incon[column].apply(lambda x: x.replace(':', '#SEMICOLON#'))
	pd_val_incon[column] = pd_val_incon[column].apply(lambda x: x.replace(' ', '#SPACE#'))
	pd_val_incon[column] = pd_val_incon[column].apply(lambda x: x.replace(',', '#COMMA#'))

pd_test = pd.read_csv('./final/train.txt', sep='\t', names=['Subject', 'Predicate', 'Object', 'Label'])

pd_merged = pd_test.append(pd_val_incon)
print(pd_merged.shape)

pd_duplicated = pd_merged[pd_merged.duplicated()]
print(pd_duplicated.shape)
print(pd_duplicated)
