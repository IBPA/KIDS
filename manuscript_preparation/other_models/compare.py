import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

names = ['Subject', 'Predicate', 'Object', 'Label', 'Probability']
subset = ['Subject', 'Predicate', 'Object']

pd_hypotheses = pd.read_csv('./7/hypotheses_confidence.txt', sep='\t', names=names)
pd_validated = pd.read_csv('./all_validated_hypothesis.txt', sep='\t').dropna()
pd_validated = pd_validated[['Subject', 'Predicate', 'Object', 'Resistance']]

pd_hypotheses = pd_hypotheses.drop(['Label'], axis=1)

pd_joined = pd_hypotheses.merge(pd_validated, how='left', left_on=subset, right_on=subset)
pd_joined = pd_joined.sort_values(by='Probability', ascending=False)
pd_joined = pd_joined.dropna(subset=['Resistance']).reset_index(drop=True)

pd_joined[pd_joined['Probability'] >= 0.1].to_csv(
    '~/Jason/UbuntuShare/temp.txt', sep='\t', index=False)

thresholds = [0.75, 0.5, 0.25, 0.0]
fop_result = {}

for threshold in thresholds:
    index = pd_joined['Probability'] >= threshold

    if threshold != 0.75:
        index &= pd_joined['Probability'] < (threshold + 0.25)

    pd_match = pd_joined[index]

    num_yes = pd_match['Resistance'].str.count('Yes').sum()
    total_num = pd_match.shape[0]
    print(num_yes, total_num, (num_yes / total_num) * 100)

print()

thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]
fop_result = {}

for threshold in thresholds:
    index = pd_joined['Probability'] >= threshold

    if threshold != 0.8:
        index &= pd_joined['Probability'] < (threshold + 0.20)

    pd_match = pd_joined[index]

    num_yes = pd_match['Resistance'].str.count('Yes').sum()
    total_num = pd_match.shape[0]
    print(num_yes, total_num, (num_yes / total_num) * 100)

print()

y_true = pd_joined['Resistance'].to_numpy()
y_scores = pd_joined['Probability'].to_numpy()
precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label='Yes')

mAP = auc(recall, precision)
print(mAP)

plt.figure()
plt.step(recall, precision, lw=2, where='mid')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
