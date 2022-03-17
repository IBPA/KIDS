import argparse
import glob
import itertools
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, kendalltau


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_folder',
        type=str,
        default='./other_seeds',
        help='Folder containing hypotheses files with different random seeds.',
    )

    args = parser.parse_args()
    return args


def get_index(filename):
    names = ['sub', 'pred', 'obj', 'label', 'probability']
    df = pd.read_csv(filename, sep='\t', names=names)
    df = df[df['probability'] > 0.2]
    df.sort_values(by='probability', ascending=False, inplace=True, kind='mergesort')
    return df.index.tolist()


def get_ranks_and_index(filename):
    names = ['sub', 'pred', 'obj', 'label', 'probability']
    df = pd.read_csv(filename, sep='\t', names=names)
    df = df[df['probability'] > 0.20]
    df.sort_values(by='probability', ascending=False, inplace=True, kind='mergesort')
    df['rank'] = df['probability'].rank(method='dense', ascending=False)
    return df.index.tolist(), df['rank'].tolist()


def get_ranks_using_index(filename, index):
    names = ['sub', 'pred', 'obj', 'label', 'probability']
    df = pd.read_csv(filename, sep='\t', names=names)
    df.sort_values(by='probability', ascending=False, inplace=True, kind='mergesort')
    df['rank'] = df['probability'].rank(method='dense', ascending=False)
    df = df.loc[index, :]
    df.sort_values(by='probability', ascending=False, inplace=True, kind='mergesort')
    return df['rank'].tolist()


def get_triples(filename):
    names = ['sub', 'pred', 'obj', 'label', 'probability']
    df = pd.read_csv(filename, sep='\t', names=names)
    df = df[df['probability'] > 0.2]
    df['triple'] = df.apply(lambda row: ' '.join([row['sub'], row['pred'], row['obj']]), axis=1)
    return df['triple'].tolist()


def rbo(l1, l2, p=0.98):
    """
    https://github.com/ragrawal/measures/blob/master/measures/rankedlist/RBO.py
    """
    """
        Calculates Ranked Biased Overlap (RBO) score. 
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []
    
    sl,ll = sorted([(len(l1), l1),(len(l2),l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l 
    # (the longer of the two lists)
    ss = set([]) # contains elements from the smaller list till depth i
    ls = set([]) # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1
        
        # if two elements are same then 
        # we don't need to add to either of the set
        if x == y: 
            x_d[d] = x_d[d-1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else: 
            ls.add(x) 
            if y != None: ss.add(y)
            x_d[d] = x_d[d-1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)     
        #calculate average overlap
        sum1 += x_d[d]/d * pow(p, d)
        
    sum2 = 0.0
    for i in range(l-s):
        d = s+i+1
        sum2 += x_d[d]*(d-s)/(d*s)*pow(p,d)

    sum3 = ((x_d[l]-x_d[s])/l+x_d[s]/s)*pow(p,l)

    # Equation 32
    rbo_ext = (1-p)/p*(sum1+sum2)+sum3
    return rbo_ext


def main():
    args = parse_argument()

    files_without_original = glob.glob(os.path.join(args.input_folder, '*.txt'))
    files_with_original = files_without_original + ['./hypotheses_confidence.txt']
    print(f'Number of files without original found: {len(files_without_original)}')
    print(f'Number of files with original found: {len(files_with_original)}')

    # #######
    # # rbo #
    # #######
    # # our hypotheses
    # all_ranks = []
    # for idx, f in enumerate(files_with_original):
    #     print(f'Processing {idx+1}/{len(files_with_original)}...')
    #     all_ranks.append(get_index(f))

    # rbo_list = []
    # for rp in list(itertools.combinations(all_ranks, 2)):
    #     rbo_list.append(rbo(rp[0], rp[1], p=0.99))

    # print(f'RBO value: {np.mean(rbo_list)} +- {np.var(rbo_list)}')

    # # random baseline
    # all_ranks_baseline = []
    # for r in all_ranks:
    #     all_ranks_baseline.append(random.sample(range(108078), len(r)))

    # rbo_baseline_list = []
    # for rp in list(itertools.combinations(all_ranks_baseline, 2)):
    #     rbo_baseline_list.append(rbo(rp[0], rp[1], p=0.99))

    # print(f'Baseline RBO value: {np.mean(rbo_baseline_list)} +- {np.var(rbo_baseline_list)}')

    # _, pval = ttest_ind(rbo_list, rbo_baseline_list)
    # print(f'p-value: {pval}')

    # ##############
    # # kendalltau #
    # ##############
    # original_index, original_rank = get_ranks_and_index('./hypotheses_confidence.txt')

    # other_ranks = [original_rank]
    # for idx, f in enumerate(files_without_original):
    #     print(f'Processing {idx+1}/{len(files_without_original)}...')
    #     other_ranks.append(get_ranks_using_index(f, original_index))

    # taus_list = []
    # for rp in list(itertools.combinations(other_ranks, 2)):
    #     tau, pval = kendalltau(rp[0], rp[1])
    #     taus_list.append(tau)

    # print(f'tau: {np.mean(taus_list)} +- {np.var(taus_list)}')

    # # random baseline
    # other_ranks_baseline = []
    # for _ in other_ranks:
    #     other_ranks_baseline.append(random.sample(range(108078), len(original_rank)))

    # taus_baseline_list = []
    # for rp in list(itertools.combinations(other_ranks_baseline, 2)):
    #     tau, pval = kendalltau(rp[0], rp[1])
    #     taus_baseline_list.append(tau)

    # print(f'Baseline tau: {np.mean(taus_baseline_list)} +- {np.var(taus_baseline_list)}')

    # _, pval = ttest_ind(taus_list, taus_baseline_list)
    # print(f'p-value: {pval}')

    #####################
    # common hypotheses #
    #####################
    # validated hypotheses
    df_validated = pd.read_csv('../figure5/all_validated_hypothesis.txt', sep='\t')
    df_validated2 = pd.read_csv('../figure5/all_validated_hypothesis_cycle_2.txt', sep='\t')
    df_validated2 = df_validated2[['Subject', 'Predicate', 'Object', 'Resistance']]
    df_validated = pd.concat([df_validated, df_validated2], ignore_index=True)
    df_validated['triple'] = df_validated.apply(lambda row: ' '.join([row['Subject'], row['Predicate'], row['Object']]), axis=1)
    df_validated = df_validated[['triple', 'Resistance']]

    # common hypotheses
    all_triples = []
    for f in files_with_original:
        all_triples.extend(get_triples(f))

    all_triples_dict = {}
    for x in all_triples:
        if x in all_triples_dict:
            all_triples_dict[x] += 1
        else:
            all_triples_dict[x] = 1

    df_all_triples = pd.DataFrame.from_dict(all_triples_dict, orient='index', columns=['count'])
    df_all_triples.sort_values(by='count', ascending=False, inplace=True)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # bins = [0, 20, 40, 60, 80, 100]
    df_all_triples['binned'] = pd.cut(df_all_triples['count'], bins, include_lowest=True)
    df_all_triples['triple'] = df_all_triples.index
    df_all_triples.reset_index(drop=True, inplace=True)

    # merge results
    df_common_merged = df_all_triples.merge(df_validated, on=['triple'], how='left')
    df_common_merged['Predicate'] = 'confers resistance to antibiotic'
    df_common_merged['Subject'] = df_common_merged['triple'].apply(lambda x: x.split(' confers resistance to antibiotic ')[0])
    df_common_merged['Object'] = df_common_merged['triple'].apply(lambda x: x.split(' confers resistance to antibiotic ')[1])
    df_common_merged = df_common_merged[['Subject', 'Predicate', 'Object', 'Resistance', 'count', 'binned']]
    df_common_merged.to_csv('./df_merged.txt', sep='\t', index=False)

    df_all_grouped = df_all_triples.groupby(['binned']).size()
    print(df_all_grouped)


if __name__ == '__main__':
    main()
