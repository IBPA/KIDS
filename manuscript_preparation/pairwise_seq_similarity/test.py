import itertools
import pandas as pd
from Bio import pairwise2
from multiprocessing import Pool
from tqdm import tqdm
import time


novel_args_lookup = {
    'NC_000913.3:932595-933089 Escherichia coli str. K-12 substr. MG1655, complete genome': 'lrp',
    'NC_000913.3:3937294-3938223 Escherichia coli str. K-12 substr. MG1655, complete genome': 'rbsK',
    'NC_000913.3:c4434024-4433164 Escherichia coli str. K-12 substr. MG1655, complete genome': 'qorB',
    'NC_000913.3:c3947967-3947128 Escherichia coli str. K-12 substr. MG1655, complete genome': 'hdfR',
    'NC_000913.3:c3162669-3161257 Escherichia coli str. K-12 substr. MG1655, complete genome': 'ftsP',
    'NC_000913.3:2804815-2806017 Escherichia coli str. K-12 substr. MG1655, complete genome': 'proV',
}
ecoli_genes_seq_dict = {}
card_args_seq_dict = {}


def _calculate_global_alignment(gene_pair):
    score = pairwise2.align.globalxx(
        ecoli_genes_seq_dict[gene_pair[0]],
        card_args_seq_dict[gene_pair[1]],
        score_only=True)
    return [gene_pair[0], gene_pair[1], score]


def _calculate_local_alignment(gene_pair):
    score = pairwise2.align.localxx(
        ecoli_genes_seq_dict[gene_pair[0]],
        card_args_seq_dict[gene_pair[1]],
        score_only=True)
    return [gene_pair[0], gene_pair[1], score]


def main():
    # read e.coli genes
    global ecoli_genes_seq_dict
    with open('./ecoli_genes.txt', mode='r', encoding='utf-8') as f:
        for row in f:
            if row.startswith('>'):
                key = row[1:-1]
                ecoli_genes_seq_dict[key] = ''
            else:
                ecoli_genes_seq_dict[key] += row[:-1]

    to_delete = []
    for k, v in ecoli_genes_seq_dict.items():
        if v == 'NIL':
            to_delete.append(k)

    for k in to_delete:
        ecoli_genes_seq_dict.pop(k, None)

    # read CARD ARGs
    global card_args_seq_dict
    with open('./card_args.txt', mode='r', encoding='utf-8') as f:
        for row in f:
            if row.startswith('>'):
                key = row[1:-1].strip()
                card_args_seq_dict[key] = ''
            else:
                card_args_seq_dict[key] += row[:-1]

    # # global alignment
    # with open('./output/global_scores.txt', mode='w', encoding='utf-8') as file:
    #     # header
    #     file.write('ecoli_gene\tbest_card_arg\tbest_score\n')

    #     start = time.time()
    #     count = 1
    #     for ecoli_gene_name, _ in ecoli_genes_seq_dict.items():
    #         # logging
    #         time_elapsed = (time.time() - start) / 60
    #         estimated_time_left = (len(ecoli_genes_seq_dict) / count) * time_elapsed
    #         print(
    #             f'Processing {count}/{len(ecoli_genes_seq_dict)}. '
    #             f'{time_elapsed:.1f} minutes elapsed. {estimated_time_left:.1f} minutes left.')

    #         # multiprocessing to calculate score
    #         pairs = list(itertools.product(
    #             [ecoli_gene_name],
    #             list(card_args_seq_dict.keys())
    #         ))

    #         result = []
    #         with Pool(20) as p:
    #             for r in list(p.imap(_calculate_global_alignment, pairs, chunksize=50)):
    #                 result.append(r)

    #         # process the results and save
    #         df_temp = pd.DataFrame(result, columns=['ecoli', 'card', 'score'])
    #         df_temp.sort_values(by='score', ascending=False, inplace=True, ignore_index=True)
    #         best_card_arg = df_temp.iloc[0]['card']
    #         best_score = df_temp.iloc[0]['score']
    #         file.write(f'{ecoli_gene_name}\t{best_card_arg}\t{best_score}\n')

    #         count += 1

    df_global_scores = pd.read_csv('./output/global_scores.txt', sep='\t')
    df_global_scores.sort_values(by='best_score', ascending=False, inplace=True, ignore_index=True)
    df_global_scores['rank'] = df_global_scores.index + 1
    print(df_global_scores)

    novel_args = list(novel_args_lookup.values())
    df_match = df_global_scores[[x in novel_args for x in df_global_scores['ecoli_gene']]]
    print(df_match)
    print('-----------------')


    # # local alignment
    # with open('./output/local_scores.txt', mode='w', encoding='utf-8') as file:
    #     # header
    #     file.write('ecoli_gene\tbest_card_arg\tbest_score\n')

    #     start = time.time()
    #     count = 1
    #     for ecoli_gene_name, _ in ecoli_genes_seq_dict.items():
    #         # logging
    #         time_elapsed = (time.time() - start) / 60
    #         estimated_time_left = (len(ecoli_genes_seq_dict) / count) * time_elapsed
    #         print(
    #             f'Processing {count}/{len(ecoli_genes_seq_dict)}. '
    #             f'{time_elapsed:.1f} minutes elapsed. {estimated_time_left:.1f} minutes left.')

    #         # multiprocessing to calculate score
    #         pairs = list(itertools.product(
    #             [ecoli_gene_name],
    #             list(card_args_seq_dict.keys())
    #         ))

    #         result = []
    #         with Pool(20) as p:
    #             for r in list(p.imap(_calculate_local_alignment, pairs, chunksize=50)):
    #                 result.append(r)

    #         # process the results and save
    #         df_temp = pd.DataFrame(result, columns=['ecoli', 'card', 'score'])
    #         df_temp.sort_values(by='score', ascending=False, inplace=True, ignore_index=True)
    #         best_card_arg = df_temp.iloc[0]['card']
    #         best_score = df_temp.iloc[0]['score']
    #         file.write(f'{ecoli_gene_name}\t{best_card_arg}\t{best_score}\n')

    #         count += 1

    # df_local_scores = pd.read_csv('./output/local_scores.txt', sep='\t')
    # df_local_scores.sort_values(by='best_score', ascending=False, inplace=True, ignore_index=True)
    # df_local_scores['rank'] = df_local_scores.index + 1
    # print(df_local_scores)

    # novel_args = list(novel_args_lookup.values())
    # df_match = df_local_scores[[x in novel_args for x in df_local_scores['ecoli_gene']]]
    # print(df_match)


if __name__ == '__main__':
    main()
