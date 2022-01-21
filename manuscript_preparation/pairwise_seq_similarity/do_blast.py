import itertools
import pandas as pd
from Bio.Blast.Applications import NcbiblastnCommandline
from multiprocessing import Pool
from tqdm import tqdm
import time
import sys


novel_args_lookup = {
    'NC_000913.3:932595-933089 Escherichia coli str. K-12 substr. MG1655, complete genome': 'lrp',
    'NC_000913.3:3937294-3938223 Escherichia coli str. K-12 substr. MG1655, complete genome': 'rbsK',
    'NC_000913.3:c4434024-4433164 Escherichia coli str. K-12 substr. MG1655, complete genome': 'qorB',
    'NC_000913.3:c3947967-3947128 Escherichia coli str. K-12 substr. MG1655, complete genome': 'hdfR',
    'NC_000913.3:c3162669-3161257 Escherichia coli str. K-12 substr. MG1655, complete genome': 'ftsP',
    'NC_000913.3:2804815-2806017 Escherichia coli str. K-12 substr. MG1655, complete genome': 'proV',
}
ecoli_genes_seq_dict = {}
no_args_seq_dict = {}
novel_args_seq_dict = {}
card_args_seq_dict = {}


def _calculate_global_alignment(gene_pair):
    score = pairwise2.align.globalxx(
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

    # no ARGs
    global no_args_seq_dict
    no_args = pd.read_csv('./no_args_validated.txt', names=['genes'])['genes'].tolist()
    no_args_seq_dict = {x: ecoli_genes_seq_dict[x] for x in no_args}

    # read novel ARGs
    global novel_args_seq_dict
    with open('./novel_args.txt', mode='r', encoding='utf-8') as f:
        for row in f:
            if row.startswith('>'):
                key = row[1:-1].strip()
                novel_args_seq_dict[key] = ''
            else:
                novel_args_seq_dict[key] += row[:-1]

    # read CARD ARGs
    global card_args_seq_dict
    with open('./card_args.txt', mode='r', encoding='utf-8') as f:
        for row in f:
            if row.startswith('>'):
                key = row[1:-1].strip()
                card_args_seq_dict[key] = ''
            else:
                card_args_seq_dict[key] += row[:-1]


if __name__ == '__main__':
    main()
