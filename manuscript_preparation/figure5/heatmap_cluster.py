# import from generic packages
import os
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import SpectralBiclustering


def save_cra_matrix(pd_known_cra, pd_genes, pd_antibiotics):
    pd_pivot_table = pd.pivot_table(
        pd_known_cra[['Subject', 'Predicate', 'Object']],
        index='Subject',
        columns='Object',
        values='Predicate',
        aggfunc='first')

    genes_not_in_cra_triples = list(set(pd_genes.values) - set(pd_pivot_table.index.values))
    antibiotics_not_in_cra_triples = list(set(pd_antibiotics.values) - set(pd_pivot_table.columns.values))

    if genes_not_in_cra_triples:
        pd_pivot_table = pd_pivot_table.reindex(index=pd_pivot_table.index.union(genes_not_in_cra_triples))

    if antibiotics_not_in_cra_triples:
        pd_pivot_table = pd_pivot_table.reindex(columns=pd_pivot_table.columns.union(antibiotics_not_in_cra_triples))

    # pd_pivot_table.to_excel('/home/jyoun/Jason/UbuntuShare/heatmap.xlsx')

    return pd_pivot_table


def plot_heatmap(pd_pivot_table):
    pd_pivot_table = pd_pivot_table.replace('confers resistance to antibiotic', 1)
    pd_pivot_table = pd_pivot_table.replace('confers no resistance to antibiotic', -1)
    pd_pivot_table = pd_pivot_table[pd_pivot_table.columns].astype(float)
    pd_pivot_table = pd_pivot_table.fillna(0)

    pd_pivot_table = pd_pivot_table.transpose()
    # pd_pivot_table = pd_pivot_table.reset_index()
    # pd_pivot_table = pd_pivot_table.set_index('Object')

    # genes = pd_pivot_table.pop("Object")
    # antibiotics = pd_pivot_table.pop("Object")

    # negative, unknown, positive
    myColors = [(1, 0.6, 0.6), (0.9, 0.9, 0.9), (0, 0, 0.7)]
    cmap = LinearSegmentedColormap.from_list('custom', myColors, len(myColors))
    g = sns.clustermap(pd_pivot_table, method="single", metric='hamming', cmap=cmap, xticklabels=False, yticklabels=1)
    ax = g.ax_heatmap

    ax.yaxis.set_tick_params(labelsize=5)
    # ax.get_xaxis().set_visible(False)

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.667, 0, 0.667])
    colorbar.set_ticklabels(['', '', ''])

    # X - Y axis labels
    ax.set_ylabel('Antibiotics')
    ax.set_xlabel('Genes')

    plt.savefig('output.png', dpi=300, bbox_inches='tight', pad_inches=0)

    # plt.show()



def main():
    """
    Main function.
    """
	# read full names dataframe
    pd_entities = pd.read_csv('./entity_full_names.txt', sep='\n', names=['Full name'])

    # split full names
    pd_entities = pd_entities['Full name'].str.split(':', n=2, expand=True)
    pd_entities.columns = ['Global', 'Type', 'Name']

    # get all genes and antibiotics in KG
    pd_genes = pd_entities[pd_entities['Type'].str.match('gene')]['Name'].reset_index(drop=True)
    pd_antibiotics = pd_entities[pd_entities['Type'].str.match('antibiotic')]['Name'].reset_index(drop=True)

    pd_genes.sort_values(inplace=True)
    pd_antibiotics.sort_values(inplace=True)

    print('Number of genes: ', pd_genes.shape[0])
    print('Number of antibiotics: ', pd_antibiotics.shape[0])

    # read knowledge graph and drop unnecessary columns
    pd_kg = pd.read_csv('./kg_final.txt', sep='\t')
    pd_kg = pd_kg.drop(['Belief', 'Source size', 'Sources'], axis=1)

    pd_known_cra = pd_kg[pd_kg['Predicate'].str.contains('resistance to antibiotic')].reset_index(drop=True)
    if pd_known_cra.duplicated(keep='first').sum() != 0:
        print('There are duplicates in the knowledge graph!')
        sys.exit()

    print('Number of CRA triples: ', pd_known_cra.shape[0])

    pd_known_pos_cra = pd_known_cra[~pd_known_cra['Predicate'].str.contains('confers no resistance')]
    pd_known_neg_cra = pd_known_cra[pd_known_cra['Predicate'].str.contains('confers no resistance')]
    print(pd_known_pos_cra)
    print(pd_known_neg_cra)
    sys.exit()

    # save heatmap as matrix
    pd_pivot_table = save_cra_matrix(pd_known_cra, pd_genes, pd_antibiotics)

    plot_heatmap(pd_pivot_table)


if __name__ == '__main__':
    main()
