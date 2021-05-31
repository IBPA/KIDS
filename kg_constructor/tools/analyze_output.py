"""
Filename: visualize.py

Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Visualize data in multiples ways as requested by the user.

To-do:
"""
# import from generic packages
import os
import warnings
import argparse
import logging as log
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import SpectralBiclustering

# default arguments
DEFAULT_DATA_DIR_STR = '../output'
DEFAULT_ENTITY_FULL_NAMES_FILENAME = 'entity_full_names.txt'
DEFAULT_KNOWLEDGE_GRAPH_FILENAME = 'kg_final.txt'

# drop these columns because it's not necessary for network training
COLUMN_NAMES_TO_DROP = ['Belief', 'Source size', 'Sources']

SORT_ANTIBIOTIC = [
    '5-Nitro-2-furaldehyde semicarbazone',
    'Furaltadone',
    'Amoxicillin',
    'Ampicillin',
    'Aztreonam',
    'Cefamandole',
    'Cefazolin',
    'Cefoxitin',
    'Cefuroxime',
    'Cefsulodin',
    'Ceftazidime',
    'Cephalothin',
    'Cerulenin',
    'Ciprofloxacin',
    'Levofloxacin',
    'Lomefloxacin',
    'Norfloxacin',
    'Ofloxacin',
    'Enoxacin',
    'Nalidixic acid',
    'Metronidazole',
    'Ornidazole',
    'Tinidazole',
    'Mitomycin C',
    'Nitrofurantoin',
    'Spectinomycin',
    'Streptonigrin',
    'Thiolactomycin',
    'Amikacin',
    'Kanamycin',
    'Tobramycin',
    'Neomycin',
    'Paromomycin',
    'Apramycin',
    'Dihydrostreptomycin',
    'Geneticin',
    'Gentamicins',
    'Sisomicin',
    'Streptomycin',
    'Azithromycin',
    'Clarythromycin',
    'Erythromycin',
    'Hygromycin B',
    'Josamycin',
    'Oleandomycin',
    'Spiramycin',
    'Troleandomycin',
    'Tylosin',
    'Novobiocin',
    'Nigericin',
    'Bicyclomycin',
    'Cefmetazole',
    'Cephradine',
    'Moxalactam',
    'Lincomycin',
    'Mecillinam',
    'Blasticidin S',
    'Capreomycin',
    'Carbenicillin',
    'Cloxacillin',
    'Nafcillin',
    'Oxacillin',
    'Penicillin G',
    'Phenethicillin',
    'Vancomycin',
    'Cefoperazone',
    'Piperacillin',
    'Bleomycin',
    'Dactinomycin',
    'Fosfomycin',
    'Chloramphenicol',
    'Oxycarboxin',
    'Radicicol',
    'Sulfachloropyridazine',
    'Sulfadiazine',
    'Sulfamethazine',
    'Sulfamethizole',
    'Sulfamethoxazole',
    'Sulfamonomethoxine',
    'Sulfanilamide',
    'Sulfathiazole',
    'Sulfisoxazole',
    'Triclosan',
    'Trimethoprim',
    'Chlortetracycline',
    'Doxycycline',
    'Doxycycline hyclate',
    'Minocycline',
    'Oxytetracycline',
    'Penimepicycline',
    'Rolitetracycline',
    'Tetracycline',
    'Rifampin',
    'Rifamycin SV',
    'Fusidic acid',
    'Puromycin',
    'CHIR090',
    'Phleomycin',
    'Polymyxin B',
    'Tunicamycin',
    'Hygromycin ',
    'Bacitracin',
    'Colistin',
    'Doxorubicin']

def set_logging():
    """
    Configure logging.
    """
    log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)

    # set logging level to WARNING for matplotlib
    logger = log.getLogger('matplotlib')
    logger.setLevel(log.WARNING)


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Visualize data.')

    parser.add_argument(
        '--data_dir',
        default=DEFAULT_DATA_DIR_STR,
        help='Path to the file data_path_file.txt')

    return parser.parse_args()


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


    pd_pivot_table = pd_pivot_table.reset_index()
    pd_pivot_table['Object'] = pd.Categorical(pd_pivot_table['Object'], SORT_ANTIBIOTIC)
    pd_pivot_table = pd_pivot_table.sort_values('Object')
    antibiotics = pd_pivot_table['Object'].to_numpy().tolist()
    pd_pivot_table = pd_pivot_table.set_index('Object')

    # build the figure instance with the desired height
    fig, ax = plt.subplots(figsize=(40, 15))

    # generate heatmap
    myColors = [(1, 0.6, 0.6), (1, 1, 1), (0, 0, 0.7)]
    cmap = LinearSegmentedColormap.from_list('custom', myColors, len(myColors))
    ax = sns.heatmap(pd_pivot_table, cmap=cmap, ax=ax)

    # x, y axis labels
    # ax.yaxis.set_label_coords(0.5, -0.04)
    # ax.xaxis.set_label_coords(-0.02, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('/home/jyoun/Jason/UbuntuShare/heatmap.png', dpi=600, transparent=True)


def main():
    """
    Main function.
    """
    # set log and parse args
    set_logging()
    args = parse_argument()

    # read full names dataframe
    pd_entities = pd.read_csv(
        os.path.join(args.data_dir, DEFAULT_ENTITY_FULL_NAMES_FILENAME),
        sep='\n',
        names=['Full name'])

    # split full names
    pd_entities = pd_entities['Full name'].str.split(':', n=2, expand=True)
    pd_entities.columns = ['Global', 'Type', 'Name']

    # get all genes and antibiotics in KG
    pd_genes = pd_entities[pd_entities['Type'].str.match('gene')]['Name'].reset_index(drop=True)
    pd_antibiotics = pd_entities[pd_entities['Type'].str.match('antibiotic')]['Name'].reset_index(drop=True)

    pd_genes.sort_values(inplace=True)
    pd_antibiotics.sort_values(inplace=True)

    log.debug('Number of genes: %d', pd_genes.shape[0])
    log.debug('Number of antibiotics: %d', pd_antibiotics.shape[0])

    # read knowledge graph and drop unnecessary columns
    pd_kg = pd.read_csv(os.path.join(args.data_dir, DEFAULT_KNOWLEDGE_GRAPH_FILENAME), sep='\t')
    pd_kg = pd_kg.drop(COLUMN_NAMES_TO_DROP, axis=1)

    pd_known_cra = pd_kg[pd_kg['Predicate'].str.contains('resistance to antibiotic')].reset_index(drop=True)
    if pd_known_cra.duplicated(keep='first').sum() != 0:
        log.warning('There are duplicates in the knowledge graph!')

    log.debug('Number of CRA triples: %d', pd_known_cra.shape[0])

    # save heatmap as matrix
    pd_pivot_table = save_cra_matrix(pd_known_cra, pd_genes, pd_antibiotics)

    plot_heatmap(pd_pivot_table)


if __name__ == '__main__':
    main()
