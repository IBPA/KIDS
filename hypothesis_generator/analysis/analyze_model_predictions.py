"""
Filename: analyze_model_predictions.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Analyze model predictions.

To-do:
"""

# standard imports
import argparse
import logging as log
import os
import sys

# third party imports
import numpy as np
import pandas as pd

# local imports
DIRECTORY = os.path.dirname(__file__)
ABS_PATH_METRICS = os.path.join(DIRECTORY, '../utils')
sys.path.insert(0, ABS_PATH_METRICS)
from config_parser import ConfigParser
from utils import create_dir
from kids_log import set_logging

EDGES_COUNT_COLUMNS = ['fold',
                       'Antibiotic',
                       'genes_resist',
                       'test_edges',
                       'dev_edges',
                       'edges_in_train',
                       'confer_edges_in_train']

TEST_STATS_COLUMNS = EDGES_COUNT_COLUMNS + ['er_tp',
                                            'er_tn',
                                            'er_fp',
                                            'pra_tp',
                                            'pra_tn',
                                            'pra_fp',
                                            'stacked_tp',
                                            'stacked_tn',
                                            'stacked_fp',
                                            'er_precision',
                                            'pra_precision',
                                            'stacked_precision',
                                            'er_recall',
                                            'pra_recall',
                                            'stacked_recall',
                                            'er_f1',
                                            'pra_f1',
                                            'stacked_f1']

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze model predictions.')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='+',
        default='./',
        help='base directory')
    parser.add_argument(
        '--results_dir',
        metavar='dir',
        nargs='?',
        action='store',
        required=True,
        help='directory to store the model')
    return parser.parse_args()

def load_data_array(filepath, root):
    my_file = os.path.join(root, filepath)

    df = pd.read_csv(my_file, sep='\t', encoding='latin-1', header=None,
                     names=["subject", "predicate", "object", "label"])

    # filter out only the predicates that has antibiotic as its object
    df = df.loc[(df['predicate'] == "confers#SPACE#resistance#SPACE#to#SPACE#antibiotic") | \
                (df['predicate'] == "upregulated#SPACE#by#SPACE#antibiotic") | \
                (df['predicate'] == "targeted#SPACE#by")]
    df = df.loc[(df['label'] == 1)]

    return df.values

def load_confers_train_data_array(filepath, root):
    my_file = os.path.join(root, filepath)

    df = pd.read_csv(my_file, sep='\t', encoding='latin-1', header=None,
                     names=["subject", "predicate", "object", "label"])

    # filter out only the CRA predicate
    df = df.loc[(df['predicate'] == "confers#SPACE#resistance#SPACE#to#SPACE#antibiotic")]
    df = df.loc[df['label'] == 1]

    return df.values

def get_edges_dic(array):
    edges_dic = {}

    for row in array:
        if row[2] not in edges_dic:
            edges_dic[row[2]] = 0

        edges_dic[row[2]] += 1

    return edges_dic

def count_antibiotic_occurrence(antibiotic, edge_dic):
    if antibiotic in edge_dic:
        return edge_dic[antibiotic]
    else:
        return 0

def load_df(filepath, names):
    return pd.read_csv(filepath, sep='\t', encoding='latin-1', header=None, names=names)

def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)

def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)

def calculate_f1(recall, precision):
    if precision + recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)

def get_stats(total_test, antibiotic):
    er_tp = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_er_mlp == 1)].count()['object']
    stacked_tp = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_stacked == 1)].count()['object']
    pra_tp = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_pra == 1)].count()['object']

    er_fp = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_er_mlp == 1)].count()['object']
    stacked_fp = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_stacked == 1)].count()['object']
    pra_fp = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_pra == 1)].count()['object']

    er_tn = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_er_mlp == 0)].count()['object']
    stacked_tn = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_stacked == 0)].count()['object']
    pra_tn = total_test[(total_test.object == antibiotic) & (total_test.label == -1) & (total_test.classification_pra == 0)].count()['object']

    er_fn = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_er_mlp == 0)].count()['object']
    stacked_fn = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_stacked == 0)].count()['object']
    pra_fn = total_test[(total_test.object == antibiotic) & (total_test.label == 1) & (total_test.classification_pra == 0)].count()['object']

    er_precision = calculate_precision(er_tp, er_fp)
    stacked_precision = calculate_precision(stacked_tp, stacked_fp)
    pra_precision = calculate_precision(pra_tp, pra_fp)

    er_recall = calculate_recall(er_tp, er_fn)
    stacked_recall = calculate_recall(stacked_tp, stacked_fn)
    pra_recall = calculate_recall(pra_tp, pra_fn)

    er_f1 = calculate_f1(er_recall, er_precision)
    stacked_f1 = calculate_f1(stacked_recall, stacked_recall)
    pra_f1 = calculate_f1(pra_recall, pra_precision)

    return er_tp, stacked_tp, pra_tp, er_fp, stacked_fp, pra_fp, \
        er_tn, stacked_tn, pra_tn, er_fn, stacked_fn, pra_fn, er_precision, stacked_precision, \
        pra_precision, er_recall, stacked_recall, pra_recall, er_f1, stacked_f1, pra_f1

def main():
    """
    Main function.
    """
    # parse args and set logging
    args = parse_argument()
    set_logging()

    # variables
    root_dir = os.path.abspath(os.path.join(DIRECTORY, '..'))

    # create results directory
    create_dir(args.results_dir)

    pd_edges_count = pd.DataFrame(columns=EDGES_COUNT_COLUMNS)
    pd_test_stats = pd.DataFrame(columns=TEST_STATS_COLUMNS)

    for fold in args.dir:
        log.info('Processing %s...', fold)

        if fold == 'final':
            is_final = True
        else:
            is_final = False

        config_file = '{}/run/configuration/{}/er_mlp.ini'.format(root_dir, fold)
        configparser = ConfigParser(config_file)

        data_path = configparser.getstr('DATA_PATH')

        if not is_final:
            data_array = load_data_array('data.txt', os.path.join(data_path, '..'))
        else:
            data_array = load_data_array('data.txt', '{}/../folds'.format(data_path))
        train_array = load_data_array('train.txt', data_path)
        train_confers_array = load_confers_train_data_array('train.txt', data_path)
        if not is_final:
            dev_array = load_data_array('dev.txt', data_path)
        test_array = load_data_array('test.txt', data_path)

        all_data_dic = get_edges_dic(data_array)
        train_edges_dic = get_edges_dic(train_array)
        train_confers_dic = get_edges_dic(train_confers_array)
        if not is_final:
            dev_edges_dic = get_edges_dic(dev_array)
        test_edges_dic = get_edges_dic(test_array)

        list_append = []
        for k, v in all_data_dic.items():
            train_count = count_antibiotic_occurrence(k, train_edges_dic)
            train_confers_count = count_antibiotic_occurrence(k, train_confers_dic)
            if not is_final:
                dev_count = count_antibiotic_occurrence(k, dev_edges_dic)
            else:
                dev_count = np.nan
            test_count = count_antibiotic_occurrence(k, test_edges_dic)

            list_append.append([fold, k, v, test_count, dev_count, train_count, train_confers_count])

        pd_edges_count = pd_edges_count.append(pd.DataFrame(list_append, columns=pd_edges_count.columns))

        if is_final:
            log.info('Skipping test statistics for final model...')
            continue

        test_file = os.path.join(data_path, "test.txt")
        er_mlp_classifications = '{}/er_mlp/model/model_instance/{}/test/classifications_er_mlp.txt'.format(root_dir, fold)
        stacked_classifications = '{}/stacked/model_instance/{}/test/classifications_stacked.txt'.format(root_dir, fold)
        pra_classifications = '{}/pra/model/model_instance/{}/instance/test/classifications_pra.txt'.format(root_dir, fold)

        test_df = load_df(filepath=test_file, names=["subject", "predicate", "object", "label"])
        er_df = load_df(filepath=er_mlp_classifications, names=["classification_er_mlp"])
        st_df = load_df(filepath=stacked_classifications, names=["classification_stacked"])
        pra_df = load_df(filepath=pra_classifications, names=["classification_pra"])

        total_test = pd.concat([test_df, er_df, st_df, pra_df], axis=1)
        test_df = test_df.loc[(test_df['label'] == 1)]
        test_antibiotic = set(test_df.object.tolist())

        list_append = []
        for antibiotic in test_antibiotic:
            get_stats_return = get_stats(total_test, antibiotic)
            er_tp, stacked_tp, pra_tp = get_stats_return[0:3]
            er_fp, stacked_fp, pra_fp = get_stats_return[3:6]
            er_tn, stacked_tn, pra_tn = get_stats_return[6:9]
            er_fn, stacked_fn, pra_fn = get_stats_return[9:12]
            er_precision, stacked_precision, pra_precision = get_stats_return[12:15]
            er_recall, stacked_recall, pra_recall = get_stats_return[15:18]
            er_f1, stacked_f1, pra_f1 = get_stats_return[18:21]

            genes_resist_count = count_antibiotic_occurrence(antibiotic, all_data_dic)
            test_count = count_antibiotic_occurrence(antibiotic, test_edges_dic)
            dev_count = count_antibiotic_occurrence(antibiotic, dev_edges_dic)
            train_count = count_antibiotic_occurrence(antibiotic, train_edges_dic)
            train_confers_count = count_antibiotic_occurrence(antibiotic, train_confers_dic)

            list_append.append(
                [fold, antibiotic, genes_resist_count, test_count,
                 dev_count, train_count, train_confers_count,
                 er_tp, er_tn, er_fp,
                 pra_tp, pra_tn, pra_fp,
                 stacked_tp, stacked_tn, stacked_fp,
                 er_precision, pra_precision, stacked_precision,
                 er_recall, pra_recall, stacked_recall,
                 er_f1, pra_f1, stacked_f1])

        pd_test_stats = pd_test_stats.append(pd.DataFrame(list_append, columns=pd_test_stats.columns))

    # save to file
    pd_edges_count.to_csv(os.path.join(args.results_dir, 'edges_count.txt'), sep='\t', index=False)
    pd_test_stats.to_csv(os.path.join(args.results_dir, 'test_stats.txt'), sep='\t', index=False)

if __name__ == '__main__':
    main()
