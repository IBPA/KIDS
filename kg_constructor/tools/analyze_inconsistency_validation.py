"""
Filename: analyze_inconsistency_validation.py

Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:

To-do:
"""
# standard imports
import argparse
import logging as log
import os
import sys

ABS_PATH_METRICS = os.path.join(os.path.dirname(__file__), '../integrate_modules')
sys.path.insert(0, ABS_PATH_METRICS)

# third party imports
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

pd.options.mode.chained_assignment = None

# default file names
DEFAULT_VALIDATED_INCONSISTENCIES_TXT = 'validated_inconsistencies.txt'
DEFAULT_MAP_FILE = '../data/name_map.txt'
CRA_STR = 'confers resistance to antibiotic'
CNRA_STR = 'confers no resistance to antibiotic'


def set_logging():
    """
    Configure logging.
    """
    log.basicConfig(format='(%(levelname)s) %(filename)s: %(message)s', level=log.DEBUG)


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze inconsistency validation experimental results.')

    parser.add_argument(
        '--threshold_key',
        default=None,
        help='String of column name to use for thresholding')

    parser.add_argument(
        '--save_only_validated',
        default=False,
        action='store_true',
        help='Remove temporal data unless this option is set')

    return parser.parse_args()


def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0

    return tp / (tp + fn)


def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def calculate_f1(recall, precision):
    if precision + recall == 0:
        return 0

    return (2 * precision * recall) / (precision + recall)


def calculate_statistics(pd_data, threshold_key=None):
    # only look at the data that we validated
    pd_data = pd_data[pd_data.Match != '']

    pd_data.loc[:, 'Resolution label'] = 0
    idx = pd_data['Predicate'].str.match(CRA_STR)
    pd_data.loc[idx, 'Resolution label'] = 1

    pd_data.loc[:, 'Validation label'] = 0
    idx = pd_data['Validation'].str.match(CRA_STR)
    pd_data.loc[idx, 'Validation label'] = 1

    cm_result = confusion_matrix(pd_data['Validation label'], pd_data['Resolution label'])

    print('-----')
    print(cm_result[1, 1], cm_result[0, 1])
    print(cm_result[1, 0], cm_result[0, 0])
    print('-----')

    if threshold_key:
        threshold_param = pd_data[threshold_key]
        param_unique = np.sort(threshold_param.unique())

        param_list = []
        f1_list = []
        for param in param_unique:
            pd_pass = pd_data[pd_data[threshold_key] >= param]
            cm_result = confusion_matrix(pd_pass['Validation label'], pd_pass['Resolution label'])

            if cm_result.shape == (1, 1):
                log.warning('Confusion matrix output has size {} for {} {}.'
                            .format(cm_result.shape, threshold_key, param))
                continue

            tp = cm_result[1, 1]
            fp = cm_result[0, 1]
            fn = cm_result[1, 0]
            tn = cm_result[0, 0]

            print('-----')
            print(tp, fp)
            print(fn, tn)
            print('-----')

            recall = calculate_recall(tp, fn)
            precision = calculate_precision(tp, fp)
            f1 = calculate_f1(recall, precision)

            param_list.append(param)
            f1_list.append(f1)

            print(param, f1, precision, recall)

        print(param_list)
        print(f1_list)

        plt.figure()
        plt.plot(param_list, f1_list)


def main():
    """
    Main function.
    """
    # set log and parse args
    set_logging()
    args = parse_argument()

    pd_validated_inconsistencies = pd.read_csv(
            os.path.join('../output', DEFAULT_VALIDATED_INCONSISTENCIES_TXT),
            sep='\t',
            na_values=[],
            keep_default_na=False)

    calculate_statistics(pd_validated_inconsistencies, threshold_key=args.threshold_key)

    plt.show()


if __name__ == '__main__':
    main()
