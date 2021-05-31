"""
Filename: figures.py

Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Draw figures.

To-do:
"""
# standard imports
import argparse
import os
import sys

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# global variables
DEFAULT_OUTDIR_STR = '../output'
DEFAULT_CONFIDENCE_FILE_STR = 'hypotheses_confidence.txt'

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze hypotheses.')

    parser.add_argument(
        '--outdir',
        default=DEFAULT_OUTDIR_STR,
        help='Output directory.')

    return parser.parse_args()

def bin_hypotheses(pd_data, intervals=[0.0, 0.25, 0.5, 0.75, 1.0]):
    intervals_sorted = sorted(intervals)
    dict_bin_count = {}

    for i in range(len(intervals_sorted)-1):
        floor = intervals_sorted[i]
        ceil = intervals_sorted[i+1]
        bin_name = '[{}, {}{}'.format(str(floor), str(ceil), ']' if ceil == 1 else ')')

        index = pd_data['Confidence'] >= floor
        if ceil != 1:
            index &= pd_data['Confidence'] < ceil
        else:
            index &= pd_data['Confidence'] <= ceil

        dict_bin_count[bin_name] = pd_data[index].shape[0]

    print(dict_bin_count)

def which_to_test(pd_data):
    floor = 0.1
    ceil = 1.0

    index = pd_data['Confidence'] >= floor
    if ceil != 1.0:
        index &= pd_data['Confidence'] < ceil
    else:
        index &= pd_data['Confidence'] <= ceil

    pd_filtered = pd_data[index]

    pd_group = pd_filtered.groupby('Object').size()
    pd_group = pd_group.sort_values(ascending=False).to_frame(name='count')
    pd_group['cum_sum'] = pd_group['count'].cumsum()
    pd_group['cum_percentage'] = (pd_group['cum_sum'] / pd_group['count'].sum()) * 100

    print(pd_group)

    pd_group.reset_index().to_csv('~/Jason/UbuntuShare/to_test.txt', sep='\t')

def main():
    """
    Main function.
    """
    args = parse_argument()

    filepath = os.path.join(args.outdir, DEFAULT_CONFIDENCE_FILE_STR)
    col_names = ['Subject', 'Predicate', 'Object', 'Label', 'Confidence']
    pd_data = pd.read_csv(filepath, sep='\t', names=col_names)

    bin_hypotheses(pd_data.copy())
    which_to_test(pd_data.copy())

    # plt.show()

if __name__ == '__main__':
    main()
