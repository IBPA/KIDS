"""
Filename: merge_queries.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Merge queries.

To-do:
"""
# standard imports
import argparse
import os

# third party imports
import pandas as pd


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='evaluate the results')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        default='./',
        help='base directory')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_argument()

    with open('./selected_relations') as _file:
        relations = _file.readlines()

    relations = [x.strip() for x in relations]
    _relations = []

    for relation in relations:
        _relations.append('_' + relation)

    relations += _relations

    for relation in relations:
        pos_df = pd.read_csv(args.dir + '/queriesR_train/' + relation, sep='\t', encoding='latin-1', names=["subject", "object"])
        if not os.path.isfile(args.dir + '/queriesR_train_neg/' + relation):
            continue

        neg_df = pd.read_csv(args.dir + '/queriesR_train_neg/' + relation, sep='\t', encoding='latin-1', names=["subject", "object"])
        result = pd.merge(pos_df, neg_df, how='left', on="subject")
        result.to_csv(args.dir + '/queriesR_train/' + relation, sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()
