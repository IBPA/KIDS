"""
Filename: create_test_queries.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Create test queries based on the data file passes as input.

To-do:
"""
# standard imports
import argparse
import random


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='generate test queries')

    parser.add_argument(
        '--predicate',
        nargs='?',
        required=True,
        help='the predicate that we will get the scores for')

    parser.add_argument(
        '--data_file',
        metavar='dir',
        nargs='?',
        default='dev.txt',
        help='file path containing the data')

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
    relation = args.predicate
    data_file = args.data_file

    with open(data_file, "r") as _file:
        lines = _file.readlines()

    queries = {}
    queries_neg = {}

    for line in lines:
        columns = line.strip().split('\t')
        columns = [x.strip() for x in columns]

        if columns[1] == relation:
            subject = "c$" + columns[0]
            _object = "c$" + columns[2]

            if columns[3] == '1':  # positive queries
                if subject not in queries:
                    queries[subject] = set()
                    queries[subject].add(_object)
                else:
                    queries[subject].add(_object)
            else:  # negative queries
                if subject not in queries_neg:
                    queries_neg[subject] = set()
                    queries_neg[subject].add(_object)
                else:
                    queries_neg[subject].add(_object)

    with open(args.dir + "/queriesR_test/" + relation, "w") as _file:
        # positive queries
        for key, val in queries.items():
            o = random.sample(val, 1)[0]
            _file.write(key + '\t' + o)
            val.remove(o)

            for o in val:
                _file.write(' ' + o)

            _file.write('\t')

            if key in queries_neg:
                neg_v = queries_neg[key]
                o = random.sample(neg_v, 1)[0]
                _file.write(o)
                neg_v.remove(o)\

                for o in neg_v:
                    _file.write(' ' + o)

            _file.write('\n')

        # negative queries
        for key, val in queries_neg.items():
            if key not in queries:
                o = random.sample(val, 1)[0]
                _file.write(key + '\t\t' + o)
                val.remove(o)

                for o in val:
                    _file.write(' ' + o)

                _file.write('\n')


if __name__ == "__main__":
    main()
