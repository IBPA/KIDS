import argparse
from glob import glob
import os
import sys
sys.path.append("..")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import load_pkl  # noqa: E402

# default variables
OUTPUT_DIR = '../../output'
DEFAULT_DATASET = 'ecoli'
DEFAULT_BASED_ON = 'f1'
MODES = ['find_best_hyperparameter', 'get_test_stats']


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Find the best hyper-parameter.')

    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET,
        help=f'Dataset to process. (Default: {DEFAULT_DATASET})',
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help='Choose mode from ' + '|'.join(MODES) + '.',
    )

    parser.add_argument(
        '--based_on',
        type=str,
        default=DEFAULT_BASED_ON,
        help=f'What metric to use for finding the best hyper-parameter. '
             f'(Default: {DEFAULT_BASED_ON})',
    )

    parser.add_argument(
        '--input_filename',
        type=str,
        default=None,
        help='Input filename(s).',
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_argument()

    if args.mode == 'find_best_hyperparameter':
        results_dir = os.path.join(OUTPUT_DIR, DEFAULT_DATASET, 'tucker')
        print(f'Finding results in directory: {results_dir}')

        files = glob(os.path.join(results_dir, args.input_filename))
        print(f'Found {len(files)} files in the directory.')

        all_results = {}
        for f in files:
            all_results.update(load_pkl(f))

        hp_score_list = []
        for k, v in all_results.items():
            # folds
            score = []
            for x in v:
                score.append(x[1][args.based_on])
            score = sum(score) / len(score)

            hp_score_list.append([k, score])

        df = pd.DataFrame(hp_score_list, columns=['hyperparameters', args.based_on])
        df.sort_values('f1', ascending=False, inplace=True)
        df.to_csv(
            os.path.join(results_dir, 'hyperparameters_results_sorted.txt'),
            sep='\t',
            index=False)

        print(f'Hyperparameters search result sorted by {args.based_on}:\n{df}')

    if args.mode == 'get_test_stats':
        results_dir = os.path.join(OUTPUT_DIR, DEFAULT_DATASET, 'tucker')
        print(f'Finding results in directory: {results_dir}')

        results = load_pkl(os.path.join(results_dir, args.input_filename))
        print(results)

        aggregated_results = {}
        for r in results:
            for k, v in r.items():
                if k not in aggregated_results:
                    aggregated_results[k] = [v]
                else:
                    aggregated_results[k].append(v)

        print(f'Aggregated results before taking average:\n{aggregated_results}')
        aggregated_results = {k: f'{np.mean(v):.3f}±{np.std(v):.3f}' for k, v in aggregated_results.items()}
        print(f'Aggregated results after taking average:\n{aggregated_results}')


if __name__ == '__main__':
    main()
