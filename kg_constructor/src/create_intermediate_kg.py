import argparse
import logging as log
from multiprocessing import cpu_count
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modules.constructor_manager import ConstructorManager  # noqa: E402
from modules.resolution_manager import ResolutionManager  # noqa: E402
from utils.logger import set_logging  # noqa: E402

# default variables
DEFAULT_DATA_DIR = '../data'
DEFAULT_OUTPUT_DIR = '../output'
DEFAULT_DATASET = 'ecoli'
DEFAULT_RESOLUTION_METHOD = 'averagelog'
DEFAULT_KG_WITH_INCONSISTENCIES_FILENAME = 'kg_with_inconsistencies.txt'
DEFAULT_RESOLVED_INCONSISTENCIES_FILENAME = 'resolved_inconsistencies.txt'
DEFAULT_KG_WITHOUT_INCONSISTENCIES_FILENAME = 'kg_without_inconsistencies.txt'
DEFAULT_INTEGRATION_SUMMARY_FILENAME = 'integration_summary.pdf'

RESOLUTION_METHODS = [
    'averagelog',
    'investment',
    'pooledinvestment',
    'sums',
    'truthfinder',
    'voting',
]


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Construct knowledge graph using input data.')

    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET,
        help=f'Dataset to process. (Default: {DEFAULT_DATASET})',
    )

    parser.add_argument(
        '--skip_mapping',
        action='store_true',
        help='Set if name mapping should be skipped.',
    )

    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='Set if inference should be skipped.',
    )

    parser.add_argument(
        '--skip_remove',
        action='store_true',
        help='Set if removing should be skipped.',
    )

    parser.add_argument(
        '--resolution_method',
        type=str,
        default=DEFAULT_RESOLUTION_METHOD,
        help=f'Set inconsistency resolution method. (Default: {DEFAULT_RESOLUTION_METHOD}).',
    )

    parser.add_argument(
        '--n_workers',
        type=int,
        default=cpu_count() - 1,
        help=f'Number of cores for parallel processing. (Default: {cpu_count() - 1})',
    )

    parser.add_argument(
        '--kg_with_inconsistencies_filename',
        type=str,
        default=DEFAULT_KG_WITH_INCONSISTENCIES_FILENAME,
        help=f'Filename of the knowledge graph with inconsistencies. '
             f'(Default: {DEFAULT_KG_WITH_INCONSISTENCIES_FILENAME})',
    )

    parser.add_argument(
        '--resolved_inconsistencies_filename',
        type=str,
        default=DEFAULT_RESOLVED_INCONSISTENCIES_FILENAME,
        help=f'Filename of the resolved inconsistencies. '
             f'(Default: {DEFAULT_RESOLVED_INCONSISTENCIES_FILENAME})',
    )

    parser.add_argument(
        '--kg_without_inconsistencies_filename',
        type=str,
        default=DEFAULT_KG_WITHOUT_INCONSISTENCIES_FILENAME,
        help=f'Filename of the knowledge graph without inconsistencies. '
             f'(Default: {DEFAULT_KG_WITHOUT_INCONSISTENCIES_FILENAME})',
    )

    parser.add_argument(
        '--integration_summary_filename',
        type=str,
        default=DEFAULT_INTEGRATION_SUMMARY_FILENAME,
        help=f'Filename of the integration summary plot. '
             f'(Default: {DEFAULT_INTEGRATION_SUMMARY_FILENAME})',
    )

    args = parser.parse_args()

    # process directories
    args.data_dir = os.path.join(DEFAULT_DATA_DIR, args.dataset)
    assert Path(args.data_dir).exists()

    args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.dataset)
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # sanity check
    if args.resolution_method not in RESOLUTION_METHODS:
        raise ValueError(f'Invalid resolution method: {args.resolution_method}')

    return args


def plot_integration_summary(
        df,
        np_trustworthiness_vector,
        inconsistencies,
        summary_plot_filepath):
    """
    Plot summary of data integration where x-axis is sources
    and y-axis is number of triplets in each source.

    Inputs:
        df: (pd.DataFrame) Integrated data inlcuding the inconsistencies.
        np_trustworthiness_vector: (np.matrix) Trustworthiness of the sources.
        inconsistencies: (dict) Dictionary containing inconsistency_id as key
                and list of inconsistent triplets + source as value.
        summary_plot_filepath: (str) Filepath to save the integration summary plot.
    """
    spo_list = ['Subject', 'Predicate', 'Object']

    df_data_copy = df.copy()
    df_grouped_data = df_data_copy.groupby(spo_list)['Source'].apply(set)
    sources = df.groupby('Source').size().index.tolist()

    list_trustworthiness_vector = \
        [float(trustworthiness) for trustworthiness in np_trustworthiness_vector]
    df_trustworthiness = pd.Series(list_trustworthiness_vector, index=sources)

    df_data_stat_column_names = ['Single source', 'Multiple sources', 'Inconsistencies']
    df_data_stat = pd.DataFrame(index=sources, columns=df_data_stat_column_names)

    def get_num_inconsistencies_of_source(source, inconsistencies):
        num_inconsistencies = 0

        for inconsistent_tuples in inconsistencies.values():
            for _, sources in inconsistent_tuples:
                if source in sources:
                    num_inconsistencies += 1

        return num_inconsistencies

    for source in sources:
        num_of_tuples_with_one_source = sum(df_grouped_data == {source})
        num_of_rest = sum(df_data_copy['Source'] == source) - num_of_tuples_with_one_source
        df_data_stat.loc[source] = [
            num_of_tuples_with_one_source,
            num_of_rest,
            get_num_inconsistencies_of_source(source, inconsistencies)]

    # start the figure
    _, ax1 = plt.subplots()
    ax1.set_yscale("log")

    sorted_sources = df_trustworthiness.sort_values(ascending=False).index.tolist()
    df_data_stat = df_data_stat.loc[sorted_sources]

    log.debug(f'Inconsistencies stat:\n{df_data_stat}')
    log.debug(f'Trustworthiness:\n{df_trustworthiness}')

    x = np.arange(len(sources))
    i = -1
    dimw = 0.5 / len(df_data_stat_column_names)
    for column_name in df_data_stat_column_names:
        ax1.bar(x + i * dimw, df_data_stat[column_name], dimw, label=column_name, bottom=0.001)
        i += 1

    ax1.set_xlabel('Sources')
    ax1.set_ylabel('Number of triplets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_sources)

    ax1.legend()
    plt.savefig(summary_plot_filepath)
    plt.close()


def main() -> None:
    args = parse_argument()
    set_logging()

    # construct the intermediate knowledge graph
    cm = ConstructorManager(
        data_dir=args.data_dir,
        n_workers=args.n_workers
    )

    df = cm.construct_intermediate_kg(
        skip_mapping=args.skip_mapping,
        skip_inference=args.skip_inference,
        skip_remove=args.skip_remove,
    )

    kg_with_inconsistencies_filepath = os.path.join(
        args.output_dir, args.kg_with_inconsistencies_filename)

    log.info(f'Saving knowledge graph with inconsistencies before resolution to '
             f'\'{kg_with_inconsistencies_filepath}\'')

    df.to_csv(kg_with_inconsistencies_filepath, sep='\t', index=False)

    # detect and resolve inconsistencies
    rm = ResolutionManager(
        data_dir=args.data_dir,
        resolution_method=args.resolution_method,
    )

    inconsistencies, resolution_result = rm.detect_and_resolve(df)
    df_resolved_inconsistencies = resolution_result[0]
    df_without_inconsistencies = resolution_result[1]
    np_trustworthiness_vector = resolution_result[2]

    # save results
    resolved_inconsistencies_filepath = os.path.join(
        args.output_dir, args.resolved_inconsistencies_filename)
    log.info(f'Saving resolved inconsistencies to '
             f'\'{resolved_inconsistencies_filepath}\'')
    df_resolved_inconsistencies.to_csv(resolved_inconsistencies_filepath, index=False, sep='\t')

    kg_without_inconsistencies_filepath = os.path.join(
        args.output_dir, args.kg_without_inconsistencies_filename)
    log.info(f'Saving knowledge graph without inconsistencies to '
             f'\'{kg_without_inconsistencies_filepath}\'')
    df_without_inconsistencies.to_csv(kg_without_inconsistencies_filepath, index=False, sep='\t')

    # report summary of integration
    integration_summary_filepath = os.path.join(
        args.output_dir, args.integration_summary_filename)

    plot_integration_summary(
        df,
        np_trustworthiness_vector,
        inconsistencies,
        integration_summary_filepath
    )


if __name__ == '__main__':
    main()
