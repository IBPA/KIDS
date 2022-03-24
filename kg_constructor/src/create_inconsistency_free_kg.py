import argparse
import logging as log
import os
from pathlib import Path

import pandas as pd  # noqa: E402

from utils.logger import set_logging  # noqa: E402

# default variables
DEFAULT_DATA_DIR = '../data'
DEFAULT_OUTPUT_DIR = '../output'
DEFAULT_DATASET = 'ecoli'
DEFAULT_RESOLVED_INCONSISTENCIES_FILENAME = 'resolved_inconsistencies.txt'
DEFAULT_KG_WITHOUT_INCONSISTENCIES_FILENAME = 'kg_without_inconsistencies.txt'
DEFAULT_VALIDATION_RESULTS_FILENAME = 'inconsistency_validation_results.txt'
DEFAULT_VALIDATION_RESULTS_MERGED_FILENAME = 'validation_results_merged.txt'
DEFAULT_FINAL_KG_FILENAME = 'kg_final.txt'
DEFAULT_REINSTATE_MODE = 'validated_match_true_and_rest_as_positives'
DEFAULT_POS_NEG_RELATION_PAIRS_FILENAME = 'pos_neg_relation_pairs.txt'


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
        '--validation_results_filename',
        type=str,
        default=DEFAULT_VALIDATION_RESULTS_FILENAME,
        help=f'Filename of the validation results. '
             f'(Default: {DEFAULT_VALIDATION_RESULTS_FILENAME})',
    )

    parser.add_argument(
        '--save_only_intersection',
        action='store_true',
        help='Set if saving only the intersection of '
             '1) computationally resolved inconsisntencies and '
             '2) manually validated inconsistencies.'
    )

    parser.add_argument(
        '--validation_results_merged_filename',
        type=str,
        default=DEFAULT_VALIDATION_RESULTS_MERGED_FILENAME,
        help=f'Filename of the merged computational and manually validation results. '
             f'(Default: {DEFAULT_VALIDATION_RESULTS_MERGED_FILENAME})',
    )

    parser.add_argument(
        '--final_kg_filename',
        type=str,
        default=DEFAULT_FINAL_KG_FILENAME,
        help=f'Filename of the final knowledge graph. '
             f'(Default: {DEFAULT_FINAL_KG_FILENAME})',
    )

    parser.add_argument(
        '--reinstate_mode',
        type=str,
        default=DEFAULT_REINSTATE_MODE,
        help=f'Select mode for reinstating the validation results to the knowledge graph. '
             f'(Default: {DEFAULT_REINSTATE_MODE})',
    )

    parser.add_argument(
        '--pos_neg_relation_pairs_filename',
        type=str,
        default=DEFAULT_POS_NEG_RELATION_PAIRS_FILENAME,
        help=f'Name of the file containing pairs of positive and negative relations. '
             f'(Default: {DEFAULT_POS_NEG_RELATION_PAIRS_FILENAME})',
    )

    args = parser.parse_args()

    # process directories
    data_dir = os.path.join(DEFAULT_DATA_DIR, args.dataset)
    assert Path(data_dir).exists()
    validation_dir = os.path.join(DEFAULT_DATA_DIR, args.dataset, 'validation')
    assert Path(validation_dir).exists()
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.dataset)
    assert Path(output_dir).exists()

    # filename to filepath
    args.resolved_inconsistencies_filename = os.path.join(
        output_dir,
        args.resolved_inconsistencies_filename)

    args.kg_without_inconsistencies_filename = os.path.join(
        output_dir,
        args.kg_without_inconsistencies_filename)

    args.validation_results_filename = os.path.join(
        validation_dir,
        args.validation_results_filename)

    args.validation_results_merged_filename = os.path.join(
        output_dir,
        args.validation_results_merged_filename)

    args.final_kg_filename = os.path.join(
        output_dir,
        args.final_kg_filename)

    args.pos_neg_relation_pairs_filename = os.path.join(
        data_dir,
        args.pos_neg_relation_pairs_filename)

    return args


def compare_resolution_with_validation(
        resolved_inconsistencies_filepath,
        validation_results_filepath,
        save_only_intersection,
        save_to) -> pd.DataFrame:
    """
    Compare resolved inconsistencies with the validation results.

    Inputs:
        df_resolved: (pd.DataFrame) Resolved inconsistencies.
        df_validated: (pd.DataFrame) Validated inconsistencies.
        save_only_intersection: (bool) True if saving only the inconsistencies
            appearing in both input dataframes, False otherwise.

    Returns:
        df_resolved_and_validated: (pd.DataFrame) Comparison result.
            Columns 'Validation' and 'Match' are added to df_resolved.
    """
    df_resolved = pd.read_csv(resolved_inconsistencies_filepath, sep='\t')
    log.debug(f'Resolved inconsistencies:\n{df_resolved}')

    df_validated = pd.read_csv(validation_results_filepath, sep='\t')
    log.debug(f'Validation results:\n{df_validated}')

    # add new columns that will be used for validation
    df_resolved_and_validated = df_resolved.copy()
    df_resolved_and_validated['Validation'] = ''
    df_resolved_and_validated['Match'] = ''

    for _, row in df_validated.iterrows():
        validated_sub = row.Subject
        validated_pred = row.Predicate
        validated_obj = row.Object

        # find matching triple in df_resolved
        match = df_resolved_and_validated[df_resolved_and_validated.Subject == validated_sub]
        match = match[match.Object == validated_obj]

        if match.shape[0] > 1:
            # if there are more than one match, there is something wrong
            raise ValueError(
                f'Found {match.shape[0]} matching resolved inconsistencies '
                f'for ({validated_sub}, {validated_obj}).')
        elif match.shape[0] == 0:
            # no match, continue to next validation
            continue

        df_resolved_and_validated.loc[match.index, 'Validation'] = validated_pred

        # check if resolution and validation results match
        resolved_and_validated_match = df_resolved_and_validated.loc[
            match.index, 'Predicate'].str.contains(validated_pred).values[0]
        if resolved_and_validated_match:
            df_resolved_and_validated.loc[match.index, 'Match'] = 'True'
        else:
            df_resolved_and_validated.loc[match.index, 'Match'] = 'False'

    if save_only_intersection:
        df_resolved_and_validated = df_resolved_and_validated[df_resolved_and_validated.Match != '']

    # save resolution validation comparison result
    log.info(f'Saving merged inconsistencies to \'{save_to}\'')
    df_resolved_and_validated.to_csv(save_to, index=False, sep='\t')

    return df_resolved_and_validated


def reinstate_resolved_and_validated(
        kg_without_inconsistencies_filepath,
        df_resolved_and_validated,
        reinstate_mode,
        pos_neg_relation_pairs_filepath,
        save_to) -> None:
    """
    Insert resolved and validated inconsistencies back into the
    original data to create the final knowledge graph.

    Inputs:
        df_kg_without_inconsistencies: (pd.DataFrame) Data without inconsistencies.
        df_resolved_and_validated: (pd.DataFrame) Inconsistencies where
            some / all of them are validated.
        reinstate_mode: (str) Which resolved inconsistencies to reinstate ('all' | 'only_validated')

    Returns:
        df_final: (pd.DataFrame) Final knowledge graph.
    """
    final_kg_columns = ['Subject', 'Predicate', 'Object', 'Belief', 'Source size', 'Sources']

    # read the files saved in phase1
    df_kg_without_inconsistencies = pd.read_csv(kg_without_inconsistencies_filepath, sep='\t')
    log.debug(f'KG without inconsistencies:\n{df_kg_without_inconsistencies}')
    log.info(f'Number of data without inconsistencies: {df_kg_without_inconsistencies.shape}')
    log.info(f'Number of resolved inconsistencies: {df_resolved_and_validated.shape}')

    # select triples to append depending on the selected reinstate_mode
    if reinstate_mode == 'all':
        df_to_append = df_resolved_and_validated
    elif reinstate_mode == 'validated_match_true':
        df_to_append = df_resolved_and_validated[df_resolved_and_validated['Validation'] != '']
        df_to_append = df_to_append[df_to_append['Match'] == 'True']
    elif reinstate_mode == 'validated_match_true_and_rest_as_positives':
        df_to_append = df_resolved_and_validated[df_resolved_and_validated['Validation'] != '']
        df_to_append = df_to_append[df_to_append['Match'] == 'True']

        df_not_validated = df_resolved_and_validated[
            df_resolved_and_validated['Validation'] == ''].copy()

        if Path(pos_neg_relation_pairs_filepath).exists():
            df_pos_neg_pairs = pd.read_csv(pos_neg_relation_pairs_filepath, sep='\t')
        else:
            df_pos_neg_pairs = None

        def _force_pos(x):
            matching_pairs = df_pos_neg_pairs[(df_pos_neg_pairs == x).any(axis=1)]
            return matching_pairs['positive'].tolist()[0]

        df_not_validated['Predicate'] = df_not_validated['Predicate'].apply(lambda x: _force_pos(x))
        df_to_append = pd.concat([df_to_append, df_not_validated], ignore_index=True, sort=False)
    else:
        raise ValueError(f'Invalid reinstate mode \'{reinstate_mode}\'')

    log.info(f'Number of resolved inconsistencies to append: {df_to_append.shape[0]}')

    # append the selected triples
    df_to_append = df_to_append.loc[:, final_kg_columns]
    df_final = pd.concat(
        [df_kg_without_inconsistencies, df_to_append],
        ignore_index=True,
        sort=False
    )

    log.info(f'Number of triples in the final knowledge graph: {df_final.shape[0]}')

    # save final KG
    log.info(f'Saving final knowledge graph to \'{save_to}\'')
    df_final.to_csv(save_to, index=False, sep='\t')


def main() -> None:
    args = parse_argument()
    set_logging()

    # compare computational resolution results with validation results
    df_resolved_and_validated = compare_resolution_with_validation(
        resolved_inconsistencies_filepath=args.resolved_inconsistencies_filename,
        validation_results_filepath=args.validation_results_filename,
        save_only_intersection=args.save_only_intersection,
        save_to=args.validation_results_merged_filename
    )

    # insert resolved and validated inconsistsencies back into the KG
    reinstate_resolved_and_validated(
        kg_without_inconsistencies_filepath=args.kg_without_inconsistencies_filename,
        df_resolved_and_validated=df_resolved_and_validated,
        reinstate_mode=args.reinstate_mode,
        pos_neg_relation_pairs_filepath=args.pos_neg_relation_pairs_filename,
        save_to=args.final_kg_filename,
    )


if __name__ == '__main__':
    main()
