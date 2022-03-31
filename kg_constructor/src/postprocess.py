import argparse
from glob import glob
import itertools
import logging as log
import os
from pathlib import Path
import random
import sys

from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from tqdm import tqdm

from utils.logger import set_logging

# default variables
DEFAULT_DATA_DIR = '../data'
DEFAULT_OUTPUT_DIR = '../output'
DEFAULT_DATASET = 'ecoli'
DEFAULT_FINAL_KG_FILENAME = 'kg_final.txt'
DEFAULT_DOMAIN_RANGE_FILENAME = 'domain_range.txt'
DEFAULT_POS_NEG_RELATION_PAIRS_FILENAME = 'pos_neg_relation_pairs.txt'
DEFAULT_FINAL_KG_WITH_LABEL_FILENAME = 'kg_final_with_label.txt'
DEFAULT_ENTITIES_FILENAME = 'entities.txt'
DEFAULT_HYPOTHESIS_RELATION = 'confers resistance to antibiotic'
DEFAULT_NUM_FOLDS = 5
DEFAULT_CORRUPT_WHERE = 'head'
DEFAULT_NUM_NEGATIVES = 49
DEFAULT_TEST_PROPORTION = 0.2
DEFAULT_RANDOM_STATE = 530
SPO_LIST = ['Subject', 'Predicate', 'Object']


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
        '--final_kg_filename',
        type=str,
        default=DEFAULT_FINAL_KG_FILENAME,
        help=f'Filename of the final knowledge graph. '
             f'(Default: {DEFAULT_FINAL_KG_FILENAME})',
    )

    parser.add_argument(
        '--domain_range_filename',
        type=str,
        default=DEFAULT_DOMAIN_RANGE_FILENAME,
        help=f'File containing domain and range information. '
             f'(Default: {DEFAULT_DOMAIN_RANGE_FILENAME})',
    )

    parser.add_argument(
        '--pos_neg_relation_pairs_filename',
        type=str,
        default=DEFAULT_POS_NEG_RELATION_PAIRS_FILENAME,
        help=f'Name of the file containing pairs of positive and negative relations. '
             f'(Default: {DEFAULT_POS_NEG_RELATION_PAIRS_FILENAME})',
    )

    parser.add_argument(
        '--final_kg_with_label_filename',
        type=str,
        default=DEFAULT_FINAL_KG_WITH_LABEL_FILENAME,
        help=f'Filename of the final knowledge graph with label column. '
             f'(Default: {DEFAULT_FINAL_KG_WITH_LABEL_FILENAME})',
    )

    parser.add_argument(
        '--entities_filename',
        type=str,
        default=DEFAULT_ENTITIES_FILENAME,
        help=f'Filename of the entities. (Default: {DEFAULT_ENTITIES_FILENAME})',
    )

    parser.add_argument(
        '--hypothesis_relation',
        type=str,
        default=DEFAULT_HYPOTHESIS_RELATION,
        help=f'Relations to generate hypothesis on. (Default: {DEFAULT_HYPOTHESIS_RELATION})',
    )

    parser.add_argument(
        '--num_folds',
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help=f'Number of folds for cross validation. (Default: {DEFAULT_NUM_FOLDS})',
    )

    parser.add_argument(
        '--corrupt_where',
        type=str,
        default=DEFAULT_CORRUPT_WHERE,
        help=f'Where to corrupt (head|tail|both) when generating negatives. '
             f'(Default: {DEFAULT_CORRUPT_WHERE})',
    )

    parser.add_argument(
        '--num_negatives',
        type=int,
        default=DEFAULT_NUM_NEGATIVES,
        help=f'Number of negatives. (Default: {DEFAULT_NUM_NEGATIVES})',
    )

    parser.add_argument(
        '--test_proportion',
        type=float,
        default=DEFAULT_TEST_PROPORTION,
        help=f'Ratio of the test set. (Default: {DEFAULT_TEST_PROPORTION})',
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f'Random state for reproducibility. (Default: {DEFAULT_RANDOM_STATE})',
    )

    args = parser.parse_args()

    # process directories
    data_dir = os.path.join(DEFAULT_DATA_DIR, args.dataset)
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.dataset)

    # update paths
    args.final_kg_filename = os.path.join(output_dir, args.final_kg_filename)
    args.domain_range_filename = os.path.join(data_dir, args.domain_range_filename)
    args.pos_neg_relation_pairs_filename = os.path.join(
        data_dir, args.pos_neg_relation_pairs_filename)
    args.final_kg_with_label_filename = os.path.join(
        output_dir, args.final_kg_with_label_filename)
    args.entities_filename = os.path.join(output_dir, args.entities_filename)
    args.output_dir = output_dir

    return args


def add_label(
        df_filepath,
        pos_neg_relation_pairs_filepath,
        final_kg_with_label_filepath):
    """
    Append column 'Label' based on the rule file.

    Returns:
        df_with_label: (pd.DataFrame) Reformated data with 'Label' column added.
    """
    df = pd.read_csv(df_filepath, sep='\t')

    if not Path(pos_neg_relation_pairs_filepath).exists():
        log.warning(f'\'{pos_neg_relation_pairs_filepath}\' does not exist!')
        log.warning('We are assuming all predicates are positives!')

        df_with_label = df[SPO_LIST].copy()
        df_with_label['Label'] = 1
        return df_with_label

    # drop unnecessary columns
    df_with_label = df[SPO_LIST].copy()
    df_with_label['Label'] = ''

    # some stats
    predicate_group = df_with_label.groupby('Predicate')
    log.debug(f'Size of data grouped by predicates before adding label:\n{predicate_group.size()}')

    df_post_neg_relation_pairs = pd.read_csv(pos_neg_relation_pairs_filepath, sep='\t')
    for _, row in df_post_neg_relation_pairs.iterrows():
        pos_pred, neg_pred = row['positive'], row['negative']

        if not pd.isna(pos_pred):
            df_with_label.loc[df_with_label['Predicate'] == pos_pred, 'Label'] = 1

        if not pd.isna(neg_pred):
            df_with_label.loc[df_with_label['Predicate'] == neg_pred, 'Label'] = -1
            df_with_label.loc[df_with_label['Predicate'] == neg_pred, 'Predicate'] = pos_pred

    # drop duplicates that results from using the Label column
    size_before = df_with_label.shape[0]
    df_with_label = df_with_label.drop_duplicates(subset=SPO_LIST, keep=False)
    if df_with_label.shape[0] - size_before > 0:
        log.warning('There were duplicates in the data. Please check!')

    # some stats
    predicate_group = df_with_label.groupby('Predicate')
    log.debug(f'Size of data grouped by predicates after adding label:\n{predicate_group.size()}')

    label_group = df_with_label.groupby('Label')
    log.debug(f'Size of data grouped by label:\n{label_group.size()}')

    log.info(f'Saving final KG with label column to \'{final_kg_with_label_filepath}\'')
    df_with_label.to_csv(final_kg_with_label_filepath, sep='\t', index=False, header=None)

    return df_with_label


def get_entity_dict(
        df_filepath,
        domain_range_filepath,
        entities_filepath):
    """
    Create the dictionary where the key is entity type and value is
    the numpy array containing all the entities belonging to that type.

    Returns:
        entity_dict: (dict) Completed entity dictionary
    """
    df = pd.read_csv(df_filepath, sep='\t')
    df_domain_range = pd.read_csv(domain_range_filepath, sep='\t')

    entity_dict = {}
    for _, row in df_domain_range.iterrows():
        df_match = df[df['Predicate'] == row['Relation']]
        subjects = df_match['Subject'].tolist()
        objects = df_match['Object'].tolist()

        dom = row['Domain']
        ran = row['Range']
        if dom in entity_dict:
            entity_dict[dom].extend(subjects)
        else:
            entity_dict[dom] = subjects

        if ran in entity_dict:
            entity_dict[ran].extend(objects)
        else:
            entity_dict[ran] = objects

    entity_dict = {k: list(set(v)) for k, v in entity_dict.items()}

    # save data
    data = []
    for k, v in entity_dict.items():
        data.extend(list(zip([k]*len(v), v)))

    df_entities = pd.DataFrame(data, columns=['type', 'name'])
    df_entities.to_csv(entities_filepath, index=False, sep='\t')

    return entity_dict


def split_and_save_folds(
        df_with_label,
        test_proportion,
        num_folds,
        domain_range_filepath,
        entity_dict,
        corrupt_where,
        num_negatives,
        random_state,
        hypothesis_relation,
        output_dir):
    """
    """
    df_domain_range = pd.read_csv(domain_range_filepath, sep='\t')

    df_positives = df_with_label.copy()[df_with_label['Label'] == 1]
    df_match_rel = df_positives[df_positives['Predicate'] == hypothesis_relation]
    df_other_rel = df_positives[df_positives['Predicate'] != hypothesis_relation]

    log.info(f'Number of only positives: {df_positives.shape[0]}')
    log.info(f'Number of matching relation positives: {df_match_rel.shape[0]}')
    log.info(f'Number of other relations positives: {df_other_rel.shape[0]}')

    log.info(f'Test proportion: {test_proportion}')
    df_match_rel_train, df_test = train_test_split(
        df_match_rel,
        test_size=test_proportion,
        random_state=random_state
    )

    # need to generate negatives for the test set only once
    if num_negatives > 0:
        log.info('Generating negatives for test set!')
        df_test = generate_negatives(
            df=df_test,
            df_known_pos=df_positives,
            df_dr=df_domain_range,
            entity_dict=entity_dict,
            corrupt_where=corrupt_where,
            num_negatives=num_negatives,
        )

    fold_save_dir = os.path.join(output_dir, 'folds')
    log.info(f'Saving folds to \'{fold_save_dir}\'')

    if Path(fold_save_dir).exists():
        log.warning(f'Fold save directory \'{fold_save_dir}\' already exists!')
    else:
        Path(fold_save_dir).mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=num_folds)
    fold = 0
    for train_idx, val_idx in kf.split(df_match_rel_train):
        log.info(f'Processing fold {fold}...')
        df_train = pd.concat([df_match_rel_train.iloc[train_idx], df_other_rel])
        df_val = df_match_rel_train.iloc[val_idx]

        if num_negatives > 0:
            log.info('Generating negatives for val set!')
            df_val = generate_negatives(
                df=df_val,
                df_known_pos=df_positives,
                df_dr=df_domain_range,
                entity_dict=entity_dict,
                corrupt_where=corrupt_where,
                num_negatives=num_negatives,
            )

        fold_n_save_dir = os.path.join(fold_save_dir, f'fold_{fold}')
        Path(fold_n_save_dir).mkdir(parents=True, exist_ok=True)

        df_train.to_csv(
            os.path.join(fold_n_save_dir, 'train.txt'), sep='\t', index=False, header=None)
        df_val.to_csv(
            os.path.join(fold_n_save_dir, 'val.txt'), sep='\t', index=False, header=None)
        df_test.to_csv(
            os.path.join(fold_n_save_dir, 'test.txt'), sep='\t', index=False, header=None)

        fold += 1


def generate_negatives(
        df,
        df_known_pos,
        df_dr,
        entity_dict,
        corrupt_where,
        num_negatives):
    """
    """
    all_triples_without_label = []
    for _, triple in df_known_pos.iterrows():
        all_triples_without_label.append(
            [triple['Subject'], triple['Predicate'], triple['Object']])

    if corrupt_where == 'both':
        num_negatives_head = int(num_negatives/2)
        num_negatives_tail = num_negatives - num_negatives_head
    elif corrupt_where == 'head':
        num_negatives_head = num_negatives
    elif corrupt_where == 'tail':
        num_negatives_tail = num_negatives
    else:
        raise ValueError(f'Invalid corrupt_where: {corrupt_where}')

    def _get_corrupt_entity(h, r, t, where, entity_type):
        candidate_entities = entity_dict[entity_type].copy()
        _random.shuffle(candidate_entities)

        if where == 'head':
            while candidate_entities:
                corrupt_entity = candidate_entities.pop()
                if [corrupt_entity, r, t] not in all_triples_without_label:
                    break
        elif where == 'tail':
            while candidate_entities:
                corrupt_entity = candidate_entities.pop()
                if [h, r, corrupt_entity] not in all_triples_without_label:
                    break

        if not candidate_entities:
            return None
        else:
            return corrupt_entity

    # TODO: convert this to use multiprocessing. it's taking too long...
    data_with_neg = []
    for _, triple in tqdm(df.iterrows(), total=df.shape[0]):
        sub, pred = triple['Subject'], triple['Predicate']
        obj, label = triple['Object'], triple['Label']

        # positive
        data_with_neg.append([sub, pred, obj, label])

        # do corrupt
        if corrupt_where in ['head', 'both']:
            entity_type = df_dr.loc[(df_dr == pred).any(axis=1), 'Domain'].tolist()[0]

            for _ in range(num_negatives_head):
                corrupt_head = _get_corrupt_entity(
                    sub, pred, obj, corrupt_where, entity_type)
                if corrupt_head:
                    data_with_neg.append([corrupt_head, pred, obj, -1])

        if corrupt_where in ['tail', 'both']:
            entity_type = df_dr.loc[(df_dr == pred).any(axis=1), 'Range'].tolist()[0]

            for _ in range(num_negatives_tail):
                corrupt_tail = _get_corrupt_entity(
                    sub, pred, obj, corrupt_where, entity_type)
                if corrupt_tail:
                    data_with_neg.append([sub, pred, corrupt_tail, -1])

    df_corrupted = pd.DataFrame(data_with_neg, columns=list(df.columns))
    return df_corrupted


def generate_data_for_final_model(
        df_with_label,
        entity_dict,
        domain_range_filepath,
        hypothesis_relation,
        output_dir):
    """
    Generate hypothesis.

    Inputs:
    """
    df_train = df_with_label.copy()[df_with_label['Label'] == 1]
    df_domain_range = pd.read_csv(domain_range_filepath, sep='\t')

    df_dr_match = df_domain_range[df_domain_range['Relation'] == hypothesis_relation]
    domain_type = df_dr_match['Domain'].tolist()[0]
    range_type = df_dr_match['Range'].tolist()[0]

    df_known = df_with_label[df_with_label['Predicate'] == hypothesis_relation]
    known_sub_obj_pairs = list(zip(df_known['Subject'], df_known['Object']))
    all_possible_sub_obj_pairs = list(itertools.product(entity_dict[domain_type], entity_dict[range_type]))
    unknown_sub_obj_pairs = set(all_possible_sub_obj_pairs) - set(known_sub_obj_pairs)

    df_hypothesis = pd.DataFrame(unknown_sub_obj_pairs, columns=['Subject', 'Object'])
    df_hypothesis['Predicate'] = hypothesis_relation
    df_hypothesis['Label'] = 1
    df_hypothesis = df_hypothesis[['Subject', 'Predicate', 'Object', 'Label']]

    # save
    final_save_dir = os.path.join(output_dir, 'final')
    log.info(f'Saving final model data to \'{final_save_dir}\'')

    if Path(final_save_dir).exists():
        log.warning(f'Final data save directory \'{final_save_dir}\' already exists!')
    else:
        Path(final_save_dir).mkdir(parents=True, exist_ok=True)

    df_train.to_csv(
        os.path.join(final_save_dir, 'train.txt'), sep='\t', index=False, header=None)
    df_hypothesis.to_csv(
        os.path.join(final_save_dir, 'test.txt'), sep='\t', index=False, header=None)


def convert_2_openke_compatible(
        entities_filepath,
        domain_range_filepath,
        output_dir):
    """
    """
    log.info('Converting data into OpenKE compatible format.')

    # folds
    folds_dir = glob(os.path.join(output_dir, 'folds/*/'))
    folds_dir = sorted(folds_dir)
    log.info(f'Folds directories found: {folds_dir}')

    final_dir = os.path.join(output_dir, 'final')
    log.info(f'Final directory found: {final_dir}')

    # entities
    df_entities = pd.read_csv(entities_filepath, sep='\t')
    df_entities['id'] = df_entities.index

    entity2id_content = f'{df_entities.shape[0]}\n'
    entity2id_content += '\n'.join(
        [f'{e}\t{i}' for e, i in list(zip(df_entities['name'], df_entities['id']))])
    entity2id_content += '\n'

    # relations
    df_relations = pd.read_csv(domain_range_filepath, sep='\t')
    df_relations['id'] = df_relations.index

    relation2id_content = f'{df_relations.shape[0]}\n'
    relation2id_content += '\n'.join(
        [f'{r}\t{i}' for r, i in list(zip(df_relations['Relation'], df_relations['id']))])
    relation2id_content += '\n'

    # lookup dict
    entity_lookup = dict(zip(df_entities['name'], df_entities['id']))
    relation_lookup = dict(zip(df_relations['Relation'], df_relations['id']))

    # process folds
    for fold_dir in folds_dir:
        log.info(f'Processing folds directory \'{fold_dir}\'...')

        # load data
        df_train = pd.read_csv(
            os.path.join(fold_dir, 'train.txt'),
            sep='\t',
            names=['Subject', 'Predicate', 'Object', 'Label'])
        df_val = pd.read_csv(
            os.path.join(fold_dir, 'val.txt'),
            sep='\t',
            names=['Subject', 'Predicate', 'Object', 'Label'])
        df_test = pd.read_csv(
            os.path.join(fold_dir, 'test.txt'),
            sep='\t',
            names=['Subject', 'Predicate', 'Object', 'Label'])

        with open(os.path.join(fold_dir, 'entity2id.txt'), mode='w', encoding='utf-8') as file:
            file.write(entity2id_content)

        with open(os.path.join(fold_dir, 'relation2id.txt'), mode='w', encoding='utf-8') as file:
            file.write(relation2id_content)

        # train
        df_train['Subject'] = df_train['Subject'].replace(entity_lookup)
        df_train['Object'] = df_train['Object'].replace(entity_lookup)
        df_train['Predicate'] = df_train['Predicate'].replace(relation_lookup)
        train2id_content = [
            f'{h} {t} {p}' for h, t, p in
            list(zip(df_train['Subject'], df_train['Object'], df_train['Predicate']))]
        with open(os.path.join(fold_dir, 'train2id.txt'), mode='w', encoding='utf-8') as file:
            file.write(f'{len(train2id_content)}\n')
            file.write('\n'.join(train2id_content))
            file.write('\n')

        # val
        df_val = df_val.replace(entity_lookup)
        df_val = df_val.replace(relation_lookup)
        val2id_content = [
            f'{h} {t} {p} {l}' for h, t, p, l in
            list(zip(df_val['Subject'], df_val['Object'], df_val['Predicate'], df_val['Label']))]
        with open(os.path.join(fold_dir, 'val2id.txt'), mode='w', encoding='utf-8') as file:
            file.write(f'{len(val2id_content)}\n')
            file.write('\n'.join(val2id_content))
            file.write('\n')

        # train + val (positives only). For OpenKE
        trainval2id_content = train2id_content + [
            f'{h} {t} {p}' for h, t, p, l in
            list(zip(df_val['Subject'], df_val['Object'], df_val['Predicate'], df_val['Label']))
            if l == 1]
        with open(os.path.join(fold_dir, 'trainval2id.txt'), mode='w', encoding='utf-8') as file:
            file.write(f'{len(trainval2id_content)}\n')
            file.write('\n'.join(trainval2id_content))
            file.write('\n')

        # test
        df_test = df_test.replace(entity_lookup)
        df_test = df_test.replace(relation_lookup)
        test2id_content = [
            f'{h} {t} {p} {l}' for h, t, p, l in
            list(zip(df_test['Subject'], df_test['Object'], df_test['Predicate'], df_test['Label']))]
        with open(os.path.join(fold_dir, 'test2id.txt'), mode='w', encoding='utf-8') as file:
            file.write(f'{df_test.shape[0]}\n')
            file.write('\n'.join(test2id_content))
            file.write('\n')


def main() -> None:
    args = parse_argument()
    set_logging()

    # Set random seed.
    global _random
    log.info(f'Random seed set to: {args.random_state}')
    _random = random.Random(args.random_state)

    # get and save complete dictionary of entities
    entity_dict = get_entity_dict(
        df_filepath=args.final_kg_filename,
        domain_range_filepath=args.domain_range_filename,
        entities_filepath=args.entities_filename,
    )

    # # reformat and save the data
    # df_with_label = add_label(
    #     df_filepath=args.final_kg_filename,
    #     pos_neg_relation_pairs_filepath=args.pos_neg_relation_pairs_filename,
    #     final_kg_with_label_filepath=args.final_kg_with_label_filename,
    # )

    # # split the dataset into specified folds
    # split_and_save_folds(
    #     df_with_label=df_with_label,
    #     test_proportion=args.test_proportion,
    #     num_folds=args.num_folds,
    #     domain_range_filepath=args.domain_range_filename,
    #     entity_dict=entity_dict,
    #     corrupt_where=args.corrupt_where,
    #     num_negatives=args.num_negatives,
    #     random_state=args.random_state,
    #     hypothesis_relation=args.hypothesis_relation,
    #     output_dir=args.output_dir,
    # )

    # # save data to use for training the final model
    # generate_data_for_final_model(
    #     df_with_label=df_with_label,
    #     entity_dict=entity_dict,
    #     domain_range_filepath=args.domain_range_filename,
    #     hypothesis_relation=args.hypothesis_relation,
    #     output_dir=args.output_dir,
    # )

    # generate OpenKE compatible data
    convert_2_openke_compatible(
        entities_filepath=args.entities_filename,
        domain_range_filepath=args.domain_range_filename,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
