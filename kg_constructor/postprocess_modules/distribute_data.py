"""
Filename: distribute_data.py

Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Distribute the knowledge base into multiple folds for evaluation
    and generate hypothesis to be used by the final model.

To-do:
"""
# standard imports
import logging as log
import os
import sys

# third party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class DistributeData():
    """
    Class for distributing the knowledge base into folds.
    """

    # global variables
    _SUB_STR = 'Subject'
    _PRED_STR = 'Predicate'
    _OBJ_STR = 'Object'
    _LABEL_STR = 'Label'

    _SPOL_LIST = [_SUB_STR, _PRED_STR, _OBJ_STR, _LABEL_STR]
    _POS_LABEL = '1'
    _NEG_LABEL = '-1'

    def __init__(self, pd_data, num_folds, num_negs, target_relation, all_genes):
        """
        Class constructor for DistributeData.

        Inputs:
            pd_data: integrated data
            num_folds: number of folds to split the data into
            all_genes: numpy array containing unique genes
        """
        self.pd_data = pd_data.sample(frac=1).reset_index(drop=True)
        self.num_folds = num_folds
        self.num_negs = num_negs
        self.target_relation = target_relation
        self.all_genes = all_genes

        # extract triplets that is needed in class methods
        self._extract_triplets()

        # appened to the cra only negatives with synthetically generated negatives
        self.updated_neg_target = self._sample_negatives()

    def _extract_triplets(self):
        """
        (Private) Extract triplets from the data
        that will be used in other class methods.
        """
        # all triplets containing target predicate (both pos and neg)
        self.all_target = self.pd_data[self.pd_data[self._PRED_STR].isin([self.target_relation])]

        # all positive & negative triplets
        self.pos_data = self.pd_data[self.pd_data[self._LABEL_STR].astype(str) == self._POS_LABEL]
        self.neg_data = self.pd_data[self.pd_data[self._LABEL_STR].astype(str) == self._NEG_LABEL]

        # positive triplets containing only or no target predicates
        self.pos_target = self.pos_data[self.pos_data[self._PRED_STR].isin([self.target_relation])]
        self.pos_except_target = self.pos_data[
            ~self.pos_data[self._PRED_STR].isin([self.target_relation])]

        # negative triplets containing only target predicates
        self.neg_target = self.neg_data[self.neg_data[self._PRED_STR].isin([self.target_relation])]

    def _sample_negatives(self):
        """
        (Private) For each positive triplet containing the target predicate,
        sample self.num_negs negatives. If there isn't enough known negatives,
        randomly generate synthetic negativs.
        """
        num_pos_to_remove_dic = {}
        random_sampled_negs = pd.DataFrame(columns=self._SPOL_LIST)

        # create dictionary where key is unique object and value is number of their occurance
        objs_values = self.pos_target[self._OBJ_STR].value_counts().keys().tolist()
        objs_counts = self.pos_target[self._OBJ_STR].value_counts().tolist()
        objs_value_counts_dic = dict(zip(objs_values, objs_counts))

        for obj, count in objs_value_counts_dic.items():
            num_negs_needed = self.num_negs * count
            num_known_neg_samples = self.neg_target[
                self.neg_target[self._OBJ_STR].isin([obj])].shape[0]

            # if we don't have enough known negatives to sample for given object
            if num_known_neg_samples < num_negs_needed:
                triplets_same_obj = self.all_target[self.all_target[self._OBJ_STR].isin([obj])]
                known_subjs = triplets_same_obj[self._SUB_STR].unique()
                unknown_subjs = np.array(
                    list(set(self.all_genes.tolist()) - set(known_subjs.tolist())))

                # find how many random subjects we need
                num_random_subjs = num_negs_needed - num_known_neg_samples

                log.debug('Number of randomly sampled subjects for %s: %d', obj, num_random_subjs)

                # if number of unknown subjects are less that what we have to
                # randomly select, remove some positives to balance
                if unknown_subjs.shape[0] < num_random_subjs:

                    num_pos_to_remove_dic[obj] = int(np.ceil((num_random_subjs -
                                                     unknown_subjs.shape[0]) / self.num_negs))
                    num_random_subjs -= (self.num_negs * num_pos_to_remove_dic[obj])

                # fill randomly samples negatives using closed world assumption
                random_generated_neg_cra = pd.DataFrame(
                    0, index=np.arange(num_random_subjs), columns=self._SPOL_LIST)
                random_generated_neg_cra[self._SUB_STR] = np.random.choice(
                    unknown_subjs, num_random_subjs, replace=False)
                random_generated_neg_cra[self._PRED_STR] = self.target_relation
                random_generated_neg_cra[self._OBJ_STR] = obj
                random_generated_neg_cra[self._LABEL_STR] = self._NEG_LABEL

                random_sampled_negs = random_sampled_negs.append(random_generated_neg_cra)
            else:
                log.info('Enough known negatives for %s', obj)

        # return updated negative triplets contaiting target predicate
        updated_neg_target = pd.concat([self.neg_target, random_sampled_negs])
        updated_neg_target = updated_neg_target.sample(frac=1).reset_index(drop=True)

        # we have some positives to remove to balance pos and neg
        if num_pos_to_remove_dic:
            for obj, count in num_pos_to_remove_dic.items():
                indices = self.pos_target.index[self.pos_target[self._OBJ_STR].isin([obj])].tolist()
                self.pd_data = self.pd_data.drop(self.pd_data.index[indices[0:count]])

            self.pd_data = self.pd_data.sample(frac=1).reset_index(drop=True)
            self._extract_triplets()

        return updated_neg_target

    def split_into_folds(self):
        """
        Perform data split.

        Returns:
            data_split_fold_dic: dictionary where the key is
                train / dev / test for each fold and the value
                is a dataframe containing the splitted
                knowledge graph
        """
        data_split_fold_dic = {}

        # distribute CRA edges among train / dev / test for specified folds
        k = 0
        for train_dev_index, test_index in KFold(n_splits=self.num_folds).split(self.pos_target):
            np.random.shuffle(train_dev_index)

            # allocate 90% of train_dev_index into train
            num_train = int(0.9 * train_dev_index.shape[0])
            train_index = train_dev_index[0:num_train]
            dev_index = train_dev_index[num_train:]

            data_split_fold_dic['fold_{}_train'.format(k)] = self.pos_target.iloc[train_index, :]
            data_split_fold_dic['fold_{}_train_local_without_neg'.format(k)] = \
                self.pos_target.iloc[train_index, :]
            data_split_fold_dic['fold_{}_dev_without_neg'.format(k)] = \
                self.pos_target.iloc[dev_index, :]
            data_split_fold_dic['fold_{}_test_without_neg'.format(k)] = \
                self.pos_target.iloc[test_index, :]

            k += 1

        # fill up train with other positive predicates for all folds
        for k in range(self.num_folds):
            data_split_fold_dic['fold_{}_train'.format(k)] = \
                data_split_fold_dic['fold_{}_train'.format(k)].append(self.pos_except_target)

        # need to do random sampling to select negatives for each positive
        for k in range(self.num_folds):
            log.info('Processing fold: %d', k)

            data_split_fold_dic = self._random_sample_negs(k, data_split_fold_dic, 'train_local')
            data_split_fold_dic = self._random_sample_negs(k, data_split_fold_dic, 'dev')
            data_split_fold_dic = self._random_sample_negs(k, data_split_fold_dic, 'test')

        for key, value in data_split_fold_dic.items():
            log.debug('Shape of %s: %d', key, value.shape[0])

        return data_split_fold_dic

    def save_folds(self, data_split_fold_dic, save_parent_dir):
        """
        Save the processed folds into the specified directory.
        save_parent_dir / folds / fold_0
                                / fold_1
                        .
                        .
                        .
                                / fold_k

        Inpus:
            data_split_fold_dic: dictionary processed in 'split_into_folds()'
            save_parent_dir: parent directory to save the folds into
        """
        sub_parent_dir = os.path.join(save_parent_dir, 'folds')

        for k in range(self.num_folds):
            each_fold_parent_dir = os.path.join(sub_parent_dir, 'fold_{}'.format(k))

            if not os.path.exists(each_fold_parent_dir):
                os.makedirs(each_fold_parent_dir)

            data_split_fold_dic['fold_{}_train'.format(k)].to_csv(
                os.path.join(each_fold_parent_dir, 'train.txt'),
                sep='\t',
                index=False,
                header=None)
            data_split_fold_dic['fold_{}_train_local'.format(k)].to_csv(
                os.path.join(each_fold_parent_dir, 'train_local.txt'),
                sep='\t',
                index=False,
                header=None)
            data_split_fold_dic['fold_{}_dev'.format(k)].to_csv(
                os.path.join(each_fold_parent_dir, 'dev.txt'),
                sep='\t',
                index=False,
                header=None)
            data_split_fold_dic['fold_{}_test'.format(k)].to_csv(
                os.path.join(each_fold_parent_dir, 'test.txt'),
                sep='\t',
                index=False,
                header=None)

    def save_train_data_for_final_model(self, output_dir):
        """
        Save data for training the final model.

        Inputs:
            output_dir: directory to save the final train data
        """
        final_dir = os.path.join(output_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)

        filepath = os.path.join(final_dir, 'train.txt')

        log.info('Saving final train data to \'%s\'', filepath)
        self.pos_data.to_csv(filepath, sep='\t', index=False, header=None)

        # init
        known_negatives_dic = {}
        new_with_neg_cra = pd.DataFrame(columns=self._SPOL_LIST)

        # for each positive, randomly sample 'self.num_negs'
        # negatives and append to 'new_with_neg_cra'
        for i in range(self.pos_target.shape[0]):
            # actual positive SPO that we're working on
            pos_triplet = self.pos_target.iloc[i, :]
            obj = pos_triplet[self._OBJ_STR]

            # before, append negatives find known negatives
            # and save to the dictionary if it already does not exist
            if obj not in known_negatives_dic:
                # find negative samples which has same object as the positive SPO
                # but different subject than that in positive SPO
                known_negatives_dic[obj] = self.updated_neg_target[
                    self.updated_neg_target[self._OBJ_STR].isin([obj])]

            if known_negatives_dic[obj].shape[0] < self.num_negs:
                continue

            # append the original positive SPO
            new_with_neg_cra = new_with_neg_cra.append(pos_triplet)

            new_with_neg_cra = new_with_neg_cra.append(
                known_negatives_dic[obj].iloc[0:self.num_negs, :])

            # remove used known negatives
            known_negatives_dic[obj] = known_negatives_dic[obj].iloc[self.num_negs:, :]

        filepath = os.path.join(final_dir, 'train_local.txt')

        log.info('Saving final train_local data to \'%s\'', filepath)
        new_with_neg_cra.to_csv(filepath, sep='\t', index=False, header=None)

    def _random_sample_negs(self, cur_fold, data_split_fold_dic, dtype):
        """
        (Private) Randomly sample 'self.num_negs' known negatives for each positive.

        Inputs:
            cur_fold: fold # currently being processed
            data_split_fold_dic: dictionary processed in 'split_into_folds()'
            dtype: data type (i.e. 'dev' | 'test' | 'train_local')
            self.num_negs: number of negatives to sample per positive

        Returns:
            data_split_fold_dic: updated dictionary containing negatives
        """
        known_negatives_dic = {}
        new_with_neg_cra = pd.DataFrame(columns=self._SPOL_LIST)

        # find how many total positives there are
        pos_size = data_split_fold_dic['fold_{}_{}_without_neg'.format(cur_fold, dtype)].shape[0]
        log.debug('Size of fold %d %s only positives: %d', cur_fold, dtype, pos_size)

        # for each positive, randomly sample 'self.num_negs'
        # negatives and append to 'new_with_neg_cra'
        for i in range(pos_size):
            # actual positive SPO that we're working on
            pos_triplet = data_split_fold_dic[
                'fold_{}_{}_without_neg'.format(cur_fold, dtype)].iloc[i, :]
            obj = pos_triplet[self._OBJ_STR]

            # before, append negatives find known negatives
            # and save to the dictionary if it already does not exist
            if obj not in known_negatives_dic:
                # find negative samples which has same object as the positive SPO
                # but different subject than that in positive SPO
                known_negatives_dic[obj] = self.updated_neg_target[
                    self.updated_neg_target[self._OBJ_STR].isin([obj])]

            if known_negatives_dic[obj].shape[0] < self.num_negs:
                continue

            # append the original positive SPO
            new_with_neg_cra = new_with_neg_cra.append(pos_triplet)

            new_with_neg_cra = new_with_neg_cra.append(
                known_negatives_dic[obj].iloc[0:self.num_negs, :])

            # remove used known negatives
            known_negatives_dic[obj] = known_negatives_dic[obj].iloc[self.num_negs:, :]

        log.debug('Newly sampled %s with negative samples: %d', dtype, new_with_neg_cra.shape[0])

        data_split_fold_dic['fold_{}_{}'.format(cur_fold, dtype)] = \
            new_with_neg_cra.reset_index(drop=True)

        return data_split_fold_dic
