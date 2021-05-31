"""
Filename: inconsistency_manager.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Do everything about the inconsistencies here.

To-do:
"""

# standard imports
from ast import literal_eval
import logging as log
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), './'))

# third party imports
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# local imports
from inconsistency_correctors.averagelog import AverageLog
from inconsistency_correctors.investment import Investment
from inconsistency_correctors.pooledinvestment import PooledInvestment
from inconsistency_correctors.sums import Sums
from inconsistency_correctors.truthfinder import TruthFinder
from utilities import get_pd_of_statement
from inconsistency_correctors.voting import Voting


class InconsistencyManager():
    """
    Class for detecting the inconsistencies.
    """
    _SPO_LIST = ['Subject', 'Predicate', 'Object']
    _RESOLVED_INCONSISTENCIES_COLUMNS = _SPO_LIST + [
        'Belief',
        'Source size',
        'Sources',
        'Total source size',
        'Mean belief of conflicting tuples',
        'Belief difference',
        'Conflicting tuple info']
    _FINAL_KG_COLUMNS = _SPO_LIST + [
        'Belief',
        'Source size',
        'Sources']

    def __init__(self, inconsistency_rule_file, resolver_mode='AverageLog'):
        """
        Class constructor for InconsistencyManager.

        Inputs:
            inconsistency_rule_file: (str) XML filepath containing the inconsistency rules.
            resolver_mode: (str) Inconsistency resolution mode.
                (AverageLog | Investment | PooledInvestment | Sums | TruthFinder | Voting)
        """
        self.inconsistency_rule_file = inconsistency_rule_file
        self.resolver_mode = resolver_mode.lower()

    def detect(self, pd_data):
        """
        Detect the inconsistencies among the data using the provided rule file.

        Inputs:
            pd_data: (pd.DataFrame) Data to detect inconsistency from

        Returns:
            inconsistencies: (dict) Dictionary containing inconsistency_id as key
                and list of inconsistent triplets + source as value.

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
        """
        inconsistency_rules = ET.parse(self.inconsistency_rule_file).getroot()
        inconsistencies = {}
        inconsistency_id = 0

        log.info('Detecting inconsistencies using inconsisty rule %s', self.inconsistency_rule_file)

        # iterate over each inconsistency rule
        for inconsistency_rule in inconsistency_rules:
            inconsistency_rule_name = inconsistency_rule.get('name')
            log.debug('Processing inconsisency rule \'%s\'', inconsistency_rule_name)

            # check if condition is met
            condition_statement = inconsistency_rule.find('condition')
            pd_condition_match = pd_data.copy()

            if condition_statement is None:
                pd_condition_match = pd_data.copy()
            else:
                pd_condition_statement = get_pd_of_statement(condition_statement)
                indices_meeting_condition = (pd_data[pd_condition_statement.index] ==
                                             pd_condition_statement).all(1).values
                pd_condition_match = pd_data[indices_meeting_condition].copy()

                # skip to the next rule if condition statement
                # exists and there are no data meeting the condition
                if pd_condition_match.shape[0] == 0:
                    log.debug('Skipping \'%s\' because there are no data meeting condition',
                              inconsistency_rule_name)
                    continue

            # check if data has conflicts
            inconsistency_statement = inconsistency_rule.find('inconsistency')
            conflict_feature_name = inconsistency_statement.get('name')
            conflict_feature_values = [inconsistency_feature.get('value')
                                       for inconsistency_feature in inconsistency_statement]

            # skip to the next rule if there are no conflicts
            if not self._data_has_conflict(
                    pd_data[conflict_feature_name], conflict_feature_values):
                log.debug('Skipping \'%s\' because there are no conflicts', inconsistency_rule_name)
                continue

            rest_feature_names = [feature_name
                                  for feature_name in self._SPO_LIST
                                  if feature_name != conflict_feature_name]
            pd_grouped_data = pd_data.groupby(rest_feature_names)[conflict_feature_name].apply(set)

            def has_conflict_values(data, conflict_feature_values):
                return data.intersection(conflict_feature_values)

            pd_nconflict_data = pd_grouped_data.apply(
                has_conflict_values, args=(set(conflict_feature_values), ))
            pd_filtered_data = pd_nconflict_data[pd_nconflict_data.apply(len) > 1]

            # create inconsistency triplet list
            for row_idx in range(pd_filtered_data.shape[0]):
                if row_idx % 100 == 0:
                    log.debug('Creating list of inconsistencies: %d/%d',
                              row_idx, pd_filtered_data.shape[0])

                pd_conflict_data = pd.Series(pd_filtered_data.index[row_idx],
                                             index=rest_feature_names)

                conflict_tuples = []
                for conflict_value in pd_filtered_data[row_idx]:
                    pd_conflict_data[conflict_feature_name] = conflict_value

                    sources = pd.unique(
                        pd_condition_match[(pd_condition_match[self._SPO_LIST] ==
                                            pd_conflict_data).all(1)]['Source'])

                    conflict_tuples.append((tuple(pd_conflict_data[self._SPO_LIST]),
                                            sources.tolist()))

                inconsistencies[inconsistency_id] = conflict_tuples
                inconsistency_id = inconsistency_id + 1

        log.info('Found %d inconsistencies', len(inconsistencies))

        return inconsistencies

    def resolve(self, pd_data, inconsistencies, **kwargs):
        """
        Wrapper function for inconsistency resolver.

        Inputs:
            pd_data: (pd.DataFrame) Data that has inconsistencies to resolve.
            inconsistencies: (dict) Dictionary containing inconsistency_id as key
                and list of inconsistent triplets + source as value.
            kwargs: Arguments specific to algorithm to use.
                    (refer to each resolution algorithm)

        Returns:
            pd_resolved_inconsistencies: (pd.DataFrame) Resolved inconsistencies.
            pd_without_inconsistencies: (pd.DataFrame) Clean data without inconsistencies.
            np_trustworthiness_vector: (np.matrix) Trustworthiness of all the sources.
        """
        if self.resolver_mode == 'averagelog':
            result = AverageLog.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
        elif self.resolver_mode == 'investment':
            result = Investment.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
        elif self.resolver_mode == 'pooledinvestment':
            result = PooledInvestment.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
        elif self.resolver_mode == 'sums':
            result = Sums.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
        elif self.resolver_mode == 'truthfinder':
            result = TruthFinder.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
        elif self.resolver_mode == 'voting':
            result = Voting.resolve_inconsistencies(pd_data, inconsistencies, **kwargs)
        else:
            raise ValueError('Invalid inconsistency corrector \'%s\'chosen.', self.resolver_mode)

        pd_resolved_inconsistencies = self._inconsistencies_dict_to_pd(result[0])
        pd_without_inconsistencies = self._without_inconsistencies_reformat(result[1])
        np_trustworthiness_vector = result[2]

        return (pd_resolved_inconsistencies, pd_without_inconsistencies, np_trustworthiness_vector)

    @staticmethod
    def compare_resolution_with_validation(pd_resolved, pd_validated, save_only_intersection=False):
        """
        Compare resolved inconsistencies with the validation results.

        Inputs:
            pd_resolved: (pd.DataFrame) Resolved inconsistencies.
            pd_validated: (pd.DataFrame) Validated inconsistencies.
            save_only_intersection: (bool) True if saving only the inconsistencies
                appearing in both input dataframes, False otherwise.

        Returns:
            pd_resolved_and_validated: (pd.DataFrame) Comparison result.
                Columns 'Validation' and 'Match' are added to pd_resolved.
        """
        # add new columns that will be used for validation
        pd_resolved_and_validated = pd_resolved.copy()
        pd_resolved_and_validated['Validation'] = ''
        pd_resolved_and_validated['Match'] = ''

        for _, row in pd_validated.iterrows():
            validated_sub = row.Subject
            validated_pred = row.Predicate
            validated_obj = row.Object

            # find matching triplet in pd_resolved
            match = pd_resolved_and_validated[pd_resolved_and_validated.Subject == validated_sub]
            match = match[match.Object == validated_obj]

            if match.shape[0] > 1:
                # if there are more than one match, there is something wrong
                raise ValueError('Found {} matching resolved inconsistencies for ({}, {}).'
                                 .format(match.shape[0], validated_sub, validated_obj))
            elif match.shape[0] == 0:
                # no match, continue to next validation
                continue

            pd_resolved_and_validated.loc[match.index, 'Validation'] = validated_pred

            # check if resolution and validation results match
            resolved_and_validated_match = pd_resolved_and_validated.loc[
                match.index, 'Predicate'].str.contains(validated_pred).values[0]
            if resolved_and_validated_match:
                pd_resolved_and_validated.loc[match.index, 'Match'] = 'True'
            else:
                pd_resolved_and_validated.loc[match.index, 'Match'] = 'False'

        if save_only_intersection:
            pd_resolved_and_validated = pd_resolved_and_validated[
                pd_resolved_and_validated.Match != '']

        return pd_resolved_and_validated

    @classmethod
    def reinstate_resolved_and_validated(
            cls,
            pd_data_without_inconsistencies,
            pd_resolved_and_validated,
            mode='only_validated'):
        """
        Insert resolved and validated inconsistencies back into the
        original data to create the final knowledge graph.

        Inputs:
            pd_data_without_inconsistencies: (pd.DataFrame) Data without inconsistencies.
            pd_resolved_and_validated: (pd.DataFrame) Inconsistencies where
                some / all of them are validated.
            mode: (str) Which resolved inconsistencies to reinstate ('all' | 'only_validated')

        Returns:
            pd_data_final: (pd.DataFrame) Final knowledge graph.
        """
        log.info('Number of data without inconsistencies: %d',
                 pd_data_without_inconsistencies.shape[0])
        log.info('Number of resolved inconsistencies: %d',
                 pd_resolved_and_validated.shape[0])

        # select triplets to append depending on the selected mode
        if mode == 'all':
            pd_to_append = pd_resolved_and_validated
        elif mode == 'only_validated':
            pd_to_append = pd_resolved_and_validated[pd_resolved_and_validated['Validation'] != '']
            pd_to_append = pd_to_append[pd_to_append['Match'] == 'True']
        elif mode == 'validated_and_rest_as_positives':
            pd_to_append = pd_resolved_and_validated[pd_resolved_and_validated['Validation'] != '']
            pd_to_append = pd_to_append[pd_to_append['Match'] == 'True']

            pd_not_validated = pd_resolved_and_validated[pd_resolved_and_validated['Validation'] == '']
            pd_not_validated['Predicate'] = 'confers resistance to antibiotic'

            pd_to_append = pd.concat([pd_to_append, pd_not_validated], ignore_index=True, sort=False)
        else:
            raise ValueError('Invalid mode \'{}\' passed'.format(mode))

        log.info('Number of resolved inconsistencies to append: %d', pd_to_append.shape[0])

        # append the selected triplets
        pd_to_append = pd_to_append.loc[:, cls._FINAL_KG_COLUMNS]
        pd_data_final = pd.concat([pd_data_without_inconsistencies, pd_to_append],
                                  ignore_index=True, sort=False)

        log.info('Number of triplets in the final knowledge graph: %d', pd_data_final.shape[0])

        return pd_data_final

    @staticmethod
    def _data_has_conflict(all_feature_values, conflict_feature_values):
        """
        (Private) Check is data has conflicting values.

        Inputs:
            all_feature_values: (pd.Series) Column from one of Subject/Predicate/Object.
            conflict_feature_values: (list) List containing the conflicting feature values.
                i.e.: ['response to antibiotic', 'no response to antibiotic']

        Returns:
            True if data has conflicting values, False otherwise.
        """
        unique_feature_values = pd.unique(all_feature_values)
        num_of_conflicts = 0

        for conflict_feature_value in conflict_feature_values:
            if conflict_feature_value in unique_feature_values:
                num_of_conflicts = num_of_conflicts + 1

        return num_of_conflicts > 1

    @classmethod
    def _inconsistencies_dict_to_pd(cls, resolved_inconsistencies_dict):
        """
        (Private) Convert inconsistencies from dictionary to pandas DataFrame.

        Inputs:
            resolved_inconsistencies_dict: (dict) Dictionary where
                the key is inconsistency_id and the value is a list of tuples
                where each tuple is of form (inconsistent_tuple, sources, belief).
                Value of the dictionary is sorted by belief from high to low.

        Returns:
            pd_resolved_inconsistencies: (pd.DataFrame) Resolved inconsistencies
                converted to DataFrame format.
        """
        log.info('Converting resolved inconsistencies from dictionary to pandas DataFrame.')

        pd_resolved_inconsistencies = pd.DataFrame(columns=cls._RESOLVED_INCONSISTENCIES_COLUMNS)

        for inconsistent_tuples in resolved_inconsistencies_dict.values():
            (selected_tuple, sources, belief) = inconsistent_tuples[0]
            conflicting_tuples = inconsistent_tuples[1:]

            total_source_size = np.sum(
                [len(inconsistent_tuple[1]) for inconsistent_tuple in inconsistent_tuples])
            mean_belief_of_conflicting_tuple = np.mean(
                [conflicting_tuple[2] for conflicting_tuple in conflicting_tuples])

            row_dict = {}
            row_dict['Subject'] = selected_tuple[0]
            row_dict['Predicate'] = selected_tuple[1]
            row_dict['Object'] = selected_tuple[2]
            row_dict['Belief'] = '{0:.10f}'.format(belief)
            row_dict['Source size'] = len(sources)
            row_dict['Sources'] = ','.join(sources)
            row_dict['Total source size'] = total_source_size
            row_dict['Mean belief of conflicting tuples'] = \
                '{0:.10f}'.format(mean_belief_of_conflicting_tuple)
            row_dict['Belief difference'] = \
                '{0:.10f}'.format(belief - mean_belief_of_conflicting_tuple)
            row_dict['Conflicting tuple info'] = str(conflicting_tuples)

            pd_resolved_inconsistencies = pd_resolved_inconsistencies.append(
                pd.DataFrame.from_records([row_dict]), ignore_index=True, sort=False)

        return pd_resolved_inconsistencies

    @staticmethod
    def _without_inconsistencies_reformat(pd_raw):
        """
        (Private) Reformat data without inconsistencies into desired format.

        Inputs:
            pd_raw: (pd.DataFrame) Data needing reformat.

        Returns:
            pd_without_inconsistencies: (pd.DataFrame) Reformatted data.
        """
        log.info('Reformating data without inconsistencies.')

        pd_without_inconsistencies = pd_raw.reset_index()
        pd_without_inconsistencies = pd_without_inconsistencies.rename(
            columns={0: 'Belief'}).astype('str')
        pd_without_inconsistencies['Belief'] = pd_without_inconsistencies['Belief'].apply(
            lambda x: '{0:.2f}'.format(float(x)))
        pd_without_inconsistencies['Source size'] = pd_without_inconsistencies['Source'].apply(
            lambda x: len(literal_eval(x)))
        pd_without_inconsistencies['Sources'] = pd_without_inconsistencies['Source'].apply(
            lambda x: ','.join(literal_eval(x)))

        pd_without_inconsistencies.drop(columns='Source', inplace=True)

        return pd_without_inconsistencies
