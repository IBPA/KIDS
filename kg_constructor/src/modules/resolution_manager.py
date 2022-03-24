from ast import literal_eval
import os
import sys
import logging as log
sys.path.append("..")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .inconsistency_correctors.averagelog import AverageLog  # noqa: E402
from .inconsistency_correctors.investment import Investment  # noqa: E402
from .inconsistency_correctors.pooledinvestment import PooledInvestment  # noqa: E402
from .inconsistency_correctors.sums import Sums  # noqa: E402
from .inconsistency_correctors.truthfinder import TruthFinder  # noqa: E402
from .inconsistency_correctors.voting import Voting  # noqa: E402


class ResolutionManager():
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

    def __init__(
            self,
            data_dir: str,
            inconsistency_rules_filename: str = 'inconsistency_rules.txt',
            resolution_method: str = 'averagelog'):
        """
        Class constructor for ResolutionManager.

        Inputs:
            inconsistency_rule_file: (str) XML filepath containing the inconsistency rules.
            resolution_method: (str) Inconsistency resolution mode.
                (AverageLog | Investment | PooledInvestment | Sums | TruthFinder | Voting)
        """
        self.data_dir = data_dir
        self.inconsistency_rules_filename = inconsistency_rules_filename
        self.resolution_method = resolution_method.lower()

    def detect_and_resolve(self, df):
        inconsistencies = self._detect(df)
        result = self._resolve(df, inconsistencies)

        return inconsistencies, result

    def _detect(self, df):
        """
        Detect the inconsistencies among the data using the provided rule file.

        Inputs:
            df: (pd.DataFrame) Data to detect inconsistency from

        Returns:
            inconsistencies: (dict) Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value.

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
        """
        df_copy = df.copy().reset_index(drop=True)

        df_rules = pd.read_csv(
            os.path.join(self.data_dir, self.inconsistency_rules_filename),
            sep='\t')

        count = 0
        inconsistencies = {}

        for idx, rule in df_rules.iterrows():
            log.info(f'Detecting inconsistencies using the following rule:\n{rule}')

            where = rule['where']
            value_1 = rule['value_1']
            value_2 = rule['value_2']

            df_match_value_1 = df_copy[df_copy[where] == value_1]
            df_match_value_2 = df_copy[df_copy[where] == value_2]

            if df_match_value_1.shape[0] == 0 or df_match_value_2.shape[0] == 0:
                log.warning('Did not find any inconsistencies using this rule!')
                continue

            df_candidates = pd.concat([df_match_value_1, df_match_value_2])
            df_candidates.reset_index(inplace=True)
            df_grouped = df_candidates.groupby([x for x in self._SPO_LIST if x != where]).agg(
                {where: set, 'index': set}).reset_index()
            df_grouped_inconsistencies = df_grouped[
                df_grouped['Predicate'].apply(lambda x: len(set(x))) > 1]

            log.debug(f'df_grouped_inconsistencies:\n{df_grouped_inconsistencies}')

            inconsistencies_index = df_grouped_inconsistencies['index'].tolist()
            num_inconsistency_sets = len(inconsistencies_index)

            log.info(f'Found {num_inconsistency_sets} sets of inconsistencies. Creating their list now.')

            for i, idx in enumerate(inconsistencies_index):
                if num_inconsistency_sets < 10:
                    log_every = 1
                else:
                    log_every = int(num_inconsistency_sets/10)

                if i % log_every == 0:
                    log.debug(f'Processing {i+1}/{num_inconsistency_sets}...')

                df_inconsistency = df_copy.iloc[list(idx)]
                df_grouped_inconsistency = df_inconsistency.groupby(self._SPO_LIST).agg(
                    {'Source': list}).reset_index()

                assert df_grouped_inconsistency.shape[0] == 2, \
                    'More than two opposing sides of inconsistencies!'

                first_triple = tuple(df_grouped_inconsistency.loc[0, self._SPO_LIST])
                first_sources = df_grouped_inconsistency.loc[0, 'Source']
                second_triple = tuple(df_grouped_inconsistency.loc[1, self._SPO_LIST])
                second_sources = df_grouped_inconsistency.loc[1, 'Source']
                inconsistencies[count] = [(first_triple, first_sources), (second_triple, second_sources)]
                count += 1

        log.info('Found %d inconsistencies', len(inconsistencies))

        return inconsistencies

    def _resolve(self, df, inconsistencies, **kwargs):
        """
        Wrapper function for inconsistency resolver.

        Inputs:
            df: (pd.DataFrame) Data that has inconsistencies to resolve.
            inconsistencies: (dict) Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value.
            kwargs: Arguments specific to algorithm to use.
                    (refer to each resolution algorithm)

        Returns:
            df_resolved_inconsistencies: (pd.DataFrame) Resolved inconsistencies.
            df_without_inconsistencies: (pd.DataFrame) Clean data without inconsistencies.
            np_trustworthiness_vector: (np.matrix) Trustworthiness of all the sources.
        """
        if self.resolution_method == 'averagelog':
            result = AverageLog.resolve_inconsistencies(df, inconsistencies, **kwargs)
        elif self.resolution_method == 'investment':
            result = Investment.resolve_inconsistencies(df, inconsistencies, **kwargs)
        elif self.resolution_method == 'pooledinvestment':
            result = PooledInvestment.resolve_inconsistencies(df, inconsistencies, **kwargs)
        elif self.resolution_method == 'sums':
            result = Sums.resolve_inconsistencies(df, inconsistencies, **kwargs)
        elif self.resolution_method == 'truthfinder':
            result = TruthFinder.resolve_inconsistencies(df, inconsistencies, **kwargs)
        elif self.resolution_method == 'voting':
            result = Voting.resolve_inconsistencies(df, inconsistencies, **kwargs)
        else:
            raise ValueError(f'Invalid inconsistency corrector \'{self.resolution_method}\'chosen.')

        df_resolved_inconsistencies = self._inconsistencies_dict_to_df(result[0])
        df_without_inconsistencies = self._without_inconsistencies_reformat(result[1])
        np_trustworthiness_vector = result[2]

        return (df_resolved_inconsistencies, df_without_inconsistencies, np_trustworthiness_vector)

    @classmethod
    def _inconsistencies_dict_to_df(cls, resolved_inconsistencies_dict):
        """
        (Private) Convert inconsistencies from dictionary to pandas DataFrame.

        Inputs:
            resolved_inconsistencies_dict: (dict) Dictionary where
                the key is inconsistency_id and the value is a list of tuples
                where each tuple is of form (inconsistent_tuple, sources, belief).
                Value of the dictionary is sorted by belief from high to low.

        Returns:
            df_resolved_inconsistencies: (pd.DataFrame) Resolved inconsistencies
                converted to DataFrame format.
        """
        log.info('Converting resolved inconsistencies from dictionary to pandas DataFrame.')

        df_resolved_inconsistencies = pd.DataFrame(columns=cls._RESOLVED_INCONSISTENCIES_COLUMNS)

        resolved_inconsistencies_list = []
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

            resolved_inconsistencies_list.append(pd.DataFrame.from_dict([row_dict]))

        df_resolved_inconsistencies = pd.concat(resolved_inconsistencies_list)

        return df_resolved_inconsistencies

    @staticmethod
    def _without_inconsistencies_reformat(df):
        """
        (Private) Reformat data without inconsistencies into desired format.

        Inputs:
            df: (pd.DataFrame) Data needing reformat.

        Returns:
            df_without_inconsistencies: (pd.DataFrame) Reformatted data.
        """
        log.info('Reformating data without inconsistencies.')

        df_without_inconsistencies = df.reset_index()
        df_without_inconsistencies = df_without_inconsistencies.rename(
            columns={0: 'Belief'}).astype('str')
        df_without_inconsistencies['Belief'] = df_without_inconsistencies['Belief'].apply(
            lambda x: '{0:.2f}'.format(float(x)))
        df_without_inconsistencies['Source size'] = df_without_inconsistencies['Source'].apply(
            lambda x: len(literal_eval(x)))
        df_without_inconsistencies['Sources'] = df_without_inconsistencies['Source'].apply(
            lambda x: ','.join(literal_eval(x)))

        df_without_inconsistencies.drop(columns='Source', inplace=True)

        return df_without_inconsistencies
