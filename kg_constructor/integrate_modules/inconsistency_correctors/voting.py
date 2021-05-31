"""
Filename: voting.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Resolve inconsistencies using voting algorithm.

To-do:
"""
# standard imports
import logging as log
import operator

# third party imports
import numpy as np
import pandas as pd


class Voting:
    """
    Inconsistency resolution using Sums.
    """
    @classmethod
    def resolve_inconsistencies(cls, data, inconsistencies, answers=None):
        """
        Resolve any inconsistency using voting algorithm.

        Inputs:
            data: integrated data that needs inconsistencies resolved
            inconsistencies: Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
            answers:

        Returns:
            inconsistent_tuples_with_max_occurrence: dictionary where the key inconsistency_id
                and the value is a list of tuples where each tuple is of form
                (inconsistent_tuple, max_occurrence)
            np_present_trustworthiness_vector: vector containing trustworthiness
                of all the sources
        """
        log.info('Resolving inconsistencies using Voting')

        np_present_trustworthiness_vector = np.array(pd.Series(data.groupby('Source').size()))
        inconsistent_tuples_with_max_occurrence = {}

        for inconsistency_id in inconsistencies:
            inconsistent_tuples = inconsistencies[inconsistency_id]
            occurrences = {inconsistent_tuple: len(sources)
                           for inconsistent_tuple, sources in inconsistent_tuples}
            inconsistent_tuple, max_occurrence = max(
                occurrences.items(), key=operator.itemgetter(1))
            inconsistent_tuples_with_max_occurrence[inconsistency_id] = \
                [(inconsistent_tuple, max_occurrence), ('dummy',)]

        return inconsistent_tuples_with_max_occurrence, None, np_present_trustworthiness_vector
