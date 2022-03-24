"""
Filename: pooledinvestment.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu

Description:
    Resolve inconsistencies using Pooled Investment algorithm.

To-do:
"""
# standard imports
import logging as log
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), './'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import numpy as np

# local imports
from investment import Investment
from sums import Sums
from utilities import measure_accuracy


MAX_NUM_ITERATIONS = 8
SPO_LIST = ['Subject', 'Predicate', 'Object']
THRESHOLD = np.power(0.1, 10)


class PooledInvestment:
    """
    Inconsistency resolution using AverageLog.
    """
    @classmethod
    def resolve_inconsistencies(cls, pd_data, inconsistencies, answers=None, exponent=1.4):
        """
        Resolve any inconsistency using investment algorithm.

        Inputs:
            pd_data: integrated data that needs inconsistencies resolved
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
            exponent: g value that is used as an exponent (refer to the paper)

        Returns:
            inconsistencies_with_max_belief: dictionary where the key inconsistency_id
                and the value is a list of tuples where each tuple is of form
                (inconsistent_tuple, sources, belief). value of the dictionary is
                sorted by belief from high to low.
            pd_belief_and_source_without_inconsistencies: belief vector and
                pd_grouped_data concatenated but without the inconsistencies
            np_trustworthiness_vector: vector containing trustworthiness
                of all the sources
        """
        log.info('Resolving inconsistencies using Pooled Investment')

        # preprocess
        pd_source_size_data = pd_data.groupby('Source').size()
        pd_grouped_data = pd_data.groupby(SPO_LIST)['Source'].apply(set)

        # initialize
        np_present_belief_vector = Investment.initialize_belief(
            pd_source_size_data, pd_grouped_data, inconsistencies)
        np_past_trustworthiness_vector = Investment.initialize_trustworthiness(pd_source_size_data)
        np_default_a_matrix, np_b_matrix = Investment.create_matrices(
            pd_grouped_data, pd_source_size_data)

        delta = 1.0
        past_accuracy = 0.0
        iteration = 1

        # update until it reaches convergence
        while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
            np_a_matrix = Investment.update_a_matrix(
                np_default_a_matrix, np_past_trustworthiness_vector, pd_source_size_data)
            np_trustworthiness_vector = Sums.normalize(np_a_matrix.dot(np_present_belief_vector))
            claims = pd_grouped_data.index.tolist()
            np_present_belief_vector = Sums.normalize(
                cls.normalize(
                    np_b_matrix.dot(np_trustworthiness_vector),
                    claims,
                    inconsistencies,
                    exponent))
            delta = Sums.measure_trustworthiness_change(
                np_past_trustworthiness_vector, np_trustworthiness_vector)
            np_past_trustworthiness_vector = np_trustworthiness_vector

            if answers is not None:
                find_tuple_with_max_belief_result = Sums.find_tuple_with_max_belief(
                    np_present_belief_vector, inconsistencies, pd_grouped_data)

                inconsistencies_with_max_belief = find_tuple_with_max_belief_result[0]
                pd_belief_and_source_without_inconsistencies = find_tuple_with_max_belief_result[1]

                accuracy = measure_accuracy(inconsistencies_with_max_belief, answers, SPO_LIST)

                if past_accuracy == accuracy:
                    log.info('\taccuracy saturation %d %f %s', iteration, delta, accuracy)
                else:
                    log.info('\titeration, delta, accuracy : %d %f %s', iteration, delta, accuracy)
                past_accuracy = accuracy
            else:
                log.info('\titeration, delta : %d %f', iteration, delta)

            # update iteration
            iteration = iteration + 1

        find_tuple_with_max_belief_result = Sums.find_tuple_with_max_belief(
            np_present_belief_vector, inconsistencies, pd_grouped_data)

        inconsistencies_with_max_belief = find_tuple_with_max_belief_result[0]
        pd_belief_and_source_without_inconsistencies = find_tuple_with_max_belief_result[1]

        return inconsistencies_with_max_belief, pd_belief_and_source_without_inconsistencies, np_trustworthiness_vector

    @staticmethod
    def normalize(np_present_belief_vector, claims, inconsistencies, exponent):
        """
        Normalize the belief vector.

        Inputs:
            np_present_belief_vector: belief vector to normalize
            claims: all the claims (a.k.a. SPOs)
            inconsistencies: Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
            exponent: g value that is used as an exponent (refer to the paper)

        Returns:
            np_new_belief_vector: normalized belief vector
        """
        np_new_belief_vector = np_present_belief_vector.copy()

        for inconsistent_tuples in inconsistencies.values():
            total_score = 0

            for (inconsistent_tuple, _) in inconsistent_tuples:
                total_score = total_score + Investment.function_s(
                    np_present_belief_vector[claims.index(inconsistent_tuple)], exponent)

            for (inconsistent_tuple, _) in inconsistent_tuples:
                present_value = np_new_belief_vector[claims.index(inconsistent_tuple)]
                claim_spepcific_value = Investment.function_s(
                    np_present_belief_vector[claims.index(inconsistent_tuple)], exponent)
                np_new_belief_vector[claims.index(inconsistent_tuple)] = \
                    present_value * claim_spepcific_value / total_score

        return np_new_belief_vector
