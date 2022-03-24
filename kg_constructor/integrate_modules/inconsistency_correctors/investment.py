"""
Filename: investment.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu

Description:
    Resolve inconsistencies using investment algorithm.

To-do:
    1. Change np.matrix into np.array for future compatibility.
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
from sums import Sums
from utilities import measure_accuracy

MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject', 'Predicate', 'Object']
THRESHOLD = np.power(0.1, 10)


class Investment:
    """
    Inconsistency resolution using AverageLog.
    """
    @classmethod
    def resolve_inconsistencies(cls, pd_data, inconsistencies, answers=None, exponent=1.2):
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
        log.info('Resolving inconsistencies using Investment')

        # preprocess
        pd_source_size_data = pd_data.groupby('Source').size()
        pd_grouped_data = pd_data.groupby(SPO_LIST)['Source'].apply(set)

        # initialize
        np_present_belief_vector = Sums.normalize(
            cls.initialize_belief(pd_source_size_data, pd_grouped_data, inconsistencies))
        np_past_trustworthiness_vector = cls.initialize_trustworthiness(pd_source_size_data)
        np_default_a_matrix, np_b_matrix = cls.create_matrices(pd_grouped_data, pd_source_size_data)

        function_s = np.vectorize(cls.function_s, otypes=[np.float])

        delta = 1.0
        past_accuracy = 0.0
        iteration = 1

        # update until it reaches convergence
        while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
            np_a_matrix = cls.update_a_matrix(
                np_default_a_matrix, np_past_trustworthiness_vector, pd_source_size_data)
            np_trustworthiness_vector = Sums.normalize(np_a_matrix.dot(np_present_belief_vector))
            np_present_belief_vector = Sums.normalize(
                function_s(np_b_matrix.dot(np_trustworthiness_vector), exponent))
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
    def function_s(x, exponent):
        """
        Return x^exponent.

        Inputs:
            x: input value
            exponent: value of the exponent

        Returns:
            x^exponent
        """
        return np.power(x, exponent)

    @staticmethod
    def create_matrices(pd_grouped_data, pd_source_size_data):
        """
        Create matrices to be used to compute the trustworthiness and belief vectors.

        Inputs:
            pd_grouped_data: data grouped by 'pd_data.groupby(SPO_LIST)['Source'].apply(set)'
                with input shape (a, )
            pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'
                with input shape (b, ) where b is number of sources

        Returns:
            np_a_matrix: collection of union of one-hot vectors of shape (b, a)
                transform belief to trustworthiness
            np_b_matrix: transpose of np_a_matrix
                transform trustworthiness to belief
        """
        sources = pd_source_size_data.index.tolist()
        pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args=(sources,))
        np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist())

        np_a_matrix = np_belief_source_matrix.T
        np_b_matrix = np_belief_source_matrix / np.array(pd_source_size_data)

        return np_a_matrix, np_b_matrix

    @staticmethod
    def update_a_matrix(np_a_matrix, np_past_trustworthiness_vector, pd_source_size_data):
        """
        Update the a_matrix which is used to transform belief to trustworthiness.

        Inputs:
            np_a_matrix: matrix to transform belief to trustworthiness
            np_past_trustworthiness_vector: previous iteration trustworthiness vector
            pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'

        Returns:
            updated a_matrix which will be used to transform belief to trustworthiness
        """
        np_a_matrix = (np_a_matrix.T / (np.array(pd_source_size_data) /
                       np_past_trustworthiness_vector.T)).T

        return np_a_matrix / np_a_matrix.sum(axis=0)

    @staticmethod
    def initialize_belief(pd_source_size_data, pd_grouped_data, inconsistencies):
        """
        Initialize the belief vector prior.

        Inputs:
            pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'
            pd_grouped_data: data grouped by 'pd_data.groupby(SPO_LIST)['Source'].apply(set)'
                with input shape (a, )
            inconsistencies: Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }

        Returns:
            numpy matrix of shape (a, 1)
        """
        pd_present_belief_vector = pd_grouped_data.apply(lambda x: 1)

        # we only need to change claim that has inconsistency.
        for inconsistent_tuples in inconsistencies.values():
            total_source_size = Investment.get_total_source_size_of_inconsistent_tuples(
                inconsistent_tuples, pd_source_size_data)

            for (inconsistent_tuple, sources) in inconsistent_tuples:
                source_size = sum([pd_source_size_data[source] for source in sources])
                pd_present_belief_vector.loc[inconsistent_tuple] = \
                    float(source_size) / float(total_source_size)

        return np.matrix(pd_present_belief_vector).T

    @staticmethod
    def get_total_source_size_of_inconsistent_tuples(inconsistent_tuples, pd_source_size_data):
        """
        Get total source size of inconsistent tuples.

        Inputs:
            inconsistent_tuples: list of tuples where each tuple is of form
                (inconsistent_tuple, sources, belief)
            pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'
        """
        total_source_size = 0

        for (_, sources) in inconsistent_tuples:
            for source in sources:
                total_source_size = total_source_size + pd_source_size_data[source]

        return total_source_size

    @staticmethod
    def initialize_trustworthiness(pd_source_size_data):
        """
        Initialize trustworthiness vector.

        Inputs:
            pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'

        Returns:
            Initialized trustworthiness vector all set to 1s.
        """
        return np.matrix(pd_source_size_data.apply(lambda x: 1)).T
