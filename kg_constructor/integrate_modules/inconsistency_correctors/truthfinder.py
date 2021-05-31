"""
Filename: truthfinder.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu

Description:
    Resolve inconsistencies using Truth Finder algorithm.

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
import pandas as pd

# local imports
from sums import Sums
from utilities import measure_accuracy

# constants
MAX_NUM_ITERATIONS = 10
SPO_LIST = ['Subject', 'Predicate', 'Object']
THRESHOLD = np.power(0.1, 10)


class TruthFinder():
    """
    Inconsistency resolution using TruthFinder.
    """
    @classmethod
    def resolve_inconsistencies(cls, pd_data, inconsistencies, answers=None, rho=0.5, gamma=0.3):
        """
        Resolve any inconsistency using Truth Finder algorithm.

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
            rho: hyperparameter (refer to the paper)
            gamma: hyperparameter (refer to the paper)

        Returns:
            inconsistencies_with_max_belief: dictionary where the key inconsistency_id
                and the value is a list of tuples where each tuple is of form
                (inconsistent_tuple, sources, belief). value of the dictionary is
                sorted by belief from high to low.
            pd_present_belief_and_source_without_inconsistencies: belief vector and
                pd_grouped_data concatenated but without the inconsistencies
            np_present_trustworthiness_vector: vector containing trustworthiness
                of all the sources
        """
        log.info('Resolving inconsistencies using Truth Finder')

        # preprocess
        pd_source_size_data = pd_data.groupby('Source').size()
        pd_grouped_data = pd_data.groupby(SPO_LIST)['Source'].apply(set)

        # initialize
        pd_present_belief_vector = cls.initialize_belief(pd_grouped_data)
        np_present_belief_vector = np.matrix(pd_present_belief_vector)
        np_past_trustworthiness_vector = cls.initialize_trustworthiness(pd_source_size_data)
        np_a_matrix, np_b_matrix = cls.create_matrices(
            pd_grouped_data, pd_source_size_data, inconsistencies, rho)

        function_s = np.vectorize(cls.function_s, otypes=[np.float])
        function_t = np.vectorize(cls.function_t, otypes=[np.float])

        delta = 1.0
        past_accuracy = 0.0
        iteration = 1

        # update until it reaches convergence
        while delta > THRESHOLD and iteration < MAX_NUM_ITERATIONS:
            np_present_belief_vector = function_s(
                np_b_matrix.dot(np_past_trustworthiness_vector), gamma)
            np_present_trustworthiness_vector = function_t(
                np_a_matrix.dot(np_present_belief_vector))
            delta = Sums.measure_trustworthiness_change(
                np_past_trustworthiness_vector, np_present_trustworthiness_vector)
            np_past_trustworthiness_vector = np_present_trustworthiness_vector

            if answers is not None:
                inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies = Sums.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)
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

        inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies = Sums.find_tuple_with_max_belief(np_present_belief_vector, inconsistencies, pd_grouped_data)

        return inconsistencies_with_max_belief, pd_present_belief_and_source_without_inconsistencies, np_present_trustworthiness_vector

    @staticmethod
    def initialize_trustworthiness(pd_source_size_data):
        """
        Initialize trustworthiness vector.

        Inputs:
            pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'

        Returns:
            Initialized trustworthiness vector.
        """
        num_of_sources = len(pd_source_size_data)
        np_trustworthiness_vector = np.full((num_of_sources, 1), -np.log(0.1))

        return np_trustworthiness_vector

    @staticmethod
    def initialize_belief(pd_grouped_data):
        """
        Initialize the belief vector prior.

        Inputs:
            pd_grouped_data: data grouped by 'pd_data.groupby(SPO_LIST)['Source'].apply(set)'
                with input shape (a, )

        Returns:
            numpy matrix of 0s of shape (a, 1)
        """
        return pd.DataFrame(pd_grouped_data.apply(lambda x: 0))

    @staticmethod
    def function_s(x, gamma=0.3):
        """
        Return 1 / (1 + np.exp(-gamma * x))

        Inputs:
            x: input value
            gamma: hyperparameter (refer to the paper)

        Returns:
            1 / (1 + np.exp(-gamma * x))
        """
        return 1 / (1 + np.exp(-gamma * x))

    @staticmethod
    def function_t(x):
        """
        Return - np.log(1 - x)

        Inputs:
            x: input value

        Returns:
            - np.log(1 - x)
        """
        return - np.log(1 - x)

    @staticmethod
    def modify_source_vector(elements, inconsistencies, rho=0.5):
        """
        Modify source vector if a source has conflicting belief.

        Inputs:
            elements: single element from pd_belief_source_matrix
            inconsistencies: Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
            rho: hyperparameter (refer to the paper)

        Returns:
            modified source vector
        """
        for idx in range(len(elements)):
            source_has_conflicting_belief = TruthFinder.source_has_conflicting_belief(
                elements.index[idx], elements.name, inconsistencies)
            if elements[idx] != 1 and source_has_conflicting_belief:
                elements[idx] = -rho

        return elements

    @staticmethod
    def create_matrices(pd_grouped_data, pd_source_size_data, inconsistencies, rho=0.5):
        """
        Create matrices to be used to compute the trustworthiness and belief vectors.

        Inputs:
            pd_grouped_data: data grouped by 'pd_data.groupby(SPO_LIST)['Source'].apply(set)'
                with input shape (a, )
            pd_source_size_data: data grouped by 'pd_data.groupby('Source').size()'
                with input shape (b, ) where b is number of sources
            inconsistencies: Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
            rho: hyperparameter (refer to the paper)

        Returns:
            np_a_matrix: collection of union of one-hot vectors of shape (b, a)
                transform belief to trustworthiness
            np_b_matrix: transpose of np_a_matrix
                transform trustworthiness to belief
        """
        sources = pd_source_size_data.index.tolist()

        pd_belief_source_matrix = pd_grouped_data.apply(Sums.create_source_vector, args=(sources,))
        np_belief_source_matrix = np.matrix(pd_belief_source_matrix.tolist())
        size_of_sources = np.array([pd_source_size_data[source] for source in sources])

        np_a_matrix = (np_belief_source_matrix / size_of_sources).T

        pd_belief_source_matrix = pd.DataFrame(
            np_belief_source_matrix, index=pd_grouped_data.index, columns=sources)
        np_b_matrix = pd_belief_source_matrix.apply(
            TruthFinder.modify_source_vector, args=(inconsistencies, rho)).as_matrix()

        return np_a_matrix, np_b_matrix

    @staticmethod
    def source_has_conflicting_belief(belief, source, inconsistencies):
        """
        Check if source has a conflicting belief.

        Inputs:
            belief:
            source: source to check if it has a conflicting belief
            inconsistencies: Dictionary containing inconsistency_id as key
                and list of inconsistent triples + source as value

                {
                0: [(('Subject 1', 'Predicate 1', 'Object 1'), ['Source 1']),
                    (('Subject 1', '!Predicate 1', 'Object 1'), ['Source 2'])],
                1: [(('Subject 2', 'Predicate 2', 'Object 2'), ['Source 1']),
                    ('Subject 2', '!Predicate 2', 'Object 2'), ['Sourcec 2'])],
                                        ...
                }
        """
        for inconsistent_tuples in inconsistencies.values():
            index_of_found_belief = -1

            for idx in range(len(inconsistent_tuples)):
                (inconsistent_tuple, sources) = inconsistent_tuples[idx]

                if tuple(belief) == tuple(inconsistent_tuple):
                    index_of_found_belief = idx
                    break

            if index_of_found_belief == -1:
                continue

            for idx in range(len(inconsistent_tuples)):
                (inconsistent_tuple, sources) = inconsistent_tuples[idx]

                if source in sources:
                    return True

        return False
