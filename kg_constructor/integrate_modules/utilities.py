"""
Filename: utilities.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Collection of utility functions specific to integrade modules.

To-do:
"""
# standard imports
import logging as log

# third party imports
import pandas as pd


def get_pd_of_statement(statement):
    """
    Get names and values of the statement.

    Inputs:
        statement: (xml.etree.ElementTree.Element) XML node containing the statement to process.

    Returns:
        (pd.Series) All found values.
    """
    feature_names = [feature.get('name') for feature in statement]
    feature_values = [feature.get('value') for feature in statement]

    return pd.Series(feature_values, index=feature_names)


def measure_accuracy(resolved_inconsistencies, answers, spo_list, iteration=0):
    correctly_resolved_inconsistencies = 0.0
    total_attempted_resolution = 0.0

    for inconsistency_id in resolved_inconsistencies:
        resolved_inconsistency = resolved_inconsistencies[inconsistency_id]
        resolved_tuple = resolved_inconsistency[0][0]
        resolved_belief = resolved_inconsistency[0][2]
        conflict_tuple = resolved_inconsistency[1][0]
        conflict_belief = resolved_inconsistency[1][2]

        pd_resolved_inconsistency = pd.Series(resolved_tuple, index=spo_list)
        pd_conflict_inconsistency = pd.Series(conflict_tuple, index=spo_list)

        if (pd_resolved_inconsistency == answers[spo_list]).all(1).any():
            correctly_resolved_inconsistencies = correctly_resolved_inconsistencies + 1
            total_attempted_resolution = total_attempted_resolution + 1
            log.debug('%d\tTRUE\t%s\t%s', iteration, resolved_belief, conflict_belief)
        elif (pd_conflict_inconsistency == answers[spo_list]).all(1).any():
            total_attempted_resolution = total_attempted_resolution + 1
            log.debug('%d\tFALSE\t%s\t%s', iteration, resolved_belief, conflict_belief)

    log.debug('%d %d', correctly_resolved_inconsistencies, total_attempted_resolution)

    if float(total_attempted_resolution) == 0:
        accuracy = 0
    else:
        accuracy = float(correctly_resolved_inconsistencies) / float(total_attempted_resolution)

    return "{0:.4f}".format(accuracy)


def measure_trustworthiness(pd_data, answers, spo_list):
    sources = pd.unique(pd_data['Source']).tolist()
    pd_trustworthiness = pd.Series(index=sources)

    for source in sources:
        source_claims = pd_data[pd_data['Source'] == source][spo_list]
        common = source_claims.merge(answers, on=spo_list)
        pd_trustworthiness[source] = float(common.shape[0]) / float(source_claims.shape[0])

    return pd_trustworthiness
