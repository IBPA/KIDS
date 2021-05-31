"""
Filename: report_manager.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu

Description:
    Functions for generating a report.

To-do:
"""
# standard imports
import logging as log

# third party imports
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set fonts
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# constants
SPO_LIST = ['Subject', 'Predicate', 'Object']
COLUMN_NAMES = SPO_LIST + ['Source']


def plot_integration_summary(
        pd_data,
        np_trustworthiness_vector,
        inconsistencies,
        summary_plot_filepath):
    """
    Plot summary of data integration where x-axis is sources
    and y-axis is number of triplets in each source.

    Inputs:
        pd_data: (pd.DataFrame) Integrated data inlcuding the inconsistencies.
        np_trustworthiness_vector: (np.matrix) Trustworthiness of the sources.
        inconsistencies: (dict) Dictionary containing inconsistency_id as key
                and list of inconsistent triplets + source as value.
        summary_plot_filepath: (str) Filepath to save the integration summary plot.
    """
    pd_data_copy = pd_data.copy()
    pd_grouped_data = pd_data_copy.groupby(SPO_LIST)['Source'].apply(set)
    sources = pd_data.groupby('Source').size().index.tolist()

    list_trustworthiness_vector = \
        [float(trustworthiness) for trustworthiness in np_trustworthiness_vector]
    pd_trustworthiness = pd.Series(list_trustworthiness_vector, index=sources)

    pd_data_stat_column_names = ['Single source', 'Multiple sources', 'Inconsistencies']
    pd_data_stat = pd.DataFrame(index=sources, columns=pd_data_stat_column_names)

    def get_num_inconsistencies_of_source(source, inconsistencies):
        num_inconsistencies = 0

        for inconsistent_tuples in inconsistencies.values():
            for _, sources in inconsistent_tuples:
                if source in sources:
                    num_inconsistencies += 1

        return num_inconsistencies

    for source in sources:
        num_of_tuples_with_one_source = sum(pd_grouped_data == {source})
        num_of_rest = sum(pd_data_copy['Source'] == source) - num_of_tuples_with_one_source
        pd_data_stat.loc[source] = [
            num_of_tuples_with_one_source,
            num_of_rest,
            get_num_inconsistencies_of_source(source, inconsistencies)]

    # start the figure
    _, ax1 = plt.subplots()
    ax1.set_yscale("log")

    sorted_sources = pd_trustworthiness.sort_values(ascending=False).index.tolist()
    pd_data_stat = pd_data_stat.loc[sorted_sources]

    log.debug(pd_data_stat)
    log.debug(pd_trustworthiness)

    x = np.arange(len(sources))
    i = -1
    dimw = 0.5 / len(pd_data_stat_column_names)
    for column_name in pd_data_stat_column_names:
        ax1.bar(x + i * dimw, pd_data_stat[column_name], dimw, label=column_name, bottom=0.001)
        i += 1

    ax1.set_xlabel('Sources')
    ax1.set_ylabel('Number of triplets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_sources)

    ax1.legend()
    plt.savefig(summary_plot_filepath)
    plt.close()
