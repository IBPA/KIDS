"""
Filename: utils.py

Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Contains utility functions.

To-do:
"""
# standard imports
import logging as log
import os
import pickle


def create_dir(directory):
    """
    Create directory only if directory does not exist already..

    Inputs:
        directory: directory to create
    """
    if os.path.isdir(directory):
        log.warning('Directory \'%s\' already exists!', directory)
    else:
        log.info('Creating directory: %s', directory)
        os.makedirs(directory)


def load_pickle(filepath_str):
    """
    Load pickled results.

    Inputs:
        filepath_str: path to the pickle file to load

    Returns:
        loaded pickle file
    """
    log.info('Loading pickle file from \'%s\'', filepath_str)
    with open(filepath_str, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def save_to_text_file(results, directory):
    """
    Save the results to the text file.

    Inputs:
        results: same as that in save_results()
        directory: that in save_results() + 'results'
    """
    with open(os.path.join(directory, 'results.txt'), 'w') as outfile:
        outfile.write('Overall metrics: \n\n')

        for metric, value in results['overall'].items():
            outfile.write('{}: {}\n'.format(metric, value))

        outfile.write('----------------------------------\n')
        outfile.write('Predicates\n\n')

        for predicate, metrics in results['predicate'].items():
            outfile.write('metrics for {}: \n'.format(predicate))

            for metric, value in metrics.items():
                outfile.write('{}: {}\n'.format(metric, value))

            outfile.write('----------------------------------\n')


def save_results(results, directory):
    """
    Save evaluation results to the specified directory.

    Inputs:
        results: python dictionary where there are keys like
            results['overall'], results['predicate'][pred_name]
            and each key again has a value of dictionary
            containing different evaluation results.
        directory: folder to save the results
    """
    directory = os.path.join(directory, 'results')
    create_dir(directory)

    filepath = os.path.join(directory, 'results.pkl')
    log.info('Saving results to \'%s\'', filepath)

    with open(filepath, 'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

    save_to_text_file(results, directory)
