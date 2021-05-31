"""
Filename: determine_thresholds.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Determine the threshold.

To-do:
    1. why set label to 0 now instead of -1?
"""
# standard imports
import argparse
import logging as log
import os
import pickle
import sys

DIRECTORY = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(DIRECTORY, '../er_mlp_imp'))
sys.path.insert(0, os.path.join(DIRECTORY, '../data_handler'))
sys.path.insert(0, os.path.join(DIRECTORY, '../../utils'))

# third party imports
import tensorflow as tf

# local imports
from er_mlp import ERMLP
from data_processor import DataProcessor
from config_parser import ConfigParser
from kids_log import set_logging


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Determine the thresholds.')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        default='./',
        help='Base directory')

    parser.add_argument(
        '--logfile',
        default='',
        help='Path to save the log')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # set log and parse args
    args = parse_argument()
    set_logging(args.logfile)

    # directory and filename setup
    model_instance_dir = 'model_instance'
    model_save_dir = os.path.join(model_instance_dir, args.dir)
    config_file = os.path.join(model_save_dir, '{}.ini'.format(args.dir))

    # setup configuration parser
    configparser = ConfigParser(config_file)

    with tf.Session() as sess:
        # load the saved parameters
        with open(os.path.join(model_save_dir, 'params.pkl'), 'rb') as file:
            params = pickle.load(file)

        log.info('Original thresholds: %s', params['thresholds'])

        # some parameters
        entity_dic = params['entity_dic']
        pred_dic = params['pred_dic']

        # init ERMLP class
        er_mlp = ERMLP(
            {'num_preds': len(pred_dic)},
            sess,
            meta_graph=os.path.join(model_save_dir, 'model.meta'),
            model_restore=os.path.join(model_save_dir, 'model'))

        # load dev data for finding the threshold
        processor = DataProcessor()

        dev_df = processor.load(os.path.join(configparser.getstr('data_path'), 'dev.txt'))
        indexed_data_dev = processor.create_indexed_triplets_with_label(
            dev_df.values, entity_dic, pred_dic)

        # change label from -1 to 0
        indexed_data_dev[:, 3][indexed_data_dev[:, 3] == -1] = 0

        # find the threshold
        thresholds = er_mlp.determine_threshold(
            sess,
            indexed_data_dev,
            use_f1=configparser.getbool('f1_for_threshold'))

        log.debug('thresholds: %s', thresholds)

        # define what to save
        save_object = {
            'entity_dic': entity_dic,
            'pred_dic': pred_dic,
            'thresholds': thresholds
        }

        if hasattr(params, 'thresholds_calibrated'):
            save_object['thresholds_calibrated'] = params['thresholds_calibrated']

        if hasattr(params, 'calibrated_models'):
            save_object['calibrated_models'] = params['calibrated_models']

        if configparser.getbool('word_embedding'):
            save_object['indexed_entities'] = params['indexed_entities']
            save_object['indexed_predicates'] = params['indexed_predicates']
            save_object['num_pred_words'] = params['num_pred_words']
            save_object['num_entity_words'] = params['num_entity_words']

        # save the parameters
        with open(os.path.join(model_save_dir, 'params.pkl'), 'wb') as output:
            pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
