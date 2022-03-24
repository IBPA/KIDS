"""
Filename: analyze_dataset.py

Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Analyze the contents of the dataset.

To-do:
"""

# standard imports
import argparse
import logging as log
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# third party imports
import pandas as pd

# local imports
from postprocess_modules.data_processor import DataProcessor
from tools.config_parser import ConfigParser
from tools.set_logging import set_logging

# default variables
DEFAULT_CONFIG_FILE = '../../configuration/postprocess_config.ini'
DEFAULT_LOG_LEVEL = 'DEBUG'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze the contents of the dataset.')

    parser.add_argument(
        '--config_file',
        default=DEFAULT_CONFIG_FILE,
        help='INI configuration file location.')

    parser.add_argument(
        '--data_filepath',
        required=True,
        help='Filepath of the data to analyze.')

    parser.add_argument(
        '--log_level',
        default=DEFAULT_LOG_LEVEL,
        help='Set log level (DEBUG | INFO | WARNING | ERROR).')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # set log and parse args
    args = parse_argument()
    set_logging(log_level=args.log_level)

    # setup config parser
    config_parser = ConfigParser(args.config_file)

    # use DataProcessor to get the stats
    data_processor = DataProcessor(
        args.data_filepath,
        config_parser.getstr('label_rules'),
        config_parser.getstr('domain_range'))

    # reformat and save the data
    pd_data = data_processor.add_label()

    # print the stats
    log.info('Total number of triplets: {}'.format(pd_data.shape[0]))
    data_processor.get_entity_dic(pd_data)

if __name__ == '__main__':
    main()
