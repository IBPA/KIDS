"""
Filename: create_kg.py

Authors:
    Minseung Kim - msgkim@ucdavis.edu
    Jason Youn -jyoun@ucdavis.edu

Description:
    Create the knowledge graph.

To-do:
"""

# standard imports
import argparse
import logging as log

# third party imports
import pandas as pd

# local imports
from integrate_modules.data_manager import DataManager
from integrate_modules.inconsistency_manager import InconsistencyManager
from integrate_modules.report_manager import plot_integration_summary
from tools.config_parser import ConfigParser
from tools.set_logging import set_logging

# default variables
DEFAULT_CONFIG_FILE = './configuration/update_kg_config.ini'
DEFAULT_LOG_LEVEL = 'DEBUG'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Integrate knowledgebase from multiple sources.')

    parser.add_argument(
        '--config_file',
        default=DEFAULT_CONFIG_FILE,
        help='INI configuration file location.')

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

    pd_kg = pd.read_csv(config_parser.getstr('input_kg'), sep='\t')
    pd_kg = pd_kg[['Subject', 'Predicate', 'Object']]

    pd_hypotheses = pd.read_csv(config_parser.getstr('validated_hypotheses'), sep='\t')
    pd_hypotheses['Predicate'] = pd_hypotheses['Resistance'].apply(
        lambda x: 'confers no resistance to antibiotic' if x == 'No' else 'confers resistance to antibiotic')
    pd_hypotheses = pd_hypotheses[['Subject', 'Predicate', 'Object']]

    pd_updated = pd.concat([pd_kg, pd_hypotheses])

    log.info('Saving updated knowledge graph to \'%s\'', config_parser.getstr('updated_kg'))
    pd_updated.to_csv(config_parser.getstr('updated_kg'), index=False, sep='\t')


if __name__ == '__main__':
    main()
