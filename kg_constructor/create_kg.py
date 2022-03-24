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
DEFAULT_CONFIG_FILE = './configuration/create_kg_config.ini'
DEFAULT_PHASE_STR = 'all'
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
        '--phase',
        default=DEFAULT_PHASE_STR,
        help='Select phase to run (phase1 | phase2 | all).')

    parser.add_argument(
        '--log_level',
        default=DEFAULT_LOG_LEVEL,
        help='Set log level (DEBUG | INFO | WARNING | ERROR).')

    # check for correct phase argument
    phase_arg = parser.parse_args().phase
    if phase_arg not in ['phase1', 'phase2', 'all']:
        raise ValueError('Invalid phase argumnet \'{}\'!'.format(phase_arg))

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

    # construct DataManager class
    data_manager = DataManager(
        config_parser.getstr('data_path'),
        config_parser.getstr('name_map'),
        config_parser.getstr('data_rule'),
        config_parser.getstr('replace_rule'))

    # construct InconsistencyManager class
    inconsistency_manager = InconsistencyManager(
        config_parser.getstr('inconsistency_rules'),
        resolver_mode=config_parser.getstr('resolver_mode'))

    if args.phase in ['phase1', 'all']:
        # perform knowledge integration
        pd_data = data_manager.integrate()

        # perform name mapping
        pd_data = data_manager.map_name(pd_data)

        # perform knowledge inferral
        pd_data = data_manager.infer(pd_data)

        # replace some parts of the data (currently used to drop temporal data)
        pd_data = data_manager.replace(pd_data)

        # perform inconsistency detection
        inconsistencies = inconsistency_manager.detect(pd_data)

        # perform inconsistency resolution and parse the results
        resolution_result = inconsistency_manager.resolve(pd_data, inconsistencies)
        pd_resolved_inconsistencies = resolution_result[0]
        pd_without_inconsistencies = resolution_result[1]
        np_trustworthiness_vector = resolution_result[2]

        # report summary of integration
        plot_integration_summary(
            pd_data,
            np_trustworthiness_vector,
            inconsistencies,
            config_parser.getstr('integration_summary_plot'))

        # save results
        log.info('Saving knowledge graph without inconsistencies to \'%s\'',
                 config_parser.getstr('without_inconsistsencies'))
        pd_without_inconsistencies.to_csv(
            config_parser.getstr('without_inconsistsencies'),
            index=False,
            sep='\t')

        log.info('Saving resolved inconsistencies to \'%s\'',
                 config_parser.getstr('resolved_inconsistencies'))
        pd_resolved_inconsistencies.to_csv(
            config_parser.getstr('resolved_inconsistencies'),
            index=False,
            sep='\t')

    if args.phase in ['phase2', 'all']:
        # read the files saved in phase1
        pd_without_inconsistencies = pd.read_csv(
            config_parser.getstr('without_inconsistsencies'), sep='\t')
        pd_resolved = pd.read_csv(
            config_parser.getstr('resolved_inconsistencies'), sep='\t')

        # read validation results
        pd_validated = pd.read_csv(
            config_parser.getstr('validation_results'), sep='\t')

        # compare computational resolution results with validation results
        pd_resolved_and_validated = inconsistency_manager.compare_resolution_with_validation(
            pd_resolved,
            pd_validated)

        # save resolution validation comparison result
        log.info('Saving resolved and validated inconsistencies to \'%s\'',
                 config_parser.getstr('validated_inconsistencies'))
        pd_resolved_and_validated.to_csv(
            config_parser.getstr('validated_inconsistencies'),
            index=False,
            sep='\t')

        # insert resolved and validated inconsistsencies back into the KG
        pd_final = inconsistency_manager.reinstate_resolved_and_validated(
            pd_without_inconsistencies,
            pd_resolved_and_validated,
            mode='validated_and_rest_as_positives')

        # save integrated data
        log.info('Saving final knowledge graph to \'%s\'', config_parser.getstr('final_kg'))
        pd_final.to_csv(config_parser.getstr('final_kg'), index=False, sep='\t')


if __name__ == '__main__':
    main()
