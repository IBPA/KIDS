"""
Filename: postprocess_data.py

Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Perform postprocessing on the integrated data so that
    it's suitable for use in hypothesis generator.

To-do:
"""
# standard imporots
import argparse
import logging as log

# local imports
from postprocess_modules.data_processor import DataProcessor
from postprocess_modules.distribute_data import DistributeData
from tools.config_parser import ConfigParser
from tools.set_logging import set_logging

# default variables
DEFAULT_CONFIG_FILE = './configuration/postprocess_config.ini'
DEFAULT_LOG_LEVEL = 'DEBUG'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Postprocess the integrated data.')

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

    # DataProcessor object
    data_processor = DataProcessor(
        config_parser.getstr('final_kg'),
        config_parser.getstr('label_rules'),
        config_parser.getstr('domain_range'))

    # reformat and save the data
    pd_with_label = data_processor.add_label()

    log.info('Saving reformatted data to \'%s\'', config_parser.getstr('all_data'))
    pd_with_label.to_csv(
        config_parser.getstr('all_data'),
        sep='\t', index=False, header=None)

    # get and save complete dictionary of entities indexed by its type
    entity_dic = data_processor.get_entity_dic(pd_with_label)

    data_processor.save_entities(
        entity_dic,
        config_parser.getstr('entities'),
        config_parser.getstr('entity_full_names'))

    # generate & save hypotheses
    pd_hypotheses = data_processor.generate_hypotheses(
        pd_with_label,
        entity_dic,
        [config_parser.getstr('hypothesis_relation')])

    log.info('Saving hypotheses to \'%s\'', config_parser.getstr('hypotheses'))
    pd_hypotheses.to_csv(
        config_parser.getstr('hypotheses'),
        sep='\t', index=False, header=None)

    # split the dataset into specified folds
    distribute_data = DistributeData(
        pd_with_label,
        config_parser.getint('num_folds'),
        config_parser.getint('num_negatives'),
        config_parser.getstr('hypothesis_relation'),
        entity_dic['gene'])

    data_split_fold_dic = distribute_data.split_into_folds()

    distribute_data.save_folds(
        data_split_fold_dic,
        config_parser.getstr('output_path'))

    # save data to use for training the final model
    distribute_data.save_train_data_for_final_model(config_parser.getstr('output_path'))


if __name__ == '__main__':
    main()
