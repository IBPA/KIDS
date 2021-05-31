"""
Filename: build_network.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Build and run the ER MLP network.

To-do:
"""
# standard imports
import argparse
import os
import sys

DIRECTORY = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(DIRECTORY, '../er_mlp_imp'))
sys.path.insert(0, os.path.join(DIRECTORY, '../data_handler'))
sys.path.insert(0, os.path.join(DIRECTORY, '../../utils'))

# local imports
from config_parser import ConfigParser
import er_mlp_max_margin
from kids_log import set_logging


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Build, run, and test ER MLP.')

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

    parser.add_argument(
        '--final_model',
        default=False,
        action='store_true',
        help='Set when training the final model')

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

    params = {
        'word_embedding': configparser.getbool('word_embedding'),
        'training_epochs': configparser.getint('training_epochs'),
        'batch_size': configparser.getint('batch_size'),
        'display_step': configparser.getint('display_step'),
        'embedding_size': configparser.getint('embedding_size'),
        'layer_size': configparser.getint('layer_size'),
        'learning_rate': configparser.getfloat('learning_rate'),
        'corrupt_size': configparser.getint('corrupt_size'),
        'lambda': configparser.getfloat('lambda'),
        'optimizer': configparser.getint('optimizer'),
        'act_function': configparser.getint('act_function'),
        'add_layers': configparser.getint('add_layers'),
        'drop_out_percent': configparser.getfloat('drop_out_percent'),
        'data_path': configparser.getstr('data_path'),
        'save_model': configparser.getbool('save_model'),
        'model_save_directory': model_save_dir,
        'train_file': configparser.getstr('train_file'),
        'train_local_file': configparser.getstr('train_local_file'),
        'separator': configparser.getstr('separator'),
        'f1_for_threshold': configparser.getbool('f1_for_threshold'),
        'margin': configparser.getfloat('margin')
    }

    if not args.final_model:
        params['dev_file'] = configparser.getstr('dev_file')
        params['test_file'] = configparser.getstr('test_file')

    # run the model
    er_mlp_max_margin.run_model(params, final_model=args.final_model)


if __name__ == '__main__':
    main()
