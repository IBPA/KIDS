"""
Filename: build_model.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Build stacked ensemble using AdaBoost.

To-do:
    1. move number of fold for final model to config.
"""
# standard imports
import argparse
import os
import pickle
import sys

DIRECTORY = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(DIRECTORY, '../utils'))

# third party imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.calibration import CalibratedClassifierCV

# local imports
from config_parser import ConfigParser
from data_processor import DataProcessor
from utils import create_dir

RANDOM_STATE = 0


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='build stacked ensemble')

    parser.add_argument(
        '--pra',
        metavar='pra_model (pra_model_2)',
        nargs='?',
        action='store',
        required=True,
        help='The pra models to add')

    parser.add_argument(
        '--er_mlp',
        metavar='er_mlp_model (er_mlp_model_2)',
        nargs='?',
        action='store',
        required=True,
        help='The er-mlp models to add')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        action='store',
        required=True,
        help='directory to store the model')

    parser.add_argument(
        '--final_model',
        default=False,
        action='store_true',
        help='Set when training the final model')

    return parser.parse_args()


def get_results_of_search(results, count=5):
    # f1
    print("F1\n")
    for idx in range(1, count + 1):
        runs = np.flatnonzero(results['rank_test_f1'] == idx)
        for run in runs:
            print("evaluation rank: {}".format(idx))
            print("score: {}".format(results['mean_test_f1'][run]))
            print("std: {}".format(results['std_test_f1'][run]))
            print(results['params'][run])
            print("")
    print("")

    # average precision
    print("average precision\n")
    for idx in range(1, count + 1):
        runs = np.flatnonzero(results['rank_test_average_precision'] == idx)
        for run in runs:
            print("evaluation rank: {}".format(idx))
            print("score: {}".format(results['mean_test_average_precision'][run]))
            print("std: {}".format(results['std_test_average_precision'][run]))
            print(results['params'][run])
            print("")

            # use average precision for reporting the results
            if idx == 1:
                ap_params = results['params'][run]
    print("")

    return ap_params


def randomized_search(configparser, train_x, train_y, test_x=None, test_y=None, final_model=False):
    """
    Given the train & test set, perform randomized search to find
    best n_estimators & learning_rate for the adaboost algorithm.

    Inputs:
        train_x: numpy array where
            train_x[:, 0] = er_mlp prediction raw output
            train_x[:, 1] = pra prediction raw output
            train_x[:, 2] = valid / invalid depending on pra
        train_y: numpy array containing the ground truth label
        test_x: (optional) same as train_x but for test data
        test_y: (optional) same as train_y but for test data

    Returns:
        dictionary of numpy (masked) ndarrays
        containing the search results
    """
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=RANDOM_STATE)

    param_distribution = {
        'adaboost__learning_rate': np.arange(
            float(configparser.getfloat('start', 'RANDOM_SEARCH_LEARNING_RATE')),
            float(configparser.getfloat('end', 'RANDOM_SEARCH_LEARNING_RATE')),
            float(configparser.getfloat('increment', 'RANDOM_SEARCH_LEARNING_RATE'))),

        'adaboost__n_estimators': np.arange(
            int(configparser.getint('start', section='RANDOM_SEARCH_ESTIMATORS')),
            int(configparser.getint('end', section='RANDOM_SEARCH_ESTIMATORS')),
            int(configparser.getint('increment', section='RANDOM_SEARCH_ESTIMATORS')))}

    # pipeline the model with SMOTE
    ros = SMOTE(sampling_strategy='minority')
    clf = Pipeline([('smote', ros), ('adaboost', clf)])

    if not final_model:
        all_x = np.vstack((train_x, test_x))
        all_y = np.vstack((train_y, test_y)).astype(int)

        # get train / test split indices for predefined split cross-validator
        train_indices = np.full(np.shape(train_x)[0], -1)
        test_indices = np.full(np.shape(test_x)[0], 0)
        test_fold = np.hstack((train_indices, test_indices))

        cv = PredefinedSplit(test_fold)
    else:
        all_x = train_x
        all_y = train_y.astype(int)

        cv = 5

    random_search = RandomizedSearchCV(
        clf,
        param_distributions=param_distribution,
        n_iter=configparser.getint('random_search_count'),
        n_jobs=configparser.getint('random_search_processes'),
        scoring=['f1', 'average_precision'],
        cv=cv,
        refit='average_precision')

    random_search.fit(all_x, all_y.ravel())

    return get_results_of_search(random_search.cv_results_)


def main():
    """
    Main function.
    """
    args = parse_argument()

    # paths
    model_instance_dir = 'model_instance'
    model_save_dir = os.path.join(model_instance_dir, args.dir)
    config_file = os.path.join(model_save_dir, '{}.ini'.format(args.dir))

    # setup configuration parser
    configparser = ConfigParser(config_file)

    # load data
    dp_train = DataProcessor(
        configparser.getstr('train_dir'),
        configparser.getstr('er_mlp_model_dir'),
        configparser.getstr('pra_model_dir'))

    pred_dic = dp_train.get_pred_dic()
    train_x, train_y, predicates_train = dp_train.get_x_y()

    if not args.final_model:
        dp_dev = DataProcessor(
            configparser.getstr('dev_dir'),
            configparser.getstr('er_mlp_model_dir'),
            configparser.getstr('pra_model_dir'))

        pred_dic = dp_dev.get_pred_dic()
        test_x, test_y, predicates_test = dp_dev.get_x_y()

    # prediction results of adaboosst
    predictions_train = np.zeros_like(predicates_train, dtype=float)
    predictions_train = predictions_train.reshape((np.shape(predictions_train)[0], 1))

    model_dic = {}

    for key, idx in pred_dic.items():
        if not args.final_model:
            test_indices, = np.where(predicates_test == idx)
            if np.shape(test_indices)[0] != 0:
                test_x_pred = test_x[test_indices]
                test_y_pred = test_y[test_indices]

        train_indices, = np.where(predicates_train == idx)
        if np.shape(train_indices)[0] != 0:
            train_x_pred = train_x[train_indices]
            train_y_pred = train_y[train_indices]
        else:
            print('No training data for predicate: {}'.format(key))
            continue

        if configparser.getbool('run_random_search'):
            if not args.final_model:
                ap_params = randomized_search(
                    configparser,
                    train_x_pred,
                    train_y_pred,
                    test_x_pred,
                    test_y_pred)
            else:
                ap_params = randomized_search(
                    configparser,
                    train_x_pred,
                    train_y_pred,
                    final_model=True)

            configparser.append(
                'RANDOM_SEARCH_BEST_PARAMS_{}'.format(key),
                {'n_estimators': ap_params['adaboost__n_estimators'],
                 'learning_rate': ap_params['adaboost__learning_rate']})

        # build & fit model using the best parameters
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1),
            n_estimators=configparser.getint('n_estimators', section='RANDOM_SEARCH_BEST_PARAMS_{}'.format(key)),
            learning_rate=configparser.getfloat('learning_rate', section='RANDOM_SEARCH_BEST_PARAMS_{}'.format(key)),
            random_state=RANDOM_STATE)

        ros = SMOTE(sampling_strategy='minority')
        pipepline = Pipeline([('smote', ros), ('adaboost', clf)])

        if args.final_model:
            final_clf = CalibratedClassifierCV(pipepline, method='isotonic', cv=5)
        else:
            final_clf = pipepline

        final_clf.fit(train_x_pred, train_y_pred.ravel())
        model_dic[idx] = final_clf

        # do prediction on the train set
        probs = final_clf.predict_proba(train_x[train_indices])[:, 1]
        probs = np.reshape(probs, (-1, 1))
        predictions_train[train_indices] = probs[:]

    with open(os.path.join(model_save_dir, 'model.pkl'), 'wb') as output:
        pickle.dump(model_dic, output, pickle.HIGHEST_PROTOCOL)

    with open('./' + config_file, 'w') as configfile:
        configparser.write(configfile)

    directory = os.path.join(model_save_dir, 'train_local')
    create_dir(directory)

    with open(os.path.join(directory, 'confidence.txt'), 'w') as t_f:
        for row in predictions_train:
            t_f.write(str(row) + '\n')


if __name__ == "__main__":
    main()
