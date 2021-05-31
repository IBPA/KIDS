"""
Filename: evaluate_network.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:

To-do:
"""
# standard imports
import argparse
import os
import pickle
import sys

DIRECTORY = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(DIRECTORY, '../er_mlp_imp'))
sys.path.insert(0, os.path.join(DIRECTORY, '../data_handler'))
sys.path.insert(0, os.path.join(DIRECTORY, '../../utils'))


# third party imports
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

# local imports
from er_mlp import ERMLP
from metrics import plot_roc, plot_pr, roc_auc_stats, pr_stats
from utils import save_results
from data_processor import DataProcessor
from config_parser import ConfigParser
from kids_log import set_logging


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate network.')

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

        # some parameters
        entity_dic = params['entity_dic']
        pred_dic = params['pred_dic']
        thresholds = params['thresholds']

        # init ERMLP class using the parameters defined above
        er_mlp = ERMLP(
            {},
            sess,
            meta_graph=os.path.join(model_save_dir, 'model.meta'),
            model_restore=os.path.join(model_save_dir, 'model'))

        # load data
        processor = DataProcessor()

        test_df = processor.load(os.path.join(configparser.getstr('data_path'), 'test.txt'))
        indexed_data_test = processor.create_indexed_triplets_with_label(
            test_df.values, entity_dic, pred_dic)

        # change label from -1 to 0
        indexed_data_test[:, 3][indexed_data_test[:, 3] == -1] = 0

        data_test = indexed_data_test[:, :3]
        labels_test = np.reshape(indexed_data_test[:, 3], (-1, 1))
        predicates_test = indexed_data_test[:, 1]

        predictions_list_test = sess.run(
            er_mlp.test_predictions,
            feed_dict={er_mlp.test_triplets: data_test, er_mlp.y: labels_test})

        # stats
        mean_average_precision_test = pr_stats(
            len(pred_dic),
            labels_test,
            predictions_list_test,
            predicates_test,
            pred_dic)

        roc_auc_test = roc_auc_stats(
            len(pred_dic),
            labels_test,
            predictions_list_test,
            predicates_test,
            pred_dic)

        classifications_test = er_mlp.classify(predictions_list_test, thresholds, predicates_test)
        classifications_test = np.array(classifications_test).astype(int)

        indexed_data_test[:, 3][indexed_data_test[:, 3] == -1] = 0
        labels_test = np.reshape(indexed_data_test[:, 3], (-1, 1))

        labels_test = labels_test.astype(int)
        fl_measure_test = f1_score(labels_test, classifications_test)
        accuracy_test = accuracy_score(labels_test, classifications_test)
        confusion_test = confusion_matrix(labels_test, classifications_test)
        precision_test = precision_score(labels_test, classifications_test)
        recall_test = recall_score(labels_test, classifications_test)

        plot_pr(
            len(pred_dic),
            labels_test,
            predictions_list_test,
            predicates_test,
            pred_dic,
            os.path.join(model_save_dir, 'test'),
            name_of_file='er_mlp_not_calibrated')

        plot_roc(
            len(pred_dic),
            labels_test,
            predictions_list_test,
            predicates_test,
            pred_dic,
            os.path.join(model_save_dir, 'test'),
            name_of_file='er_mlp_not_calibrated')

        results = {}

        results['overall'] = {
            'map': mean_average_precision_test,
            'roc_auc': roc_auc_test,
            'f1': fl_measure_test,
            'accuracy': accuracy_test,
            'cm': confusion_test,
            'precision': precision_test,
            'recall': recall_test
        }

        results['predicate'] = {}

        for i in range(len(pred_dic)):
            for key, value in pred_dic.items():
                if value == i:
                    pred_name = key
            indices, = np.where(predicates_test == i)

            if np.shape(indices)[0] != 0:
                classifications_predicate = classifications_test[indices]
                labels_predicate = labels_test[indices]
                fl_measure_predicate = f1_score(labels_predicate, classifications_predicate)
                accuracy_predicate = accuracy_score(labels_predicate, classifications_predicate)
                confusion_predicate = confusion_matrix(labels_predicate, classifications_predicate)
                precision_predicate = precision_score(labels_predicate, classifications_predicate)
                recall_predicate = recall_score(labels_predicate, classifications_predicate)

                print(' - test f1 measure for {}: {}'.format(pred_name, fl_measure_predicate))
                print(' - test accuracy for {}: {}'.format(pred_name, accuracy_predicate))
                print(' - test precision for {}: {}'.format(pred_name, precision_predicate))
                print(' - test recall for {}: {}'.format(pred_name, recall_predicate))
                print(' - test confusion matrix for {}:'.format(pred_name))
                print(confusion_predicate)
                print(' ')
                predicate_predictions = predictions_list_test[indices]
                fpr_pred, tpr_pred, _ = roc_curve(
                    labels_predicate.ravel(),
                    predicate_predictions.ravel())
                roc_auc_pred = auc(fpr_pred, tpr_pred)
                ap_pred = average_precision_score(
                    labels_predicate.ravel(),
                    predicate_predictions.ravel())

                results['predicate'][pred_name] = {
                    'map': ap_pred,
                    'roc_auc': roc_auc_pred,
                    'f1': fl_measure_predicate,
                    'accuracy': accuracy_predicate,
                    'cm': confusion_predicate,
                    'precision': precision_predicate,
                    'recall': recall_predicate
                }

        print('test mean average precision: {}'.format(mean_average_precision_test))
        print('test f1 measure: {}'.format(fl_measure_test))
        print('test accuracy: {}'.format(accuracy_test))
        print('test roc auc: {}'.format(roc_auc_test))
        print('test precision: {}'.format(precision_test))
        print('test recall: {}'.format(recall_test))
        print('test confusion matrix:')
        print(confusion_test)
        print('thresholds:')
        print(thresholds)
        save_results(results, os.path.join(model_save_dir, 'test'))

    with open(os.path.join(model_save_dir, 'test/classifications_er_mlp.txt'), 'w') as file:
        for row in classifications_test:
            file.write(str(row) + '\n')


if __name__ == '__main__':
    main()
