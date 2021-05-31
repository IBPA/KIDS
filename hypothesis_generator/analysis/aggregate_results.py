"""
Filename: aggregate_results.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Aggregate all the results of different models.

To-do:
"""

# standard imports
import argparse
import glob
import logging as log
import os
import sys

# third party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc

# local imports
DIRECTORY = os.path.dirname(__file__)
ABS_PATH_METRICS = os.path.join(DIRECTORY, '../utils')
sys.path.insert(0, ABS_PATH_METRICS)
from utils import save_results, create_dir, load_pickle
from kids_log import set_logging

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Aggregate all the results.')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='+',
        default='./',
        help='base directory')
    parser.add_argument(
        '--results_dir',
        metavar='dir',
        nargs='?',
        action='store',
        required=True,
        help='directory to store the model')

    return parser.parse_args()

def create_results_list(results):
    """
    Create dictionary of lists to hold the results.

    Inputs:
        results: loaded pickle file using load_pickle()

    Returns:
        results_list: initialized dictionary of lists
    """
    results_list = {}
    results_list['overall'] = {}

    for metric, _ in results['overall'].items():
        results_list['overall'][metric] = []

    results_list['predicate'] = {}
    for predicate, metrics in results['predicate'].items():
        results_list['predicate'][predicate] = {}
        for metric, _ in metrics.items():
            results_list['predicate'][predicate][metric] = []

    return results_list

def add_result(results, results_overall):
    """
    Append the results.

    Inputs:
        results: loadd pickle file using load_pickle()
        results_overall: results that will be appended to
    """
    for metric, value in results['overall'].items():
        results_overall['overall'][metric].append(value)

    for predicate, metrics in results['predicate'].items():
        for metric, value in metrics.items():
            results_overall['predicate'][predicate][metric].append(value)

def get_latest_fig(path, prefix):
    """
    Load the figure that was saved the latest.

    Inputs:
        path: directory where the figures are saved at
        prefix: prefix of the file that uses time as its suffix

    Returns:
        latest figure's pickle
    """
    list_of_files = glob.glob('{}/{}*.pkl'.format(path, prefix))
    latest_file = max(list_of_files, key=os.path.getctime)

    return load_pickle(latest_file)

def combine_results(results_list):
    """
    Given different metrics based on the result lists provided.

    Inputs:
        results_list: dictionary containing the lists of all the results

    Returns:
        results_overall: dictionary containing the lists of all the results
            now with additional performance metric data added
    """
    results_overall = {}
    results_overall['overall'] = {}
    results_overall['predicate'] = {}

    # find metrics for overall
    for metric, value in results_list['overall'].items():
        if metric in ('map', 'roc_auc'):
            results_overall['overall'][metric] = np.mean(np.array(results_list['overall'][metric]))
            results_overall['overall']['{}_std'.format(metric)] = np.std(np.array(results_list['overall'][metric]))
        elif metric == 'cm':
            TN = np.array([v[0][0] for v in value])
            FP = np.array([v[0][1] for v in value])
            FN = np.array([v[1][0] for v in value])
            TP = np.array([v[1][1] for v in value])

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            accuracy = (TP + TN) / (TN + FP + FN + TP)
            known_true = TP + FN
            known_neg = TN + FP
            predicted_true = TP + FP
            predicted_neg = FN + TN
            specificity = TN / (FP + TN)
            npv = TN / (TN + FN)
            fdr = FP / (FP + TP)

            results_overall['overall']['cm'] = np.mean(np.array(value), axis=0)
            results_overall['overall']['cm_std'] = np.std(np.array(value), axis=0)
            results_overall['overall']['precision'] = np.mean(precision)
            results_overall['overall']['precision_std'] = np.std(precision)
            results_overall['overall']['recall'] = np.mean(recall)
            results_overall['overall']['recall_std'] = np.std(recall)
            results_overall['overall']['accuracy'] = np.mean(accuracy)
            results_overall['overall']['accuracy_std'] = np.std(accuracy)
            results_overall['overall']['known_true'] = np.mean(known_true)
            results_overall['overall']['known_true_std'] = np.std(known_true)
            results_overall['overall']['known_neg'] = np.mean(known_neg)
            results_overall['overall']['known_neg_std'] = np.std(known_neg)
            results_overall['overall']['predicted_true'] = np.mean(predicted_true)
            results_overall['overall']['predicted_true_std'] = np.std(predicted_true)
            results_overall['overall']['predicted_neg'] = np.mean(predicted_neg)
            results_overall['overall']['predicted_neg_std'] = np.std(predicted_neg)
            results_overall['overall']['specificity'] = np.mean(specificity)
            results_overall['overall']['specificity_std'] = np.std(specificity)
            results_overall['overall']['npv'] = np.mean(npv)
            results_overall['overall']['npv_std'] = np.std(npv)
            results_overall['overall']['fdr'] = np.mean(fdr)
            results_overall['overall']['fdr_std'] = np.std(fdr)

            f1 = 2 * (precision * recall) / (precision + recall)
            results_overall['overall']['f1'] = np.mean(f1)
            results_overall['overall']['f1_std'] = np.std(f1)

    # find metrics for each predicate
    for predicate, metrics in results_list['predicate'].items():
        results_overall['predicate'][predicate] = {}

        for metric, value in metrics.items():
            if metric in ('map', 'roc_auc'):
                results_overall['predicate'][predicate][metric] = np.mean(np.array(results_list['overall'][metric]))
                results_overall['predicate'][predicate]['{}_std'.format(metric)] = np.std(np.array(results_list['overall'][metric]))
            elif metric == 'cm':
                TN = np.array([v[0][0] for v in value])
                FP = np.array([v[0][1] for v in value])
                FN = np.array([v[1][0] for v in value])
                TP = np.array([v[1][1] for v in value])

                precision = TP / (TP + FP)
                recall = TP / (TP +FN)
                accuracy = (TP + TN) / (TN + FP + FN + TP)
                known_true = TP + FN
                known_neg = TN + FP
                predicted_true = TP + FP
                predicted_neg = FN + TN
                specificity = TN / (FP + TN)
                npv = TN / (TN + FN)
                fdr = FP / (FP + TP)

                results_overall['predicate'][predicate]['cm'] = np.mean(np.array(value), axis=0)
                results_overall['predicate'][predicate]['cm_std'] = np.std(np.array(value), axis=0)
                results_overall['predicate'][predicate]['precision'] = np.mean(precision)
                results_overall['predicate'][predicate]['precision_std'] = np.std(precision)
                results_overall['predicate'][predicate]['recall'] = np.mean(recall)
                results_overall['predicate'][predicate]['recall_std'] = np.std(recall)
                results_overall['predicate'][predicate]['accuracy'] = np.mean(accuracy)
                results_overall['predicate'][predicate]['accuracy_std'] = np.std(accuracy)
                results_overall['predicate'][predicate]['known_true'] = np.mean(known_true)
                results_overall['predicate'][predicate]['known_true_std'] = np.std(known_true)
                results_overall['predicate'][predicate]['known_neg'] = np.mean(known_neg)
                results_overall['predicate'][predicate]['known_neg_std'] = np.std(known_neg)
                results_overall['predicate'][predicate]['predicted_true'] = np.mean(predicted_true)
                results_overall['predicate'][predicate]['predicted_true_std'] = np.std(predicted_true)
                results_overall['predicate'][predicate]['predicted_neg'] = np.mean(predicted_neg)
                results_overall['predicate'][predicate]['predicted_neg_std'] = np.std(predicted_neg)
                results_overall['predicate'][predicate]['specificity'] = np.mean(specificity)
                results_overall['predicate'][predicate]['specificity_std'] = np.std(specificity)
                results_overall['predicate'][predicate]['npv'] = np.mean(npv)
                results_overall['predicate'][predicate]['npv_std'] = np.std(npv)
                results_overall['predicate'][predicate]['fdr'] = np.mean(fdr)
                results_overall['predicate'][predicate]['fdr_std'] = np.std(fdr)

                f1 = 2 * (precision * recall) / (precision + recall)
                results_overall['predicate'][predicate]['f1'] = np.mean(f1)
                results_overall['predicate'][predicate]['f1_std'] = np.std(f1)

    return results_overall

def get_mean_average_roc(roc, dir_list):
    """
    Generate one ROC curve using average of different ROC curves.

    Inputs:
        roc: pickle containing the ROC curves
        dir_list: list containing different fold directory names
    """
    all_fpr = {}
    mean_tpr = {}
    predicates_fpr = {}
    predicates_tpr = {}
    predicates_auc = {}
    mean_fpr = {}
    mean_tpr = {}
    mean_auc = {}
    mean_auc_std = {}

    for fold in dir_list:
        for predicate, values in roc[fold].items():
            if predicate not in predicates_fpr:
                predicates_auc[predicate] = []
                predicates_fpr[predicate] = []
                predicates_tpr[predicate] = []

            fpr, tpr, _auc = values
            predicates_auc[predicate].append(_auc)
            predicates_fpr[predicate].append(fpr)
            predicates_tpr[predicate].append(tpr)

        for predicate, fpr_list in predicates_fpr.items():
            all_fpr[predicate] = np.unique(np.concatenate(fpr_list))
            mean_tpr[predicate] = np.zeros_like(all_fpr[predicate])

    for predicate, tprs in predicates_tpr.items():
        for i in range(len(tprs)):
            mean_tpr[predicate] += interp(all_fpr[predicate], predicates_fpr[predicate][i], tprs[i])

        mean_tpr[predicate] /= len(tprs)
        mean_fpr[predicate] = all_fpr[predicate]
        mean_tpr[predicate] = mean_tpr[predicate]
        mean_auc[predicate] = auc(mean_fpr[predicate], mean_tpr[predicate])
        mean_auc_std[predicate] = np.std(np.array(predicates_auc[predicate]))

    return mean_fpr, mean_tpr, mean_auc, mean_auc_std

def plot_combined_pr(pr, dir_list, model, results_dir):
    """
    Combine different PR curves into one plot.

    Inputs:
        pr: pickle containing the PR curve
        dir_list: list containing different fold directory names
        model: model name (PRA, MLP, Stacked)
        results_dir: directory to save the results to
    """
    plt.figure()
    for fold in dir_list:
        # There only exists one predicate: Confers resistance to antibiotic.
        # No need to show the name in the plot.
        for _, values in pr[fold].items():
            recall, precision, ap = values
            plt.step(recall, precision, label="{} (AP:{:.3f})".format(fold, ap), where='post')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("{} - Precision Recall".format(model))
            plt.legend(loc="upper right", prop={'size': 6})
            plt.savefig(os.path.join(results_dir, '{}_pr.pdf'.format(model)))

def main():
    """
    Main function.
    """
    # parse args and set logging
    args = parse_argument()
    set_logging()

    # variables
    dir_list = args.dir
    results_dir = args.results_dir
    root_dir = os.path.abspath(os.path.join(DIRECTORY, '..'))

    # load previously saved results
    first_flag = True
    log.info('Loading previously saved results...')

    for fold in dir_list:
        pra_results = load_pickle(
            '{}/pra/model/model_instance/{}/instance/test/results/results.pkl'.format(root_dir, fold))
        er_mlp_results = load_pickle(
            '{}/er_mlp/model/model_instance/{}/test/results/results.pkl'.format(root_dir, fold))
        stacked_results = load_pickle(
            '{}/stacked/model_instance/{}/test/results/results.pkl'.format(root_dir, fold))

        if first_flag is True:
            pra_results_list = create_results_list(pra_results)
            er_mlp_results_list = create_results_list(er_mlp_results)
            stacked_results_list = create_results_list(stacked_results)
            first_flag = False

        add_result(pra_results, pra_results_list)
        add_result(er_mlp_results, er_mlp_results_list)
        add_result(stacked_results, stacked_results_list)

    # create directories where the aggregated results will be saved to
    create_dir(results_dir)
    pra_dir = os.path.join(results_dir, 'pra')
    er_mlp_dir = os.path.join(results_dir, 'er_mlp')
    stacked_dir = os.path.join(results_dir, 'stacked')
    create_dir(pra_dir)
    create_dir(er_mlp_dir)
    create_dir(stacked_dir)

    # calculate additional performance metrics from the loaded results
    pra_results_overall = combine_results(pra_results_list)
    er_mlp_results_overall = combine_results(er_mlp_results_list)
    stacked_results_overall = combine_results(stacked_results_list)

    # save these calculated results
    save_results(pra_results_overall, pra_dir)
    save_results(er_mlp_results_overall, er_mlp_dir)
    save_results(stacked_results_overall, stacked_dir)

    # load the figures
    pra_pr, pra_roc, er_mlp_pr, er_mlp_roc, stacked_pr, stacked_roc = {}, {}, {}, {}, {}, {}
    for fold in dir_list:
        pra_pr[fold] = get_latest_fig('{}/pra/model/model_instance/{}/instance/test/fig'.format(root_dir, fold), 'pr_pra')
        pra_roc[fold] = get_latest_fig('{}/pra/model/model_instance/{}/instance/test/fig'.format(root_dir, fold), 'roc_pra')
        er_mlp_pr[fold] = get_latest_fig('{}/er_mlp/model/model_instance/{}/test/fig'.format(root_dir, fold), 'pr_er_mlp')
        er_mlp_roc[fold] = get_latest_fig('{}/er_mlp/model/model_instance/{}/test/fig'.format(root_dir, fold), 'roc_er_mlp')
        stacked_pr[fold] = get_latest_fig('{}/stacked/model_instance/{}/fig'.format(root_dir, fold), 'pr_model')
        stacked_roc[fold] = get_latest_fig('{}/stacked/model_instance/{}/fig'.format(root_dir, fold), 'roc_model')

    # plot the aggregated PR curves
    plot_combined_pr(pra_pr, dir_list, 'PRA', results_dir)
    plot_combined_pr(er_mlp_pr, dir_list, 'MLP', results_dir)
    plot_combined_pr(stacked_pr, dir_list, 'Stacked', results_dir)

    # plot the aggregated ROC curve
    pra_mean_fpr, pra_mean_tpr, pra_mean_auc, pra_mean_auc_std = get_mean_average_roc(pra_roc, dir_list)
    er_mlp_mean_fpr, er_mlp_mean_tpr, er_mlp_mean_auc, er_mlp_mean_auc_std = get_mean_average_roc(er_mlp_roc, dir_list)
    stacked_mean_fpr, stacked_mean_tpr, stacked_mean_auc, stacked_mean_auc_std = get_mean_average_roc(stacked_roc, dir_list)

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="baseline (AUC:{:.3f})".format(0.5))

    for predicate, _ in pra_mean_tpr.items():
        plt.plot(pra_mean_fpr[predicate], pra_mean_tpr[predicate], label="PRA (AUC:{:.3f} +- {:.3f})".format(pra_mean_auc[predicate], pra_mean_auc_std[predicate]))
        plt.plot(er_mlp_mean_fpr[predicate], er_mlp_mean_tpr[predicate], label="MLP (AUC:{:.3f} +- {:.3f})".format(er_mlp_mean_auc[predicate], er_mlp_mean_auc_std[predicate]))
        plt.plot(stacked_mean_fpr[predicate], stacked_mean_tpr[predicate], label="Stacked (AUC:{:.3f} +- {:.3f})".format(stacked_mean_auc[predicate], stacked_mean_auc_std[predicate]))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right", prop={'size': 6})
    plt.savefig(os.path.join(results_dir, 'roc.pdf'))

if __name__ == '__main__':
    main()
