"""
Filename: metrics.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu
    Jason Youn - jyoun@ucdavis.edu

Description:
    Contains utility functions.

To-do:
    1. print baseline pr inside pr_stats()?
    2. combine roc_auc_stats and pr_stats into one maybe
        because they share lots of stuff.
"""
# standard imports
import datetime
import logging as log
import os
import pickle
import time

# third party imports
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import interp

# local imports
from utils import create_dir


def roc_auc_stats(num_preds, Y, predictions, predicates, pred_dic):
    """
    Find different statistics receiver operating characteristic (ROC).

    Inputs:
        num_preds: number of predicates
        Y: ground truth label to be compared with predictions_list
        predictions: prediction found by running a feed-forward of the model
        predicates: numpy array containing all the predicates within the dataset
        pred_dic: dictionary whose key is entity / relation and
            value is the index assigned to that entity / relation

    Returns:
        AUC of ROC
    """
    # some inits
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    predicates_included = []

    # for each predicate
    for i in range(num_preds):
        # find which lines (indeces) contain the predicate of interest
        predicate_indices = np.where(predicates == i)[0]

        if np.shape(predicate_indices)[0] == 0:
            continue
        else:
            predicates_included.append(i)

        # find actual prediction and label for each predicate
        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]

        # compute ROC
        fpr[i], tpr[i], _ = roc_curve(predicate_labels.ravel(), predicate_predictions.ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in predicates_included]))
    mean_tpr = np.zeros_like(all_fpr)

    # find mean of interpolated true positive rate
    for i in predicates_included:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(predicates_included)

    # for each predicate that was actually found above
    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name = key

        log.info('ROC auc for class %s (area = %f)', pred_name, roc_auc[i])

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return roc_auc['macro']


def pr_stats(num_preds, Y, predictions, predicates, pred_dic):
    """
    Find different statistics related to precision and recall.

    Inputs:
        num_preds: number of predicates
        Y: ground truth label to be compared with predictions_list
        predictions: prediction found by running a feed-forward of the model
        predicates: numpy array containing all the predicates within the dataset
        pred_dic: dictionary whose key is entity / relation and
            value is the index assigned to that entity / relation

    Returns:
        mean_average_precision: mAP
    """
    # some inits
    ap = dict()
    aucPR = dict()
    recall = dict()
    precision = dict()
    predicates_included = []
    sum_ap = 0.0

    # set baseline performance equal to all zeros
    baseline = np.zeros(np.shape(predictions))
    # find baseline precision / recall
    baseline_precision, baseline_recall, _ = precision_recall_curve(Y.ravel(), baseline.ravel())

    ######################################
    # print baseline precision & recall? #
    ######################################

    # for each predicate
    for i in range(num_preds):
        # find which lines (indeces) contain the predicate of interest
        predicate_indices = np.where(predicates == i)[0]

        if np.shape(predicate_indices)[0] == 0:
            continue
        else:
            predicates_included.append(i)

        # find actual prediction and label for each predicate
        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]

        # using the actual prediction and label, find their statistics
        precision[i], recall[i], _ = precision_recall_curve(
            predicate_labels.ravel(), predicate_predictions.ravel())
        ap[i] = average_precision_score(
            predicate_labels.ravel(), predicate_predictions.ravel())
        sum_ap += ap[i]
        aucPR[i] = auc(recall[i], precision[i])

    # for each predicate that was actually found above
    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name = key

        log.info('PR auc for class %s (area = %f)', pred_name, aucPR[i])

    # calculate mAP
    mean_average_precision = sum_ap / len(predicates_included)

    return mean_average_precision


def plot_roc(num_preds, Y, predictions, predicates, pred_dic, directory, name_of_file='model'):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('_%Y_%m_%d_%H_%M_%S')
    baseline = np.zeros(np.shape(predictions))
    baseline_fpr, baseline_tpr, _ = roc_curve(Y.ravel(), baseline.ravel())
    baseline_aucROC = auc(baseline_fpr, baseline_tpr)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    predicates_included = []

    for i in range(num_preds):
        predicate_indices = np.where(predicates == i)[0]

        if np.shape(predicate_indices)[0] == 0:
            continue
        else:
            predicates_included.append(i)

        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]

        fpr[i], tpr[i], _ = roc_curve(predicate_labels.ravel(), predicate_predictions.ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in predicates_included]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in predicates_included:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(predicates_included)
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    plt.figure()
    plt.plot(fpr['macro'], tpr['macro'], lw=2, color='darkorange', label='Macro Average ROC curve (AUC:{:.3f})'.format(roc_auc['macro']))
    saved_data_points = (fpr['macro'], tpr['macro'], roc_auc['macro'])
    plt.plot(baseline_fpr, baseline_tpr, lw=2, color='green', label='baseline (AUC:{:.3f})'.format(baseline_aucROC))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right', prop={'size': 6})

    directory = directory + '/fig'
    create_dir(directory)

    filename = directory + '/roc.png'
    plt.savefig(filename)

    with open(directory + '/roc_macro_' + name_of_file + st + '.pkl', 'wb') as output:
        pickle.dump(saved_data_points, output, pickle.HIGHEST_PROTOCOL)
    print("saved:{!s}".format(filename))

    plt.figure()
    pred_name = None
    lines = []
    labels = []
    saved_data_points = {}
    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name = key

        saved_data_points[pred_name] = (fpr[i], tpr[i], roc_auc[i])
        l, = plt.plot(fpr[i], tpr[i], lw=2)
        lines.append(l)
        labels.append('ROC for class {} (area = {:.3f})'.format(pred_name, roc_auc[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Predicate ROC')
    plt.legend(lines, labels, loc='upper right', prop={'size': 6})

    filename = directory + '/roc_' + name_of_file + st + '.png'
    plt.savefig(filename)
    with open(directory + '/roc_' + name_of_file + st + '.pkl', 'wb') as output:
        pickle.dump(saved_data_points, output, pickle.HIGHEST_PROTOCOL)

    print('saved:{!s}'.format(filename))


def plot_pr(num_preds, Y, predictions, predicates, pred_dic, directory, name_of_file='model'):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('_%Y_%m_%d_%H_%M_%S')
    baseline = np.zeros(np.shape(predictions))
    baseline_precision, baseline_recall, _ = precision_recall_curve(Y.ravel(), baseline.ravel())

    precision = dict()
    recall = dict()
    aucPR = dict()
    predicates_included = []
    sum_ap = 0.0
    ap = dict()

    for i in range(num_preds):
        predicate_indices = np.where(predicates == i)[0]

        if np.shape(predicate_indices)[0] == 0:
            continue
        else:
            predicates_included.append(i)

        predicate_predictions = predictions[predicate_indices]
        predicate_labels = Y[predicate_indices]
        precision[i], recall[i], _ = precision_recall_curve(predicate_labels.ravel(), predicate_predictions.ravel())
        aucPR[i] = auc(recall[i], precision[i])
        ap[i] = average_precision_score(predicate_labels.ravel(), predicate_predictions.ravel())
        sum_ap += ap[i]

    mean_average_precision = sum_ap / len(predicates_included)

    plt.figure()
    pred_name = None
    lines = []
    labels = []
    saved_data_points = {}

    for i in predicates_included:
        for key, value in pred_dic.items():
            if value == i:
                pred_name = key

        saved_data_points[pred_name] = (recall[i], precision[i], ap[i])
        l, = plt.step(recall[i], precision[i], lw=2, where='post')
        # l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {} (area = {:.3f})'.format(pred_name, aucPR[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall (mAP = {:.3f})'.format(mean_average_precision))
    plt.legend(lines, labels, loc='upper right', prop={'size': 6})

    directory = directory + '/fig'
    create_dir(directory)

    filename = directory + '/pr_' + name_of_file + st + '.png'
    plt.savefig(filename)
    print('saved:{!s}'.format(filename))
    with open(directory + '/pr_' + name_of_file + st + '.pkl', 'wb') as output:
        pickle.dump(saved_data_points, output, pickle.HIGHEST_PROTOCOL)


def plot_cost(iterations, cost_list, directory):
    """
    Plot and save cost graph.

    Inputs:
        iterations: list containing the iteration
        cost_list: list containing the cost at each corresponding iteration
        directory: directory to save the cost plot
    """
    plt.figure()
    plt.plot(iterations, cost_list, lw=1, color='darkorange')
    plt.xlabel('Iteration #')
    plt.ylabel('Loss')
    plt.title('Loss per iteration of training')

    directory = directory + '/fig'
    create_dir(directory)

    filename = directory + '/cost.png'
    plt.savefig(filename)


def plot_map(iterations, map_list, directory, filename='map.png'):
    """
    Plot and save map graph.

    Inputs:
        iterations: list containing the iteration
        map_list: list containing the map at each corresponding iteration
        directory: directory to save the map plot
    """
    plt.figure()
    plt.plot(iterations, map_list, lw=1, color='darkorange')
    plt.xlabel('Iteration #')
    plt.ylabel('MAP')
    plt.title('MAP per iteration of training')

    directory = os.path.join(directory, 'fig')
    create_dir(directory)

    filename = os.path.join(directory, filename)
    plt.savefig(filename)
