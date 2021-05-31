"""
Filename: determine_thresholds.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Given the prediction scores, find the best value for threshold.

To-do:
"""
# standard imports
import argparse

# third party imports
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='determine the threshold')

    parser.add_argument(
        '--predicate',
        nargs='?',
        required=True,
        help='the predicate that we will get the scores for')

    parser.add_argument(
        '--dir',
        metavar='dir',
        nargs='?',
        default='./',
        help='base directory')

    return parser.parse_args()


def compute_threshold(predictions_list, dev_labels, f1=True):
    """
    Determine the best threshold to use for classification.

    Inputs:
        predictions_list: prediction found by running the model
        dev_labels: ground truth label to be compared with predictions_list
        f1: True is using F1 score, False if using accuracy score

    Returns:
        best_threshold: threshold that yields the best accuracy
    """
    predictions_list = predictions_list.reshape(-1, 1)
    dev_labels = dev_labels.reshape(-1, 1)
    both = np.column_stack((predictions_list, dev_labels))
    both = both[both[:, 0].argsort()]
    predictions_list = both[:, 0].ravel()
    dev_labels = both[:, 1].ravel()
    accuracies = np.zeros(np.shape(predictions_list))

    for i in range(np.shape(predictions_list)[0]):
        score = predictions_list[i]
        predictions = (predictions_list >= score) * 2 - 1
        accuracy = accuracy_score(predictions, dev_labels)

        if f1:
            accuracy = f1_score(dev_labels, predictions)

        accuracies[i] = accuracy

    indices = np.argmax(accuracies)
    best_threshold = np.mean(predictions_list[indices])

    return best_threshold


def main():
    """
    Main function.
    """
    args = parse_argument()

    relation = args.predicate
    scores_file = args.dir + '/scores/' + relation
    thresholds_file = args.dir + '/thresholds/' + relation

    labels_file = args.dir + '/queriesR_labels/' + relation

    with open(labels_file, "r") as l_file:
        labels = l_file.readlines()

    with open(scores_file, "r") as _file:
        scores = _file.readlines()

    # read scores and corresponding labels
    scores = [x.strip().split('\t') for x in scores]
    labels = [x.strip().split('\t') for x in labels]

    # convert to np array
    scores = np.array(scores)
    labels = np.array(labels)

    labels = labels[:, 2]
    labels = labels.astype(np.int)
    labels[:][labels[:] == 0] = -1

    threshold = compute_threshold(scores[:, 0].astype(np.float), labels, f1=True)

    with open(thresholds_file, "w") as _file:
        _file.write(str(threshold) + '\n')


if __name__ == "__main__":
    main()
