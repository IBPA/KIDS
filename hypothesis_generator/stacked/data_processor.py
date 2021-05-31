"""
Filename: data_processor.py

Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Read the model predictions and parser them.

To-do:
"""
# standard imports
import os
import pickle

# third party imports
import numpy as np


class DataProcessor():
    _ER_MLP_MODEL_HOME = '../er_mlp/model/model_instance/'
    _PRA_MODEL_HOME = '../pra/model/model_instance/'

    def __init__(self, data_type, mlp_dir, pra_dir):
        self.data_type = data_type
        self.mlp_model_base_dir = os.path.join(self._ER_MLP_MODEL_HOME, mlp_dir)
        self.pra_model_base_dir = os.path.join(self._PRA_MODEL_HOME, pra_dir, 'instance')

        with open(os.path.join(self.mlp_model_base_dir, 'params.pkl'), 'rb') as file:
            self.pred_dic = pickle.load(file)['pred_dic']

    @staticmethod
    def _load(filepath):
        with open(filepath, 'r') as file:
            lines = [x.strip() for x in file.readlines()]

        return lines

    def get_pred_dic(self):
        return self.pred_dic

    def get_x_y(self):
        """
        Load the model predictions information.

        Returns:
            pred_dic: dictionary where key is a predicate and value is the
                index assigned to that specific predicate
            x: numpy array where
                x[:, 0] = er_mlp prediction raw output
                x[:, 1] = pra prediction raw output
                x[:, 2] = valid / invalid depending on pra
            y: numpy array containing the ground truth label
            predicates: numpy array containing which predicate
                the (x, y) pair belong to using the indexing from pred_dic
        """
        #######
        # mlp #
        #######
        filepath = os.path.join(self.mlp_model_base_dir, self.data_type, 'predictions.txt')

        labels = []
        er_mlp_features = []

        for line in self._load(filepath):
            for field in line.split('\t'):
                if 'predicate' in field:
                    predicate = int(field.replace('predicate: ', ''))
                elif 'prediction' in field:
                    pred = float(field.replace('prediction: ', ''))
                elif 'label' in field:
                    label = int(field.replace('label: ', ''))

            labels.append([label])
            er_mlp_features.append([predicate, pred, 1])

        # convert to numpy arrays
        np_labels = np.array(labels)
        np_mlp_features = np.array(er_mlp_features)

        #######
        # pra #
        #######
        pra_features = []

        for key, val in self.pred_dic.items():
            scores_file = os.path.join(
                self.pra_model_base_dir,
                self.data_type,
                'scores',
                key)

            if os.path.isfile(scores_file):
                for line in self._load(scores_file):
                    fields = line.split('\t')

                    pred = float(fields[0].strip())
                    valid = int(fields[1].strip())

                    pra_features.append([int(val), pred, valid])
            else:
                continue

        np_pra_features = np.array(pra_features)

        ##################################
        # process the extracted features #
        ##################################
        predicates_mlp = np_mlp_features[:, 0].astype(int)
        predicates_pra = np_pra_features[:, 0].astype(int)

        labels_list = []
        combined_list = []
        predicates_list = []

        for key, val in self.pred_dic.items():
            mlp_predicate_indices = np.where(predicates_mlp[:] == val)[0]
            pra_predicate_indices = np.where(predicates_pra[:] == val)[0]

            labels_list.append(np_labels[mlp_predicate_indices])
            predicates_list.append(predicates_mlp[mlp_predicate_indices])

            selected_mlp_features = np_mlp_features[mlp_predicate_indices][:, 1:]
            selected_pra_features = np_pra_features[pra_predicate_indices][:, 1:]
            combined_list.append(np.hstack((selected_mlp_features, selected_pra_features)))

        # make returns
        x = np.vstack(combined_list)[:, [0, 2, 3]]
        y = np.vstack(labels_list)
        y[:][y[:] == -1] = 0
        predicates = np.hstack(predicates_list)

        return x, y, predicates
