"""
Filename: data_processor.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Class containing multiple functions that will be used to process the data.

To-do:
"""
# standard imports
import logging as log
import re

# third party imports
import numpy as np
import pandas as pd
import scipy.io as spio


class DataProcessor:
    """
    Collection of functions to process the data.
    """
    _SPOL_LIST = ['Subjct', 'Predicate', 'Object', 'Label']

    def __init__(self):
        """
        Constructor for DataProcessor class.
        """

    @classmethod
    def load(cls, filename):
        """
        Load the data and return the pandas dataframe.

        Inputs:
            filename: filename containing the full path

        Returns:
            pd_data: dataframe containing the data
        """
        log.info('Loading data from \'%s\'...', filename)

        pd_data = pd.read_csv(
            filename,
            sep='\t',
            encoding='latin-1',
            names=cls._SPOL_LIST)

        return pd_data

    @staticmethod
    def machine_translate_using_word(fname, init_embed_file=None, separator='_'):
        """
        Given a file containing either entities or relations, translate them into
        machine friendly setting using word embddings. Assume we are given two entities
        'A_C' and 'B_C'. First, separate them into words and generate word pool
        'A', 'B', and 'C'. Then represent each entity as combination of these words.

        Inputs:
            fname: (str) filename containing all the entities / relations
            init_embed_file: (optional) mat file to use instead
            separator: (optional) separator which separates words within an entity / relation

        Returns:
            indexed_items: List of lists [[], [], []] where each list
                contains word index id(s) for a single entity / relation.
            num_words: (int) Total number of words inside the all the entities / relations.
            item_dic: (dict) Dictionary whose key is a single entity / relation
                and value is the index assigned to that entity / relation.
        """
        # some inits
        item_dic = {}
        item_index_id = 0

        log.info('Translating items in \'%s\' using word embedding...', fname)

        # open file
        with open(fname, encoding='utf8') as _file:
            items = [l.split() for l in _file.read().strip().split('\n')]

        # create item dictionary where key is item
        # and value is the index assigned to that item
        for item in items:
            if item[0] not in item_dic:
                item_dic[item[0]] = item_index_id
                item_index_id += 1

        if init_embed_file:
            log.info('Using init embed file \'%s\' for word embedding...', init_embed_file)
            mat = spio.loadmat(init_embed_file, squeeze_me=True)
            indexed_items = [
                [mat['tree'][i][()][0] - 1]
                if isinstance(mat['tree'][i][()][0], int)
                else [x - 1 for x in mat['tree'][i][()][0]] for i in range(len(mat['tree']))]
            num_words = len(mat['words'])
        else:
            word_index_id = 0
            word_ids = {}
            items_to_words = {}

            # for each item
            for item in items:
                words = []

                # item[0] = molecular_function
                for s in re.split(separator, item[0]):
                    words.append(s)

                    if s not in word_ids:
                        word_ids[s] = word_index_id
                        word_index_id += 1

                # words = ['molecular', 'function']
                items_to_words[item[0]] = words

            # create list of of length len(item_dic) where all entries are None
            indexed_items = [None] * len(item_dic)

            for key, val in item_dic.items():
                indexed_items[val] = []

                # items_to_words[key] = ['molecular', 'function']
                for s in items_to_words[key]:
                    indexed_items[val].append(word_ids[s])

            num_words = len(word_ids)

        return indexed_items, num_words, item_dic

    @staticmethod
    def machine_translate(fname):
        """
        Given a file containing entities, assign each entity
        with unique entity index id which machine can use.

        Inputs:
            fname: filename containing all the entities

        Returns:
            entity_dic: dictionary of entities where the key is entity,
            and the value is a index id assigned to that specific entity.
        """
        # some inits
        entity_dic = {}
        entity_index_id = 0

        log.info('Translating items in \'%s\' without using word embedding...', fname)

        # open file
        with open(fname, encoding='utf-8') as _file:
            entities = [l.split() for l in _file.read().strip().split('\n')]

        for entity in entities:
            if entity[0] not in entity_dic:
                entity_dic[entity[0]] = entity_index_id
                entity_index_id += 1

        return entity_dic

    @staticmethod
    def create_indexed_triplets_without_label(df_data, entity_dic, pred_dic):
        """
        Same as self.create_indexed_triplets_with_label() except for the lack
        of true/false field in the returning numpy array.

        Inputs:
            df_data: dataframe containing the data
            entity_dic: dictionary of entities where the key is entity,
                and the value is a index id assigned to that specific entity.
            pred_dic: dictionary of entities where the key is pred,
                and the value is a index id assigned to that specific pred.

        Returns:
            list of lists where each list has length equal to 3.
            [sub_index, pred_index, obj_index]
        """
        indexed_data = []

        for i in range(len(df_data)):
            sub_index = entity_dic[df_data[i][0]]
            pred_index = pred_dic[df_data[i][1]]
            obj_index = entity_dic[df_data[i][2]]

            indexed_data.append([sub_index, pred_index, obj_index])

        return np.array(indexed_data)

    @staticmethod
    def create_indexed_triplets_with_label(df_data, entity_dic, pred_dic):
        """
        Given a train / dev / test dataset, create a numpy array
        which consists of indices of all the items in the triple.

        Inputs:
            df_data: dataframe containing the data
            entity_dic: dictionary of entities where the key is entity,
                and the value is a index id assigned to that specific entity.
            pred_dic: dictionary of entities where the key is pred,
                and the value is a index id assigned to that specific pred.

        Returns:
            list of lists where each list has length equal to 4.
            [sub_index, pred_index, obj_index, 1/-1]
        """
        indexed_data = []

        for i in range(len(df_data)):
            sub_index = entity_dic[df_data[i][0]]
            pred_index = pred_dic[df_data[i][1]]
            obj_index = entity_dic[df_data[i][2]]
            label = df_data[i][3]

            indexed_data.append([sub_index, pred_index, obj_index, label])

        return np.array(indexed_data)
