"""
Filename: data_processor.py

Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Process the integrated data by 1) removing temporal data
    and 2) creating a 'Label' column to denote positive / negative.

To-do:
"""
# standard imports
import logging as log

# third party imports
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


class DataProcessor():
    """
    Class for processing the integrated data generated by 'integrate_data.py'.
    """
    # global variables
    _SUB_STR = 'Subject'
    _PRED_STR = 'Predicate'
    _OBJ_STR = 'Object'
    _LABEL_STR = 'Label'

    _DOMAIN_STR = 'Domain'
    _RELATION_STR = 'Relation'
    _RANGE_STR = 'Range'

    _SPO_LIST = [_SUB_STR, _PRED_STR, _OBJ_STR]
    _SPOL_LIST = _SPO_LIST + [_LABEL_STR]

    def __init__(self, data_filepath, label_rule_file, domain_relation_file):
        """
        Class constructor for DataProcessor.

        Inputs:
            label_rule_file: (str) Filepath of label rule file.
        """
        self.pd_data = pd.read_csv(data_filepath, sep='\t', keep_default_na=False)
        self.label_rule_file = label_rule_file
        self.pd_dr, self.relations, self.entity_types = self._read_dr(domain_relation_file)

    def add_label(self):
        """
        Append column 'Label' based on the rule file.

        Returns:
            pd_with_label: (pd.DataFrame) Reformated data with 'Label' column added.
        """
        # drop unnecessary columns
        pd_with_label = self.pd_data[self._SPO_LIST].copy()
        pd_with_label[self._LABEL_STR] = ''

        # predicates before applying the label
        predicate_group = pd_with_label.groupby(self._PRED_STR)
        log.debug('Size of data grouped by raw predicates: \n%s', predicate_group.size())

        log.info('Applying labels using %s', self.label_rule_file)

        # iterate over each label rule
        for label_rule in ET.parse(self.label_rule_file).getroot():
            log.debug('Processing label rule %s', label_rule.get('label'))

            # check if data has conflicts
            label_feature_ifs = [label_feature.get('if') for label_feature in label_rule]
            label_feature_thens = [label_feature.get('then') for label_feature in label_rule]
            label_feature_if_then = list(zip(label_feature_ifs, label_feature_thens))

            for feature_if, feature_then in label_feature_if_then:
                pd_filtered = pd_with_label[pd_with_label[self._PRED_STR] == feature_if]
                pd_with_label.loc[pd_filtered.index, self._PRED_STR] = feature_then
                pd_with_label.loc[pd_filtered.index, self._LABEL_STR] = label_rule.get('label')

        # drop duplicates that results from using the Label column
        log.info('Dropping duplicates...')
        pd_with_label = pd_with_label.drop_duplicates(subset=self._SPO_LIST, keep=False)

        # logging
        predicate_group = pd_with_label.groupby(self._PRED_STR)
        log.debug('Size of data grouped by predicates: \n%s', predicate_group.size())

        label_group = pd_with_label.groupby(self._LABEL_STR)
        log.debug('Size of data grouped by label: \n%s', label_group.size())

        return pd_with_label

    def get_entity_dic(self, pd_with_label):
        """
        Create the dictionary where the key is entity type and value is
        the numpy array containing all the entities belonging to that type.

        Returns:
            entity_dic: (dict) Completed entity dictionary
        """
        entity_dic = dict.fromkeys(self.entity_types, np.array([]))

        # loop through each relation and fill in the
        # entity dictionary based on their domain / range type
        for relation in self.relations:
            log.debug('Processing relation: %s', relation)
            single_grr = self.pd_dr.loc[self.pd_dr[self._RELATION_STR] == relation]

            domain_type = single_grr[self._DOMAIN_STR].item()
            range_type = single_grr[self._RANGE_STR].item()

            # find unique domain & range for given relation type
            matching_relation_idx = pd_with_label[self._PRED_STR].isin([relation])
            domains = pd_with_label[matching_relation_idx][self._SUB_STR].unique()
            ranges = pd_with_label[matching_relation_idx][self._OBJ_STR].unique()

            entity_dic[domain_type] = np.append(entity_dic[domain_type], domains)
            entity_dic[range_type] = np.append(entity_dic[range_type], ranges)

        for key, value in entity_dic.items():
            entity_dic[key] = np.unique(value)
            log.debug('Count of entity type \'%s\': %d', key, entity_dic[key].shape[0])

        return entity_dic

    def save_entities(self, entity_dic, entities_filepath, entity_full_names_filepath):
        """
        Save all the entities to the specified file location.

        Inputs:
            file_path: file path to save all the entities
        """
        entities = np.array([])
        entity_full_names = np.array([])

        for entity_type in list(entity_dic.keys()):
            # entities
            entities = np.append(entities, entity_dic[entity_type])

            # entity full names
            pd_entities_for_type = pd.DataFrame(np.copy(entity_dic[entity_type]))
            pd_entities_for_type.iloc[:, 0] = 'concept:{}:'.format(entity_type) + \
                pd_entities_for_type.iloc[:, 0].astype(str)
            entity_full_names = np.append(entity_full_names, pd_entities_for_type.values.ravel())

        # make sure the array is unique
        entities = np.unique(entities)
        # entity_full_names = np.unique(entity_full_names)

        # save files
        log.info('Saving entities to \'%s\'...', entities_filepath)
        np.savetxt(entities_filepath, entities, fmt='%s')

        log.info('Saving entity full names to \'%s\'...', entity_full_names_filepath)
        np.savetxt(entity_full_names_filepath, entity_full_names, fmt='%s')

    def generate_hypotheses(self, pd_with_label, entity_dic, relations):
        """
        Generate hypotheses.

        Inputs:
            relations: list of strings of relations
                       e.g. ['represses', 'confers resistance to antibiotic']
        """
        pd_hypotheses = pd.DataFrame(columns=self._SPOL_LIST)

        for relation in relations:
            log.info('Processing hypotheses for relation \'%s\'...', relation)

            # extract domain and range type for chosen relation
            row = self.pd_dr[self.pd_dr[self._RELATION_STR] == relation]
            np_domain_entities = entity_dic[row[self._DOMAIN_STR].iloc[0]]
            np_range_entities = entity_dic[row[self._RANGE_STR].iloc[0]]

            # generate list of tuples (sub, obj) from known triplets
            pd_known = pd_with_label[pd_with_label[self._PRED_STR] == relation]
            known_sub_obj_tuple_list = \
                [tuple(x) for x in pd_known[[self._SUB_STR, self._OBJ_STR]].values]

            # generate list of tuples (sub, obj) of all possible combinations
            all_sub_obj_tuple_list = np.array(
                np.meshgrid(np_domain_entities, np_range_entities)).T.reshape(-1, 2)
            all_sub_obj_tuple_list = list(map(tuple, all_sub_obj_tuple_list))

            log.debug('Size of all possible combinations of subject and object: %d',
                      len(all_sub_obj_tuple_list))
            log.debug('Size of known combinations of subject and object: %d',
                      len(known_sub_obj_tuple_list))

            # generate list of tuples (sub, obj) of unknown combinations
            unknown_combinations = list(set(all_sub_obj_tuple_list) - set(known_sub_obj_tuple_list))

            log.debug('Size of unknown combinations of subject and object: %d',
                      len(unknown_combinations))

            # append the unknown triplets to generate hypothesis on
            pd_hypotheses_to_append = pd.DataFrame(
                unknown_combinations,
                columns=[self._SUB_STR, self._OBJ_STR])
            pd_hypotheses_to_append.insert(1, column=self._PRED_STR, value=relation)
            pd_hypotheses_to_append.insert(3, column=self._LABEL_STR, value=1)
            pd_hypotheses = pd_hypotheses.append(pd_hypotheses_to_append, sort=False)

        return pd_hypotheses

    def _read_dr(self, domain_relation_file):
        """
        (Private) Read the relation / domain / range text file.

        Returns:
            pd_dr: (pd.DataFrame) Domain / range information.
            all_relations_list: (list) All relations.
            entity_types: (list) All unique entity types.
        """
        pd_dr = pd.read_csv(domain_relation_file, sep='\t')

        # get all the relations in the dataset working on
        relation_group = pd_dr.groupby(self._RELATION_STR)
        all_relations_list = list(relation_group.groups.keys())

        # find unique entity types
        entity_types = []
        for dr_tuple, _ in pd_dr.groupby([self._DOMAIN_STR, self._RANGE_STR]):
            entity_types.extend(list(dr_tuple))
        entity_types = list(set(entity_types))

        return pd_dr, all_relations_list, entity_types
