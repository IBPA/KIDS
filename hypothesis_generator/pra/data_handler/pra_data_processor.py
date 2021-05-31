"""
Filename: pra_data_processor.py

Authors:
    Nicholas Joodi - npjoodi@ucdavis.edu

Description:
    Process data to be used by PRA program.

To-do
"""
# standard imports
import argparse
import csv

# third party imports
import numpy as np
import pandas as pd


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Process data')

    parser.add_argument(
        '--data_path',
        metavar='dir',
        nargs='?',
        default='./',
        help='data directory')
    parser.add_argument(
        '--train_file',
        metavar='dir',
        nargs='?',
        default='./',
        help='The train file')
    parser.add_argument(
        '--use_domain',
        action='store_const',
        default=False,
        const=True)

    return parser.parse_args()


def create_type_subsets_dic(data_array):
    """
    Create dictionaries given a file entity_full_names.txt.

    Inputs:
        data_array: numpy array containing data

    Returns:
        type_dic: dictionary where key is entity name
            and value is entity type
        subsets_dic: dictionary where key is entity type
            and value is a set of entity names
    """
    type_dic = {}
    subsets_dic = {}

    for row in data_array:
        clean_row(row)

        if row[1] not in subsets_dic:
            subsets_dic[row[1]] = set()

        subsets_dic[row[1]].add(row[2])
        type_dic[row[2]] = row[1]

    return type_dic, subsets_dic


def clean_row(row):
    """
    Remove all whitespace for given row at the start and end,
    including spaces, tabs, newlines and carriage returns.

    Inputs:
        row: numpy array containing one row from 'entity_full_names.txt'
    """
    for i in range(np.shape(row)[0]):
        if isinstance(row[i], str):
            row[i] = row[i].strip()


class DataProcessor:
    def __init__(self, data_path, use_domain=False):
        """
        Constructor for DataProcessor class.

        Inputs:
            data_path: path to the data
            use_domain: True if using domain / range information
        """
        self.use_domain = use_domain
        self.data_path = data_path

        if self.use_domain:
            my_file = "entity_full_names.txt"
            pd_data = pd.read_csv(
                self.data_path + '/' + my_file,
                sep=':',
                encoding='latin-1',
                header=None)
            data_array = pd_data.values
            self.type_dic, self.subsets_dic = create_type_subsets_dic(data_array)

            my_file = "domain_range.txt"
            pd_data = pd.read_csv(
                self.data_path + '/' + my_file,
                sep='\t',
                encoding='latin-1',
                header=None)
            data_array = pd_data.values
            self.domain_range_dic = {}
            for row in data_array:
                self.domain_range_dic[row[0].strip()] = (row[1].strip(), row[2].strip())

            self.no_negatives = set()
            self.no_negatives.add('has')
            self.no_negatives.add('is')
            self.no_negatives.add('is#SPACE#involved#SPACE#in')
            self.no_negatives.add('upregulated#SPACE#by#SPACE#antibiotic')
            self.no_negatives.add('targeted#SPACE#by')

    def load(self, data_file):
        """
        Load data file.

        Inputs:
            data_file: file under self.data_path to load

        Returns:
            pd_data: dataframe containing the data
        """
        pd_data = pd.read_csv(
            self.data_path + '/' + data_file,
            sep='\t',
            encoding='latin-1',
            header=None)

        return pd_data

    def create_selected_relations_file(self, data):
        """
        Given a file, get all the relations in that file
        and save it to the file 'selected_relations'.

        Inputs:
            data: numpy array containing the contents of file
        """
        selected_relation_set = set()

        # add all the relations in the data
        for i in range(len(data)):
            selected_relation_set.add(data[i][1])

        self.selected_relation_set = selected_relation_set

        # write to file 'selected_relations'
        with open('selected_relations', 'a') as the_file:
            for relation in selected_relation_set:
                the_file.write(relation + '\n')

    def create_sets(self):
        """
        Create relation and entity sets.
        """
        entities_file = self.data_path + "/entities.txt"
        relations_file = self.data_path + "/relations.txt"
        entity_set = set()
        relation_set = set()

        with open(entities_file) as _file:
            for line in _file:
                entity_set.add(line.strip())

        with open(relations_file) as _file:
            for line in _file:
                relation_set.add(line.strip())

        self.relation_set = relation_set
        self.entity_set = entity_set

    def create_relations_file(self):
        """
        Create relations file based on whether
        if they use domain information or not.
        """
        with open('relations', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['relationName',
                             'humanFormat',
                             'populate',
                             'generalizations',
                             'domain',
                             'range',
                             'antisymmetric',
                             'mutexExceptions',
                             'knownNegatives',
                             'inverse',
                             'seedInstances',
                             'seedExtractionPatterns',
                             'nrOfValues',
                             'nrOfInverseValues',
                             'requiredForDomain',
                             'requiredForRange',
                             'editDate',
                             'author',
                             'description',
                             'freebaseID',
                             'coment'])

            for relation in self.relation_set:
                if self.use_domain:
                    writer.writerow(['concept:' + relation,
                                     '{{}}',
                                     'true',
                                     '{"object"}',
                                     self.domain_range_dic[relation][0],
                                     self.domain_range_dic[relation][1],
                                     'true',
                                     'concept:' + self.domain_range_dic[relation][0],
                                     'concept:' + self.domain_range_dic[relation][1],
                                     '',
                                     '(empty set)',
                                     '(empty set)',
                                     'any',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE'])
                else:
                    writer.writerow([relation,
                                     '{{}}',
                                     'true',
                                     '{}',
                                     'object',
                                     'object',
                                     'true',
                                     '(empty set)',
                                     '(empty set)',
                                     '',
                                     '(empty set)',
                                     '(empty set)',
                                     'any',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE',
                                     'NO_THEO_VALUE'])

    def create_triplets_generalizations_file(self, data, positive=True):
        filename = 'ecoli_generalizations.csv'

        # if data only contains postive triplets without the indicator,
        # concatenate 1s to denote that it is positive data
        if np.shape(data)[1] != 4:
            new_col = np.tile('1', data.shape[0])[None].T
            data = np.concatenate((data, new_col), 1)

        # get all the positives from the data
        positives = data[data[:, 3] == '1']
        if np.shape(positives)[0] == 0:
            positives = data[data[:, 3] == 1]

        # get all negatives from the data
        if not positive:
            positives = data[data[:, 3] == '-1']
            if np.shape(positives)[0] == 0:
                positives = data[data[:, 3] == -1]
            if np.shape(positives)[0] == 0:
                positives = data[data[:, 3] == '0']
            if np.shape(positives)[0] == 0:
                positives = data[data[:, 3] == 0]

            filename = 'ecoli_generalizations_neg.csv'

        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Entity', 'Relation', 'Value', 'Iteration of Promotion', 'Probability', 'Source', 'Candidate Source'])

            if self.use_domain:
                for row in positives:
                    writer.writerow(['concept:' + self.type_dic[row[0]] + ':' + row[0],
                                     'concept:' + row[1],
                                     'concept:' + self.type_dic[row[2]] + ':' + row[2],
                                     '',
                                     '1.0',
                                     '',
                                     ''])
                for key in self.entity_set:
                    writer.writerow(['concept:' + key,
                                     'generalizations',
                                     'concept:' + self.type_dic[key],
                                     '',
                                     '1.0',
                                     '',
                                     ''])
                for key, _ in self.subsets_dic.items():
                    writer.writerow(['concept:' + key,
                                     'generalizations',
                                     'concept:object',
                                     '',
                                     '1.0',
                                     '',
                                     ''])
            else:
                for row in positives:
                    writer.writerow([row[0], row[1], row[2], '', '1.0', '', ''])
                for key in self.entity_set:
                    writer.writerow([key, 'generalizations', 'object', '', '1.0', '', ''])


def main():
    """
    Main function.
    """
    args = parse_argument()

    processor = DataProcessor(args.data_path, use_domain=args.use_domain)
    test_df = processor.load('test.txt')
    processor.create_selected_relations_file(test_df.values)
    processor.create_sets()
    processor.create_relations_file()

    train_df = processor.load(args.train_file)
    processor.create_triplets_generalizations_file(train_df.values)

    if args.use_domain:
        if np.shape(train_df.values)[1] == 4:
            processor.create_triplets_generalizations_file(train_df.values, positive=False)


if __name__ == "__main__":
    main()
