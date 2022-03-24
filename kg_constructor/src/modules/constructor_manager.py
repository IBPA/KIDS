import os
from typing import Dict
import sys
import logging as log
sys.path.append("..")

import pandas as pd  # noqa: E402
from pandarallel import pandarallel  # noqa: E402


class ConstructorManager:
    def __init__(
            self,
            data_dir: str,
            n_workers: int = 1,
            input_data_filename: str = 'input_data.txt',
            name_map_filename: str = 'name_map.txt',
            infer_filename: str = 'infer_rules.txt',
            remove_filename: str = 'remove_rules.txt',
            ):
        """
        """
        if n_workers > 1:
            log.info(f'Using pandarallel with {n_workers} cores.')
            pandarallel.initialize(nb_workers=n_workers, progress_bar=True)

        self.n_workers = n_workers
        self.data_dir = data_dir
        self.input_data_filepath = os.path.join(self.data_dir, input_data_filename)
        self.name_map_filepath = os.path.join(self.data_dir, name_map_filename)
        self.infer_filepath = os.path.join(self.data_dir, infer_filename)
        self.remove_filepath = os.path.join(self.data_dir, remove_filename)

        self.input_data = self._load_files()

    def _load_files(self,) -> Dict[str, str]:
        log.info(f'Input data filename: {self.input_data_filepath}')
        df_input_data = pd.read_csv(self.input_data_filepath, sep='\t')

        files_dict = dict(zip(df_input_data['Source'], df_input_data['Path']))
        log.debug(f'File names and paths: {files_dict}')

        return files_dict

    def construct_intermediate_kg(
            self,
            skip_mapping=False,
            skip_inference=False,
            skip_remove=False,
            ):
        df = self._integrate()
        if not skip_mapping:
            df = self._map_name(df)
        if not skip_inference:
            df = self._infer(df)
        if not skip_remove:
            df = self._remove(df)

        return df

    def _integrate(self) -> pd.DataFrame:
        """
        Integrate data from multiple sources.

        Returns:
        """

        data = []
        for dataset, path in self.input_data.items():
            full_path = os.path.join(self.data_dir, path)
            log.info(f'Loading \'{dataset}\' from \'{full_path}\'')

            df = pd.read_csv(full_path, sep='\t')

            # remove missing values
            before = df.shape[0]
            df.dropna(inplace=True)
            after = df.shape[0]

            if before > after:
                log.warning(f'Dropping {before - after} missing values!')

            # remove duplicates
            before = df.shape[0]
            df.drop_duplicates(inplace=True)
            after = df.shape[0]

            if before > after:
                log.warning(f'Dropping {before - after} duplicates!')

            df['Source'] = dataset
            data.append(df)

            log.info(f'Adding {df.shape[0]} triples.')

        # concatenate data
        df_output = pd.concat(data, sort=True)
        df_output = df_output[['Subject', 'Predicate', 'Object', 'Source']]

        log.info(
            f'Total of {df_output.shape[0]} triples integrated from '
            f'{len(self.input_data)} sources.')

        return df_output.reset_index(drop=True)

    def _map_name(self, df: pd.DataFrame):
        """
        Perform name mapping given data from single source.

        Inputs:

        Returns:
        """
        log.info('Cleaning name mapping table...')

        # open name mapping file
        df_map = pd.read_csv(self.name_map_filepath, sep='\t')
        df_map.drop_duplicates(inplace=True)

        # remove mapping cases where source == target
        df_map = df_map[df_map['Source'] != df_map['Target']]

        # remove infinite mapping cases where both a->b and b->a exist
        df_map_flipped = df_map.copy()
        df_map_flipped.rename(columns={'Source': 'Target', 'Target': 'Source'}, inplace=True)
        df_map_original_and_flipped = pd.concat([df_map, df_map_flipped])
        df_infinite_loops = df_map_original_and_flipped[df_map_original_and_flipped.duplicated()]

        def _unique(x):
            last = object()
            for item in x:
                if item == last:
                    continue
                yield item
                last = item

        unique_infinite_loops = []
        for _, row in df_infinite_loops.iterrows():
            item = [row['Source'], row['Target']]
            item.sort()
            unique_infinite_loops.append(item)

        df_infinite_loops = pd.DataFrame(
            list(_unique(unique_infinite_loops)),
            columns=['Source', 'Target'])

        log.warning(f'Removing {df_infinite_loops.shape[0]} infinite name mapping case!')
        log.debug(f'Looped mapping cases are as follows:\n{df_infinite_loops}')

        df_map = pd.concat([df_map, df_infinite_loops])
        df_map.drop_duplicates(keep=False, inplace=True)

        # remove non-infinite loop cases
        sources = set(df_map['Source'].tolist())
        targets = set(df_map['Target'].tolist())
        intersection = sources.intersection(targets)

        while intersection:
            df_loop = df_map[df_map['Source'].isin(intersection)]
            loop_dict = dict(zip(df_loop['Source'], df_loop['Target']))

            df_map['Target'] = df_map['Target'].replace(loop_dict)
            df_map.drop_duplicates(inplace=True)

            sources = set(df_map['Source'].tolist())
            targets = set(df_map['Target'].tolist())
            intersection = sources.intersection(targets)

        log.info(f'Name mapping size after cleaning: {df_map.shape[0]}')

        # now the real mapping
        log.info('Performing name mapping...')

        map_dict = dict(zip(df_map['Source'], df_map['Target']))

        def _map(x):
            if x in map_dict:
                return map_dict[x]
            else:
                return x

        if self.n_workers > 1:
            df['Subject'] = df['Subject'].parallel_apply(lambda x: _map(x))
            df['Object'] = df['Object'].parallel_apply(lambda x: _map(x))
        else:
            df['Subject'] = df['Subject'].apply(lambda x: _map(x))
            df['Object'] = df['Object'].apply(lambda x: _map(x))

        return df

    def _infer(self, df: pd.DataFrame):
        """
        Apply data rule and infer new data.

        Inputs:
            df: (pd.DataFrame) Data to infer new knowledge from.

        Returns:
            pd_updated: (pd.DataFrame) Data with new inferred data added.
        """
        df_output = df.copy()

        # open name mapping file
        df_rules = pd.read_csv(self.infer_filepath, sep='\t')
        df_rules.drop_duplicates(inplace=True)
        log.info(f'Found {df_rules.shape[0]} inference rules...')

        # apply the rules
        for _, rule in df_rules.iterrows():
            log.info(f'Applying rule:\n{rule}')

            if_where = rule['if_where']
            if_rule = rule['if_rule']
            if_value = rule['if_value']
            replace_where = rule['replace_where'].split(',')
            replace_value = rule['replace_value'].split(',')
            replace_dict = dict(zip(replace_where, replace_value))

            if if_rule == 'is':
                df_inferred = df_output[df_output[if_where] == if_value].copy()

            if if_rule == 'startswith':
                df_inferred = df_output[df_output[if_where].str.startswith(if_value)].copy()

            for where, value in replace_dict.items():
                df_inferred[where] = value

            log.info(f'Found {df_inferred.shape[0]} new triples using this rule.')

            df_output = pd.concat([df_output, df_inferred])

        df_output.drop_duplicates(inplace=True)

        num_new_triples = df_output.shape[0] - df.shape[0]
        log.info(f'Inferred {num_new_triples} triples using inference rules.')

        return df_output.reset_index(drop=True)

    def _remove(self, df: pd.DataFrame):
        """
        Replace any parts of the data if necessary.
        (Currently used specifically to drop temporal data.)

        Inputs:
            df: (pd.DataFrame) Data that has parts to be replaced.

        Returns:
            pd_replaced: (pd.DataFrame) Data with parts replaced.
        """
        df_output = df.copy()

        # open removing file
        df_rules = pd.read_csv(self.remove_filepath, sep='\t')
        df_rules.drop_duplicates(inplace=True)
        log.info(f'Found {df_rules.shape[0]} remove rules...')

        # apply the rules
        for _, rule in df_rules.iterrows():
            log.info(f'Applying rule:\n{rule}')

            if_where = rule['if_where']
            if_rule = rule['if_rule']
            if_value = rule['if_value']

            if if_rule == 'is':
                df_output = df_output[df_output[if_where] != if_value]

            if if_rule == 'startswith':
                df_output = df_output[~df_output[if_where].str.startswith(if_value)]

        num_new_triples = df.shape[0] - df_output.shape[0]
        log.info(f'Removed {num_new_triples} triples using remove rules.')

        return df_output.reset_index(drop=True)
