import os


class Data:
    def __init__(self, data_dir, mode, reverse=False):
        self.mode = mode
        if self.mode == 'gridsearch':
            self.train_data = self.load_data(data_dir, "train", reverse=reverse)
            self.test_data = self.load_data(data_dir, "val", reverse=reverse)
        elif self.mode == 'evaluate':
            train_data = self.load_data(data_dir, "train", reverse=reverse)
            val_data = self.load_data(data_dir, "val", reverse=reverse)
            self.train_data = train_data + [x for x in val_data if x[-1] == '1']
            self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        elif self.mode == 'final':
            self.train_data = self.load_data(data_dir, "train", reverse=reverse)
            self.test_data = self.load_data(data_dir, "test", reverse=reverse)

        self.data = self.train_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + \
            [i for i in self.test_relations if i not in self.train_relations]

    def load_data(self, data_dir, data_type, reverse=False):
        print(f'Loading {data_type} data...')

        filepath = os.path.join(data_dir, f'{data_type}.txt')
        with open(filepath, "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split('\t') for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]

        if data_type == 'train':
            labels = set([x[-1] for x in data])
            assert {'1'} == labels

        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
