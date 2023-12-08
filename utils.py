import datasets

class LookupDatasetHF(datasets.Dataset):

    # def __init__(self, args, kwargs):
    #     super().__init__(*args, **kwargs)
    def add_index(self):
        # Add an index column to the dataset
        self = self.add_column("index", range(len(self)))

        # Create a mapping function from index to sentence_id
        self.id2idx_map = dict(zip(self["id"], self["index"]))
        print('added map')
        print(self.id2idx_map)

    def __getitem__(self, ids):
        if type(ids) != list():
            ids = list(ids)
        idx = [self.id2idx_map[id_] for id_ in ids]
        return self[idx]
