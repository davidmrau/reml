import datasets

def get_by_ids(dataset, ids):
    idxs = [ dataset.id2index[id_] for id_ in ids]
    return dataset[idxs]['sentence']

def make_hf_dataset(dataset, q_ids, d_ids, multi_doc=False):
    dataset_dict = {'query': [], 'doc': [], 'q_id': []}
    if not multi_doc:
        dataset_dict['d_id'] = []
    
    queries = get_by_ids(dataset['query'], q_ids)
    for i, q_id in enumerate(q_ids):
        if multi_doc:
            docs = get_by_ids(dataset['doc'], d_ids[i])
            if multi_doc:
                dataset_dict['doc'].append(docs)
            else:
                for d_id, doc in zip(d_ids[i], docs):
                    dataset_dict['d_id'].append(d_id)
                    dataset_dict['doc'].append(doc)
            dataset_dict['query'].append(queries[i])
            dataset_dict['q_id'].append(q_id)
    return datasets.Dataset.from_dict(dataset_dict)



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


