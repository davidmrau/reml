import datasets

def get_by_ids(dataset, ids):
    # if single id is passed cast it to list
    if not isinstance(ids, list):
        ids = [ids]
    idxs = [ dataset.id2index[id_] for id_ in ids]
    return dataset[idxs]['content']


# gets q_ids and d_ids and does a lookup by id to get the content
# then constructs hf_dataset out of it
def make_hf_dataset(dataset, q_ids, d_ids, multi_doc=False, pyserini_docs=None):
    dataset_dict = {'query': [], 'doc': [], 'q_id': []}
    labels = dataset['query']['label'] if 'label' in dataset['query'].features else None
    if labels!= None:
        dataset_dict['label'] = []
    if not multi_doc:
        dataset_dict['d_id'] = []
    assert len(d_ids) == len(q_ids)

    queries = get_by_ids(dataset['query'], q_ids)
    for i, q_id in enumerate(q_ids):
        if pyserini_docs == None:
            docs = get_by_ids(dataset['doc'], d_ids[i])
        else:
            docs = pyserini_docs[i]
        # for multi_doc=True, all documents are saved to the 'doc' entry
        if multi_doc:
            dataset_dict['doc'].append(docs)
            dataset_dict['query'].append(queries[i])
            dataset_dict['q_id'].append(q_id)
            if labels != None:
                dataset_dict['label'].append(labels[i])
        else:
            # for multi_doc=False, we save every document to a new entry
                for d_id, doc in zip(d_ids[i], docs):
                    dataset_dict['d_id'].append(d_id)
                    dataset_dict['doc'].append(doc)
                    dataset_dict['query'].append(queries[i])
                    dataset_dict['q_id'].append(q_id)
                    if labels != None:
                        dataset_dict['label'].append(labels[i])
    return datasets.Dataset.from_dict(dataset_dict)

def print_generate_out(gen_out, first_n=10):
    ids, instructions, responses, labels = gen_out['q_id'], gen_out['instruction'], gen_out['response'], gen_out['labels']
    for i, (id_,instr, response, label) in enumerate(zip(ids, instructions, responses, labels)):
        if i > first_n:
            break
        print('_'*50)
        print('Query ID:', id_)
        print('_'*50)
        print('Instruction to Generator:')
        print(instr)
        print()
        print('Generated Response:')
        print(response)
        print('Label:')
        print(label)
        print()
        print()


def print_rag_model(rag, retriever_kwargs,reranker_kwargs, generator_kwargs):
    print(':'*100)
    print('RAG Model:')
    # init modules
    print(f"Retriever: {retriever_kwargs['model_name']}")
    print(f"Reranker: {reranker_kwargs['model_name']}")
    print(f"Generator: {generator_kwargs['model_name']}")
    print(':'*100)
    print()




class LookupDatasetHF(datasets.Dataset):

    # def __init__(self, args, kwargs):
    #     super().__init__(*args, **kwargs)
    def add_index(self):
        # Add an index column to the dataset
        self = self.add_column("index", range(len(self)))

        # Create a mapping function from index to content_id
        self.id2idx_map = dict(zip(self["id"], self["index"]))
        print('added map')
        print(self.id2idx_map)

    def __getitem__(self, ids):
        if type(ids) != list():
            ids = list(ids)
        idx = [self.id2idx_map[id_] for id_ in ids]
        return self[idx]