import os 
from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
from datasets import Dataset
import torch
from collections import defaultdict

from datasets.fingerprint import Hasher

class RAG:
    def __init__(self, generator_kwargs=None, retriever_kwargs=None, reranker_kwargs=None, experiment_folder=None, datasets=None):

        self.datasets = datasets
        
        # init modules 

        print(f"Loading Retriever: {retriever_kwargs['model_name']}")
        self.retriever = Retrieve(**retriever_kwargs, datasets=defaultdict(dict))

        print(f"Loading Reranker: {reranker_kwargs['model_name']}")
        self.reranker = Rerank(**reranker_kwargs) 

        print(f"Loading Generator: {generator_kwargs['model_name']}")
        self.generator = Generate(**generator_kwargs)

        self.experiment_folder = experiment_folder
        self.modules = {"retriever": self.retriever, "reranker": self.reranker, 'generator': self.generator}

        # process data on init
        self.process_data()

    def zero_shot_single_retrieval(self):
        # todo save ids in emb perhaps
        # index
        split = 'eval'
        doc_embs = self.index(split=split, subset='doc') 
        # retrieve
        out_retrieve = self.retriever.retrieve(doc_embs, return_embeddings=False)
        query_ids, doc_ids = out_retrieve['q_id'], out_retrieve['doc_id']
        assert len(doc_ids) == len(query_ids)

        # rerank
        if self.reranker.model != None:
            rerank_dataset = self.make_hf_dataset(self.datasets[split], query_ids, doc_ids, multi_doc=False)
            out_rerank = self.reranker.eval(rerank_dataset)
            query_ids, doc_ids = out_retrieve['q_id'], out_retrieve['doc_id']
        
        gen_dataset = self.make_hf_dataset(self.datasets[split], query_ids, doc_ids, multi_doc=True)
        query_ids, instructions, responses  = self.generator.eval(gen_dataset)
        return {
                "instruction": instructions,
                "response": responses,
                "q_id": query_ids
            }

    @torch.no_grad()
    def index(self, split, subset):
        hash_ = Hasher.hash(self.retriever.datasets[split][subset])
        dataset_path = f'{self.experiment_folder}/{hash_}'
        if os.path.exists(dataset_path):
            embs = torch.load(f"{dataset_path}_embedding.pt").to(self.retriever.model.device)
        else:
            os.makedirs(dataset_path)
            embs = self.retriever.encode(self.retriever.datasets[split][subset])
            torch.save(embs.detach().cpu(), f'{dataset_path}_embedding.pt')
        return embs 
            
        
    def process_data(self):        
        # make mapping from str id to index for easy lookup by id
        for split in self.datasets:
            for subset in self.datasets[split]:
                if self.datasets[split][subset] != None:
                    self.datasets[split][subset] = self.datasets[split][subset].add_column("index", range(len(self.datasets[split][subset])))
                    self.datasets[split][subset].id2index = dict(zip(self.datasets[split][subset]["id"], self.datasets[split][subset]["index"]))

        # tokenize collection
        # process data for modules individually as they can have differnt tokenizers
        for mod_name, mod in self.modules.items():
            if mod_name == 'retriever':
                print(f'Processing dataset for {mod_name}...')
                for split in self.datasets:
                    for subset in self.datasets[split]:
                        if self.datasets[split][subset] != None:
                            print(f'Processing {split} {subset}')
                            mod.datasets[split][subset] = self.datasets[split][subset].map(mod.tokenize, batched=True)
                            mod.datasets[split][subset] = mod.datasets[split][subset].remove_columns(['sentence'])
                            mod.datasets[split][subset] = mod.datasets[split][subset].remove_columns(['index'])

    def get_by_ids(self, dataset, ids):
        idxs = [ dataset.id2index[id_] for id_ in ids]
        return dataset[idxs]['sentence']
    
    def make_hf_dataset(self, dataset, q_ids, d_ids, multi_doc=False):
        dataset_dict = {'query': [], 'doc': [], 'q_id': []}
        if not multi_doc:
            dataset_dict['d_id'] = []
        
        queries = self.get_by_ids(dataset['query'], q_ids)
        for i, q_id in enumerate(q_ids):
            if multi_doc:
                docs = self.get_by_ids(dataset['doc'], d_ids[i])
                dataset_dict['query'].append(queries[i])
                dataset_dict['doc'].append(docs)
                dataset_dict['q_id'].append(q_id)
            else:
                docs = self.get_by_ids(dataset['doc'], d_ids[i])
                for d_id, doc in zip(d_ids[i], docs):
                    dataset_dict['d_id'].append(d_id)
                    dataset_dict['query'].append(queries[i])
                    dataset_dict['doc'].append(doc)
                    dataset_dict['q_id'].append(q_id) 
        return Dataset.from_dict(dataset_dict)

