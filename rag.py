import os
from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
import torch
from collections import defaultdict
from utils import make_hf_dataset
from datasets.fingerprint import Hasher
from shutil import rmtree

class RAG:
    def __init__(self, generator_kwargs=None, retriever_kwargs=None, reranker_kwargs=None, experiment_folder=None, run_name=None, datasets=None):

        self.datasets = datasets

        # add index for lookup by id
        self.add_index()

        print(':'*100)
        print('RAG Model:')
        # init modules
        print(f"Loading Retriever: {retriever_kwargs['model_name']}")
        self.retriever = Retrieve(**retriever_kwargs, datasets=datasets)

        print(f"Loading Reranker: {reranker_kwargs['model_name']}")
        self.reranker = Rerank(**reranker_kwargs)

        print(f"Loading Generator: {generator_kwargs['model_name']}")
        self.generator = Generate(**generator_kwargs)
        print(':'*100)
        print()

        self.experiment_folder = experiment_folder
        self.run_name = run_name


    def zero_shot_single_retrieval(self):
        # todo save ids in emb perhaps
        split = 'eval'
        # index
        doc_embs = self.index(split=split, subset='doc')
        # retrieve
        out_retrieve = self.retriever.retrieve(doc_embs, return_embeddings=False)
        query_ids, doc_ids = out_retrieve['q_id'], out_retrieve['doc_id']


        # rerank
        if self.reranker.model != None:
            rerank_dataset = make_hf_dataset(self.datasets[split], query_ids, doc_ids, multi_doc=False)
            out_rerank = self.reranker.eval(rerank_dataset)
            query_ids, doc_ids = out_rerank['q_id'], out_rerank['doc_id']
        gen_dataset = make_hf_dataset(self.datasets[split], query_ids, doc_ids, multi_doc=True)
        query_ids, instructions, responses  = self.generator.eval(gen_dataset)
        return {
                "instruction": instructions,
                "response": responses,
                "q_id": query_ids
            }
    
    @torch.no_grad()
    def index(self, split, subset):
        hash_ = Hasher.hash(self.retriever.datasets[split][subset])
        dataset_path = f'{self.experiment_folder}/{self.run_name}/{hash_}'
        # makes folder, encodes, and saves embeddings
        def create_index(dataset_path):
            os.makedirs(dataset_path)
            embs = self.retriever.encode(self.retriever.datasets[split][subset])
            torch.save(embs.detach().cpu(), f'{dataset_path}_embedding.pt')
            return embs
        # if dataset has been encoded before
        if os.path.exists(dataset_path):
            print(f'Loading Dataset {split} {subset} from disk...')
            try:
                embs = torch.load(f"{dataset_path}_embedding.pt").to(self.retriever.model.device)
            except:
                # if loading fails, delete and build again
                print('Index is corrupted bulding new index!')
                rmtree(dataset_path)
                embs = create_index(dataset_path)
        else:
            embs = create_index(dataset_path)
        return embs

    def add_index(self):
        # make mapping from str id to index for easy lookup by id
        for split in self.datasets:
            for subset in self.datasets[split]:
                if self.datasets[split][subset] != None:
                    self.datasets[split][subset] = self.datasets[split][subset].add_column("index", range(len(self.datasets[split][subset])))
                    self.datasets[split][subset].id2index = dict(zip(self.datasets[split][subset]["id"], self.datasets[split][subset]["index"]))
