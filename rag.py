import os
from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
import torch
from collections import defaultdict
from utils import make_hf_dataset, get_by_ids
from datasets.fingerprint import Hasher

class RAG:
    def __init__(self, generator_kwargs=None, retriever_kwargs=None, reranker_kwargs=None, experiment_folder=None, run_name=None, datasets=None):

        self.datasets = datasets

        # add index for lookup by id
        self.add_index()

        # init modules
        print(f"Loading Retriever: {retriever_kwargs['model_name']}")
        self.retriever = Retrieve(**retriever_kwargs, datasets=datasets)

        print(f"Loading Reranker: {reranker_kwargs['model_name']}")
        self.reranker = Rerank(**reranker_kwargs)

        print(f"Loading Generator: {generator_kwargs['model_name']}")
        self.generator = Generate(**generator_kwargs)

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

        assert len(doc_ids) == len(query_ids)

        # rerank
        if self.reranker.model != None:
            rerank_dataset = make_hf_dataset(self.datasets[split], query_ids, doc_ids, multi_doc=False)
            out_rerank = self.reranker.eval(rerank_dataset)
            query_ids, doc_ids = out_retrieve['q_id'], out_retrieve['doc_id']

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
        if os.path.exists(dataset_path):
            print(f'Loading Dataset {split} {subset} from disk...')
            embs = torch.load(f"{dataset_path}_embedding.pt").to(self.retriever.model.device)
        else:
            os.makedirs(dataset_path)
            embs = self.retriever.encode(self.retriever.datasets[split][subset])
            torch.save(embs.detach().cpu(), f'{dataset_path}_embedding.pt')
        return embs


    def add_index(self):
        # make mapping from str id to index for easy lookup by id
        for split in self.datasets:
            for subset in self.datasets[split]:
                if self.datasets[split][subset] != None:
                    self.datasets[split][subset] = self.datasets[split][subset].add_column("index", range(len(self.datasets[split][subset])))
                    self.datasets[split][subset].id2index = dict(zip(self.datasets[split][subset]["id"], self.datasets[split][subset]["index"]))
