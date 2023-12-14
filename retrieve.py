# Retrieve
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from datasets.fingerprint import Hasher
import os 
from shutil import rmtree

class Retrieve:
    def __init__(self, 
                 model_name=None, 
                 datasets=None, 
                 batch_size=1, 
                 batch_size_sim=1, 
                 top_k_documents=1, 
                 index_folder=None, 
                 pyserini_num_threads=1,
                 processing_num_proc=1,
                 prebuild_indexes=None,
                 ):
        
        self.prebuild_indexes = prebuild_indexes
        self.model_name = model_name
        self.batch_size = batch_size
        self.batch_size_sim = batch_size_sim
        self.top_k_documents = top_k_documents
        self.index_folder = index_folder
        self.pyserini_num_threads = pyserini_num_threads
        self.processing_num_proc = processing_num_proc
        # match model class
        if self.model_name == 'splade':
            from models.retrievers.splade import Splade
            retriever_class = Splade
        elif self.model_name == 'bm25':
            from models.retrievers.bm25 import BM25
            retriever_class = BM25
        else:
            #raise ValueError(f"Model {kwargs['model_name']} not implemented!")
            from models.retrievers.dpr import DPR
            retriever_class = DPR

        # instaniate model
        self.model = retriever_class(model_name=self.model_name)
        # preprocess dataset on init
        self.datasets = self.preprocess_datasets(datasets)

    def index(self, split, subset):
        dataset = self.datasets[split][subset]
        prebuild_index_name = self.get_prebuild_index_name(dataset.name)
        if prebuild_index_name == None:
            index_path = self.get_index_path(split, subset)
            # if dataset has not been encoded before
            if not os.path.exists(index_path):
                if self.model_name == 'bm25':
                    self.model.index(dataset, index_path, num_threads=self.pyserini_num_threads)
                else: 
                    os.makedirs(self.index_folder, exist_ok=True)
                    embs = self.encode(dataset)
                    torch.save(embs.detach().cpu(), index_path)
        return 

    @torch.no_grad()
    def retrieve(self, split, return_embeddings=False, sort_by_score=True, return_docs=False):
        dataset = self.datasets[split]
        prebuild_index_name = self.get_prebuild_index_name(dataset['doc'].name)
        doc_ids = dataset['doc']['id']
        q_ids = dataset['query']['id']
        if self.model_name == "bm25":
            index_path = self.get_index_path(split, 'doc')
            bm25_out = self.model(
                dataset['query'], 
                index_path=index_path, 
                top_k_documents=self.top_k_documents, 
                prebuild_index=prebuild_index_name, 
                batch_size=self.batch_size, 
                num_threads=self.pyserini_num_threads,
                return_docs=return_docs,
                )
            
            return bm25_out
        else:
            doc_embs = torch.load(index_path).to(self.model.device)
            q_embs = self.encode(dataset['query'])
            scores = self.sim_dot(q_embs, doc_embs)
            if sort_by_score:
                idxs_sorted = self.sort_by_score_indexes(scores)
                # get top-k indices
                idxs_sorted_top_k = idxs_sorted[:, :self.top_k_documents]
                # use sorted top-k indices indices to retrieve corresponding document embeddings
                doc_embs = doc_embs[idxs_sorted_top_k]
                # use sorted top-k indices to gather scores
                scores = scores.gather(dim=1, index=idxs_sorted_top_k)
                # Use sorted top-k indices indices to retrieve corresponding document IDs
                doc_ids = [[doc_ids[i] for i in q_idxs] for q_idxs in idxs_sorted_top_k]

            return {
                "doc_emb": doc_embs if return_embeddings else None,
                "q_emb": q_embs if return_embeddings else None,
                "score": scores,
                "q_id": q_ids,
                "doc_id": doc_ids
                }

    def encode(self, dataset):
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.model.collate_fn
            )
        embs_list = list()
        for batch in tqdm(dataloader, desc=f'Encoding: {self.model_name}'):
            outputs = self.model(batch)
            emb = outputs['embedding']
            embs_list.append(emb)
        embs = torch.cat(embs_list)
        return embs
    
    def sort_by_score_indexes(self, scores):
        idxs_sorted = list()
        for q_scores in scores:
            idx_sorted = torch.argsort(q_scores, descending=True)
            idxs_sorted.append(idx_sorted)
        idxs_sorted = torch.stack(idxs_sorted)
        return idxs_sorted

    def sim_dot(self, emb_q, emb_doc):
        scores = list()
        with torch.inference_mode():
            # !! perhaps OOM for very large corpora, might need to be batched for documents as well
            for emb_q_single in emb_q:
                scores_q = torch.matmul(emb_q_single, emb_doc.t())
                scores.append(scores_q)
        scores = torch.stack(scores)
        return scores

    def tokenize(self, example):
       return self.model.tokenize(example)
    
    def get_prebuild_index_name(self, dataset_name):
        return self.prebuild_indexes[self.model_name].get(dataset_name, None)
    
    def get_index_path(self, split, subset):
        return f'{self.index_folder}/{self.datasets[split][subset].name}_{split}_{subset}_{self.model_name.split("/")[-1]}'
    

    def preprocess_datasets(self, datasets):
        datasets = datasets
        # copy dataset to retriever class and tokenize when not using bm25
        # only tokenize if not using bm25
        if self.model_name != 'bm25':
            print(f'Processing datasets for Retriever...')
            for split in datasets:
                for subset in datasets[split]:
                    if datasets[split][subset] != None:
                        # apply tokenizer 
                        datasets[split][subset] = datasets[split][subset].map(self.tokenize, batched=True, num_proc=self.processing_num_proc)
                        # remove all non-model input fields
                        datasets[split][subset] = datasets[split][subset].remove_columns(['sentence', 'index'])

        return datasets

    # code in case dot product between query and all docs leads to OOM, needs to be revised!!
    def sim_dot_batch(self, q_emb_dataset, doc_emb_dataset, batch_size):
        raise NotImplementedError('this code is wrong and not finished!')
        dataloader_doc = DataLoader(doc_emb_dataset, batch_size=batch_size, shuffle=False)
        # Iterate through document embeddings batches
        scores_list = []
        print(len(dataloader_doc))
        for doc_emb in dataloader_doc:
            dataloader_queries = DataLoader(q_emb_dataset, batch_size=batch_size, shuffle=False)
            print(len(dataloader_queries))
            for q_emb in dataloader_queries:
                with torch.inference_mode():
                    batch_score = torch.matmul(q_emb.transpose(0,), doc_emb.t())
    
                scores_list.append(batch_score)
    
        # Concatenate results to obtain the final dot product matrix
        scores = torch.cat(scores_list)
    
        return scores





