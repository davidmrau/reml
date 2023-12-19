# Retrieve
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
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
                 ):
        
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
        elif self.model_name == 'retromae':
            from models.retrievers.retromae import RetroMAE
            retriever_class = RetroMAE
        else:
            #raise ValueError(f"Model {kwargs['model_name']} not implemented!")
            from models.retrievers.dpr import DPR
            retriever_class = DPR

        # instaniate model
        self.model = retriever_class(model_name=self.model_name)
        # preprocess dataset on init
        self.datasets = datasets
        self.datasets = self.preprocess_datasets(datasets)

    def index(self, split, subset):
        print(self.datasets)
        dataset = self.datasets[split][subset]
        index_path = self.get_index_path(split, subset)
        # if dataset has not been encoded before
        if not os.path.exists(index_path):
            if self.model_name == 'bm25':
                self.model.index(dataset, index_path, num_threads=self.pyserini_num_threads)
            else: 
                os.makedirs(self.index_folder, exist_ok=True)
                embs = self.encode(dataset)
                self.save_index(embs, index_path)
    
    @torch.no_grad 
    def save_index(embs, index_path):
        embs = embs.detach().cpu()
        chunk_size = 250000  # Set the size of each chunk
        num_chunks = embedding.size(0) // chunk_size + 1

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, embedding.size(0))
            chunk_embedding = embedding[start_idx:end_idx, :] 
            # Save each chunk
            chunk_save_path = self.get_chunk_path(index_path, chunk)
            torch.save(chunk_embedding, chunk_save_path)
        
    @torch.no_grad()
    def retrieve(self, split, return_embeddings=False, sort_by_score=True, return_docs=False):
        dataset = self.datasets[split]
        doc_ids = dataset['doc']['id']
        q_ids = dataset['query']['id']
        if self.model_name == "bm25":
            index_path = self.get_index_path(split, 'doc')
            bm25_out = self.model(
                dataset['query'], 
                index_path=index_path, 
                top_k_documents=self.top_k_documents, 
                batch_size=self.batch_size, 
                num_threads=self.pyserini_num_threads,
                return_docs=return_docs,
                )
            
            return bm25_out
        else:
            doc_embs = self.load_index(index_path)
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

    def load_index(self, index_path):
        num_emb_files = len(glob.glob(f'{index_path}/*.pt'))
        embs = list()
        for i in range(num_emb_files):
            chunk_path = self.get_chunk_path(index_path, i)
            emb = torch.load(chunk_path)
            embs.append(emb)
        embs = torch.concat(embs, dim=1)
        return embs.to(self.model.device)
            
    def encode(self, dataset):
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.model.collate_fn
            )
        embs_list = list()
        for i, batch in tqdm(enumerate(dataloader, desc=f'Encoding: {self.model_name}')):
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

    @torch.no_grad
    def sim_dot(self, emb_q, emb_doc):
        scores = list()
        # !! perhaps OOM for very large corpora, might need to be batched for documents as well
        for emb_q_single in emb_q:
            scores_q = torch.matmul(emb_q_single, emb_doc.t())
            scores.append(scores_q)
        scores = torch.stack(scores)
        return scores

    def tokenize(self, example):
       return self.model.tokenize(example)
    
    
    def get_index_path(self, split, subset):
        return f'{self.index_folder}/{self.datasets[split][subset].name}_{subset}_{self.model_name.split("/")[-1]}'
    
    def get_chunk_path(self, index_path, chunk):
        return f'{index_path}/embedding_chunk_{chunk + 1}.pt'

    def preprocess_datasets(self, datasets):
        # copy dataset to retriever class and tokenize when not using bm25
        # only tokenize if not using bm25
        if self.model_name != 'bm25':
            for split in datasets:
                for subset in datasets[split]:
                    if datasets[split][subset] != None:
                        index_folder = self.get_index_path(split, subset)
                        if os.path.exists(index_folder):
                            dataset = datasets.Dataset.load_from_disk(index_folder)
                        else: 
                            dataset = datasets[split][subset]
                            # apply tokenizer 
                            dataset = dataset.map(self.tokenize, batched=True, num_proc=self.processing_num_proc, desc=f'Processing dataset {split} {subset} for {self.model_name}')
                            # remove all non-model input fields
                            dataset = dataset.remove_columns(['content'])
                            dataset.save_to_disk(index_folder)
                        datasets[split][subset] = dataset
         
        return datasets
