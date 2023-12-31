# Retrieve
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
import os 
from shutil import rmtree
from datasets import Dataset
import glob
class Retrieve:
    def __init__(self, 
                 model_name=None, 
                 datasets=None, 
                 batch_size=1, 
                 batch_size_sim=1, 
                 top_k_documents=1, 
                 index_folder=None, 
                 pyserini_num_threads=1,
                 ):
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.batch_size_sim = batch_size_sim
        self.top_k_documents = top_k_documents
        self.index_folder = index_folder
        self.pyserini_num_threads = pyserini_num_threads
        # match model class
        if self.model_name == 'naver/splade-cocondenser-selfdistil':
            from models.retrievers.splade import Splade
            retriever_class = Splade
        elif self.model_name == 'bm25':
            from models.retrievers.bm25 import BM25
            retriever_class = BM25
        elif self.model_name == 'Shitao/RetroMAE':
            from models.retrievers.retromae import RetroMAE
            retriever_class = RetroMAE
        elif self.model_name == 'castorini/repllama-v1-7b-lora-passage':
            from models.retrievers.repllama import RepLlama
            retriever_class = RepLlama
        else:
            #raise ValueError(f"Model {kwargs['model_name']} not implemented!")
            from models.retrievers.dpr import DPR
            retriever_class = DPR

        # instaniate model
        self.model = retriever_class(model_name=self.model_name)

        self.datasets = datasets

    def index(self, split, query_or_doc):
        dataset = self.datasets[split][query_or_doc]
        dataset = dataset.remove_columns(['id'])
        index_path = self.get_index_path(split, query_or_doc)
        # if dataset has not been encoded before
        if not os.path.exists(index_path):
            if self.model_name == 'bm25':
                self.model.index(dataset, index_path, num_threads=self.pyserini_num_threads)
            else: 
                self.encode(dataset, query_or_doc=query_or_doc, detach=True, index_path=index_path)
                # self.save_index(embs, index_path)
    
    def save_index(self, embedding, index_path):
        os.makedirs(index_path, exist_ok=True)
        chunk_size = 250000  # Set the size of each chunk
        num_chunks = embedding.size(0) // chunk_size + 1

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, embedding.size(0))
            chunk_embedding = embedding[start_idx:end_idx, :].clone() 
            # Save each chunk
            chunk_save_path = self.get_chunk_path(index_path, i)
            torch.save(chunk_embedding, chunk_save_path)
        
    def retrieve(self, split, return_embeddings=False, return_docs=False):
        dataset = self.datasets[split]
        doc_ids = dataset['doc']['id']
        q_ids = dataset['query']['id']
        index_path = self.get_index_path(split, 'doc')
        if self.model_name == "bm25":
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
            q_embs = self.encode(dataset['query'], query_or_doc='query')
            #doc_embs = self.load_index(index_path)
            #scores_sorted_topk, indices_sorted_topk = self.similarity_dot(q_embs, doc_embs)
            scores_sorted_topk, indices_sorted_topk = self.load_collection_and_retrieve(q_embs, index_path)
            if return_embeddings:
                # use sorted top-k indices indices to retrieve corresponding document embeddings
                doc_embs = doc_embs[indices_sorted_topk]
            # Use sorted top-k indices indices to retrieve corresponding document IDs
            doc_ids = [[doc_ids[i] for i in q_idxs] for q_idxs in indices_sorted_topk]
            return {
                "doc_emb": doc_embs if return_embeddings else None,
                "q_emb": q_embs if return_embeddings else None,
                "score": scores_sorted_topk,
                "q_id": q_ids,
                "doc_id": doc_ids
                }

    @torch.no_grad() 
    def encode(self, dataset, query_or_doc=None, detach=False, index_path=None, chunk_size=1000):
        #continue_example = 135000
        if index_path != None:
            os.makedirs(index_path, exist_ok=True)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            # collate_fn=lambda batch: self.model.collate_fn(batch, query_or_doc) if query_or_doc != None else self.model.collate_fn(batch),
            collate_fn=lambda batch: self.model.collate_fn(batch, query_or_doc),
            num_workers=4
            )
        
        embs_list = list()
        for i, batch in tqdm(enumerate(dataloader), total=len(dataset)//self.batch_size, desc=f'Encoding: {self.model_name}', file=sys.stdout):
            # if i <= continue_example:
            #    continue
            outputs = self.model(batch)
            emb = outputs['embedding']
            if detach:
                emb = emb.detach().cpu()
            embs_list.append(emb)
            if index_path != None and i % chunk_size == 0 and i != 0:
                chunk_save_path = self.get_chunk_path(index_path, i)
                embs = torch.cat(embs_list)
                if 'splade' in self.model_name:
                    embs = embs.to_sparse()
                torch.save(embs, chunk_save_path)
                embs_list = list()
        if index_path != None:
            chunk_save_path = self.get_chunk_path(index_path, i)
            embs = torch.cat(embs_list)
            if 'splade' in self.model_name:
                embs = embs.to_sparse()
            torch.save(embs, chunk_save_path)

        if index_path != None:
            return None
        else:
            return torch.cat(embs_list)

    @torch.no_grad()
    def load_collection_and_retrieve(self, emb_q, index_path):
        emb_files = glob.glob(f'{index_path}/*.pt')
        sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

        top_k_scores_list, top_k_indices_list = [], []
        num_emb = 0
        for emb_file in tqdm(sorted_emb_files, total=len(sorted_emb_files), desc=f'Load and retrieve...'):
            emb_chunk = torch.load(emb_file)
            emb_chunk = emb_chunk.to(self.model.device)
            scores_q = torch.sparse.mm(emb_q, emb_chunk.t())  
            scores_sorted_q, indices_sorted_q = torch.topk(scores_q, self.top_k_documents, dim=1)
            top_k_scores_list.append(scores_sorted_q)
            top_k_indices_list.append(indices_sorted_q+num_emb)
            num_emb += emb_chunk.shape[0]
        # Concatenate top-k scores and indices from each chunk
        all_top_k_scores = torch.cat(top_k_scores_list, dim=1)
        all_top_k_indices = torch.cat(top_k_indices_list, dim=1)
        # Get final top-k scores and indices across all chunks
        final_top_k_scores, top_k_indices = torch.topk(all_top_k_scores, self.top_k_documents, dim=1)
        # Extract corresponding indices for final top-k scores
        final_top_k_indices = torch.gather(all_top_k_indices, 1, top_k_indices)
        return final_top_k_scores, final_top_k_indices

    def tokenize(self, example):
       return self.model.tokenize(example)
    
    def get_model_name(self):
        return self.model_name.replace('/', '_')

    def get_index_path(self, split, query_or_doc):
        return f'{self.index_folder}{self.datasets[split][query_or_doc].name}_{query_or_doc}_{self.get_model_name()}'
    
    def get_dataset_path(self, split, query_or_doc):
        return f'datasets/{self.datasets[split][query_or_doc].name}_{query_or_doc}_{self.get_model_name()}'

    def get_chunk_path(self, index_path, chunk):
        return f'{index_path}/embedding_chunk_{chunk+1}.pt'




    @torch.no_grad() 
    def similarity_dot(self, emb_q, emb_doc, batch_size=250):
        scores_sorted, indices_sorted = list(), list()
        emb_q = emb_q.half()
        # !! perhaps OOM for very large corpora, might need to be batched for documents as well
        for i in tqdm(range(0, emb_q.shape[0], self.batch_size_sim), desc=f'Retrieving docs...', total=emb_q.shape[0]//batch_size):
            emb_q_batch= emb_q[i:i+self.batch_size_sim]
            scores_q = torch.matmul(emb_q_batch, emb_doc.t())
            scores_sorted_q, indices_sorted_q = torch.topk(scores_q, self.top_k_documents, dim=1)
            scores_sorted_q = scores_sorted_q.detach().cpu()
            scores_sorted.append(scores_sorted_q)
            indices_sorted.append(indices_sorted_q)
        sorted_scores = torch.cat(scores_sorted, dim=0)
        indices_sorted = torch.cat(indices_sorted, dim=0)
        return sorted_scores, indices_sorted

    def load_index(self, index_path):
        emb_files = glob.glob(f'{index_path}/*.pt')
        sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        embs = list()
        for emb_file in tqdm(sorted_emb_files, total=len(sorted_emb_files), desc=f'Loading saved index {index_path}'):
            emb = torch.load(emb_file)
            embs.append(emb)
        embs = torch.cat(embs)
        return embs.to(self.model.device)

        