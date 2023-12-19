
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from collections import defaultdict

# reranking
class Rerank():
    def __init__(self, model_name=None, batch_size=1, top_k_documents=1):

        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k_documents = top_k_documents

        # init model
        from models.rerankers.crossencoder import CrossEncoder
        # instaniate model
        self.model = CrossEncoder(model_name=self.model_name)

    def eval(self, dataset, return_embeddings=False):
        # get dataloader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.model.collate_fn)
        q_ids, d_ids, scores, embs_list = list(), list(), list(), list()
        # run inference on the dataset
        for batch in tqdm(dataloader, desc=f'Reranking: {self.model_name}'):
            q_ids += batch.pop('q_id')
            d_ids += batch.pop('d_id')
            outputs = self.model(batch)
            score = outputs['score']
            scores.append(score)
            if return_embeddings:
                emb = outputs['embedding']
                embs_list.append(emb)
 
        # get flat tensor of scores        
        scores = torch.cat(scores).ravel()
        # sort by scores 
        q_ids_sorted, d_ids_sorted, scores_sorted = self.sort_by_score_indexes(scores, q_ids, d_ids)

        return {
            "emb": embs_list if return_embeddings else None,
            "score": scores_sorted,
            "doc_id": d_ids_sorted,
            "q_id": q_ids_sorted,
            }
    
   # takes scores (num_queries x top_k) and q_ids and sortes them per q_id by score and returns list of sorted idxs
    def sort_by_score_indexes(self, scores, q_ids, d_ids):
        ranking = defaultdict(list)
        q_ids_sorted, doc_ids_sorted, scores_sorted = list(), list(), list()
        for i, (q_id, d_id) in enumerate(zip(q_ids, d_ids)):
            ranking[q_id].append((scores[i], d_id))
        for q_id in ranking:
            # sort (score, doc_id) by score
            sorted_list = sorted(ranking[q_id], key=lambda x: x[0])
            score_sorted, d_id_sorted = zip(*sorted_list)
            scores_sorted.append(torch.stack(score_sorted))
            doc_ids_sorted.append(list(d_id_sorted))
            q_ids_sorted.append(q_id)
        return q_ids_sorted, doc_ids_sorted, scores_sorted
