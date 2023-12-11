
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

# reranking
class Rerank():
    def __init__(self, model_name=None, batch_size=1, top_k_documents=1):

        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k_documents = top_k_documents


        if self.model_name == None:
            self.model = None
        else:

            from models.rerankers.crossencoder import CrossEncoder
            # instaniate model
            self.model = CrossEncoder(model_name=self.model_name)

    def eval(self, dataset, return_embeddings=False):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.model.collate_fn)
        q_ids = list()
        d_ids = list()
        scores = list()
        for batch in tqdm(dataloader, desc='Encoding'):
            q_ids += batch.pop('q_id')
            d_ids += batch.pop('d_id')
            outputs = self.model(batch)
            scores.append(outputs['score'])
            if return_embeddings:
                emb = outputs['embedding']
                embs_list.append(emb)

        raise NotImplementedError('check d_ids')
        scores = torch.cat(scores)
        idxs_sorted = self.sort_by_score_indexes(scores)
        # get top-k indices
        idxs_sorted_top_k = idxs_sorted[:, :self.top_k_documents]
        if return_embeddings:
            # use sorted top-k indices indices to retrieve corresponding document embeddings
            doc_embs = doc_embs[idxs_sorted_top_k]
        # use sorted top-k indices to gather scores
        scores = scores.gather(dim=1, index=idxs_sorted_top_k)
        print(scores)
        # Use sorted top-k indices indices to retrieve corresponding document IDs
        d_ids = [[d_ids[i] for i in q_idxs] for q_idxs in idxs_sorted_top_k]

        ## Use sorted top-k indices indices to retrieve corresponding document IDs
        #d_ids = [d_ids[i] for i in idxs_sorted]

        return {
            "doc_emb": doc_embs if return_embeddings else None,
            "q_emb": q_embs if return_embeddings else None,
            "score": scores,
            "q_id": q_ids,
            "doc_id": d_ids
            }

    def sort_by_score_indexes(self, scores):
        idxs_sorted = list()
        for q_scores in scores:
            idx_sorted = torch.argsort(q_scores, descending=True)
            idxs_sorted.append(idx_sorted)
        idxs_sorted = torch.stack(idxs_sorted)
        return idxs_sorted
