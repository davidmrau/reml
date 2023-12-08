# Retrieve



from torch.utils.data import DataLoader
import torch

class Retrieve():
    def __init__(self, kwargs):
        self.model_name = kwargs['model_name']
        self.eval_dataset_q = kwargs['eval_dataset_q']
        self.eval_dataset_doc = kwargs['eval_dataset_doc']

        # match model class
        if self.model_name == 'splade':
            from models.retrievers.splade import Splade
            retriever_class = Splade
        elif self.model_name == 'BM25':
            from models.retrievers.bm25 import BM25
            retriever_class = BM25
        else:
            #raise ValueError(f"Model {kwargs['model_name']} not implemented!")
            from models.retrievers.dpr import DPR
            retriever_class = DPR

        # instaniate model
        self.module = retriever_class(kwargs)

        self.batch_size = kwargs['batch_size']
        self.batch_size_sim = kwargs['batch_size_sim']
        self.top_k_documents = kwargs['top_k_documents']

    def eval(self, return_embeddings=False, sort_by_score=True):
        output_q = self.module.encode(self.eval_dataset_q, self.batch_size)
        output_doc = self.module.encode(self.eval_dataset_doc, self.batch_size)
        q_ids, q_embs = output_q['ids'], output_q['embeddings']
        doc_ids, doc_embs = output_doc['ids'], output_doc['embeddings']
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
            "doc_embs": doc_embs if return_embeddings else None,
            "q_embs": q_embs if return_embeddings else None,
            "scores": scores,
            "q_ids": q_ids,
            "doc_ids": doc_ids
            }

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

    # def sim_dot_batch(self, q_emb_dataset, doc_emb_dataset, batch_size):
    #     raise NotImplemtedError('this code is wrong and not finished!')
    #     dataloader_doc = DataLoader(doc_emb_dataset, batch_size=batch_size, shuffle=False)
    #     # Iterate through document embeddings batches
    #     scores_list = []
    #     print(len(dataloader_doc))
    #     for doc_emb in dataloader_doc:
    #         dataloader_queries = DataLoader(q_emb_dataset, batch_size=batch_size, shuffle=False)
    #         print(len(dataloader_queries))
    #         for q_emb in dataloader_queries:
    #             with torch.inference_mode():
    #                 batch_score = torch.matmul(q_emb.transpose(0,), doc_emb.t())
    #
    #             scores_list.append(batch_score)
    #
    #     # Concatenate results to obtain the final dot product matrix
    #     scores = torch.cat(scores_list)
    #
    #     return scores
