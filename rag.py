from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
import torch
from collections import defaultdict
from utils import make_hf_dataset, print_rag_model
from dataset import ProcessDatasets
from metric import Metrics

class RAG:
    def __init__(self, 
                 generator_kwargs=None, 
                 retriever_kwargs=None, 
                 reranker_kwargs=None, 
                 experiment_folder=None, 
                 index_folder=None,
                 run_name=None, 
                 dataset_names=None, 
                 processing_num_proc=1,
                 dataset_folder='datasets',
                 overwrite_datasets=False,
                 prebuild_indexes=defaultdict(dict),
                ):

        self.dataset_folder = dataset_folder
        self.experiment_folder = experiment_folder
        self.run_name = run_name
        self.processing_num_proc = processing_num_proc
        self.index_folder = index_folder

        self.metrics = {
            "train": None, 
            "test": Metrics(dataset_names['test']['query'][0]), 
            "dev": None
        }

        # process datasets, downloading, loading, covert to format
        self.datasets = ProcessDatasets.process(
            dataset_names, 
            out_folder=self.dataset_folder, 
            num_proc=processing_num_proc,
            overwrite=overwrite_datasets,
            )

        print_rag_model(self, retriever_kwargs,reranker_kwargs, generator_kwargs)
        # init modules
        self.retriever = Retrieve(
                    **retriever_kwargs, 
                    datasets=self.datasets, 
                    index_folder=self.index_folder,
                    processing_num_proc=processing_num_proc,
                    prebuild_indexes=prebuild_indexes,
                    )
        self.reranker = Rerank(**reranker_kwargs)
        self.generator = Generate(**generator_kwargs)

              
    
    def retrieve(self):
        # todo save ids in emb perhaps
        split = 'test'
        # index
        self.index(split=split, subset='doc')

        # retrieve
        out_retrieve = self.retriever.retrieve(split, return_embeddings=False)
        print(out_retrieve)

    def generate_simple(self):
        # todo save ids in emb perhaps
        split = 'test'
        # index
        self.retriever.index(split, 'doc')
        # retrieve

        fetch_pyserini_docs=False

        out_ranking = self.retriever.retrieve(
            split, 
            return_embeddings=False,
            return_docs=fetch_pyserini_docs,
            )
        query_ids, doc_ids, docs = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['doc']

        # rerank
        if self.reranker.model != None:
            rerank_dataset = make_hf_dataset(
                self.datasets[split], 
                query_ids, 
                doc_ids,
                multi_doc=False,
                pyserini_docs=docs
                )
            
            out_ranking = self.reranker.eval(rerank_dataset)
            query_ids, doc_ids, docs = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['doc']
        gen_dataset = make_hf_dataset(
            self.datasets[split], 
            query_ids, 
            doc_ids, 
            multi_doc=True, 
            pyserini_docs=docs
            )
        query_ids, instructions, responses, labels  = self.generator.eval(gen_dataset)
        metrics_out = self.metrics[split].compute(predictions=responses, references=labels)

        return {
                "instruction": instructions,
                "response": responses,
                "q_id": query_ids, 
                "labels": labels,
                "metrics": metrics_out
            }
