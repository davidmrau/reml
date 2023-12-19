from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
import torch
from collections import defaultdict
from utils import make_hf_dataset, print_rag_model, print_generate_out
from dataset_processor import ProcessDatasets
from metric import Metrics

class RAG:
    def __init__(self, 
                 generator_config=None, 
                 retriever_config=None, 
                 reranker_config=None, 
                 experiment_folder=None, 
                 index_folder=None,
                 run_name=None, 
                 dataset_config=None, 
                 processing_num_proc=1,
                 dataset_folder=None,
                 overwrite_datasets=False,
                debug=False,
                ):

        self.dataset_folder = dataset_folder
        self.experiment_folder = experiment_folder
        self.run_name = run_name
        self.processing_num_proc = processing_num_proc
        self.index_folder = index_folder

        self.metrics = {
            "train": None,
            # lookup metric with dataset name (tuple: dataset_name, split) 
            "test": Metrics(dataset_config['test']['query'][0]), 
            "dev": None
        }
        with torch.no_grad(): 
            # process datasets, downloading, loading, covert to format
            self.datasets = ProcessDatasets.process(
                dataset_config, 
                out_folder=self.dataset_folder, 
                num_proc=processing_num_proc,
                overwrite=overwrite_datasets,
                debug=debug,
                )

        print_rag_model(self, retriever_config,reranker_config, generator_config)
        # init modules
        self.retriever = Retrieve(
                    **retriever_config, 
                    datasets=self.datasets, 
                    index_folder=self.index_folder,
                    processing_num_proc=processing_num_proc,
                    ) if retriever_config != None else None
        self.reranker = Rerank(**reranker_config) if reranker_config != None else None
        self.generator = Generate(**generator_config) if generator_config != None else None

               
    def retrieve(self):
        split = 'test'
        # index
        self.retriever.index(split=split, subset='doc')

        # retrieve
        out_retrieve = self.retriever.retrieve(split, return_embeddings=False)
        print(out_retrieve)

    def generate_simple(self):
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
        if self.reranker == self.generator == None:
            print(out_ranking)
            return out_ranking

        # rerank
        if self.reranker !=  None:
            rerank_dataset = make_hf_dataset(
                self.datasets[split], 
                query_ids, 
                doc_ids,
                multi_doc=False,
                pyserini_docs=docs
                )
            
            out_ranking = self.reranker.eval(rerank_dataset)
            query_ids, doc_ids, docs = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['doc']
       
        if self.generator != None: 
            gen_dataset = make_hf_dataset(
                self.datasets[split], 
                query_ids, 
                doc_ids, 
                multi_doc=True, 
                pyserini_docs=docs
                )
            query_ids, queries, instructions, responses, labels  = self.generator.eval(gen_dataset)
            metrics_out = self.metrics[split].compute(predictions=responses, references=labels, questions=queries)

            out_dict =  {
                    "instruction": instructions,
                    "response": responses,
                    "q_id": query_ids, 
                    "labels": labels,
                    "metrics": metrics_out
                }
            print_generate_out(out_dict)
            return out_dict
