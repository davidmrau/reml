from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
import torch
from collections import defaultdict
from utils import make_hf_dataset, print_rag_model, print_generate_out, write_trec, write_generated, load_trec
from dataset_processor import ProcessDatasets
from metric import Metrics
import time 
import os 

class RAG:
    def __init__(self, 
                 generator_config=None, 
                 retriever_config=None, 
                 reranker_config=None, 
                 experiment_folder=None, 
                 run_folder=None,
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
        self.run_folder = run_folder
        self.run_name = run_name
        self.processing_num_proc = processing_num_proc
        self.index_folder = index_folder
        self.dataset_config = dataset_config
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
                    ) if retriever_config != None else None
        self.reranker = Rerank(**reranker_config) if reranker_config != None else None
        self.generator = Generate(**generator_config) if generator_config != None else None      

    def retrieve(self):
        split = 'test'
        # index
        self.retriever.index(split=split, query_or_doc='doc')

        # retrieve
        out_retrieve = self.retriever.retrieve(split, return_embeddings=False)
        print(out_retrieve)

    def default(self, split):
        fetch_pyserini_docs=False
        dataset_name = self.dataset_config[split]["query"][0]
        if self.retriever !=  None:
            ranking_file = f'{self.run_folder}/run.retrieve.top_{self.retriever.top_k_documents}.{dataset_name}.{split}.{self.retriever.get_model_name()}.trec'
            if not os.path.exists(ranking_file):
                # index
                self.retriever.index(split, query_or_doc='doc')
                # retrieve
                out_ranking = self.retriever.retrieve(
                    split, 
                    return_embeddings=False,
                    return_docs=fetch_pyserini_docs,
                    )
                query_ids, doc_ids, scores = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['score']
                docs = out_ranking['doc'] if fetch_pyserini_docs else None
                write_trec(ranking_file, query_ids, doc_ids, scores)
            else:
                query_ids, doc_ids, scores = load_trec(ranking_file)
                docs = None
        else:
            docs, query_ids, doc_ids = None, None, None

        # rerank
        if self.reranker !=  None:
            reranking_file = f'{self.run_folder}/run.rerank.top_{self.reranker.top_k_documents}.{dataset_name}.{split}.{self.retriever.get_model_name()}.trec'
            if not os.path.exists(reranking_file):
                rerank_dataset = make_hf_dataset(
                        self.datasets[split], 
                        query_ids, 
                        doc_ids,
                        multi_doc=False,
                        pyserini_docs=docs
                    )
                out_ranking = self.reranker.eval(rerank_dataset)
                query_ids, doc_ids, docs, scores = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['doc'], out_ranking['score']
                write_trec(reranking_file, query_ids, doc_ids, scores)
            else:
                query_ids, doc_ids, scores = load_trec(reranking_file)

        if self.generator != None: 
            gen_dataset = make_hf_dataset(
                self.datasets[split], 
                query_ids, 
                doc_ids, 
                multi_doc=True, 
                pyserini_docs=docs
                )
            generation_start = time.time()
            query_ids, queries, instructions, responses, labels  = self.generator.eval(gen_dataset)
            generation_time_seconds = time.time() - generation_start
            metrics_out = self.metrics[split].compute(
                predictions=responses, 
                references=labels, 
                questions=queries
                )
            out_folder = f'{self.experiment_folder}/{self.run_name}/'
            write_generated(
                out_folder,
                query_ids, 
                instructions, 
                responses, 
                labels, 
                metrics_out, 
                generation_time_seconds
                )

            print_generate_out(
                responses,
                query_ids, 
                labels, 
                metrics_out)
