from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
from utils import *
from dataset_processor import ProcessDatasets
from metric import Metrics
import time 
import os 
import json
import shutil
from datasets.fingerprint import Hasher
from omegaconf import OmegaConf


class RAG:
    def __init__(self, 
                generator_config=None, 
                retriever_config=None, 
                reranker_config=None, 
                runs_folder=None,
                run_name=None, 
                dataset_config=None, 
                processing_num_proc=1,
                dataset_folder='datasets/',
                index_folder='indexes/',
                experiments_folder='experiments/', 
                overwrite_datasets=False,
                retrieve_top_k=1,
                rerank_top_k=1,
                pyserini_num_threads=1,
                config=None,
                debug=False,

                ):
        
        
        
        self.dataset_folder = dataset_folder
        self.experiments_folder = experiments_folder
        self.runs_folder = runs_folder
        self.run_name = run_name
        self.processing_num_proc = processing_num_proc
        self.index_folder = index_folder
        self.dataset_config = dataset_config
        self.config = config
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.pyserini_num_threads = pyserini_num_threads
        # print RAG model
        print_rag_model(self, retriever_config, reranker_config, generator_config)
        # init experiment (set run name, create dirs)
        self.run_name, self.experiment_folder = self.init_experiment(config, experiments_folder, index_folder, runs_folder, run_name)

        # process datasets, downloading, loading, covert to format
        self.datasets = ProcessDatasets.process(
            dataset_config, 
            out_folder=self.dataset_folder, 
            num_proc=processing_num_proc,
            overwrite=overwrite_datasets,
            debug=debug,
            )
        self.metrics = {
            "train": None,
            # lookup metric with dataset name (tuple: dataset_name, split) 
            "test": Metrics(self.datasets['test']['query'].name), 
            "dev": None
        }
        # init modules
        self.retriever = Retrieve(
                    **retriever_config, 
                    datasets=self.datasets, 
                    index_folder=self.index_folder,
                    top_k_documents=self.retrieve_top_k,
                    pyserini_num_threads=self.pyserini_num_threads,
                    ) if retriever_config != None else None
        self.reranker = Rerank(
            **reranker_config,
            top_k_documents=self.rerank_top_k,
            ) if reranker_config != None else None
        self.generator = Generate(**generator_config) if generator_config != None else None   

    def default(self, split):
        fetch_pyserini_docs=False
        # retrieve
        query_ids, doc_ids, docs, scores = self.retrieve(split)
        # rerank
        query_ids, doc_ids, docs, scores = self.rerank(split)
        # generate
        questions, instructions, predictions, references = self.generate(split, query_ids, doc_ids, docs, scores)
        # eval metrics
        self.eval_metrics(split, questions, predictions, references)

        move_finished_experiment(self.experiment_folder)


    def retrieve(self, split):
        
        if self.retriever ==  None:
            return None, None, None, None

        dataset_name = self.datasets[split]["query"].name
        ranking_file = f'{self.runs_folder}/run.retrieve.top_{self.retriever.top_k_documents}.{dataset_name}.{split}.{self.retriever.get_model_name()}.trec'
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
            # copy ranking file to experiment folder                 
            shutil.copyfile(ranking_file, f'{self.experiment_folder}/{ranking_file.split("/")[-1]}')
            query_ids, doc_ids, scores = load_trec(ranking_file)
            docs = None

        return query_ids, doc_ids, docs, scores

    def rerank(self, split):
        
        if self.reranker ==  None:
            return None, None, None, None

        dataset_name = self.datasets[split]["query"].name
        reranking_file = f'{self.runs_folder}/run.rerank.top_{self.reranker.top_k_documents}.{dataset_name}.{split}.{self.retriever.get_model_name()}.trec'
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
            # copy reranking file to experiment folder 
            shutil.copyfile(reranking_file, f'{self.experiment_folder}/{reranking_file.split("/")[-1]}')
            query_ids, doc_ids, scores = load_trec(reranking_file)
            docs = None
        
        return query_ids, doc_ids, docs, scores


    def generate(self, split, query_ids, doc_ids, docs, scores):
        if self.generator == None: 
            return None, None, None, None, None

        gen_dataset = make_hf_dataset(
            self.datasets[split], 
            query_ids, 
            doc_ids, 
            multi_doc=True, 
            pyserini_docs=docs
            )

        generation_start = time.time()
        query_ids, questions, instructions, predictions, references  = self.generator.eval(gen_dataset)
        generation_time = time.time() - generation_start
        write_generated(
            self.experiment_folder,
            f"eval_{split}_out.json",
            query_ids, 
            instructions, 
            predictions, 
            references, 
        )

        print_generate_out(
            questions,
            instructions,
            predictions,
            query_ids, 
            references)

        formated_time_dict = {"Generation time": time.strftime("%H:%M:%S.{}".format(str(generation_time % 1)[2:])[:11], time.gmtime(generation_time))}
        write_dict(self.experiment_folder, f"eval_{split}_generation_time.json", formated_time_dict)

        return questions, instructions, predictions, references

    def eval_metrics(self, split, questions, predictions, references):
        if predictions == references == questions == None:
            return

        metrics_out = self.metrics[split].compute(
        predictions=predictions, 
        references=references, 
        questions=questions
        )
        write_dict(self.experiment_folder, f"eval_{split}_metrics.json", metrics_out)


    def init_experiment(self, config, experiments_folder, index_folder, runs_folder, run_name):
        # if run_name != None hash self to get run_name to avoid overwriting and exp. folder mess
        run_name = f'tmp_{Hasher.hash(str(config))}' if run_name == None else run_name
        experiment_folder = f"{experiments_folder}/{run_name}"
        # get name of finished experiment
        finished_exp_name = get_finished_experiment_name(experiment_folder)
        print(finished_exp_name)
        # if experiment already exists finished quit
        if os.path.exists(finished_exp_name):
            raise OSError(f"Experiment {finished_exp_name} already exists!")

        # make dirs
        os.makedirs(experiments_folder, exist_ok=True)
        os.makedirs(index_folder, exist_ok=True)
        os.makedirs(experiment_folder, exist_ok=True)
        # save entire config 
        OmegaConf.save(config=config, f=f"{experiment_folder}/config.yaml")
        # print config
        print(OmegaConf.to_yaml(config))
        return run_name, experiment_folder

