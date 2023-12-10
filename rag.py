
from retrieve import Retrieve
from rerank import Rerank
from generate import Generate
from datasets import Dataset
import torch
from collections import defaultdict

class RAG:
    def __init__(self, generator_kwargs=None, retriever_kwargs=None, reranker_kwargs=None, experiment_folder=None, datasets=None):

        self.datasets = datasets
        
        # init modules 

        print(f"Loading Retriever: {retriever_kwargs['model_name']}")
        self.retriever = Retrieve(**retriever_kwargs, datasets=defaultdict(dict))

        print(f"Loading Reranker: {reranker_kwargs['model_name']}")
        self.reranker = Rerank(**reranker_kwargs, datasets=defaultdict(dict)) 

        print(f"Loading Generator: {generator_kwargs['model_name']}")
        self.generator = Generate(**generator_kwargs)

        self.experiment_folder = experiment_folder
        self.modules = {"retriever": self.retriever, "reranker": self.reranker, 'generator': self.generator}


        # process data on init
        self.process_data()

    def zero_shot_single_retrieval(self):
        out_retrieve = self.retriever.eval(return_embeddings=False)
        query_ids, doc_ids = out_retrieve['q_id'], out_retrieve['doc_id']
        assert len(doc_ids) == len(query_ids)
        print(query_ids)
        q_idxs = [ self.datasets['eval']['query'].id2index[id_] for id_ in query_ids]
        queries = self.datasets['eval']['query'][q_idxs]['sentence']
        responses = list()
        instructions = list()
        for i in range(len(query_ids)):
            d_idxs = [ self.datasets['eval']['doc'].id2index[id_] for id_ in doc_ids[i]]
            docs = self.datasets['eval']['doc'][d_idxs]
            instruction, response  = self.generator.eval(queries[i], docs)
            responses.append(response)
            instructions.append(instruction)
        return {
                "instruction": instructions,
                "response": responses
            }
    def index(self, ):
        #datatset_path = f'{}'
        output = retriever.encode(self.eval_dataset_doc, self.batch_size)
        ids, embs = output['id'], output['embedding']
        dataset = Dataset.from_dict({'id': ids, 'embedding': embs})
        
    def process_data(self):
        
        # make mapping from str id to index for easy lookup by id
        for split in self.datasets:
            for dataset in self.datasets[split]:
                if self.datasets[split][dataset] != None:
                    self.datasets[split][dataset] = self.datasets[split][dataset].add_column("index", range(len(self.datasets[split][dataset])))
                    self.datasets[split][dataset].id2index = dict(zip(self.datasets[split][dataset]["id"], self.datasets[split][dataset]["index"]))

        # process data for modules individually as they can have differnt tokenizers
        for mod_name, mod in self.modules.items():
            if mod_name == 'retriever':
                print(f'Processing dataset for {mod_name}...')
                for split in self.datasets:
                    for dataset in self.datasets[split]:
                        if self.datasets[split][dataset] != None:
                            print(f'Processing {split} {dataset}')
                            mod.datasets[split][dataset] = self.datasets[split][dataset].map(mod.tokenize, batched=True)
                            mod.datasets[split][dataset] = mod.datasets[split][dataset].remove_columns(['sentence'])
                            mod.datasets[split][dataset] = mod.datasets[split][dataset].remove_columns(['index'])
