
from retrieve import Retrieve
from rerank import Rerank
from generate import Generate

class RAG:
    def __init__(self, generator_kwargs, retriever=None, reranker=None):

        self.eval_dataset_q = generator_kwargs['eval_dataset_q']
        self.eval_dataset_doc = generator_kwargs['eval_dataset_doc']
        self.model_name = generator_kwargs['model_name']
        self.retriever = retriever
        self.reranker = reranker
        self.modules = {"retriever": self.retriever, "reranker": self.reranker}
        # match model class
        if self.model_name == None:
            print('Not using a Generator!')
            self = None
        elif self.model_name == 'dummy':
            from models.generators.dummy import Dummy
            generator_class = Dummy
        elif self.model_name == 'meta-llama/Llama-2-7b-chat-hf':
            from models.generators.llama2 import Llama2
            generator_class = Llama2
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented!")

        # instatiate model
        self.generator = generator_class(generator_kwargs)

        # make idx to id mapping
        self.eval_dataset_q = self.eval_dataset_q.add_column("index", range(len(self.eval_dataset_q)))
        self.q_map_eval = dict(zip(self.eval_dataset_q["id"], self.eval_dataset_q["index"]))

        self.eval_dataset_doc = self.eval_dataset_doc.add_column("index", range(len(self.eval_dataset_doc)))
        self.d_map_eval = dict(zip(self.eval_dataset_doc["id"], self.eval_dataset_doc["index"]))


        self.process_data()

    def process_data(self):
        for mod_name, mod in self.modules.items():
            if mod.module != None:
                print(f'Processing dataset for {mod_name}...')
                mod.eval_dataset_q = mod.eval_dataset_q.map(mod.module.tokenize, batched=True)
                mod.eval_dataset_doc = mod.eval_dataset_doc.map(mod.module.tokenize, batched=True) #fn_kwargs={"sentence_field": mod.eval_dataset_doc.sentence_field, "id_field": mod.eval_dataset_doc.id_field})
                mod.eval_dataset_q = mod.eval_dataset_q.remove_columns(['sentence'])
                mod.eval_dataset_doc = mod.eval_dataset_doc.remove_columns(['sentence'])
        # init datasets retriever

    def zero_shot_single_retrieval(self):
        out_retrieve = self.retriever.eval(return_embeddings=False)
        query_ids, doc_ids = out_retrieve['q_ids'], out_retrieve['doc_ids']
        assert len(doc_ids) == len(query_ids)
        q_idxs = [ self.q_map_eval[id_] for id_ in query_ids]
        queries = self.eval_dataset_q[q_idxs]['sentence']
        responses = list()
        instructions = list()
        for i in range(len(query_ids)):
            d_idxs = [ self.d_map_eval[id_] for id_ in doc_ids[i]]
            docs = self.eval_dataset_doc[d_idxs]
            instruction, response  = self.generator.eval(queries[i], docs)
            responses.append(response)
            instructions.append(instruction)
        return {
                "instructions": instructions,
                "responses": responses
            }
