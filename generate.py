from models.generators.dummy import Dummy
from models.generators.llama2 import Llama2

# Generate
class Generate():
    def __init__(self, kwargs):
        self.model_name = kwargs['model_name']
        self.eval_dataset_q = kwargs['eval_dataset_q']
        self.eval_dataset_doc = kwargs['eval_dataset_doc']
        # match model class
        if self.model_name == None:
            print('Not using a Generator!')
            self = None
        elif self.model_name == 'dummy':
            generator_class = Dummy
        elif self.model_name == 'meta-llama/Llama-2-7b-chat-hf':
            generator_class = Llama2
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented!")

        # instatiate model
        self.generator = generator_class(kwargs)

        # make idx to id mapping
        self.eval_dataset_q = self.eval_dataset_q.add_column("index", range(len(self.eval_dataset_q)))
        self.q_map_eval = dict(zip(self.eval_dataset_q["id"], self.eval_dataset_q["index"]))

        self.eval_dataset_docs = self.eval_dataset_docs.add_column("index", range(len(self.eval_dataset_docs)))
        self.d_map_eval = dict(zip(self.eval_dataset_docs["id"], self.eval_dataset_docs["index"]))



    def eval(self, query_ids, doc_ids):
        assert len(doc_ids) == len(query_ids)
        q_idxs = [ self.q_map[id_] for id_ in query_ids]
        queries = self.eval_dataset_q[q_idxs]['sentence']
        responses = list()
        instructions = list()
        for i in range(len(query_ids)):
            d_idxs = [ self.d_map[id_] for id_ in doc_ids[i]]
            docs = self.eval_dataset_doc[d_idxs]
            instruction, response  = self.generator.eval(queries[i], docs)
            responses.append(response)
        return {
                "instructions": instructions,
                "responses": responses
            }
